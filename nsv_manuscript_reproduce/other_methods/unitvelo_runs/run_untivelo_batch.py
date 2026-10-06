"""
Batch wrapper for running UniTVelo on multiple datasets from a config YAML.

Core processing is UniTVelo's integrated `utv.run_model(...)`, driven entirely
by a config file (see README "RNA Velocity on New Dataset").

USAGE
-----
    python run_unitvelo_batch.py --config unitvelo_config_example.yaml

Run from the UniTVelo repo root (the folder containing the `unitvelo/` package)
so `import unitvelo as utv` resolves.

CONFIG SCHEMA
-------------
    log_file: /abs/path/to/run.log        # global; detailed log for ALL datasets
    defaults:                              # optional; applied to every dataset
        label: clusters                    # .obs column with cell-type labels
                                           # (REQUIRED by UniTVelo)
        normalize: true                    # run scVelo filter_and_normalize+moments
        config:                            # UniTVelo Configuration overrides
            FIT_OPTION: '1'                # '1' unified-time (default), '2' independent
            GPU: 0                         # GPU card index; -1 = CPU
            R2_ADJUST: true
            IROOT: null
            N_TOP_GENES: 2000
            # ...any UPPERCASE attribute from unitvelo/config.py
    datasets:
        - name: my_dataset                 # label used in logs / output names
          h5ad_path: /abs/path/to/adata.h5ad
          output_dir: /abs/path/to/output_dir
          label: clusters                  # overrides defaults.label if given
          gene_col: gene_symbol            # OPTIONAL: adata.var column holding gene
                                           # symbols. Omit if var_names are already
                                           # gene symbols.
          # `config`, `normalize`, `label` may be overridden per-dataset

OUTPUTS (clearly labeled with "unitvelo")
-----------------------------------------
    <output_dir>/<name>_unitvelo_out.h5ad          # final AnnData with velocity
    <output_dir>/<name>_unitvelo_input/            # UniTVelo's own temp folder
        temp_<FIT_OPTION>.h5ad, logging.txt
"""

import os
import sys


# --------------------------------------------------------------------------- #
# Thread settings -- must be set BEFORE tensorflow / unitvelo import.          #
#                                                                             #
# UniTVelo runs on TensorFlow 2 (single process). On CPU, TF/BLAS default to   #
# a thread per core on the whole node, not the slice allocated to your job ->  #
# oversubscription on HPC. Pin thread counts to the CPUs actually allocated.   #
# (On GPU this is largely irrelevant; harmless to set.)                        #
# --------------------------------------------------------------------------- #
def _available_cpus():
    try:
        return max(1, len(os.sched_getaffinity(0)))      # Linux: honors SLURM/cgroup
    except AttributeError:
        return max(1, os.cpu_count() or 1)               # non-Linux fallback


_NCPU = str(_available_cpus())
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS'):
    os.environ.setdefault(_v, _NCPU)
os.environ.setdefault('TF_NUM_INTEROP_THREADS', str(min(2, _available_cpus())))

# UniTVelo uses the Keras-2 legacy optimizer API (tf.keras.optimizers.legacy),
# which TF 2.16+ (bundling Keras 3) no longer provides. Routing tf.keras back to
# Keras 2 restores it. REQUIRES the `tf_keras` package installed in the env:
#     pip install tf_keras
# Must be set before tensorflow / unitvelo are imported (below).
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')

import argparse
import logging
import traceback
from contextlib import contextmanager

import yaml
import anndata as ad
import scanpy as sc
import scvelo as scv
import unitvelo as utv


DEFAULTS = {
    'label': 'clusters',
    'preprocess': True,    # do scVelo preprocessing here + call run_model(normalize=False)
    'normalize': True,     # only used when preprocess=False (UniTVelo's own normalize)
    'config': {},          # UniTVelo Configuration attribute overrides
}


# --------------------------------------------------------------------------- #
# Logging + output-capture helpers                                            #
# --------------------------------------------------------------------------- #
def setup_logging(log_file):
    """Configure a logger that writes to `log_file` and to the console."""
    log_dir = os.path.dirname(os.path.abspath(log_file))
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger('unitvelo_batch')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter('%(asctime)s | %(levelname)-7s | %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file, mode='a')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


@contextmanager
def stdout_to_log(log_file):
    """
    Redirect OS-level stdout (fd 1) into the log file, leaving stderr (fd 2)
    untouched.

    UniTVelo prints its informative messages (config summary, gene selection,
    normalization, timing) to stdout -- these are the "detailed log" we keep.
    Its 12000-iteration loss progress bar is a tqdm bar on STDERR, which we
    deliberately leave on the terminal so it shows live but never bloats the
    log file. (In a headless batch job, redirect stderr yourself if you don't
    want the bar in the .err file.)
    """
    f = open(log_file, 'a')
    saved_out = os.dup(1)
    try:
        sys.stdout.flush()
        os.dup2(f.fileno(), 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved_out, 1)
        os.close(saved_out)
        f.close()


# --------------------------------------------------------------------------- #
# Config handling                                                             #
# --------------------------------------------------------------------------- #
def merge_params(dataset_cfg, defaults):
    """Merge global defaults + per-dataset overrides into a plain dict."""
    merged = {k: (dict(v) if isinstance(v, dict) else v)
              for k, v in DEFAULTS.items()}
    for k, v in (defaults or {}).items():
        if k == 'config' and isinstance(v, dict):
            merged['config'] = {**merged['config'], **v}
        else:
            merged[k] = v
    for k, v in dataset_cfg.items():
        if k in ('name', 'h5ad_path', 'output_dir', 'gene_col'):
            continue
        if k == 'config' and isinstance(v, dict):
            merged['config'] = {**merged['config'], **v}
        else:
            merged[k] = v

    name = dataset_cfg.get('name')
    if not name:
        raise ValueError("Each dataset must have a 'name'.")
    if not dataset_cfg.get('h5ad_path'):
        raise ValueError("Dataset '%s' is missing 'h5ad_path'." % name)
    if not dataset_cfg.get('output_dir'):
        raise ValueError("Dataset '%s' is missing 'output_dir'." % name)
    if not merged.get('label'):
        raise ValueError("Dataset '%s' has no 'label' (cell-type .obs column)." % name)

    merged['name'] = name
    merged['h5ad_path'] = dataset_cfg['h5ad_path']
    merged['output_dir'] = dataset_cfg['output_dir']
    merged['gene_col'] = dataset_cfg.get('gene_col')
    return merged


def build_config(overrides, logger, name):
    """Create a UniTVelo Configuration and apply UPPERCASE attribute overrides."""
    velo = utv.config.Configuration()
    for k, v in (overrides or {}).items():
        if not hasattr(velo, k):
            logger.warning("[%s] Unknown UniTVelo config key '%s' (ignored). "
                           "Expected an UPPERCASE attribute from unitvelo/config.py.",
                           name, k)
            continue
        setattr(velo, k, v)
    return velo


# --------------------------------------------------------------------------- #
# Core UniTVelo step                                                          #
# --------------------------------------------------------------------------- #
def set_total_X(adata, name, logger):
    """
    Set adata.X to the total-count matrix, preferring existing precomputed
    layers: 'matrix' -> 'total' -> ('spliced' + 'unspliced').

    A copy is used so adata.X does not alias a layer (downstream normalization
    modifies X in place). If none of the sources exist, X is left unchanged.
    """
    if 'matrix' in adata.layers:
        adata.X = adata.layers['matrix'].copy()
        src = 'matrix'
    elif 'total' in adata.layers:
        adata.X = adata.layers['total'].copy()
        src = 'total'
    elif 'spliced' in adata.layers and 'unspliced' in adata.layers:
        adata.X = adata.layers['spliced'] + adata.layers['unspliced']
        src = 'spliced + unspliced'
    else:
        logger.warning("[%s] No 'matrix'/'total' layer and no spliced+unspliced "
                       "layers found; leaving adata.X unchanged.", name)
        return
    logger.info("[%s] Set adata.X from layer(s): %s", name, src)


def preprocess_scvelo(adata, cfg, name, logger):
    """
    scVelo preprocessing done here (so UniTVelo can be called with normalize=False).

    This mirrors what UniTVelo's normalize=True path does internally
    (filter_genes -> normalize_per_cell -> HVG selection -> log1p -> moments),
    as individual calls, which avoids the newer-scvelo bug where
    filter_and_normalize forwards `n_top_genes` into normalize_per_cell.

    HVG selection uses sc.pp.highly_variable_genes (scvelo's filter_genes_dispersion
    was removed in newer versions). Because scanpy's default 'seurat' flavor expects
    log-normalized data, log1p is applied BEFORE HVG selection here (scvelo's old
    filter_genes_dispersion log-transformed internally, so this keeps the same
    dispersion-on-log-data behavior).

    PCA and neighbors are computed explicitly with scanpy before scv.pp.moments,
    because scvelo>=0.4.0 deprecated automatic neighbor calculation inside moments.
    moments is then called with n_pcs/n_neighbors=None so it reuses that graph
    instead of recomputing (which would re-trigger the deprecation warning).
    Parameters mirror UniTVelo's config defaults.
    """
    min_shared = cfg.get('MIN_SHARED_COUNTS', 20)
    n_top = cfg.get('N_TOP_GENES', 2000)
    n_pcs = cfg.get('N_PCS', 30)
    n_neighbors = cfg.get('N_NEIGHBORS', 30)
    logger.info("[%s] scVelo preprocessing (min_shared_counts=%s, n_top_genes=%s, "
                "n_pcs=%s, n_neighbors=%s)", name, min_shared, n_top, n_pcs, n_neighbors)
    scv.pp.filter_genes(adata, min_shared_counts=min_shared)
    scv.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top, subset=True)
    sc.pp.pca(adata, n_comps=n_pcs)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    scv.pp.moments(adata, n_pcs=None, n_neighbors=None)


def run_one(p, log_file, logger):
    """Run the UniTVelo pipeline for a single dataset (dict `p`)."""
    name = p['name']
    out_dir = os.path.abspath(p['output_dir'])   # absolute: survives the chdir below
    os.makedirs(out_dir, exist_ok=True)

    logger.info("[%s] Reading input: %s", name, p['h5ad_path'])
    adata = ad.read_h5ad(p['h5ad_path'])

    if p.get('gene_col'):
        if p['gene_col'] not in adata.var.columns:
            raise KeyError("gene_col '%s' not in adata.var (available: %s)"
                           % (p['gene_col'], list(adata.var.columns)))
        logger.info("[%s] Setting var_names from adata.var['%s']", name, p['gene_col'])
        adata.var_names = adata.var[p['gene_col']].astype(str).values
        adata.var_names_make_unique()

    if p['label'] not in adata.obs.columns:
        raise KeyError("label '%s' not found in adata.obs (available: %s)"
                       % (p['label'], list(adata.obs.columns)))

    # Set adata.X to the total counts (matrix -> total -> spliced+unspliced).
    set_total_X(adata, name, logger)

    velo = build_config(p['config'], logger, name)
    # When we preprocess here, tell UniTVelo not to normalize again.
    run_normalize = False if p['preprocess'] else p['normalize']
    logger.info("[%s] Running utv.run_model (label='%s', FIT_OPTION=%s, GPU=%s, "
                "preprocess_here=%s, run_model normalize=%s). Loss progress bar "
                "prints to the terminal, not the log.",
                name, p['label'], getattr(velo, 'FIT_OPTION', '?'),
                getattr(velo, 'GPU', '?'), p['preprocess'], run_normalize)

    # Pass the in-memory AnnData object (NOT a path). UniTVelo's run_model calls
    # scv.read() only when given a path string -- and recent scvelo versions have
    # dropped the top-level `scv.read` alias (AttributeError). Handing it the
    # already-loaded object skips that broken call entirely. When given an object,
    # UniTVelo writes its own temp files under `<cwd>/res/`, so we chdir into a
    # per-dataset temp folder to keep those artifacts isolated and clearly named.
    out_path = os.path.join(out_dir, "%s_unitvelo_out.h5ad" % name)
    tmp_dir = os.path.join(out_dir, "%s_unitvelo_tmp" % name)
    os.makedirs(tmp_dir, exist_ok=True)
    prev_cwd = os.getcwd()
    with stdout_to_log(log_file):
        try:
            os.chdir(tmp_dir)   # -> UniTVelo temp lands in <tmp_dir>/res/
            if p['preprocess']:
                preprocess_scvelo(adata, p['config'], name, logger)
            adata_out = utv.run_model(adata, p['label'],
                                      config_file=velo, normalize=run_normalize)
        finally:
            os.chdir(prev_cwd)

    adata_out.write(out_path)
    logger.info("[%s] UniTVelo temp artifacts under: %s", name, os.path.join(tmp_dir, 'res'))
    logger.info("[%s] Saved -> %s (%d cells x %d genes)",
                name, out_path, adata_out.n_obs, adata_out.n_vars)


def run_dataset(dataset_cfg, defaults, log_file, logger):
    """Run one dataset; never raises (errors logged so the batch continues)."""
    name = dataset_cfg.get('name', '<unnamed>')
    logger.info("=" * 70)
    logger.info("START dataset: %s", name)
    try:
        p = merge_params(dataset_cfg, defaults)
        logger.info("[%s] Params: %s", name, p)
        run_one(p, log_file, logger)
        logger.info("FINISH dataset: %s -> SUCCESS", name)
        return True
    except Exception:
        logger.error("FAILED dataset: %s\n%s", name, traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch-run UniTVelo over datasets defined in a config YAML.")
    parser.add_argument('--config', required=True, help='Path to the config YAML.')
    cli = parser.parse_args()

    with open(cli.config) as f:
        cfg = yaml.safe_load(f)

    log_file = cfg.get('log_file')
    if not log_file:
        sys.exit("Config must specify a global 'log_file' path.")
    defaults = cfg.get('defaults', {})
    datasets = cfg.get('datasets', [])
    if not datasets:
        sys.exit("Config must specify at least one dataset under 'datasets'.")

    logger = setup_logging(log_file)
    logger.info("#" * 70)
    logger.info("UniTVelo batch run started | config=%s | %d dataset(s) | CPUs=%s",
                cli.config, len(datasets), _available_cpus())

    results = {}
    for dataset_cfg in datasets:
        results[dataset_cfg.get('name', '<unnamed>')] = \
            run_dataset(dataset_cfg, defaults, log_file, logger)

    n_ok = sum(1 for v in results.values() if v)
    logger.info("#" * 70)
    logger.info("Batch complete: %d/%d succeeded", n_ok, len(results))
    for name, ok in results.items():
        logger.info("  %-30s %s", name, "OK" if ok else "FAILED")


if __name__ == '__main__':
    main()
