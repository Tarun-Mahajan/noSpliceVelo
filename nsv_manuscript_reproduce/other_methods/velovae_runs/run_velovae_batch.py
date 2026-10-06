"""
Batch wrapper for running VeloVAE on multiple datasets from a config YAML.

Core processing logic is extracted from `notebooks/velovae_example.ipynb`
(preprocess -> create VAE -> train -> save), generalized to read arbitrary
.h5ad inputs and driven entirely by a config file.

USAGE
-----
    python run_velovae_batch.py --config velovae_config_example.yaml

Run from the VeloVAE repo root (the folder containing the `velovae/` package),
so `import velovae as vv` resolves.

CONFIG SCHEMA
-------------
    log_file: /abs/path/to/run.log        # global; detailed log for ALL datasets
    defaults:                              # optional; applied to every dataset
        n_gene: 2000                       # genes kept by vv.preprocess
        tmax: 20
        dim_z: 5
        device: auto                       # auto | cpu | cuda:0 | cuda:1 ...
        full_vb: false                     # VeloVAE Full VB variant
        discrete: false                    # discrete (Poisson/NB) variant
        init_method: steady                # steady | tprior
        cluster_key: clusters              # .obs key with cell types (if present)
        seed: 2022
        model_key: velovae                 # prefix for all saved outputs
        train_config: {}                   # dict of training hyperparameters
                                           # (batch_size, early_stop, n_neighbors...)
    datasets:
        - name: my_dataset                 # label used in logs / output names
          h5ad_path: /abs/path/to/adata.h5ad
          output_dir: /abs/path/to/output_dir
          gene_col: gene_symbol            # OPTIONAL: adata.var column holding gene
                                           # symbols. Omit if var_names are already
                                           # gene symbols.
          # any key from `defaults` may be overridden here per-dataset

OUTPUTS (all clearly labeled with "velovae" via model_key)
----------------------------------------------------------
    <output_dir>/<name>_velovae_out.h5ad          # AnnData with results
    <output_dir>/checkpoints/encoder_velovae.pt   # trained encoder
    <output_dir>/checkpoints/decoder_velovae.pt   # trained decoder
    <output_dir>/figures/                         # (only if plotting enabled)
Inside the .h5ad, all model quantities are prefixed with the model_key, e.g.
`velovae_velocity`, `velovae_time`, `velovae_alpha`.
"""

import os
import sys


# --------------------------------------------------------------------------- #
# Thread settings -- must be set BEFORE numpy / torch / velovae import.        #
#                                                                             #
# Unlike TFvelo (many joblib workers -> 1 thread each), VeloVAE is a single    #
# torch process whose CPU parallelism IS its thread pool. On HPC, os.cpu_count #
# sees the whole node, so torch/BLAS would spawn threads for every core on the #
# machine, not the slice allocated to your job -> oversubscription. Pin the    #
# thread count to the CPUs actually allocated to this process instead.         #
# --------------------------------------------------------------------------- #
def _available_cpus():
    try:
        return max(1, len(os.sched_getaffinity(0)))      # Linux: honors SLURM/cgroup
    except AttributeError:
        return max(1, os.cpu_count() or 1)               # non-Linux fallback


_NCPU = str(_available_cpus())
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, _NCPU)

import argparse
import logging
import traceback
from contextlib import contextmanager

import yaml
import numpy as np
import anndata as ad
import scanpy as sc
import scvelo as scv
import torch
import velovae as vv


# --------------------------------------------------------------------------- #
# Default parameters (mirror velovae_example.ipynb)                            #
# --------------------------------------------------------------------------- #
PARAM_DEFAULTS = {
    'n_gene': 2000,
    'min_shared_counts': 20,
    'n_pcs': 30,
    'n_neighbors': 30,
    'tmax': 20,
    'dim_z': 5,
    'device': 'auto',
    'full_vb': False,
    'discrete': False,
    'init_method': 'steady',
    'cluster_key': 'clusters',
    'seed': 2022,
    'model_key': 'velovae',
    'hidden_size': (500, 250, 250, 500),
    'rate_prior': {'alpha': (0.0, 1.0), 'beta': (0.0, 0.5), 'gamma': (0.0, 0.5)},
    'train_config': {},
}


# --------------------------------------------------------------------------- #
# Logging + output-capture helpers (same strategy as the TFvelo wrapper)      #
# --------------------------------------------------------------------------- #
def setup_logging(log_file):
    """Configure a logger that writes to `log_file` and to the console."""
    log_dir = os.path.dirname(os.path.abspath(log_file))
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger('velovae_batch')
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
def capture_to_log(log_file):
    """
    Send OS-level stdout AND stderr into the log file (via os.dup2), capturing
    Python prints, C-extension output, and scanpy/scvelo messages during the
    preprocessing phase.
    """
    f = open(log_file, 'a')
    saved_out, saved_err = os.dup(1), os.dup(2)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(f.fileno(), 1)
        os.dup2(f.fileno(), 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        f.close()


@contextmanager
def stdout_to_terminal():
    """
    Route OS-level stdout (fd 1) to the controlling terminal (/dev/tty), so the
    verbose training progress (per-epoch 'Epoch N: Train ELBO = ...' prints and
    tqdm bars) shows LIVE on an interactive terminal but is NOT written to a
    redirected stdout -- e.g. an HPC job's captured .out file or a `> log`.

    If there is no controlling terminal (a pure batch job), that output is sent
    to /dev/null instead. stderr (fd 2) is left untouched, so tqdm bars and any
    warnings/errors keep flowing for live monitoring.
    """
    try:
        tgt = open('/dev/tty', 'w')
    except OSError:
        tgt = open(os.devnull, 'w')
    saved_out = os.dup(1)
    try:
        sys.stdout.flush()
        os.dup2(tgt.fileno(), 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved_out, 1)
        os.close(saved_out)
        tgt.close()


# --------------------------------------------------------------------------- #
# Config handling                                                             #
# --------------------------------------------------------------------------- #
def merge_params(dataset_cfg, defaults):
    """Merge global defaults + per-dataset overrides into a plain dict."""
    merged = dict(PARAM_DEFAULTS)
    merged.update(defaults or {})
    merged.update({k: v for k, v in dataset_cfg.items()
                   if k not in ('name', 'h5ad_path', 'output_dir', 'gene_col')})

    name = dataset_cfg.get('name')
    if not name:
        raise ValueError("Each dataset must have a 'name'.")
    if not dataset_cfg.get('h5ad_path'):
        raise ValueError("Dataset '%s' is missing 'h5ad_path'." % name)
    if not dataset_cfg.get('output_dir'):
        raise ValueError("Dataset '%s' is missing 'output_dir'." % name)

    merged['name'] = name
    merged['h5ad_path'] = dataset_cfg['h5ad_path']
    merged['output_dir'] = dataset_cfg['output_dir']
    merged['gene_col'] = dataset_cfg.get('gene_col')
    return merged


def resolve_device(device):
    """Resolve 'auto' to cuda:0 if a GPU is available, else cpu."""
    if device in (None, 'auto'):
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device


# --------------------------------------------------------------------------- #
# Core VeloVAE steps (extracted / generalized from velovae_example.ipynb)     #
# --------------------------------------------------------------------------- #
def set_total_X(adata, name, logger):
    """
    Set adata.X to the total-count matrix, preferring existing precomputed
    layers: 'matrix' -> 'total' -> ('spliced' + 'unspliced').

    Gives VeloVAE's preprocessing the same raw total-count .X convention used
    for the other methods. A copy is used so adata.X does not alias a layer.
    If none of the sources exist, X is left unchanged.
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


def preprocess_scvelo(adata, params, name, logger):
    """
    scVelo preprocessing done here for VeloVAE.

    Performs individual scVelo/Scanpy calls (filter_genes -> normalize_per_cell ->
    log1p -> highly_variable_genes -> pca -> neighbors -> moments), matching
    the preprocessing style from UniTVelo's batch wrapper.
    """
    min_shared = params.get('min_shared_counts', 20)
    n_top = params.get('n_gene', params.get('n_top_genes', 2000))
    n_pcs = params.get('n_pcs', 30)
    n_neighbors = params.get('n_neighbors', 30)
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
    """Run the full VeloVAE pipeline for a single dataset (dict `p`)."""
    name = p['name']
    out_dir = p['output_dir']
    ckpt_dir = os.path.join(out_dir, 'checkpoints')
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(out_dir, exist_ok=True)

    # ------- Load input -------
    logger.info("[%s] Reading input: %s", name, p['h5ad_path'])
    adata = ad.read_h5ad(p['h5ad_path'])

    if p.get('gene_col'):
        if p['gene_col'] not in adata.var.columns:
            raise KeyError("gene_col '%s' not in adata.var (available: %s)"
                           % (p['gene_col'], list(adata.var.columns)))
        logger.info("[%s] Setting var_names from adata.var['%s']", name, p['gene_col'])
        adata.var_names = adata.var[p['gene_col']].astype(str).values
        adata.var_names_make_unique()

    # Set adata.X to the total counts (matrix -> total -> spliced+unspliced).
    set_total_X(adata, name, logger)

    # ------- Preprocess (detailed output captured to the log) -------
    logger.info("[%s] Preprocessing using preprocess_scvelo", name)
    with capture_to_log(log_file):
        preprocess_scvelo(adata, p, name, logger)
    logger.info("[%s] After preprocess: %d cells x %d genes", name, adata.n_obs, adata.n_vars)

    # ------- Build + train model (progress -> terminal, not log) -------
    device = resolve_device(p['device'])
    if device == 'cpu':
        torch.set_num_threads(_available_cpus())
    torch.manual_seed(p['seed'])
    np.random.seed(p['seed'])

    variant = ('Full VB' if p['full_vb'] else 'VeloVAE')
    variant += ' (discrete)' if p['discrete'] else ''
    logger.info("[%s] Building %s | device=%s tmax=%s dim_z=%s init=%s",
                name, variant, device, p['tmax'], p['dim_z'], p['init_method'])

    vae_kwargs = dict(
        tmax=p['tmax'], dim_z=p['dim_z'], device=device,
        hidden_size=tuple(p['hidden_size']), full_vb=p['full_vb'],
        discrete=p['discrete'], init_method=p['init_method'],
    )
    if p['full_vb']:
        vae_kwargs['rate_prior'] = p['rate_prior']

    logger.info("[%s] Training (progress prints live to terminal, not the log)...", name)
    with stdout_to_terminal():
        vae = vv.VAE(adata, **vae_kwargs)
        vae.train(adata, config=p['train_config'], plot=False,
                  cluster_key=p['cluster_key'], figure_path=fig_dir)

        # ------- Save (outputs labeled with model_key, e.g. "velovae") -------
        key = p['model_key']
        out_h5ad = "%s_velovae_out.h5ad" % name
        vae.save_model(ckpt_dir, "encoder_%s" % key, "decoder_%s" % key)
        vae.save_anndata(adata, key, out_dir, file_name=out_h5ad)

    logger.info("[%s] Saved -> %s (key='%s') + models in %s",
                name, os.path.join(out_dir, out_h5ad), key, ckpt_dir)


def run_dataset(dataset_cfg, defaults, log_file, logger):
    """Run one dataset; never raises (errors are logged so the batch continues)."""
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
        description="Batch-run VeloVAE over datasets defined in a config YAML.")
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
    logger.info("VeloVAE batch run started | config=%s | %d dataset(s) | CPUs=%s | cuda=%s",
                cli.config, len(datasets), _available_cpus(), torch.cuda.is_available())

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
