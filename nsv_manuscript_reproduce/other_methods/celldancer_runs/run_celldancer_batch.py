"""
Batch wrapper for running cellDancer on multiple datasets from a config YAML.

Core processing mirrors the cellDancer reproducibility scripts
(Reproducibility/cellDancer_script_notebook/*): scVelo preprocessing, then
    adata_to_df_with_embed -> velocity -> compute_cell_velocity -> pseudo_time,
generalized to read arbitrary .h5ad inputs and driven entirely by a config file.
This is the cellDancer analogue of run_unitvelo_batch.py / run_tfvelo_batch.py.

USAGE
-----
    python run_celldancer_batch.py --config celldancer_config_example.yaml

cellDancer must be importable (`import celldancer as cd`); install it into the
environment first (`pip install celldancer`).

CONFIG SCHEMA
-------------
    log_file: /abs/path/to/run.log        # global; detailed log for ALL datasets
    defaults:                              # optional; applied to every dataset
        label: clusters                    # .obs column with cell-type labels
                                           #   (-> cellDancer cell_type_para)
        embed_key: X_umap                  # adata.obsm key for the embedding
                                           #   (-> cellDancer embed_para)
        compute_umap: false                # if the embed_key is missing, compute
                                           #   a UMAP with scanpy instead of failing
        preprocess: true                   # run scVelo preprocessing in the wrapper
        # ---- cellDancer stage parameters (all optional; defaults shown) --------
        MIN_SHARED_COUNTS: 20              # preprocess: scv.pp.filter_genes
        N_TOP_GENES: 2000                  # preprocess: HVG selection
        N_PCS: 30                          # preprocess: PCA / neighbors
        N_NEIGHBORS: 30                    # preprocess: neighbors
        max_epoches: 200                   # cd.velocity
        permutation_ratio: 0.125           # cd.velocity
        n_jobs: 8                          # cd.velocity / cd.pseudo_time
        projection_neighbor_choice: gene   # cd.compute_cell_velocity
        expression_scale: power10          # cd.compute_cell_velocity
        projection_neighbor_size: 30       # cd.compute_cell_velocity
        speed_up: [100, 100]               # cd.compute_cell_velocity / cd.pseudo_time
        dt: 0.05                           # cd.pseudo_time
        n_repeats: 10                      # cd.pseudo_time
        grid: [30, 30]                     # cd.pseudo_time
        n_paths: 3                         # cd.pseudo_time
        plot_long_trajs: true              # cd.pseudo_time
        save_plot: true                    # save the pseudotime scatter PNG
    datasets:
        - name: my_dataset                 # label used in logs / output names
          h5ad_path: /abs/path/to/adata.h5ad
          output_dir: /abs/path/to/output_dir
          label: clusters                  # overrides defaults.label if given
          gene_col: gene_symbol            # OPTIONAL: adata.var column holding gene
                                           #   symbols. Omit if var_names are already
                                           #   gene symbols.
          # any key from `defaults` may be overridden here per-dataset

OUTPUTS (clearly labeled with "celldancer")
-------------------------------------------
    <output_dir>/<name>_celldancer_out.h5ad     # AnnData with cellDancer results:
        layers: cd_alpha, cd_beta, cd_gamma, cd_velocity_u, cd_velocity
        obs:    cd_time
    <output_dir>/<name>_celldancer_data/         # cellDancer's own working folder
        <name>.csv                               # adata_to_df_with_embed output
        <name>_cellDancer_out.csv                # full cellDancer df (all columns)
        cd_time_<name>.png                       # pseudotime scatter (if save_plot)
"""

import os
import sys


# --------------------------------------------------------------------------- #
# Thread settings -- set BEFORE numpy / torch / celldancer import.            #
#                                                                             #
# cellDancer parallelizes across genes with joblib (n_jobs workers). If each  #
# worker also spawns many BLAS threads you oversubscribe an HPC node. Pin the #
# inner thread counts to 1 so the joblib fan-out is the only parallelism.     #
# setdefault lets a SLURM script still override these.                        #
# --------------------------------------------------------------------------- #
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')


def _available_cpus():
    """CPUs actually allocated to this job (honors SLURM/cgroup on Linux)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)

import argparse
import logging
import threading
import traceback
from contextlib import contextmanager

import yaml
import numpy as np
import anndata as ad
import scanpy as sc
import scvelo as scv
import celldancer as cd


DEFAULTS = {
    'label': 'clusters',            # -> cellDancer cell_type_para
    'embed_key': 'X_umap',          # -> cellDancer embed_para (adata.obsm key)
    'compute_umap': False,          # compute UMAP if embed_key missing
    'preprocess': True,             # run scVelo preprocessing here
    # preprocessing params (mirror UniTVelo/scVelo defaults)
    'MIN_SHARED_COUNTS': 20,
    'N_TOP_GENES': 2000,
    'N_PCS': 30,
    'N_NEIGHBORS': 30,
    # cd.velocity -- this is the slow step: cellDancer trains one small neural
    # net PER GENE, and genes run in parallel batches of size n_jobs. Throughput
    # is ~linear in n_jobs, so n_jobs is the dominant speed lever.
    'max_epoches': 200,
    'permutation_ratio': 0.125,
    'n_jobs': None,                 # None/'auto' -> use all allocated CPU cores
    'check_val_every_n_epoch': 10,  # early-stopping check cadence
    'patience': 3,                  # early-stopping patience (smaller = stop sooner)
    'velocity_speed_up': True,      # fit on downsampled cells (big speed win)
    # cd.compute_cell_velocity
    'projection_neighbor_choice': 'gene',
    'expression_scale': 'power10',
    'projection_neighbor_size': 30,
    'speed_up': [100, 100],
    # cd.pseudo_time
    'dt': 0.05,
    'n_repeats': 10,
    'grid': [30, 30],
    'n_paths': 3,
    'plot_long_trajs': True,
    # Pseudotime is cellDancer's slow, fragile, OPTIONAL final step. Its recursive
    # intra-cluster time assignment (recur_cell_time_assignment_intracluster) both
    # (a) can exceed Python's default 1000-frame recursion limit -> RecursionError,
    # and (b) re-spawns a multiprocessing Pool at every recursion level, so on
    # large continuous trajectories (e.g. CD34 bone marrow) it can grind for hours.
    # We therefore run it in a killable subprocess (big C stack + raised recursion
    # limit) with a wall-clock budget, and can skip it entirely. Velocity + kinetic
    # results are saved regardless; only obs['cd_time'] depends on pseudotime.
    'compute_pseudotime': True,     # set False to skip the slow pseudotime step
    'pseudotime_timeout_min': 45,   # abort pseudotime after N minutes (None = no cap)
    'recursion_limit': 50000,
    'thread_stack_mb': 256,
    # output
    'save_plot': True,
}

# Keys that are per-dataset identity (not merged as tunable params).
_IDENTITY_KEYS = ('name', 'h5ad_path', 'output_dir', 'gene_col')


# --------------------------------------------------------------------------- #
# Logging + output-capture helpers                                            #
# --------------------------------------------------------------------------- #
def setup_logging(log_file):
    """Configure a logger that writes to `log_file` and to the console."""
    log_dir = os.path.dirname(os.path.abspath(log_file))
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger('celldancer_batch')
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

    cellDancer prints stage/progress messages to stdout -- these are the
    "detailed log" we keep. Its per-gene tqdm progress bars are on STDERR, which
    we deliberately leave on the terminal so they show live but never bloat the
    log file. (In a headless batch job, redirect stderr yourself if you don't
    want the bars in the .err file.)
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
    merged = dict(DEFAULTS)
    for k, v in (defaults or {}).items():
        merged[k] = v
    for k, v in dataset_cfg.items():
        if k in _IDENTITY_KEYS:
            continue
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


# --------------------------------------------------------------------------- #
# Data helpers                                                                #
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


def ensure_embedding(adata, p, name, logger):
    """Make sure the embedding cellDancer needs (embed_key) is present in obsm."""
    embed_key = p['embed_key']
    if embed_key in adata.obsm:
        return
    if p.get('compute_umap'):
        logger.info("[%s] obsm['%s'] missing; computing UMAP with scanpy.",
                    name, embed_key)
        if 'neighbors' not in adata.uns:
            sc.pp.neighbors(adata, n_neighbors=p['N_NEIGHBORS'], n_pcs=p['N_PCS'])
        sc.tl.umap(adata)
        if embed_key != 'X_umap' and 'X_umap' in adata.obsm:
            adata.obsm[embed_key] = adata.obsm['X_umap']
    else:
        raise KeyError(
            "embed_key '%s' not found in adata.obsm (available: %s). "
            "Provide a precomputed embedding or set compute_umap: true."
            % (embed_key, list(adata.obsm.keys())))


def map_df_to_adata(adata, df, name, logger):
    """
    Write cellDancer per-cell/per-gene results back onto an AnnData.

    cellDancer's output df has one row per (gene, cell), ordered gene-major with
    cells in adata order within each gene. Rather than assuming every input gene
    survived (cellDancer can drop genes it cannot fit), we recover the genes
    actually present in the df -- in first-appearance (var) order -- and subset a
    copy of `adata` to them, so the reshape is always exactly n_genes x n_cells.
    """
    gene_col = 'gene_name' if 'gene_name' in df.columns else 'gene_list'
    genes = list(dict.fromkeys(df[gene_col].tolist()))   # unique, order-preserving
    n_cells = adata.n_obs
    n_genes = len(genes)

    if len(df) != n_genes * n_cells:
        raise ValueError(
            "[%s] cellDancer df has %d rows, expected n_genes(%d) x n_cells(%d)=%d; "
            "cannot reshape unambiguously." % (name, len(df), n_genes, n_cells,
                                               n_genes * n_cells))

    missing = [g for g in genes if g not in set(adata.var_names)]
    if missing:
        raise KeyError("[%s] %d cellDancer genes not in adata.var_names (e.g. %s)."
                       % (name, len(missing), missing[:5]))

    out = adata[:, genes].copy()

    def col_grid(col):
        return df[col].to_numpy().reshape(n_genes, n_cells).T   # -> (cells, genes)

    out.layers['cd_alpha'] = col_grid('alpha')
    out.layers['cd_beta'] = col_grid('beta')
    out.layers['cd_gamma'] = col_grid('gamma')
    out.layers['cd_velocity_u'] = col_grid('unsplice_predict') - col_grid('unsplice')
    out.layers['cd_velocity'] = col_grid('splice_predict') - col_grid('splice')

    # cd_time only exists if the (optional) pseudotime step ran successfully.
    if 'pseudotime' in df.columns:
        out.obs['cd_time'] = \
            df['pseudotime'].to_numpy().reshape(n_genes, n_cells).T[:, 0]
        time_note = "with cd_time"
    else:
        time_note = "WITHOUT cd_time (pseudotime skipped/timed out)"

    logger.info("[%s] Mapped cellDancer results onto AnnData (%d cells x %d genes) %s.",
                name, out.n_obs, out.n_vars, time_note)
    return out


def _pseudotime_child(kwargs, recursion_limit, stack_mb, result_path):
    """
    Subprocess entry point: run cd.pseudo_time and pickle its result to disk.

    Runs inside a forked child process so a hung/pathologically-slow pseudotime
    can be killed with Process.terminate() (which also tears down the child's
    internal multiprocessing Pool). Within the child, the actual call runs on a
    thread with a large C stack + raised recursion limit, because cellDancer's
    recursive intra-cluster time assignment can otherwise overflow the default
    ~8 MB stack (segfault) or the 1000-frame recursion limit (RecursionError).
    """
    import pickle
    sys.setrecursionlimit(recursion_limit)
    try:
        threading.stack_size(stack_mb * 1024 * 1024)
    except (ValueError, RuntimeError):
        pass

    box = {}

    def _target():
        sys.setrecursionlimit(recursion_limit)
        box['result'] = cd.pseudo_time(**kwargs)

    t = threading.Thread(target=_target)
    t.start()
    t.join()
    with open(result_path, 'wb') as f:
        pickle.dump(box.get('result'), f)


def run_pseudotime(kwargs, recursion_limit, stack_mb, timeout_min, result_path,
                   logger, name):
    """
    Run cd.pseudo_time(**kwargs) in a killable subprocess with a wall-clock cap.

    Returns the pseudotime dataframe on success, or None if it exceeded
    `timeout_min` (in which case the subprocess is terminated) or the child
    failed. `timeout_min=None` means no cap. On fork platforms the large
    in-memory dataframe in `kwargs` is inherited copy-on-write (not re-pickled).
    """
    import multiprocessing as mp
    ctx = mp.get_context('fork')
    proc = ctx.Process(target=_pseudotime_child,
                       args=(kwargs, recursion_limit, stack_mb, result_path))
    proc.start()
    proc.join(timeout_min * 60 if timeout_min else None)

    if proc.is_alive():
        logger.warning("[%s] pseudo_time exceeded %s min; terminating and "
                       "continuing without cd_time.", name, timeout_min)
        proc.terminate()
        proc.join()
        return None
    if proc.exitcode != 0:
        logger.warning("[%s] pseudo_time subprocess exited with code %s; "
                       "continuing without cd_time.", name, proc.exitcode)
        return None

    import pickle
    try:
        with open(result_path, 'rb') as f:
            return pickle.load(f)
    except (OSError, EOFError, pickle.UnpicklingError) as e:
        logger.warning("[%s] Could not read pseudo_time result (%s); "
                       "continuing without cd_time.", name, e)
        return None


# --------------------------------------------------------------------------- #
# Core cellDancer step                                                        #
# --------------------------------------------------------------------------- #
def run_one(p, log_file, logger):
    """Run the cellDancer pipeline for a single dataset (dict `p`)."""
    name = p['name']
    out_dir = os.path.abspath(p['output_dir'])
    os.makedirs(out_dir, exist_ok=True)
    data_path = os.path.join(out_dir, "%s_celldancer_data" % name)
    os.makedirs(data_path, exist_ok=True)

    logger.info("[%s] Reading input: %s", name, p['h5ad_path'])
    adata = ad.read_h5ad(p['h5ad_path'])
    adata.obs_names_make_unique()

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

    if p['preprocess']:
        preprocess_scvelo(adata, p, name, logger)

    ensure_embedding(adata, p, name, logger)

    speed_up = tuple(p['speed_up'])
    grid = tuple(p['grid'])
    t_total = int(10 / p['dt'])

    # Resolve n_jobs: None/'auto'/<=0 -> all allocated CPU cores. cd.velocity
    # trains one net per gene and fans genes out across n_jobs workers, so this
    # is the single biggest speed lever. We pinned inner BLAS/OMP threads to 1,
    # so one worker == one core and n_jobs should match the cores you allocated.
    n_jobs = p['n_jobs']
    if n_jobs in (None, 'auto') or (isinstance(n_jobs, int) and n_jobs <= 0):
        n_jobs = _available_cpus()

    csv_path = os.path.join(data_path, "%s.csv" % name)
    logger.info("[%s] Running cellDancer (cell_type_para='%s', embed_para='%s', "
                "max_epoches=%s, n_jobs=%s). Progress bars print to the terminal, "
                "not the log.", name, p['label'], p['embed_key'],
                p['max_epoches'], n_jobs)

    with stdout_to_log(log_file):
        # 1) AnnData -> cellDancer dataframe (spliced/unspliced + embedding + celltype)
        df = cd.adata_to_df_with_embed(
            adata,
            cell_type_para=p['label'],
            embed_para=p['embed_key'],
            save_path=csv_path,
        )

        # 2) Estimate kinetic parameters + predicted un/spliced abundances
        df_loss, df = cd.velocity(
            df,
            max_epoches=p['max_epoches'],
            check_val_every_n_epoch=p['check_val_every_n_epoch'],
            patience=p['patience'],
            permutation_ratio=p['permutation_ratio'],
            speed_up=p['velocity_speed_up'],
            n_jobs=n_jobs,
            save_path=data_path,
        )

        # 3) Project cell-level velocities onto the embedding
        df = cd.compute_cell_velocity(
            cellDancer_df=df,
            projection_neighbor_choice=p['projection_neighbor_choice'],
            expression_scale=p['expression_scale'],
            projection_neighbor_size=p['projection_neighbor_size'],
            speed_up=speed_up,
        )

    # Persist velocity/kinetic results NOW, before the optional (slow) pseudotime
    # step -- so a hung/timed-out pseudotime never costs us the velocity output.
    df_out_csv = os.path.join(data_path, "%s_cellDancer_out.csv" % name)
    df.to_csv(df_out_csv, index=False)
    logger.info("[%s] cellDancer velocity dataframe -> %s", name, df_out_csv)

    # 4) Estimate pseudotime by diffusion (OPTIONAL). cellDancer's recursive
    #    intra-cluster time assignment can exceed Python's recursion limit and, on
    #    large continuous trajectories, re-spawn a Pool per recursion level and
    #    grind for hours. Run it in a killable subprocess (big stack + raised
    #    recursion limit) with a wall-clock cap; on skip/timeout we keep going and
    #    just omit obs['cd_time'].
    if p.get('compute_pseudotime', True):
        logger.info("[%s] pseudo_time (timeout=%s min, recursion_limit=%s, "
                    "thread_stack=%s MB)", name, p['pseudotime_timeout_min'],
                    p['recursion_limit'], p['thread_stack_mb'])
        pt_kwargs = dict(
            cellDancer_df=df,
            grid=grid,
            dt=p['dt'],
            t_total=t_total,
            n_repeats=p['n_repeats'],
            speed_up=speed_up,
            n_paths=p['n_paths'],
            plot_long_trajs=p['plot_long_trajs'],
            psrng_seeds_diffusion=[i for i in range(p['n_repeats'])],
            n_jobs=n_jobs,
        )
        pt_pickle = os.path.join(data_path, "%s_pseudotime.pkl" % name)
        with stdout_to_log(log_file):
            df_pt = run_pseudotime(pt_kwargs, p['recursion_limit'],
                                   p['thread_stack_mb'], p['pseudotime_timeout_min'],
                                   pt_pickle, logger, name)
        if df_pt is not None:
            df = df_pt
            df.to_csv(df_out_csv, index=False)   # overwrite with pseudotime added
            logger.info("[%s] pseudo_time complete; dataframe -> %s", name, df_out_csv)
        try:
            os.remove(pt_pickle)
        except OSError:
            pass
    else:
        logger.info("[%s] compute_pseudotime=False; skipping pseudotime.", name)

    if p.get('save_plot') and 'pseudotime' in df.columns:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import celldancer.cdplt as cdplt
            fig, ax = plt.subplots(figsize=(8, 6))
            cdplt.scatter_cell(ax, df, colors='pseudotime', alpha=0.5, velocity=True)
            ax.axis('off')
            png_path = os.path.join(data_path, "cd_time_%s.png" % name)
            fig.savefig(png_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            logger.info("[%s] Pseudotime scatter -> %s", name, png_path)
        except Exception as e:
            logger.warning("[%s] Could not render pseudotime plot: %s", name, e)

    # Map results back onto a gene-subset copy of the AnnData and save.
    out_path = os.path.join(out_dir, "%s_celldancer_out.h5ad" % name)
    adata_out = map_df_to_adata(adata, df, name, logger)
    adata_out.write(out_path)
    logger.info("[%s] Saved -> %s (%d cells x %d genes)",
                name, out_path, adata_out.n_obs, adata_out.n_vars)


def rebuild_one(p, logger):
    """
    Rebuild <name>_celldancer_out.h5ad from an EXISTING cellDancer CSV, without
    re-running velocity/pseudotime. Useful when a run was killed (e.g. during the
    slow pseudotime step) but the per-cell/per-gene CSV was already written.

    The CSV is located via, in order of preference:
      * p['csv_path']                                         (explicit override)
      * <output_dir>/<name>_celldancer_data/<name>_cellDancer_out.csv  (wrapper)
      * <output_dir>/<name>_celldancer_data/cellDancer_velocity_*/cellDancer_estimation.csv
                                                              (cellDancer's own output)
    Any of these works: obs['cd_time'] is written only if a 'pseudotime' column
    is present. Cells are aligned by sorting on ('gene_name','cellIndex') so the
    reshape is order-robust regardless of how the CSV rows were saved.
    """
    import glob
    import pandas as pd

    name = p['name']
    out_dir = os.path.abspath(p['output_dir'])
    data_path = os.path.join(out_dir, "%s_celldancer_data" % name)

    candidates = []
    if p.get('csv_path'):
        candidates.append(p['csv_path'])
    candidates.append(os.path.join(data_path, "%s_cellDancer_out.csv" % name))
    candidates.extend(sorted(glob.glob(os.path.join(
        data_path, "cellDancer_velocity_*", "cellDancer_estimation.csv"))))

    csv_path = next((c for c in candidates if c and os.path.exists(c)), None)
    if csv_path is None:
        raise FileNotFoundError(
            "[%s] No cellDancer CSV found. Looked for: %s" % (name, candidates))
    logger.info("[%s] Rebuilding from CSV: %s", name, csv_path)
    df = pd.read_csv(csv_path)

    required = {'gene_name', 'alpha', 'beta', 'gamma',
                'unsplice', 'splice', 'unsplice_predict', 'splice_predict'}
    missing_cols = required.difference(df.columns)
    if missing_cols:
        raise KeyError("[%s] CSV %s is missing required columns: %s"
                       % (name, csv_path, sorted(missing_cols)))

    # Order-robust alignment: gene-major, cells in original (cellIndex) order.
    if 'cellIndex' in df.columns:
        df = df.sort_values(['gene_name', 'cellIndex'], kind='stable')
        df = df.reset_index(drop=True)

    logger.info("[%s] Reading input: %s", name, p['h5ad_path'])
    adata = ad.read_h5ad(p['h5ad_path'])
    adata.obs_names_make_unique()
    if p.get('gene_col'):
        if p['gene_col'] not in adata.var.columns:
            raise KeyError("gene_col '%s' not in adata.var (available: %s)"
                           % (p['gene_col'], list(adata.var.columns)))
        adata.var_names = adata.var[p['gene_col']].astype(str).values
        adata.var_names_make_unique()

    out_path = os.path.join(out_dir, "%s_celldancer_out.h5ad" % name)
    os.makedirs(out_dir, exist_ok=True)
    adata_out = map_df_to_adata(adata, df, name, logger)
    adata_out.write(out_path)
    logger.info("[%s] Rebuilt -> %s (%d cells x %d genes)",
                name, out_path, adata_out.n_obs, adata_out.n_vars)


def run_dataset(dataset_cfg, defaults, log_file, logger, rebuild=False):
    """Run one dataset; never raises (errors logged so the batch continues)."""
    name = dataset_cfg.get('name', '<unnamed>')
    logger.info("=" * 70)
    logger.info("%s dataset: %s", "REBUILD" if rebuild else "START", name)
    try:
        p = merge_params(dataset_cfg, defaults)
        logger.info("[%s] Params: %s", name, p)
        if rebuild:
            rebuild_one(p, logger)
        else:
            run_one(p, log_file, logger)
        logger.info("FINISH dataset: %s -> SUCCESS", name)
        return True
    except Exception:
        logger.error("FAILED dataset: %s\n%s", name, traceback.format_exc())
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Batch-run cellDancer over datasets defined in a config YAML.")
    parser.add_argument('--config', required=True, help='Path to the config YAML.')
    parser.add_argument('--rebuild', action='store_true',
                        help="Skip cellDancer; rebuild <name>_celldancer_out.h5ad "
                             "from existing CSVs (velocity layers, plus cd_time if "
                             "a pseudotime column is present).")
    parser.add_argument('--only', default=None,
                        help="Comma-separated dataset name(s) to process "
                             "(default: all datasets in the config).")
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

    if cli.only:
        wanted = {n.strip() for n in cli.only.split(',') if n.strip()}
        datasets = [d for d in datasets if d.get('name') in wanted]
        if not datasets:
            sys.exit("--only matched no datasets in the config (wanted: %s)."
                     % sorted(wanted))

    logger = setup_logging(log_file)
    logger.info("#" * 70)
    logger.info("cellDancer batch %s started | config=%s | %d dataset(s)",
                "REBUILD" if cli.rebuild else "run", cli.config, len(datasets))

    results = {}
    for dataset_cfg in datasets:
        results[dataset_cfg.get('name', '<unnamed>')] = \
            run_dataset(dataset_cfg, defaults, log_file, logger, rebuild=cli.rebuild)

    n_ok = sum(1 for v in results.values() if v)
    logger.info("#" * 70)
    logger.info("Batch complete: %d/%d succeeded", n_ok, len(results))
    for name, ok in results.items():
        logger.info("  %-30s %s", name, "OK" if ok else "FAILED")


if __name__ == '__main__':
    main()
