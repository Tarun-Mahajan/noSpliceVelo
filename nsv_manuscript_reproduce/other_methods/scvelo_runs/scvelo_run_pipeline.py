# load required packages
import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pandas as pd
import scvelo as scv
import scanpy as sc
import seaborn as sns
import gc
import importlib


def _set_total_X(adata):
    """Set adata.X to the total-count matrix.

    Uses layers['matrix'] or layers['total'] if present; otherwise falls back
    to layers['spliced'] + layers['unspliced']. Raises if none are available.
    """
    if 'matrix' in adata.layers:
        adata.X = adata.layers['matrix'].copy()
    elif 'total' in adata.layers:
        adata.X = adata.layers['total'].copy()
    elif 'spliced' in adata.layers and 'unspliced' in adata.layers:
        adata.X = adata.layers['spliced'] + adata.layers['unspliced']
    else:
        raise KeyError(
            "adata must have layers['matrix'], layers['total'], or both "
            "layers['spliced'] and layers['unspliced'] to set .X"
        )
    return adata


def filter_normalize_adata(adata, n_top_genes=2000, min_shared_counts=30):
    """Filter, normalize, select HVGs and compute moments for scVelo.

    Returns the processed (gene-subset) AnnData.
    """
    adata = _set_total_X(adata)
    # filter_and_normalize does not apply log1p to .X for scVelo 0.3.4, so we apply log1p.
    scv.pp.filter_and_normalize(adata, min_shared_counts=min_shared_counts)
    sc.pp.log1p(adata)

    # HVG selection with scanpy
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata = adata[:, adata.var.highly_variable].copy()

    # moments for velocity (uses spliced/unspliced layers)
    sc.pp.pca(adata, n_comps=30)
    sc.pp.neighbors(adata, n_neighbors=30, n_pcs=30)
    scv.pp.moments(adata)

    return adata


def run_scvelo(
    adata,
    dir_path,
    continue_from_prev=False,
    n_top_genes=2000,
    min_shared_counts=30,
    n_jobs=-1,
    mode='dynamical',
):
    """Run the scVelo pipeline: preprocess -> velocity -> velocity graph -> save.

    Parameters
    ----------
    adata :
        Input AnnData with spliced/unspliced layers.
    dir_path :
        Directory where the result AnnData is saved / loaded.
    continue_from_prev :
        If True and a saved result ('adata_scvelo.h5ad') already exists in
        `dir_path`, load and return it instead of recomputing. scVelo has no
        resumable training state, so this is all-or-nothing: it skips the
        expensive `recover_dynamics` when a completed result is present.
    n_top_genes :
        Number of highly-variable genes to keep.
    min_shared_counts :
        `min_shared_counts` for scvelo filter_and_normalize.
    n_jobs :
        Parallel jobs for `recover_dynamics` and `velocity_graph`.
    mode :
        scVelo velocity mode: 'dynamical' (default; runs `recover_dynamics`
        first), 'stochastic', or 'deterministic'.

    Returns
    -------
    The processed AnnData with velocity fields populated.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    out_path = os.path.join(dir_path, 'adata_scvelo.h5ad')

    if continue_from_prev and os.path.exists(out_path):
        print(f"[continue_from_prev] Found existing result at {out_path}; "
              f"loading it and skipping recomputation")
        return ad.read_h5ad(out_path)

    adata = filter_normalize_adata(
        adata.copy(), n_top_genes=n_top_genes, min_shared_counts=min_shared_counts
    )

    if mode == 'dynamical':
        # dynamical mode requires the full splicing kinetics to be recovered
        scv.tl.recover_dynamics(adata, n_jobs=n_jobs)

    scv.tl.velocity(adata, mode=mode)
    scv.tl.velocity_graph(adata, n_jobs=n_jobs)

    adata.write_h5ad(out_path)
    return adata


# =============================================================================
# Command-line runner: drive one or more datasets from a YAML config file
# =============================================================================

# Optional pipeline parameters accepted per dataset (and in the global
# `defaults:` block). Anything else is ignored with a warning, except the
# required `adata_path` / `dir_path` and the cosmetic `name`.
_PIPELINE_PARAMS = (
    "continue_from_prev",
    "n_top_genes",
    "min_shared_counts",
    "n_jobs",
    "mode",
)


class _Tee:
    """Write everything to several streams at once (console + log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass


# Matches tqdm / progress lines, e.g.
#   "  0%| | 12/2000 [00:03<08:01, 2.14it/s]"  or  "Epoch 12/300 ..."
import re as _re
_PROGRESS_RE = _re.compile(
    r"\d{1,3}%\s*\|"     # "  0%|"
    r"|it/s[,\]]"        # "2.14it/s]" / "it/s,"
    r"|s/it[,\]]"        # "s/it]" / "s/it,"
    r"|^Epoch\s+\d+/\d+" # "Epoch 12/300"
)


class _FilteredFile:
    """File wrapper that drops in-place progress-bar updates from the log."""

    def __init__(self, fh):
        self._fh = fh
        self._buf = ""

    def write(self, data):
        self._buf += data
        while True:
            nl = self._buf.find("\n")
            if nl == -1:
                cr = self._buf.rfind("\r")
                if cr != -1:
                    self._buf = self._buf[cr + 1:]
                break
            line = self._buf[:nl]
            self._buf = self._buf[nl + 1:]
            eff = line.rsplit("\r", 1)[-1]
            if eff.strip() and _PROGRESS_RE.search(eff):
                continue  # drop progress-bar line
            self._fh.write(eff + "\n")
        self._fh.flush()

    def flush(self):
        try:
            self._fh.flush()
        except Exception:
            pass

    def close(self):
        rem = self._buf.rsplit("\r", 1)[-1]
        if rem.strip() and not _PROGRESS_RE.search(rem):
            self._fh.write(rem)
        self._buf = ""
        self._fh.flush()
        self._fh.close()


def _timestamp():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg):
    """Timestamped line; goes to console and (via the Tee) the log file."""
    print(f"[{_timestamp()}] {msg}", flush=True)


def _load_adata(adata_path):
    """Read an AnnData from disk (.h5ad preferred, falls back to scanpy)."""
    if not os.path.exists(adata_path):
        raise FileNotFoundError(f"adata_path does not exist: {adata_path}")
    if adata_path.endswith(".h5ad"):
        return ad.read_h5ad(adata_path)
    return sc.read(adata_path)


def _normalize_config(cfg):
    """Return (global_dict, [dataset_dict, ...]) from a parsed YAML config."""
    if isinstance(cfg, list):
        return {}, cfg
    if isinstance(cfg, dict):
        datasets = cfg.get("datasets")
        if datasets is None:
            raise ValueError("Config dict must contain a 'datasets:' list")
        if not isinstance(datasets, list):
            raise ValueError("'datasets' must be a list")
        return cfg, datasets
    raise ValueError("Top-level YAML must be a list or a mapping")


def _resolve_params(defaults, entry):
    """Merge global defaults with a dataset entry -> kwargs for run_scvelo."""
    merged = dict(defaults or {})
    for k, v in entry.items():
        if k in ("name", "adata_path", "dir_path"):
            continue
        if k in _PIPELINE_PARAMS:
            merged[k] = v
        else:
            _log(f"  WARNING: ignoring unknown parameter '{k}'")
    return {k: merged[k] for k in merged if k in _PIPELINE_PARAMS}


def run_from_config(config_path):
    """Run the scVelo pipeline for every dataset in a YAML config, in sequence."""
    try:
        import yaml
    except ImportError as e:
        raise SystemExit(
            "PyYAML is required to read the config file. "
            "Install it with `pip install pyyaml`."
        ) from e

    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    global_cfg, datasets = _normalize_config(cfg)
    defaults = global_cfg.get("defaults", {}) if isinstance(global_cfg, dict) else {}

    # ---- set up the detailed log (console + file) ----
    from datetime import datetime
    default_log = f"scvelo_pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_file = global_cfg.get("log_file", default_log) if isinstance(global_cfg, dict) else default_log
    log_dir = os.path.dirname(os.path.abspath(log_file))
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_fh = _FilteredFile(open(log_file, "a"))
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    sys.stderr = _Tee(sys.__stderr__, log_fh)

    import time as _time
    import traceback as _traceback

    _log(f"=== scVelo batch run started ===")
    _log(f"config file : {os.path.abspath(config_path)}")
    _log(f"log file    : {os.path.abspath(log_file)}")
    _log(f"datasets    : {len(datasets)}")
    if defaults:
        _log(f"defaults    : {defaults}")

    summary = []
    for i, entry in enumerate(datasets, start=1):
        name = entry.get("name", f"dataset_{i}")
        _log("")
        _log(f"----- [{i}/{len(datasets)}] {name} -----")

        if "adata_path" not in entry or "dir_path" not in entry:
            _log(f"  ERROR: entry '{name}' must define 'adata_path' and 'dir_path'; skipping")
            summary.append((name, "SKIPPED (missing adata_path/dir_path)", 0.0))
            continue

        params = _resolve_params(defaults, entry)
        _log(f"  adata_path : {entry['adata_path']}")
        _log(f"  dir_path   : {entry['dir_path']}")
        _log(f"  params     : {params}")

        t0 = _time.time()
        try:
            adata = _load_adata(entry["adata_path"])
            _log(f"  loaded adata: {adata.shape[0]} cells x {adata.shape[1]} genes")
            run_scvelo(adata, entry["dir_path"], **params)
            dt = _time.time() - t0
            _log(f"  DONE in {dt/60:.1f} min")
            summary.append((name, "OK", dt))
        except Exception as e:
            dt = _time.time() - t0
            _log(f"  FAILED after {dt/60:.1f} min: {type(e).__name__}: {e}")
            _traceback.print_exc()
            summary.append((name, f"FAILED ({type(e).__name__})", dt))

        gc.collect()

    _log("")
    _log("=== batch run summary ===")
    for name, status, dt in summary:
        _log(f"  {name:<30s} {status:<28s} {dt/60:6.1f} min")
    _log("=== scVelo batch run finished ===")

    log_fh.flush()
    log_fh.close()


def _parse_args(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        description="Run the scVelo pipeline over one or more datasets "
                    "described in a YAML config file."
    )
    parser.add_argument(
        "config",
        help="Path to the YAML config file (see scvelo_config_template.yaml).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_from_config(args.config)
