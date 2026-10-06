from velovi import preprocess_data, VELOVI
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
import torch
import gc
import importlib
import torch.nn.functional as F
import scvi


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
    """Filter, normalize, select HVGs and compute moments for veloVI.

    Returns the processed (gene-subset) AnnData. NOTE: the original version
    reassigned `adata` locally after the HVG subset, so the subset + moments
    were lost to the caller; returning the object fixes that.
    """
    adata = _set_total_X(adata)
    scv.pp.filter_and_normalize(adata, min_shared_counts=min_shared_counts)
    sc.pp.log1p(adata)

    # HVG selection with scanpy
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata = adata[:, adata.var.highly_variable].copy()

    # moments for velocity (uses spliced/unspliced layers)
    sc.pp.pca(adata, n_comps=30)
    sc.pp.neighbors(adata, n_neighbors=30, n_pcs=30)
    scv.pp.moments(adata)

    # veloVI-specific preprocessing: min-max scale Ms/Mu and drop genes with a
    # poor steady-state fit (r2 filter). This is the standard veloVI step and
    # can further reduce the gene set, so it must run before setup_anndata.
    adata = preprocess_data(adata)

    return adata


def run_velovi(
    adata,
    dir_path,
    continue_from_prev=False,
    n_top_genes=2000,
    min_shared_counts=30,
    n_hidden=256,
    n_latent=10,
    n_epochs_kl_warmup=500,
    batch_size=None,
):
    """Run the veloVI pipeline: preprocess -> setup -> train (-> save).

    Parameters
    ----------
    adata :
        Input AnnData with spliced/unspliced layers.
    dir_path :
        Directory where the trained model is saved / loaded.
    continue_from_prev :
        If True and a saved veloVI checkpoint ('model_velovi.pt') already exists
        in `dir_path`, resume training from it instead of building a fresh model.
        When resuming, `n_epochs_kl_warmup` is forced to 0.
    n_top_genes :
        Number of highly-variable genes to keep.
    min_shared_counts :
        `min_shared_counts` for scvelo filter_and_normalize.
    n_hidden, n_latent :
        Hidden width / latent dim of the VELOVI model.
    n_epochs_kl_warmup :
        KL warmup epochs for the training plan (forced to 0 when resuming).
    batch_size :
        Training minibatch size. If None (default), veloVI's default is used.
    """
    scvi.settings.seed = 0
    torch.cuda.empty_cache()
    gc.collect()

    # Preprocess a copy so the caller's object is untouched, and so resume can
    # rebuild the exact same gene set deterministically.
    adata = filter_normalize_adata(
        adata.copy(), n_top_genes=n_top_genes, min_shared_counts=min_shared_counts
    )

    VELOVI.setup_anndata(adata, spliced_layer="Ms", unspliced_layer="Mu")

    # Ensure the output directory exists (needed for both saving and resuming).
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    checkpoint_path = os.path.join(dir_path, 'model_velovi.pt')
    resume = continue_from_prev and os.path.exists(checkpoint_path)

    vae = None
    if resume:
        # Guard: the loaded checkpoint is validated against the rebuilt adata
        # (var names / layers). If the HVG selection drifted and the registries
        # no longer match, load() raises -> fall back to a fresh model.
        try:
            print(f"[continue_from_prev] Found checkpoint at {checkpoint_path}; "
                  f"resuming training and forcing n_epochs_kl_warmup = 0")
            vae = VELOVI.load(checkpoint_path, adata=adata)
            n_epochs_kl_warmup = 0
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[continue_from_prev] Could not resume from {checkpoint_path} "
                  f"({type(e).__name__}: {e}). Falling back to training a new "
                  f"model from scratch.")
            resume = False

    if not resume:
        if continue_from_prev:
            print(f"[continue_from_prev] No usable checkpoint at {checkpoint_path}; "
                  f"training a new model from scratch")
        vae = VELOVI(adata, n_hidden=n_hidden, n_latent=n_latent)

    plan_kwargs = {"n_epochs_kl_warmup": n_epochs_kl_warmup}

    train_kwargs = dict(
        max_epochs=300000,
        early_stopping=True,
        early_stopping_patience=100,
        early_stopping_monitor='elbo_validation',
        plan_kwargs=plan_kwargs,
    )
    if batch_size is not None:
        train_kwargs["batch_size"] = batch_size

    vae.train(**train_kwargs)

    vae.save(checkpoint_path, overwrite=True, save_anndata=True)

    return vae


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
    "n_hidden",
    "n_latent",
    "n_epochs_kl_warmup",
    "batch_size",
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


# Matches tqdm / PyTorch-Lightning training progress lines, e.g.
#   "Epoch 1028/300000:   0%| | 1027/300000 [08:01<38:49:13, 2.14it/s, ...]"
import re as _re
_PROGRESS_RE = _re.compile(
    r"\d{1,3}%\s*\|"     # "  0%|"
    r"|it/s[,\]]"        # "2.14it/s]" / "it/s,"
    r"|s/it[,\]]"        # "s/it]" / "s/it,"
    r"|^Epoch\s+\d+/\d+" # "Epoch 1028/300000"
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
    """Merge global defaults with a dataset entry -> kwargs for run_velovi."""
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
    """Run the veloVI pipeline for every dataset in a YAML config, in sequence."""
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
    default_log = f"velovi_pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_file = global_cfg.get("log_file", default_log) if isinstance(global_cfg, dict) else default_log
    log_dir = os.path.dirname(os.path.abspath(log_file))
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_fh = _FilteredFile(open(log_file, "a"))
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    sys.stderr = _Tee(sys.__stderr__, log_fh)

    import time as _time
    import traceback as _traceback

    _log(f"=== veloVI batch run started ===")
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
            run_velovi(adata, entry["dir_path"], **params)
            dt = _time.time() - t0
            _log(f"  DONE in {dt/60:.1f} min")
            summary.append((name, "OK", dt))
        except Exception as e:
            dt = _time.time() - t0
            _log(f"  FAILED after {dt/60:.1f} min: {type(e).__name__}: {e}")
            _traceback.print_exc()
            summary.append((name, f"FAILED ({type(e).__name__})", dt))

        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    _log("")
    _log("=== batch run summary ===")
    for name, status, dt in summary:
        _log(f"  {name:<30s} {status:<28s} {dt/60:6.1f} min")
    _log("=== veloVI batch run finished ===")

    log_fh.flush()
    log_fh.close()


def _parse_args(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        description="Run the veloVI pipeline over one or more datasets "
                    "described in a YAML config file."
    )
    parser.add_argument(
        "config",
        help="Path to the YAML config file (see velovi_config_template.yaml).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_from_config(args.config)
