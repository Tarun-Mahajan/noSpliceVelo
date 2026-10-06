"""Config-driven wrapper around compute_velo for noSpliceVelo.

For each dataset in a YAML config, this:
  1. loads the trained model from  <dir_path>/<model_filename>  (default
     'model_nosplicevelo.pt'),
  2. runs compute_velo to generate the velocity layers / fitted moments /
     kinetic parameters,
  3. saves the annotated AnnData to  <dir_path>/<output_filename>  (default
     'adata_nosplicevelo.h5ad') and the per-state probabilities to
     <dir_path>/<probs_filename>  (default 'prob_state_avg_nosplicevelo.npy'),
     saved next to the output AnnData.

Usage:
    python compute_velo_run.py compute_velo_config_template_.yaml
"""

# Make ../nsv (model code) and this folder importable, independent of the
# working directory. Set NSV_SRC to use a different copy of the model code.
import sys
import os
_HERE = os.path.dirname(os.path.abspath(__file__))
NSV_SRC = os.environ.get("NSV_SRC", os.path.abspath(os.path.join(_HERE, "..", "nsv")))
for _p in (NSV_SRC, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import os
import sys
import gc

import numpy as np
import anndata as ad
import scanpy as sc
import torch

from nosplicevelo_model_v5_polar import noSpliceVelo as noSpliceVelo_v4_polar
from compute_velo import compute_velo


# Optional per-dataset parameters (also allowed in the global `defaults:` block).
_PIPELINE_PARAMS = (
    "continue_from_prev",
    "nboot",
    "prob_thresh",
    "rep",
    "n_neighbors",
    "cell_batch_size",
    "gpu_batch_size",
    "model_filename",
    "scvi_model_filename",
    "output_filename",
    "probs_filename",
    "save_probs",
    "state_reduction",
    "extra_reductions",
    "knn_rep",
    "knn_k",
    "knn_n_iter",
    "temper_taus",
)


def run_compute_velo(
    dir_path,
    continue_from_prev=False,
    nboot=10,
    prob_thresh=0.5,
    rep="X_pca",
    n_neighbors=15,
    cell_batch_size=4096,
    gpu_batch_size=512,
    model_filename="model_nosplicevelo.pt",
    scvi_model_filename="model_scvi_modified.pt",
    output_filename="adata_nosplicevelo.h5ad",
    probs_filename="prob_state_avg_nosplicevelo.npy",
    save_probs=True,
    state_reduction="argmax_stable",
    extra_reductions=None,
    knn_rep="X_latent",
    knn_k=30,
    knn_n_iter=1,
    temper_taus=(4.0,),
):
    """Load a trained noSpliceVelo model from `dir_path`, compute velocity, save.

    Also loads the companion SCVIModified model from `dir_path`/`scvi_model_filename`
    if present, and passes it to compute_velo (as model_scvi). Returns the path
    to the written AnnData.
    """
    model_path = os.path.join(dir_path, model_filename)
    scvi_path = os.path.join(dir_path, scvi_model_filename)
    out_path = os.path.join(dir_path, output_filename)
    probs_path = os.path.join(dir_path, probs_filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model not found: {model_path}")

    if continue_from_prev and os.path.exists(out_path):
        print(f"[continue_from_prev] {out_path} already exists; skipping")
        return out_path

    # Load the model together with its saved AnnData (save_anndata=True at train time).
    model = noSpliceVelo_v4_polar.load(model_path)

    # Load the companion SCVIModified model if it exists (imported lazily so the
    # dependency is only required when the file is present).
    model_scvi = None
    if os.path.exists(scvi_path):
        from scvi_modified_capture_efficiency_model import SCVIModified
        print(f"[compute_velo] loading SCVIModified from {scvi_path}")
        model_scvi = SCVIModified.load(scvi_path)
    else:
        print(f"[compute_velo] SCVIModified checkpoint not found at {scvi_path}; "
              f"passing model_scvi=None")

    adata, prob_state_avg = compute_velo(
        model,
        model_scvi=model_scvi,
        nboot=nboot,
        prob_thresh=prob_thresh,
        rep=rep,
        n_neighbors=n_neighbors,
        cell_batch_size=cell_batch_size,
        gpu_batch_size=gpu_batch_size,
        state_reduction=state_reduction,
        extra_reductions=extra_reductions,
        knn_rep=knn_rep,
        knn_k=knn_k,
        knn_n_iter=knn_n_iter,
        temper_taus=temper_taus,
    )

    adata.write_h5ad(out_path)
    if save_probs:
        # (n_cells, n_genes, n_states); float32 keeps the file compact
        np.save(probs_path, prob_state_avg.astype(np.float32))

    del model, model_scvi, adata
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()
    return out_path


# =============================================================================
# Command-line runner: drive one or more datasets from a YAML config file
# =============================================================================

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


import re as _re
_PROGRESS_RE = _re.compile(
    r"\d{1,3}%\s*\|"
    r"|it/s[,\]]"
    r"|s/it[,\]]"
    r"|^Epoch\s+\d+/\d+"
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
                continue
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
    print(f"[{_timestamp()}] {msg}", flush=True)


def _normalize_config(cfg):
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
    merged = dict(defaults or {})
    for k, v in entry.items():
        if k in ("name", "dir_path"):
            continue
        if k in _PIPELINE_PARAMS:
            merged[k] = v
        else:
            _log(f"  WARNING: ignoring unknown parameter '{k}'")
    return {k: merged[k] for k in merged if k in _PIPELINE_PARAMS}


def run_from_config(config_path):
    """Run compute_velo for every dataset in a YAML config, in sequence."""
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

    from datetime import datetime
    default_log = f"compute_velo_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_file = global_cfg.get("log_file", default_log) if isinstance(global_cfg, dict) else default_log
    log_dir = os.path.dirname(os.path.abspath(log_file))
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_fh = _FilteredFile(open(log_file, "a"))
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    sys.stderr = _Tee(sys.__stderr__, log_fh)

    import time as _time
    import traceback as _traceback

    _log("=== compute_velo batch run started ===")
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

        if "dir_path" not in entry:
            _log(f"  ERROR: entry '{name}' must define 'dir_path'; skipping")
            summary.append((name, "SKIPPED (missing dir_path)", 0.0))
            continue

        params = _resolve_params(defaults, entry)
        _log(f"  dir_path : {entry['dir_path']}")
        _log(f"  params   : {params}")

        t0 = _time.time()
        try:
            out_path = run_compute_velo(entry["dir_path"], **params)
            dt = _time.time() - t0
            _log(f"  DONE in {dt/60:.1f} min -> {out_path}")
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
    _log("=== compute_velo batch run finished ===")

    log_fh.flush()
    log_fh.close()


def _parse_args(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        description="Compute noSpliceVelo velocity for one or more trained "
                    "models described in a YAML config file."
    )
    parser.add_argument(
        "config",
        help="Path to the YAML config file (see compute_velo_config_template_.yaml).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_from_config(args.config)
