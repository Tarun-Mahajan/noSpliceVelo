"""Config-driven wrapper to compute velocity and kinetic parameters from trained veloVI models.

Fashioned after `nosplicevelo/compute_velo_run.py`.

For each dataset in a YAML config:
  1. Loads a trained veloVI model checkpoint (`model_velovi.pt`) and companion AnnData.
  2. Extracts latent time and velocity using `vae.get_latent_time()` and `vae.get_velocity()`.
  3. Computes scaling factors and updates `adata.layers` and `adata.var` kinetic parameters.
  4. Saves the annotated AnnData to `output_filename` (default `adata_velovi.h5ad`).

Usage:
    python compute_velovi_velocity_run.py compute_velovi_velocity_config_template.yaml
"""

import os
import sys
import gc
import time
import traceback
import numpy as np
import anndata as ad
import torch

try:
    from velovi import VELOVI
except ImportError:
    VELOVI = None

# Pipeline parameters allowed in global `defaults:` block or per-dataset.
_PIPELINE_PARAMS = (
    "continue_from_prev",
    "n_samples",
    "velo_statistic",
    "model_filename",
    "output_filename",
)


def add_velovi_outputs_to_adata(adata, vae, n_samples=25, velo_statistic="mean"):
    """Extract velocity and fitted parameters from vae model and add to adata."""
    latent_time = vae.get_latent_time(n_samples=n_samples)
    velocities = vae.get_velocity(n_samples=n_samples, velo_statistic=velo_statistic)

    t = latent_time
    t_max = t.max(0)
    t_max_vals = np.maximum(np.asarray(t_max), 1e-6)
    scaling = 20 / t_max_vals

    adata.layers["velocity"] = velocities / scaling
    adata.layers["latent_time_velovi"] = latent_time

    rates = vae.get_rates()
    adata.var["fit_alpha"] = rates["alpha"] / scaling
    adata.var["fit_beta"] = rates["beta"] / scaling
    adata.var["fit_gamma"] = rates["gamma"] / scaling
    
    if hasattr(vae.module, "switch_time_unconstr"):
        switch_time = torch.nn.functional.softplus(vae.module.switch_time_unconstr).detach().cpu().numpy()
        adata.var["fit_t_"] = switch_time * scaling
    
    t_vals = latent_time.values if hasattr(latent_time, 'values') else np.asarray(latent_time)
    adata.layers["fit_t"] = t_vals * scaling[np.newaxis, :]
    adata.var["fit_scaling"] = 1.0
    return adata


def run_compute_velovi_velocity(
    dir_path,
    adata_path=None,
    continue_from_prev=False,
    n_samples=25,
    velo_statistic="mean",
    model_filename="model_velovi.pt",
    output_filename="adata_velovi.h5ad",
):
    """Load a trained veloVI model from `dir_path`, compute velocity, save AnnData."""
    model_path = os.path.join(dir_path, model_filename)
    out_path = os.path.join(dir_path, output_filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"velovi model checkpoint file not found: {model_path}")

    if continue_from_prev and os.path.exists(out_path):
        _log(f"[continue_from_prev] Output {out_path} already exists; skipping computation.")
        return out_path

    # Determine AnnData path if not explicitly provided
    if adata_path is None:
        for fn in ["adata_scvelo.h5ad", "adata.h5ad"]:
            candidate = os.path.join(dir_path, fn)
            if os.path.exists(candidate):
                adata_path = candidate
                break
        if adata_path is None:
            raise FileNotFoundError(f"No AnnData file specified and none found in {dir_path}")

    if not os.path.exists(adata_path):
        raise FileNotFoundError(f"adata_path not found: {adata_path}")

    _log(f"Reading AnnData from {adata_path}...")
    adata = ad.read_h5ad(adata_path)
    
    if VELOVI is None:
        raise RuntimeError("velovi package is not installed in the environment.")

    _log(f"Loading trained veloVI model from {model_path}...")
    vae = VELOVI.load(model_path, adata=adata)

    _log(f"Computing velocity & latent time (n_samples={n_samples}, velo_statistic='{velo_statistic}')...")
    add_velovi_outputs_to_adata(adata, vae, n_samples=n_samples, velo_statistic=velo_statistic)

    _log(f"Saving velocity annotated AnnData to {out_path}...")
    adata.write_h5ad(out_path)

    del vae, adata
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()
    return out_path


# =============================================================================
# Logging / Output capture wrappers
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

    def close(self):
        self.flush()


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
        if k in ("name", "dir_path", "adata_path"):
            continue
        if k in _PIPELINE_PARAMS:
            merged[k] = v
        else:
            _log(f"  WARNING: ignoring unknown parameter '{k}'")

    default_fallbacks = {
        "continue_from_prev": False,
        "n_samples": 25,
        "velo_statistic": "mean",
        "model_filename": "model_velovi.pt",
        "output_filename": "adata_velovi.h5ad",
    }

    resolved = {}
    for k in _PIPELINE_PARAMS:
        if k in merged:
            resolved[k] = merged[k]
        else:
            resolved[k] = default_fallbacks[k]

    return resolved


def run_from_config(config_path):
    """Run veloVI compute_velocity for every dataset in a YAML config, in sequence."""
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
    default_log = f"velovi_compute_velocity_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_file = global_cfg.get("log_file", default_log) if isinstance(global_cfg, dict) else default_log
    log_dir = os.path.dirname(os.path.abspath(log_file))
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_fh = _FilteredFile(open(log_file, "a"))
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    sys.stderr = _Tee(sys.__stderr__, log_fh)

    try:
        _log("=== veloVI compute_velocity batch run started ===")
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
            adata_path = entry.get("adata_path")
            _log(f"  dir_path   : {entry['dir_path']}")
            if adata_path:
                _log(f"  adata_path : {adata_path}")
            _log(f"  params     : {params}")

            t0 = time.time()
            try:
                out_path = run_compute_velovi_velocity(
                    dir_path=entry["dir_path"],
                    adata_path=adata_path,
                    **params
                )
                dt = time.time() - t0
                _log(f"  DONE in {dt/60:.2f} min -> {out_path}")
                summary.append((name, "OK", dt))
            except Exception as e:
                dt = time.time() - t0
                _log(f"  FAILED after {dt/60:.2f} min: {type(e).__name__}: {e}")
                traceback.print_exc()
                summary.append((name, f"FAILED ({type(e).__name__})", dt))

            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()

        _log("")
        _log("=== batch run summary ===")
        for name, status, dt in summary:
            _log(f"  {name:<30s} {status:<28s} {dt/60:6.2f} min")
        _log("=== veloVI compute_velocity batch run finished ===")
    finally:
        log_fh.flush()
        log_fh.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__


def _parse_args(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        description="Compute veloVI velocity and kinetic parameters for trained "
                    "models described in a YAML config file."
    )
    parser.add_argument(
        "config",
        help="Path to the YAML config file (see compute_velovi_velocity_config_template.yaml).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_from_config(args.config)
    sys.exit(0)
