"""Config-driven wrapper around score_gene_fit_v3 for batch evaluation.

For each dataset in a YAML config, this:
  1. loads the AnnData from <dir_path>/<h5ad_filename> (default 'adata_nosplicevelo.h5ad'),
  2. loads per-state probabilities from <dir_path>/<prob_state_filename> if present,
  3. computes gene fit quality metrics (task 1: mean-null R2, task 2: quadratic-vs-line),
  4. writes scores to <out_dir>/<out_csv> (default 'gene_fit_scores_v3.csv') and scatter plots.

Usage:
    python score_gene_fit_v3_run.py score_gene_fit_v3_config_template_.yaml
    python score_gene_fit_v3_run.py score_gene_fit_v3_config_template_.yaml --select mouse_pancreas_studentT_time
"""

from __future__ import annotations

import os
import sys
import gc
import time
import traceback
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from score_gene_fit_v3 import run_on_h5ad


_PIPELINE_PARAMS = (
    "h5ad_filename",
    "mu_obs_layer",
    "var_obs_layer",
    "mu_fit_layer",
    "var_fit_layer",
    "combine",
    "thresh_mu_meannull",
    "thresh_var_meannull",
    "prob_state_filename",
    "up_states",
    "down_states",
    "branch_min_cells",
    "branch_min_range_ratio",
    "branch_agg",
    "branch_conf_thresh",
    "corner_quantile",
    "sep_n_bins",
    "run_ftest",
    "fdr_alpha",
    "run_cv",
    "cv_folds",
    "cv_seed",
    "n_jobs",
    "progress_every",
    "write_h5ad",
    "out_csv",
    "out_dir",
    "diagnose_only",
    "continue_from_prev",
    # task 4: bow + signed separation (gene_bow_separation.py)
    "bow_scores",
    "bow_split",
    "bow_velocity_layer",
    "mu_naive_layer",
    "var_naive_layer",
    "bow_embedding",
    "bow_block_size",
    "bow_n_boot",
    "bow_n_bins",
    "bow_min_cells",
    "bow_min_per_bin",
    "bow_z_thresh",
    "signed_sep_n_bins",
    "signed_sep_min_per_bin",
    "bow_seed",
)


def run_score_gene_fit(dir_path: str, **params) -> str:
    """Score gene fit for one dataset in `dir_path`."""
    h5ad_filename = params.pop("h5ad_filename", "adata_nosplicevelo.h5ad")
    prob_state_filename = params.pop("prob_state_filename", "prob_state_avg_nosplicevelo.npy")
    continue_from_prev = params.pop("continue_from_prev", False)

    h5ad_path = os.path.join(dir_path, h5ad_filename)
    prob_state_path = os.path.join(dir_path, prob_state_filename)

    if params.get("out_dir") is None:
        params["out_dir"] = dir_path

    out_dir = params["out_dir"]
    out_csv = params.get("out_csv") or os.path.join(out_dir, "gene_fit_scores_v3.csv")

    if continue_from_prev and os.path.exists(out_csv):
        print(f"[continue_from_prev] {out_csv} already exists; skipping")
        return out_csv

    if not os.path.exists(h5ad_path):
        raise FileNotFoundError(f"h5ad file not found: {h5ad_path}")

    if os.path.exists(prob_state_path):
        params["prob_state"] = prob_state_path

    # Convert state list params if needed
    if "up_states" in params and isinstance(params["up_states"], list):
        params["up_states"] = tuple(params["up_states"])
    if "down_states" in params and isinstance(params["down_states"], list):
        params["down_states"] = tuple(params["down_states"])

    run_on_h5ad(h5ad_path, **params)
    return out_csv


# =============================================================================
# Logging utilities
# =============================================================================

class _Tee:
    """Write output to multiple streams simultaneously."""

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
    """File wrapper that filters out progress bar spam."""

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
    return time.strftime("%Y-%m-%d %H:%M:%S")


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


def run_from_config(config_path: str, select: str | None = None) -> None:
    """Run score_gene_fit_v3 for every dataset in a YAML config file."""
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

    if select:
        datasets = [d for d in datasets if d.get("name") == select]
        if not datasets:
            print(f"[WARN] No dataset named '{select}' in config.")
            return

    default_log = f"score_gene_fit_v3_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_file = global_cfg.get("log_file", default_log) if isinstance(global_cfg, dict) else default_log
    log_dir = os.path.dirname(os.path.abspath(log_file))
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_fh = _FilteredFile(open(log_file, "a"))
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    sys.stderr = _Tee(sys.__stderr__, log_fh)

    _log("=== score_gene_fit_v3 batch run started ===")
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

        t0 = time.time()
        try:
            out_csv = run_score_gene_fit(entry["dir_path"], **params)
            dt = time.time() - t0
            _log(f"  DONE in {dt/60:.1f} min -> {out_csv}")
            summary.append((name, "OK", dt))
        except Exception as e:
            dt = time.time() - t0
            _log(f"  FAILED after {dt/60:.1f} min: {type(e).__name__}: {e}")
            traceback.print_exc()
            summary.append((name, f"FAILED ({type(e).__name__})", dt))

        gc.collect()

    _log("")
    _log("=== batch run summary ===")
    for name, status, dt in summary:
        _log(f"  {name:<30s} {status:<28s} {dt/60:6.1f} min")
    _log("=== score_gene_fit_v3 batch run finished ===")

    log_fh.flush()
    log_fh.close()


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate score_gene_fit_v3 for one or more datasets "
                    "described in a YAML config file."
    )
    parser.add_argument(
        "config",
        help="Path to the YAML config file (see score_gene_fit_v3_config_template_.yaml).",
    )
    parser.add_argument(
        "--select",
        default=None,
        help="Run only the dataset with this name (optional).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_from_config(args.config, select=args.select)
