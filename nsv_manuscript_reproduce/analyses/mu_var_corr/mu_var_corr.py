#!/usr/bin/env python
"""Correlation of the smoothed scVI moments with their naive counterparts.

For every dataset in the config, reads one h5ad and correlates each layer pair
(default: mu_scvi_smooth vs mu_naive_smooth, and sqrt(var_scvi_smooth) vs
sqrt(var_naive_smooth), i.e. the var pair is compared on the standard-deviation
scale; set with `pre_transform` in the config):

  per gene   r over cells   -> how well scVI reproduces each gene's trajectory
  per cell   r over genes   -> how well scVI reproduces each cell's profile
  global     r over all (cell, gene) entries (one number per dataset)

Correlations: pearson (on each requested transform: none, log1p, applied after
the pre_transform, so var_pearson_log1p = r(log1p(sqrt(var_a)), log1p(sqrt(var_b))))
and spearman
(rank-based, so it is the same for every monotone transform and computed once).
Non-finite entries are dropped pairwise; a correlation needs >= min_valid finite
pairs and non-zero variance in both vectors, otherwise it is NaN.

Output: ONE h5ad
  X                (dataset x gene) rows  x  metric columns, e.g. mu_pearson,
                   mu_pearson_log1p, mu_spearman, var_pearson, ...
  obs              dataset, gene, gene_id, mean_<layer> for every layer used
  var              quantity, layer_a, layer_b, pre_transform, method, transform
  layers['n_valid'] number of finite pairs behind each per-gene value
  uns['per_cell']  {dataset: DataFrame (cells x metrics)}   when per_cell: true
  uns['summary']   per (dataset, level, metric): n, mean, median, q25, q75, and
                   the global (flattened) correlation
  uns['datasets']  {dataset: h5ad path};  uns['config'] the YAML used

Usage:
  python mu_var_corr.py mu_var_corr_config.yaml
"""
import argparse
import os
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.stats import rankdata

DEFAULT_PAIRS = OrderedDict(mu=["mu_scvi_smooth", "mu_naive_smooth"],
                            var=["var_scvi_smooth", "var_naive_smooth"])
TRANSFORMS = {"none": lambda a: a, "log1p": np.log1p}
# Applied to both layers of a pair BEFORE the correlation transforms above.
# Default: var pairs are correlated as standard deviations, sqrt(var).
PRE_TRANSFORMS = {"none": lambda a: a, "sqrt": np.sqrt, "log1p": np.log1p}
DEFAULT_PRE = {"var": "sqrt"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# correlation kernels: column-wise r between A[:, j] and B[:, j]
# ---------------------------------------------------------------------------

def _pearson_cols(A, B):
    """Column-wise Pearson for fully finite A, B (n x m)."""
    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)
    num = (A * B).sum(axis=0)
    den = np.sqrt((A * A).sum(axis=0) * (B * B).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    r[~(den > 0)] = np.nan
    return np.clip(r, -1.0, 1.0)


def _rank_cols(A):
    return rankdata(A, axis=0, method="average")


def corr_cols(A, B, method, min_valid):
    """Column-wise correlation with pairwise removal of non-finite entries.

    Returns (r, n_valid), both of length A.shape[1]."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    ok = np.isfinite(A) & np.isfinite(B)
    n_valid = ok.sum(axis=0)
    r = np.full(A.shape[1], np.nan)

    clean = ok.all(axis=0)                       # fast path: no missing values
    if clean.any():
        a, b = A[:, clean], B[:, clean]
        if method == "spearman":
            a, b = _rank_cols(a), _rank_cols(b)
        r[clean] = _pearson_cols(a, b)

    for j in np.where(~clean)[0]:                # columns with missing values
        m = ok[:, j]
        if m.sum() < 2:
            continue
        a, b = A[m, j][:, None], B[m, j][:, None]
        if method == "spearman":
            a, b = _rank_cols(a), _rank_cols(b)
        r[j] = _pearson_cols(a, b)[0]

    r[n_valid < min_valid] = np.nan
    return r, n_valid


def corr_flat(A, B, method):
    a, b = np.ravel(A).astype(np.float64), np.ravel(B).astype(np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    a, b = a[m][:, None], b[m][:, None]
    if method == "spearman":
        a, b = _rank_cols(a), _rank_cols(b)
    return float(_pearson_cols(a, b)[0])


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def read_yaml(path):
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def resolve(path, base):
    path = os.path.expanduser(str(path))
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(base, path))


def metric_specs(pairs, methods, transforms):
    """[(metric_name, quantity, layer_a, layer_b, method, transform)]"""
    specs = []
    for q, (la, lb) in pairs.items():
        for meth in methods:
            if meth == "spearman":
                specs.append((f"{q}_spearman", q, la, lb, "spearman", "rank"))
                continue
            for t in transforms:
                name = f"{q}_{meth}" if t == "none" else f"{q}_{meth}_{t}"
                specs.append((name, q, la, lb, meth, t))
    return specs


def dataset_jobs(cfg, base):
    defaults = cfg.get("defaults", {}) or {}
    jobs = OrderedDict()
    for name, entry in (cfg.get("datasets") or {}).items():
        if isinstance(entry, str):
            entry = {"h5ad": entry}
        entry = {**defaults, **(entry or {})}
        if not entry.get("h5ad"):
            raise ValueError(f"dataset '{name}': 'h5ad' path is required")
        entry["h5ad"] = resolve(entry["h5ad"], base)
        jobs[name] = entry
    if not jobs:
        raise ValueError("config has no datasets")
    return jobs


# ---------------------------------------------------------------------------
# per-dataset computation
# ---------------------------------------------------------------------------

def dense(x, rows=None, cols=None):
    if rows is not None:
        x = x[rows]
    if cols is not None:
        x = x[:, cols]
    return x.toarray() if sp.issparse(x) else np.asarray(x)


def run_dataset(name, job, specs, pairs, min_valid, per_cell, pre):
    import anndata as ad

    log(f"[{name}] reading {job['h5ad']}")
    adata = ad.read_h5ad(job["h5ad"])

    # per-dataset layer-name overrides: {quantity: [layer_a, layer_b]}
    lay = OrderedDict((q, list(v)) for q, v in pairs.items())
    for q, v in (job.get("layers") or {}).items():
        lay[q] = list(v)
    needed = sorted({l for v in lay.values() for l in v})
    missing = [l for l in needed if l not in adata.layers]
    if missing:
        raise KeyError(f"[{name}] layers missing from h5ad: {missing}; "
                       f"available: {list(adata.layers.keys())}")

    cells = np.arange(adata.n_obs)
    if job.get("cell_query"):
        cells = np.where(adata.obs.eval(job["cell_query"]).to_numpy(bool))[0]
    genes = np.arange(adata.n_vars)
    if job.get("gene_query"):
        genes = np.where(adata.var.eval(job["gene_query"]).to_numpy(bool))[0]
    log(f"[{name}] {len(cells)}/{adata.n_obs} cells, {len(genes)}/{adata.n_vars} genes")
    if len(cells) < 3 or len(genes) < 1:
        raise ValueError(f"[{name}] too few cells/genes after filtering")

    cell_names = adata.obs_names[cells].astype(str)
    gcol = job.get("gene_names_col")
    gene_ids = adata.var_names[genes].astype(str)
    gene_names = (adata.var[gcol].iloc[genes].astype(str).to_numpy() if gcol
                  else np.asarray(gene_ids))

    metric_names = [s[0] for s in specs]
    R = np.full((len(genes), len(specs)), np.nan)
    N = np.zeros((len(genes), len(specs)), dtype=np.int32)
    PC = np.full((len(cells), len(specs)), np.nan) if per_cell else None
    glob = {}

    X = {l: dense(adata.layers[l], cells, genes).astype(np.float64) for l in needed}
    del adata
    means = {}
    for l in needed:
        with np.errstate(all="ignore"):
            means[l] = np.nanmean(np.where(np.isfinite(X[l]), X[l], np.nan), axis=0)

    # pre-transform each pair once (e.g. var -> sqrt(var)); negatives under sqrt
    # become NaN and are dropped pairwise like any other non-finite entry
    P = {}
    for q, (la, lb) in lay.items():
        pname = pre.get(q, "none")
        for l in (la, lb):
            if pname == "sqrt":
                n_neg = int((X[l] < 0).sum())
                if n_neg:
                    log(f"[{name}] WARNING {l}: {n_neg} negative entries -> NaN under sqrt")
            with np.errstate(invalid="ignore", divide="ignore"):
                P[(q, l)] = PRE_TRANSFORMS[pname](X[l])
    del X

    for k, (mname, q, _, _, meth, t) in enumerate(specs):
        la, lb = lay[q]
        f = TRANSFORMS.get(t, lambda a: a)        # spearman: 'rank' -> identity
        with np.errstate(invalid="ignore", divide="ignore"):
            A, B = f(P[(q, la)]), f(P[(q, lb)])
        R[:, k], N[:, k] = corr_cols(A, B, meth, min_valid)            # per gene, over cells
        if per_cell:
            PC[:, k], _ = corr_cols(A.T, B.T, meth, min_valid)        # per cell, over genes
        glob[mname] = corr_flat(A, B, meth)                            # all entries
    del P

    pc_df = (pd.DataFrame(PC, index=cell_names, columns=metric_names) if per_cell else None)

    obs = pd.DataFrame({"dataset": name, "gene": gene_names, "gene_id": np.asarray(gene_ids)})
    for l in needed:
        obs[f"mean_{l}"] = means[l]
    obs.index = [f"{name}::{g}" for g in gene_ids]

    for k, m in enumerate(metric_names):
        v = R[:, k]
        msg = f"[{name}] {m:<24s} per-gene median {np.nanmedian(v) if np.isfinite(v).any() else np.nan:.3f}"
        if pc_df is not None:
            c = pc_df[m].to_numpy()
            msg += f" | per-cell median {np.nanmedian(c) if np.isfinite(c).any() else np.nan:.3f}"
        msg += f" | global {glob[m]:.3f}"
        log(msg)

    used_layers = {q: list(v) for q, v in lay.items()}
    return obs, R, N, pc_df, glob, len(cells), used_layers


def summarise(name, metric_names, R, pc_df, glob):
    rows = []

    def stats(level, m, v):
        v = v[np.isfinite(v)]
        rows.append(dict(dataset=name, level=level, metric=m, n=len(v),
                         mean=v.mean() if len(v) else np.nan,
                         median=np.median(v) if len(v) else np.nan,
                         q25=np.quantile(v, 0.25) if len(v) else np.nan,
                         q75=np.quantile(v, 0.75) if len(v) else np.nan,
                         frac_above_0p9=(v > 0.9).mean() if len(v) else np.nan,
                         global_r=glob[m]))

    for k, m in enumerate(metric_names):
        stats("per_gene", m, R[:, k])
        if pc_df is not None:
            stats("per_cell", m, pc_df[m].to_numpy())
    return rows


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--datasets", nargs="*", default=None, help="subset of dataset names")
    args = ap.parse_args()

    import anndata as ad

    cfg_path = os.path.abspath(args.config)
    base = os.path.dirname(cfg_path)
    cfg = read_yaml(cfg_path)

    pairs = OrderedDict((q, list(v)) for q, v in (cfg.get("pairs") or DEFAULT_PAIRS).items())
    for q, v in pairs.items():
        if len(v) != 2:
            raise ValueError(f"pairs.{q} must be [layer_a, layer_b]")
    methods = [m.lower() for m in (cfg.get("methods") or ["pearson", "spearman"])]
    bad = set(methods) - {"pearson", "spearman"}
    if bad:
        raise ValueError(f"unknown methods {bad}; use pearson / spearman")
    transforms = list(cfg.get("transforms") or ["none", "log1p"])
    bad = set(transforms) - set(TRANSFORMS)
    if bad:
        raise ValueError(f"unknown transforms {bad}; use {list(TRANSFORMS)}")
    pre = dict(DEFAULT_PRE if cfg.get("pre_transform") is None else cfg["pre_transform"])
    pre = {q: str(v or "none").lower() for q, v in pre.items()}
    bad = set(pre.values()) - set(PRE_TRANSFORMS)
    if bad:
        raise ValueError(f"unknown pre_transform {bad}; use {list(PRE_TRANSFORMS)}")
    log("pre_transform: " + ", ".join(f"{q}={pre.get(q, 'none')}" for q in pairs))
    min_valid = int(cfg.get("min_valid", 10))
    per_cell = bool(cfg.get("per_cell", True))
    skip_failed = bool(cfg.get("skip_failed", False))
    out_path = resolve(cfg.get("out_h5ad", "./mu_var_corr.h5ad"), base)

    specs = metric_specs(pairs, methods, transforms)
    metric_names = [s[0] for s in specs]
    jobs = dataset_jobs(cfg, base)
    if args.datasets:
        jobs = OrderedDict((k, v) for k, v in jobs.items() if k in args.datasets)

    obs_l, R_l, N_l, per_cell_d, summ, used = [], [], [], {}, [], {}
    for name, job in jobs.items():
        try:
            obs, R, N, pc_df, glob, n_cells, lay = run_dataset(
                name, job, specs, pairs, min_valid, per_cell, pre)
        except Exception as e:
            if not skip_failed:
                raise
            log(f"[{name}] FAILED, skipped: {e}")
            continue
        obs_l.append(obs); R_l.append(R); N_l.append(N)
        if pc_df is not None:
            per_cell_d[name] = pc_df
        summ += summarise(name, metric_names, R, pc_df, glob)
        used[name] = dict(h5ad=job["h5ad"], n_cells=int(n_cells), n_genes=int(len(obs)),
                          layers={q: ",".join(v) for q, v in lay.items()})

    if not obs_l:
        raise SystemExit("no dataset produced results")

    obs = pd.concat(obs_l)
    obs["dataset"] = pd.Categorical(obs["dataset"], categories=list(used))
    var = pd.DataFrame([dict(quantity=q, layer_a=a, layer_b=b,
                             pre_transform=pre.get(q, "none"), method=m, transform=t)
                        for _, q, a, b, m, t in specs], index=metric_names)
    out = ad.AnnData(X=np.vstack(R_l).astype(np.float32), obs=obs, var=var)
    out.layers["n_valid"] = np.vstack(N_l)
    if per_cell_d:
        out.uns["per_cell"] = per_cell_d
    out.uns["summary"] = pd.DataFrame(summ)
    out.uns["datasets"] = used
    with open(cfg_path) as fh:
        out.uns["config"] = fh.read()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out.write_h5ad(out_path, compression="gzip")
    log(f"wrote {out_path}  ({out.n_obs} gene rows x {out.n_vars} metrics, "
        f"{len(used)} datasets)")
    with pd.option_context("display.width", 200, "display.max_rows", 500,
                           "display.float_format", "{:.3f}".format):
        print(out.uns["summary"][["dataset", "level", "metric", "n", "median",
                                  "q25", "q75", "global_r"]].to_string(index=False))


if __name__ == "__main__":
    main()
