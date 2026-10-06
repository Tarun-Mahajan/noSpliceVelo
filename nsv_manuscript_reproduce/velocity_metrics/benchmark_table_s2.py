#!/usr/bin/env python
"""Supplementary Table S2 and the nSV vs TFvelo test from the CBDir tables.

Reads the per-cell CBDir tables written by compute_cbdir_run.py
(`{dataset}_cbdir_long{suffix}.csv`) for every dataset and method in the CBDir
global config, then reproduces Methods, "Statistical analysis of the benchmark":

  dataset score     per-cell CBDir -> Fisher z (arctanh) -> mean over the cells
                    of each transition -> mean over the transitions of the
                    dataset -> tanh                     (benchmark heatmap, Table S2 top)
  pairwise tests    two-sided exact Wilcoxon signed-rank test between every pair
                    of methods across the datasets, on the Fisher-z scale, with
                    Benjamini-Hochberg adjustment over all pairs (Table S2 bottom)
  primary test      nSV vs TFvelo, one-sided sign-flip permutation test of the
                    mean paired difference over all 2^n_datasets sign assignments

Outputs (in --out-dir):
  table_s2_mean_cbdir.csv          dataset x method mean CBDir, plus mean, s.d.
                                   and number of datasets with CBDir > 0
  table_s2_pairwise_wilcoxon.csv   one row per pair: p, BH-adjusted p
  table_s2_pairwise_matrix.csv     the upper-triangle matrix as in the table
  table_s2_primary_signflip.csv    nSV vs TFvelo
  table_s2.tex                     both LaTeX tabulars

Usage (from velocity_metrics/):
  python benchmark_table_s2.py --global-config cbdir_global_config_4throot.yaml
  python benchmark_table_s2.py --means table_s2_mean_cbdir.csv     # tests only
"""

import argparse
import glob
import itertools
import os
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

DATASET_ORDER = ["human_erythroid", "mouse_erythroid", "mouse_organoid", "human_bonemarrow",
                 "mouse_neural", "mouse_dentategyrus", "mouse_pancreas"]
DATASET_LABELS = {"human_erythroid": "Hum. ery.", "mouse_erythroid": "Mouse ery.",
                  "mouse_organoid": "Organoid", "human_bonemarrow": "Bone marrow",
                  "mouse_neural": "Cortex", "mouse_dentategyrus": "Dent. gyrus",
                  "mouse_pancreas": "Pancreas"}
METHOD_LABELS = {"nSV": "nSV", "veloVAE": "VeloVAE", "uniTVelo": "UniTVelo",
                 "celldancer": "cellDancer", "scVelo-dynamical": "scVelo-dynamical",
                 "scVelo-stochastic": "scVelo-stochastic", "veloVI": "veloVI", "TFvelo": "TFvelo"}
EPS = 1e-6


def load_long_tables(global_config, suffix=None):
    """Concatenate every {ds}_cbdir_long{suffix}.csv reachable from the CBDir configs."""
    gdir = os.getcwd()   # compute_cbdir_run.py resolves config paths from its working directory
    g = yaml.safe_load(open(global_config))
    suffix = suffix if suffix is not None else (g.get("str_suffix") or "")
    frames, seen = [], set()
    for method, mpath in (g.get("methods") or {}).items():
        mpath = mpath if os.path.isabs(mpath) else os.path.join(gdir, mpath)
        m = yaml.safe_load(open(mpath))
        for e in m.get("datasets", []) or []:
            d = e.get("dir_path")
            if not d:
                continue
            d = d if os.path.isabs(d) else os.path.join(gdir, d)
            for f in glob.glob(os.path.join(d, f"{e['name']}_cbdir_long{suffix}.csv")):
                f = os.path.abspath(f)
                if f not in seen:
                    seen.add(f)
                    frames.append(pd.read_csv(f, usecols=["dataset", "method", "edge", "cell_barcode", "cbdir"]))
    if not frames:
        sys.exit("no CBDir long tables found; run compute_cbdir_run.py first")
    long = pd.concat(frames, ignore_index=True)
    long = long.drop_duplicates(["dataset", "method", "edge", "cell_barcode"])
    print(f"read {len(seen)} long table(s): {long['dataset'].nunique()} datasets, "
          f"{long['method'].nunique()} methods, {len(long)} scored cells")
    return long


def dataset_scores(long):
    """Fisher-z mean per transition, mean over transitions; returns z-scale table."""
    long = long.dropna(subset=["cbdir"]).copy()
    long["z"] = np.arctanh(np.clip(long["cbdir"].astype(float), -1 + EPS, 1 - EPS))
    per_edge = long.groupby(["dataset", "method", "edge"])["z"].mean()
    per_ds = per_edge.groupby(["dataset", "method"]).mean().unstack("method")
    return per_ds, per_edge


def pairwise_wilcoxon(z, methods):
    rows = []
    for a, b in itertools.combinations(methods, 2):
        pair = z[[a, b]].dropna()
        d = pair[a] - pair[b]
        if len(d) < 2 or np.allclose(d, 0):
            p = np.nan
        else:
            p = wilcoxon(pair[a], pair[b], alternative="two-sided", mode="exact").pvalue
        rows.append(dict(method_a=a, method_b=b, n_datasets=len(d), p=p))
    res = pd.DataFrame(rows)
    ok = res["p"].notna()
    res["p_bh"] = np.nan
    res.loc[ok, "p_bh"] = multipletests(res.loc[ok, "p"], method="fdr_bh")[1]
    return res


def signflip_one_sided(x, y):
    """P(mean(s*d) >= mean(d)) over all 2^n sign vectors s, d = x - y."""
    d = np.asarray(x, float) - np.asarray(y, float)
    obs = d.mean()
    signs = np.array(list(itertools.product([1, -1], repeat=len(d))))
    null = (signs * d).mean(axis=1)
    return obs, float(np.mean(null >= obs - 1e-12)), len(d)


def fmt(v):
    return f"\\({v:.2f}\\)" if v < 0 else f"{v:.2f}"


def fmt_p(p):
    if not np.isfinite(p):
        return "NA"
    return f"{p:.3f}".rstrip("0") if p < 0.1 else f"{p:.2f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--global-config", help="CBDir global config (run from velocity_metrics/)")
    src.add_argument("--means", help="dataset x method CSV of mean CBDir (skips the long tables)")
    ap.add_argument("--str-suffix", default=None)
    ap.add_argument("--out-dir", default="./results_table_s2")
    ap.add_argument("--reference", default="nSV")
    ap.add_argument("--primary-comparator", default="TFvelo")
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)

    if a.global_config:
        z, per_edge = dataset_scores(load_long_tables(a.global_config, a.str_suffix))
        per_edge.reset_index().assign(cbdir=lambda t: np.tanh(t["z"])).to_csv(
            os.path.join(a.out_dir, "table_s2_per_transition.csv"), index=False)
    else:
        m = pd.read_csv(a.means, index_col=0)
        m = m[[c for c in m.columns if c not in ("mean", "sd", "n_positive")]]
        z = np.arctanh(np.clip(m.astype(float), -1 + EPS, 1 - EPS))

    datasets = [d for d in DATASET_ORDER if d in z.index] + [d for d in z.index if d not in DATASET_ORDER]
    z = z.loc[datasets]
    cb = np.tanh(z)
    # methods ordered by mean CBDir across datasets (as in the benchmark heatmap)
    methods = list(cb.mean(axis=0).sort_values(ascending=False).index)
    cb = cb[methods]
    z = z[methods]

    table = cb.T.copy()
    table["mean"] = cb.mean(axis=0)
    table["sd"] = cb.std(axis=0, ddof=1)
    table["n_positive"] = (cb > 0).sum(axis=0)
    table.index.name = "method"
    table.to_csv(os.path.join(a.out_dir, "table_s2_mean_cbdir.csv"))
    print("\nmean CBDir (dataset scores)\n" + table.round(2).to_string())

    pw = pairwise_wilcoxon(z, methods)
    pw.to_csv(os.path.join(a.out_dir, "table_s2_pairwise_wilcoxon.csv"), index=False)
    mat = pd.DataFrame("", index=methods[:-1], columns=methods[1:])
    for _, r in pw.iterrows():
        mat.loc[r.method_a, r.method_b] = f"{fmt_p(r.p)} ({fmt_p(r.p_bh)})"
    mat.to_csv(os.path.join(a.out_dir, "table_s2_pairwise_matrix.csv"))
    print("\npairwise two-sided exact Wilcoxon, p (BH)\n" + mat.to_string())

    if a.reference in z and a.primary_comparator in z:
        pair = z[[a.reference, a.primary_comparator]].dropna()
        obs, p, n = signflip_one_sided(pair[a.reference], pair[a.primary_comparator])
        wins = int((pair[a.reference] > pair[a.primary_comparator]).sum())
        pd.DataFrame([dict(reference=a.reference, comparator=a.primary_comparator, n_datasets=n,
                           mean_z_difference=obs, reference_higher=wins, p_one_sided=p,
                           smallest_attainable_p=1 / 2 ** n)]).to_csv(
            os.path.join(a.out_dir, "table_s2_primary_signflip.csv"), index=False)
        print(f"\n{a.reference} vs {a.primary_comparator}: higher on {wins}/{n} datasets, "
              f"one-sided sign-flip p = {p:.4f} (smallest attainable {1 / 2 ** n:.4f})")

    # LaTeX
    lines = ["\\begin{tabular}{l " + "r" * len(datasets) + " c}", "\\toprule",
             "Method & " + " & ".join(DATASET_LABELS.get(d, d) for d in datasets) + " & Datasets \\(>0\\) \\\\",
             "\\midrule"]
    for mth in methods:
        lab = METHOD_LABELS.get(mth, mth) + (" (our method)" if mth == a.reference else "")
        lines.append(lab + " & " + " & ".join(fmt(cb.loc[d, mth]) for d in datasets)
                     + f" & {int(table.loc[mth, 'n_positive'])}/{len(datasets)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "",
              "\\begin{tabular}{l " + "c" * (len(methods) - 1) + "}", "\\toprule",
              " & " + " & ".join(METHOD_LABELS.get(m, m) for m in methods[1:]) + " \\\\", "\\midrule"]
    for i, ma in enumerate(methods[:-1]):
        cells = ["" if j <= i else mat.loc[ma, mb] for j, mb in enumerate(methods[1:], start=1)]
        lines.append(METHOD_LABELS.get(ma, ma) + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(a.out_dir, "table_s2.tex"), "w").write("\n".join(lines) + "\n")
    print(f"\nwrote tables to {a.out_dir}/")


if __name__ == "__main__":
    main()
