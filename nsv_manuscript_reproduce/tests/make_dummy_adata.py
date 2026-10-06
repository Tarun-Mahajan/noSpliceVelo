"""Synthetic stand-in for a real dataset, for smoke-testing the pipeline.

Simulates a single linear trajectory with bursty transcription. Each kinetic
gene switches from a basal state (f0, B0) to an induced state (f1, B1) at t=0
and back at t_off; mean and variance follow the moment equations of the bursty
model (degradation rate gamma = 1):

    d mu / dt     = f B - mu
    d sigma2 / dt = f B (1 + 2B) + mu - 2 sigma2

so that, at the same mean, the variance is higher during induction than during
repression (the signal noSpliceVelo uses). Counts are drawn from a negative
binomial with that mean and variance and then thinned with a per-cell capture
efficiency. A few genes are constant (noise genes for the model selection step).

The output has the same fields as a real `adata_pan.h5ad` (see
preprocessing/prepare_adata.py), plus `obs['true_time']` and a fake
`var['MURK_gene']` flag so that the erythroid MURK filter can be exercised.

    python make_dummy_adata.py --out ../data/dummy/adata_pan.h5ad
"""

import argparse
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "preprocessing"))
from prepare_adata import prepare_adata  # noqa: E402


def moments(t, f0, B0, f1, B1, t_off, dt=0.01):
    """Integrate (mu, sigma2) on a grid and interpolate at times t."""
    grid = np.arange(0.0, t.max() + dt, dt)
    mu = np.empty_like(grid); s2 = np.empty_like(grid)
    mu[0] = f0 * B0; s2[0] = mu[0] * (1 + B0)
    for i in range(1, len(grid)):
        f, B = (f1, B1) if grid[i - 1] < t_off else (f0, B0)
        dmu = f * B - mu[i - 1]
        ds2 = f * B * (1 + 2 * B) + mu[i - 1] - 2 * s2[i - 1]
        mu[i] = mu[i - 1] + dt * dmu
        s2[i] = s2[i - 1] + dt * ds2
    return np.interp(t, grid, mu), np.interp(t, grid, s2)


def simulate(n_cells=800, n_kinetic=80, n_noise=20, t_max=8.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.0, t_max, n_cells))
    capture = rng.uniform(0.25, 0.45, n_cells)
    G = n_kinetic + n_noise
    counts = np.zeros((n_cells, G), dtype=np.int64)
    for g in range(n_kinetic):
        f0, B0 = rng.uniform(0.2, 0.6), rng.uniform(1.0, 3.0)
        f1, B1 = f0 * rng.uniform(2.0, 4.0), B0 * rng.uniform(3.0, 8.0)
        t_off = rng.uniform(0.3, 0.7) * t_max
        mu, s2 = moments(t, f0, B0, f1, B1, t_off)
        s2 = np.maximum(s2, mu * 1.05)
        p = mu / s2                      # NB(n, p): mean n(1-p)/p, var n(1-p)/p^2
        n = mu * p / (1 - p)
        counts[:, g] = rng.negative_binomial(n, p)
    for g in range(n_kinetic, G):
        counts[:, g] = rng.poisson(rng.uniform(1.0, 5.0), n_cells)
    counts = rng.binomial(counts, capture[:, None])
    spliced = rng.binomial(counts, 0.8)
    unspliced = counts - spliced

    obs = pd.DataFrame(index=[f"cell{i}" for i in range(n_cells)])
    obs["true_time"] = t
    obs["capture_true"] = capture
    obs["clusters"] = pd.Categorical(
        pd.cut(t, bins=[-1, 0.25 * t_max, 0.5 * t_max, 0.75 * t_max, t_max + 1],
               labels=["A", "B", "C", "D"]).astype(str))
    var = pd.DataFrame(index=[f"gene{g}" for g in range(G)])
    var["kinetic_true"] = np.arange(G) < n_kinetic
    adata = ad.AnnData(X=sp.csr_matrix(counts.astype(np.float32)), obs=obs, var=var,
                       layers={"spliced": sp.csr_matrix(spliced),
                               "unspliced": sp.csr_matrix(unspliced)})
    # a 2-D embedding that follows the trajectory (published objects ship one)
    adata.obsm["X_umap"] = np.c_[t + rng.normal(0, 0.2, n_cells),
                                 np.sin(t / t_max * np.pi) + rng.normal(0, 0.1, n_cells)]
    return adata


def main(argv=None):
    ap = argparse.ArgumentParser(description="Write a synthetic adata_pan.h5ad")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-cells", type=int, default=800)
    ap.add_argument("--n-kinetic", type=int, default=80)
    ap.add_argument("--n-noise", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    raw = simulate(a.n_cells, a.n_kinetic, a.n_noise, seed=a.seed)
    adata = prepare_adata(raw, min_genes=10)
    # One fake MURK gene: the filter drops every latent-time cluster holding > 10 % of
    # the MURK genes, so with a single flagged gene at most one cluster is removed.
    # (Randomly flagging 10 % of 100 genes puts MURK genes in every cluster and
    # removes all genes.)
    adata.var["MURK_gene"] = False
    adata.var.iloc[0, adata.var.columns.get_loc("MURK_gene")] = True
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    adata.write_h5ad(a.out)
    print(f"wrote {a.out}: {adata.n_obs} cells x {adata.n_vars} genes; "
          f"layers={list(adata.layers)}; obsm={list(adata.obsm)}")


if __name__ == "__main__":
    main()
