import numpy as np


def eff_genes_report(V, power=0.25, coverage=(0.5, 0.9), gene_names=None):
    """
    Effective number of genes carrying the velocity signal, per cell and across
    the population.

    V      : (n_cells, n_genes) velocity array (dense; .toarray() sparse first)
    power  : the elementwise transform the cosine actually sees --
             1.0 = untransformed, 0.5 = sqrt_transform, 0.25 = fourth root.
             MUST match your velocity_graph / confidence settings.

    per cell    q_ig = |v_ig|^(2p) / sum_g |v_ig|^(2p)     (rows sum to 1)
                PR_i = 1 / sum_g q_ig^2
    population  w_g  = mean_i q_ig                          (sums to 1)
                PR   = 1 / sum_g w_g^2

    Cells are normalised before pooling so that cells with large velocity norms
    don't dominate -- the cosine is per-cell magnitude-invariant, so each cell
    should get one vote on which genes matter.

    By Jensen, PR_population >= harmonic_mean_i(PR_i), equality iff every cell
    has the same weight distribution. diversity = PR_pop / PR_harmonic is the
    cell-to-cell turnover in *which* genes carry the signal. Note the practical
    floor is ~3, not 1: random weight fluctuation inside a fixed panel already
    produces ~2.6. Read it comparatively across methods.
    """
    V = np.asarray(V, float)
    keep = np.isfinite(V).all(0)
    V = V[:, keep]

    p = np.abs(V) ** (2.0 * power)
    tot = p.sum(1, keepdims=True)
    ok = tot.ravel() > 0
    q = np.zeros_like(p)
    q[ok] = p[ok] / tot[ok]

    pr_cell = np.full(V.shape[0], np.nan)
    pr_cell[ok] = 1.0 / (q[ok] ** 2).sum(1)

    w = q[ok].mean(0)
    pr_pop = 1.0 / (w ** 2).sum()
    pr_h = 1.0 / np.nanmean(1.0 / pr_cell[ok])

    order = np.argsort(-w)
    cum = np.cumsum(w[order])

    out = dict(
        n_genes=int(V.shape[1]),
        n_cells_used=int(ok.sum()),
        power=float(power),
        pr_cell_median=float(np.nanmedian(pr_cell)),
        pr_cell_harmonic=float(pr_h),
        pr_population=float(pr_pop),
        diversity=float(pr_pop / pr_h),
    )
    for c in coverage:
        out[f"n_genes_{int(c * 100)}pct"] = int(np.searchsorted(cum, c) + 1)

    if gene_names is not None:
        names = np.asarray(gene_names)[keep]
        out["top_genes"] = list(names[order[:20]])
        out["gene_weights"] = w[order]        # descending, aligned to top_genes
    return out, pr_cell