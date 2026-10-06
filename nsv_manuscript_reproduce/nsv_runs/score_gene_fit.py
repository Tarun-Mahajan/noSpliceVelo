"""
Per-gene fit quality from OBSERVED vs FITTED moments (no pseudotime, no
velocity graph -> non-circular). Uses the four noSpliceVelo layers:

    observed : mu_scvi_smooth , var_scvi_smooth
    fitted   : mu_fit         , var_fit

Two nulls, each scored consistently in the SAME space it is defined in:

  --null mean : per-gene R2 of observed-vs-fitted for mu and var separately,
                relative to a flat mean. Combined by `min` (a gene must fit
                both moments). Standard reconstruction R2.

  --null line : GEOMETRIC R2 in the (mu, var) plane. Both axes are scaled by
                their OBSERVED per-gene std (std of mu_obs, std of var_obs), so
                the two moments are comparable. The null is a single straight
                line fit by total least squares (TLS: orthogonal distance,
                symmetric in the two noisy moments), and BOTH the model residual
                (obs -> fitted point) and the null residual (obs -> orthogonal
                projection on the line) are summed squared distances in that
                same scaled plane:

                    R2_2D = 1 - SSE_model_2D / SSE_TLSline_2D

                R2_2D > 0 : noSpliceVelo's (curved) fit beats a straight line.
                R2_2D < 0 : the fit is WORSE than a straight line -> broken
                            (wrong scale / collapsed). This is the drop flag.
                No log transform: the null is a literal straight line in the raw
                (std-scaled) (mu, var) plane, fit and scored in one metric.

Writes per-gene columns to adata.var (r2_mu, r2_var, fit_r2, geom_r2_line,
scale_ratio_mu/var) and a boolean `velocity_reliable`, which feeds the existing
--gene_filter in generate_stream_plots.py / diagnose_backflow.py.

Detailed run log is written to a .log file (see --log_file).
"""

import os
import sys
import time
import argparse
import numpy as np


# ---------------------------------------------------------------------------
# core (numpy; testable without anndata)
# ---------------------------------------------------------------------------

def per_gene_r2(obs, fit, eps=1e-8):
    """R2 per gene (column) across cells, relative to the flat mean of obs."""
    obs = np.asarray(obs, float); fit = np.asarray(fit, float)
    ss_res = ((obs - fit) ** 2).sum(axis=0)
    ss_tot = ((obs - obs.mean(axis=0, keepdims=True)) ** 2).sum(axis=0) + eps
    return 1.0 - ss_res / ss_tot


def geometric_r2_line(mu_obs, var_obs, mu_fit, var_fit, eps=1e-12):
    """
    Geometric R2 in the std-scaled (mu, var) plane, null = TLS straight line.

    Scaling: each axis divided by the OBSERVED per-gene std (so the metric does
    not move with a possibly-broken fit). Model SSE = sum of squared scaled
    distances observed->fitted. Line SSE = sum of squared orthogonal distances
    observed->best line (= smallest eigenvalue of the 2x2 scatter of the scaled,
    centered points; TLS). Returns (r2_2d, sse_model, sse_line).
    """
    mu_obs = np.asarray(mu_obs, float); var_obs = np.asarray(var_obs, float)
    mu_fit = np.asarray(mu_fit, float); var_fit = np.asarray(var_fit, float)

    s_mu = mu_obs.std(axis=0); s_var = var_obs.std(axis=0)
    s_mu = np.where(s_mu < eps, eps, s_mu)
    s_var = np.where(s_var < eps, eps, s_var)

    # scale both observed and fitted by the SAME observed std
    x = mu_obs / s_mu; y = var_obs / s_var          # [N, G] scaled observed
    xf = mu_fit / s_mu; yf = var_fit / s_var          # [N, G] scaled fitted

    sse_model = ((x - xf) ** 2 + (y - yf) ** 2).sum(axis=0)   # [G]

    # TLS line residual: orthogonal distance to best line through the centroid.
    xc = x - x.mean(axis=0); yc = y - y.mean(axis=0)
    Sxx = (xc ** 2).sum(axis=0)
    Syy = (yc ** 2).sum(axis=0)
    Sxy = (xc * yc).sum(axis=0)
    half = 0.5 * (Sxx + Syy)
    disc = np.sqrt(np.clip((0.5 * (Sxx - Syy)) ** 2 + Sxy ** 2, 0, None))
    sse_line = np.clip(half - disc, eps, None)        # smallest eigenvalue = SSE_line

    r2 = 1.0 - sse_model / sse_line
    return r2, sse_model, sse_line


def state_entropy(prob, normalize=True, eps=1e-12):
    """Shannon entropy of the per-cell-gene state posterior. prob: [N, G, S].
    normalize -> /log(S) so it is in [0,1] (1 = uniform over all states).
    NOTE: full-state entropy flags within-branch ambiguity (e.g. upreg vs upper
    steady) too; for velocity/branch filtering prefer branch_posterior()."""
    p = np.clip(np.asarray(prob, float), eps, 1.0)
    H = -(p * np.log(p)).sum(axis=-1)
    if normalize:
        H = H / np.log(p.shape[-1])
    return H


def branch_posterior(prob, up_states=(0, 1), down_states=(2, 3), eps=1e-12):
    """Collapse the 4 kinetic states to UP vs DOWN branch probabilities.

    up_states / down_states index the last axis of prob [N, G, S]
    (default 0=upreg,1=upper-steady -> up; 2=downreg,3=down-steady -> down).

    Returns (p_up, p_down, branch_entropy, branch_conf), each [N, G]:
      branch_entropy : binary entropy of {p_up, p_down} in [0,1] (1 = 50/50 tie)
      branch_conf    : max(p_up, p_down) in [0.5, 1] (1 = certain branch)
    """
    p = np.asarray(prob, float)
    p_up = p[..., list(up_states)].sum(-1)
    p_down = p[..., list(down_states)].sum(-1)
    s = p_up + p_down + eps
    p_up = p_up / s
    p_down = p_down / s
    Hb = -(p_up * np.log(p_up + eps) + p_down * np.log(p_down + eps)) / np.log(2)
    return p_up, p_down, Hb, np.maximum(p_up, p_down)


def auto_conf_threshold(branch_conf, target_error=0.15, min_conf=0.5):
    """Automatically pick a branch-confidence threshold.

    branch_conf is (under a calibrated posterior) P(branch call correct), so
    1 - branch_conf is the expected misassignment rate. This returns the LEAST
    strict threshold whose RETAINED cells have mean expected misassignment
    <= target_error -- i.e. you choose the error budget you tolerate and the
    threshold follows the data (no arbitrary cutoff, no knee needed).

    Returns (threshold, retained_fraction, realized_error).
    """
    c = np.sort(np.asarray(branch_conf, float).ravel())[::-1]   # descending conf
    if c.size == 0:
        return min_conf, 0.0, 0.0
    err = 1.0 - c                                                # ascending error
    cum_mean_err = np.cumsum(err) / np.arange(1, c.size + 1)     # non-decreasing
    ok = cum_mean_err <= target_error
    if not ok.any():
        # can't meet the budget even with the single most-confident cell
        return float(max(c[0], min_conf)), 1.0 / c.size, float(err[0])
    k = int(np.max(np.where(ok)[0]))                             # largest prefix
    thr = float(max(c[k], min_conf))
    keep = np.asarray(branch_conf, float) >= thr
    return thr, float(keep.mean()), float(np.mean(1.0 - np.asarray(branch_conf)[keep]))


def confident_cell_mask(prob, velocity=None, up_states=(0, 1), down_states=(2, 3),
                        conf_thresh=0.9, min_frac=0.7, weight_by_velocity=True):
    """Per-cell [N] boolean mask (True = keep) of branch-confident cells.

    A cell is kept if the (optionally |velocity|-weighted) fraction of genes
    whose UP/DOWN branch posterior is confident (branch_conf >= conf_thresh) is
    at least `min_frac`. Weighting by |velocity| lets the directionally-active
    genes decide and stops uninformative near-steady genes from dominating.

    prob     : [N, G, S] state posterior (e.g. adata.uns/obsm prob_state or the
               saved prob_state_avg).
    velocity : [N, G] velocity_mu (for the per-gene weight); optional.
    """
    _, _, _, conf = branch_posterior(prob, up_states, down_states)   # [N, G]
    confident = conf >= conf_thresh                                   # [N, G] bool
    if weight_by_velocity and velocity is not None:
        w = np.abs(np.asarray(velocity, float)).mean(axis=0)
        w = w / (w.sum() + 1e-12)
        frac = (confident * w[None, :]).sum(axis=1)                   # weighted keep-frac
    else:
        frac = confident.mean(axis=1)
    return frac >= min_frac


def top_right_mask(mu, var, q=0.9, q_mu=None, q_var=None):
    """Vectorized [N, G] boolean mask of the 'top-right corner' of each gene's
    (mu, var) scatter: cells whose mu is above the gene's q_mu quantile AND var
    above the gene's q_var quantile. Thresholds are per gene (np.quantile over
    cells, axis=0), so they adapt to each gene's scale. No Python loop.

    q       : shared quantile if q_mu/q_var not given (e.g. 0.9 -> top 10%).
    Returns : mask [N, G]  (True = top-right corner). Use ~mask to KEEP the rest.
    """
    mu = np.asarray(mu, float); var = np.asarray(var, float)
    q_mu = q if q_mu is None else q_mu
    q_var = q if q_var is None else q_var
    mu_thr = np.quantile(mu, q_mu, axis=0)      # (G,)
    var_thr = np.quantile(var, q_var, axis=0)   # (G,)
    return (mu > mu_thr[None, :]) & (var > var_thr[None, :])


def tls_predict_mu_var(mu_obs, var_obs, eps=1e-12):
    """Predicted (mu, var) from a per-gene TOTAL-LEAST-SQUARES line.

    For each gene, fit a straight line to the (mu_obs, var_obs) cloud by TLS
    (orthogonal distance, symmetric in the two noisy moments) in the std-scaled
    plane, and return each cell's ORTHOGONAL PROJECTION onto that line, mapped
    back to the original scale. This is the null-model reconstruction used by the
    line-null geometric R2 -- a straight line in the (mu, var) plane, giving both
    a predicted mu and a predicted var per cell.

    Returns (mu_pred [N,G], var_pred [N,G]).
    """
    mu = np.asarray(mu_obs, float); var = np.asarray(var_obs, float)
    s_mu = mu.std(axis=0); s_mu = np.where(s_mu < eps, eps, s_mu)
    s_var = var.std(axis=0); s_var = np.where(s_var < eps, eps, s_var)

    x = mu / s_mu; y = var / s_var                 # scale axes to be comparable
    xm = x.mean(axis=0); ym = y.mean(axis=0)
    xc = x - xm; yc = y - ym                        # centre

    Sxx = (xc ** 2).sum(axis=0)
    Syy = (yc ** 2).sum(axis=0)
    Sxy = (xc * yc).sum(axis=0)
    theta = 0.5 * np.arctan2(2 * Sxy, Sxx - Syy)    # major-axis angle per gene
    ux = np.cos(theta); uy = np.sin(theta)          # unit line direction (G,)

    t = xc * ux + yc * uy                           # projection coordinate (N,G)
    x_hat = t * ux + xm
    y_hat = t * uy + ym
    return x_hat * s_mu, y_hat * s_var              # back to original scale


def scale_ratio(obs, fit, eps=1e-8):
    """median(fit)/median(obs) per gene -- != 1 flags systematic scale bias."""
    obs = np.asarray(obs, float); fit = np.asarray(fit, float)
    return (np.median(fit, axis=0) + eps) / (np.median(obs, axis=0) + eps)


def per_gene_corr(A, B, eps=1e-12):
    """Pearson correlation per gene (column) between two [N, G] arrays."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    Ac = A - A.mean(0, keepdims=True); Bc = B - B.mean(0, keepdims=True)
    num = (Ac * Bc).sum(0)
    den = np.sqrt((Ac ** 2).sum(0) * (Bc ** 2).sum(0)) + eps
    return num / den


def _pct(x):
    x = np.asarray(x, float)
    return (f"min={np.nanmin(x):.3g} p1={np.nanpercentile(x,1):.3g} "
            f"p50={np.nanpercentile(x,50):.3g} p99={np.nanpercentile(x,99):.3g} "
            f"max={np.nanmax(x):.3g}")


def report_layer_health(mu_obs, var_obs, mu_fit, var_fit):
    """Diagnose whether observed and fitted layers are on the same scale.

    Prints per-layer value ranges + NaN/inf counts, the global fit/obs scale
    factor, and the distribution of per-gene correlation(obs, fit). High
    correlation but off-scale => a units/normalization/per-cell-size mismatch
    (R2 is over-penalising a scale difference, not bad fits). Low correlation
    => genuinely poor fits or a wrong layer pairing.
    """
    layers = dict(mu_obs=mu_obs, var_obs=var_obs, mu_fit=mu_fit, var_fit=var_fit)
    print("\n--- layer health ---")
    for name, L in layers.items():
        L = np.asarray(L, float)
        n_nan = int(np.isnan(L).sum()); n_inf = int(np.isinf(L).sum())
        Lf = L[np.isfinite(L)]
        print(f"  {name:8s}: {_pct(Lf)}   NaN={n_nan}  inf={n_inf}")

    gm_mu = np.median(np.asarray(mu_fit, float)) / (np.median(np.asarray(mu_obs, float)) + 1e-8)
    gm_var = np.median(np.asarray(var_fit, float)) / (np.median(np.asarray(var_obs, float)) + 1e-8)
    print(f"  global median fit/obs:  mu={gm_mu:.3g}   var={gm_var:.3g}")

    c_mu = per_gene_corr(mu_obs, mu_fit)
    c_var = per_gene_corr(var_obs, var_fit)
    print(f"  per-gene corr(mu_obs, mu_fit) : median={np.nanmedian(c_mu):.3f} "
          f"(>0.7 for {np.mean(c_mu > 0.7)*100:.0f}% of genes)")
    print(f"  per-gene corr(var_obs, var_fit): median={np.nanmedian(c_var):.3f} "
          f"(>0.7 for {np.mean(c_var > 0.7)*100:.0f}% of genes)")
    print("  interpretation:")
    print("    high corr + off-scale  -> units/normalization/per-cell-size mismatch")
    print("                              (align spaces before R2; layers not comparable as-is)")
    print("    low corr               -> genuinely poor fits or wrong layer pairing")
    return dict(corr_mu=c_mu, corr_var=c_var, global_ratio_mu=gm_mu, global_ratio_var=gm_var)


def score_gene_fit(mu_obs, var_obs, mu_fit, var_fit, null="line", threshold=None,
                   combine="min", gene_names=None):
    """
    Returns per-gene arrays and a `reliable` mask based on the chosen null.
      null='line' : reliable = geom_r2_line >= threshold (default 0.0)
      null='mean' : reliable = fit_r2       >= threshold (default 0.5)
    """
    r2_mu = per_gene_r2(mu_obs, mu_fit)
    r2_var = per_gene_r2(var_obs, var_fit)
    fit_r2 = np.minimum(r2_mu, r2_var) if combine == "min" else 0.5 * (r2_mu + r2_var)
    geom_r2_line, sse_model, sse_line = geometric_r2_line(
        mu_obs, var_obs, mu_fit, var_fit)

    if null == "line":
        score = geom_r2_line
        thr = 0.0 if threshold is None else threshold
    elif null == "mean":
        score = fit_r2
        thr = 0.5 if threshold is None else threshold
    else:
        raise ValueError("null must be 'line' or 'mean'")
    reliable = score >= thr

    G = len(fit_r2)
    if gene_names is None:
        gene_names = np.array([f"gene_{g}" for g in range(G)])
    return dict(
        gene=np.asarray(gene_names),
        r2_mu=r2_mu, r2_var=r2_var, fit_r2=fit_r2,
        geom_r2_line=geom_r2_line, sse_model=sse_model, sse_line=sse_line,
        scale_ratio_mu=scale_ratio(mu_obs, mu_fit),
        scale_ratio_var=scale_ratio(var_obs, var_fit),
        reliable=reliable, null=null, threshold=thr, score=score,
    )


def summarize(res, known_bad=None, top=15):
    score = res["score"]; genes = res["gene"]; null = res["null"]; thr = res["threshold"]
    score_name = "geom_r2_line" if null == "line" else "fit_r2"
    G = len(score)
    print(f"\nnull = {null}   score = {score_name}   threshold = {thr}")
    print(f"genes: {G}   reliable ({score_name} >= {thr}): "
          f"{int(res['reliable'].sum())}   dropped: {int((~res['reliable']).sum())}")
    for q in (1, 5, 10, 25, 50, 75, 90):
        print(f"  {score_name} {q:>2d}th pct = {np.percentile(score, q): .3f}")
    print(f"  {score_name} < 0 (worse than the null): {int((score < 0).sum())} genes")

    order = np.argsort(score)
    print(f"\nworst {top} genes by {score_name}:")
    print(f"{'gene':<20}{score_name:>14}{'r2_mu':>8}{'r2_var':>8}"
          f"{'scale_mu':>10}{'scale_var':>10}")
    for g in order[:top]:
        print(f"{str(genes[g]):<20}{score[g]:>14.2f}{res['r2_mu'][g]:>8.2f}"
              f"{res['r2_var'][g]:>8.2f}{res['scale_ratio_mu'][g]:>10.2g}"
              f"{res['scale_ratio_var'][g]:>10.2g}")

    if known_bad:
        idx = {str(gn): i for i, gn in enumerate(genes)}
        print(f"\ncalibration -- where your known-bad genes fall ({score_name}):")
        for gn in known_bad:
            if gn in idx:
                i = idx[gn]
                pct = 100.0 * (score < score[i]).mean()
                flag = "DROP" if not res["reliable"][i] else "kept"
                print(f"  {gn}: {score_name}={score[i]:.2f}  "
                      f"(below {pct:.0f}% of genes)  [{flag}]")
            else:
                print(f"  {gn}: not found")
        print("  -> set --threshold just above these to catch them as a group.")


# ---------------------------------------------------------------------------
# per-cell geometric distances + confidence-restricted reporting
# ---------------------------------------------------------------------------

def _geom_percell_d2(mu_obs, var_obs, mu_fit, var_fit, eps=1e-12):
    """Per-cell squared distances [N,G] in the std-scaled (mu,var) plane:
    model_d2 (obs->fit) and line_d2 (orthogonal dist to the per-gene TLS line).
    Summing each over cells reproduces geometric_r2_line's SSE_model / SSE_line."""
    mu = np.asarray(mu_obs, float); var = np.asarray(var_obs, float)
    mf = np.asarray(mu_fit, float); vf = np.asarray(var_fit, float)
    s_mu = mu.std(0); s_mu = np.where(s_mu < eps, eps, s_mu)
    s_var = var.std(0); s_var = np.where(s_var < eps, eps, s_var)
    x = mu / s_mu; y = var / s_var; xf = mf / s_mu; yf = vf / s_var
    model_d2 = (x - xf) ** 2 + (y - yf) ** 2
    xc = x - x.mean(0); yc = y - y.mean(0)
    Sxx = (xc ** 2).sum(0); Syy = (yc ** 2).sum(0); Sxy = (xc * yc).sum(0)
    theta = 0.5 * np.arctan2(2 * Sxy, Sxx - Syy)
    ux = np.cos(theta); uy = np.sin(theta)
    perp = (-uy[None, :]) * xc + (ux[None, :]) * yc        # dist to line along normal
    line_d2 = perp ** 2
    return model_d2, line_d2


def _three_metrics(model_d2, line_d2, mask=None, eps=1e-12):
    """Per-gene (geom_r2_line, frac_beats_line, median_ratio), optionally over a
    [N,G] keep-mask (NaN-aware). All robust-friendly; median_ratio and
    frac_beats_line are insensitive to a few outlier cells."""
    md = np.asarray(model_d2, float); ld = np.asarray(line_d2, float)
    if mask is not None:
        md = np.where(mask, md, np.nan); ld = np.where(mask, ld, np.nan)
    geom = 1.0 - np.nansum(md, 0) / (np.nansum(ld, 0) + eps)
    beats = np.where(np.isnan(md), np.nan, (md < ld).astype(float))
    frac = np.nanmean(beats, 0)
    med = 1.0 - np.nanmedian(md, 0) / (np.nanmedian(ld, 0) + eps)
    return geom, frac, med


def _geom_r2_masked(mu, var, mf, vf, mask, eps=1e-12):
    """Per-gene geometric line-null R2 using ONLY the cells in `mask` [N,G].
    Scaling std and the TLS line are computed within the masked (branch) subset,
    NaN-aware. Returns (r2 [G], n_cells [G])."""
    mu = np.where(mask, mu, np.nan); var = np.where(mask, var, np.nan)
    mf = np.where(mask, mf, np.nan); vf = np.where(mask, vf, np.nan)
    s_mu = np.nanstd(mu, axis=0); s_mu = np.where(s_mu < eps, eps, s_mu)
    s_var = np.nanstd(var, axis=0); s_var = np.where(s_var < eps, eps, s_var)
    x = mu / s_mu; y = var / s_var; xf = mf / s_mu; yf = vf / s_var
    sse_model = np.nansum((x - xf) ** 2 + (y - yf) ** 2, axis=0)
    xc = x - np.nanmean(x, axis=0); yc = y - np.nanmean(y, axis=0)
    Sxx = np.nansum(xc ** 2, axis=0); Syy = np.nansum(yc ** 2, axis=0)
    Sxy = np.nansum(xc * yc, axis=0)
    half = 0.5 * (Sxx + Syy)
    disc = np.sqrt(np.clip((0.5 * (Sxx - Syy)) ** 2 + Sxy ** 2, 0, None))
    sse_line = np.clip(half - disc, eps, None)
    return 1.0 - sse_model / sse_line, mask.sum(0)


def _aggregate_branches(r2_up, r2_dn, n_up, n_dn, min_cells, agg):
    """Gate branches with too few cells (-> NaN) and aggregate up/down R2."""
    r2_up = np.where(n_up >= min_cells, r2_up, np.nan)
    r2_dn = np.where(n_dn >= min_cells, r2_dn, np.nan)
    stack = np.vstack([r2_up, r2_dn])
    with np.errstate(invalid="ignore"):
        if agg == "max":
            r2 = np.nanmax(stack, axis=0)
        elif agg == "min":
            r2 = np.nanmin(stack, axis=0)
        elif agg == "weighted":
            w = np.vstack([np.where(n_up >= min_cells, n_up, 0.0),
                           np.where(n_dn >= min_cells, n_dn, 0.0)]).astype(float)
            wsum = w.sum(0)
            r2 = np.where(wsum > 0,
                          np.nansum(np.nan_to_num(stack) * w, axis=0) / (wsum + 1e-12),
                          np.nan)
        else:
            raise ValueError("agg must be max|min|weighted")
    r2 = np.where(np.all(np.isnan(stack), axis=0), np.nan, r2)
    return r2, r2_up, r2_dn


def geometric_r2_perbranch(mu_obs, var_obs, mu_fit, var_fit, prob_state,
                           up_states=(0, 1), down_states=(2, 3), min_cells=30,
                           agg="max", keep_mask=None):
    """Per-BRANCH geometric line-null R2, then aggregate.

    Split cells per gene into UP vs DOWN by argmax branch posterior; fit a
    SEPARATE straight-line null within each branch (single arc -> fair line) and
    compute the geometric R2 there. A branch with fewer than `min_cells` assigned
    cells is gated out (NaN). Aggregate the (up-to-two) branch R2s by `agg`:
      "max"      -> best branch (lenient; a gene fits well on its dominant branch)
      "weighted" -> cell-count-weighted mean (overall fit across branches)
      "min"      -> strict (both branches must fit)

    keep_mask : optional [N,G] boolean; if given, each branch is further
                restricted to these cells (e.g. branch-confident cells), so the
                per-branch lines aren't contaminated by weakly-assigned cells.

    Returns dict: r2 [G] (aggregated), r2_up, r2_down, n_up, n_down.
    """
    p_up, p_down, _, _ = branch_posterior(prob_state, up_states, down_states)
    up_mask = p_up >= p_down
    dn_mask = ~up_mask
    if keep_mask is not None:
        up_mask = up_mask & keep_mask
        dn_mask = dn_mask & keep_mask
    r2_up, n_up = _geom_r2_masked(mu_obs, var_obs, mu_fit, var_fit, up_mask)
    r2_dn, n_dn = _geom_r2_masked(mu_obs, var_obs, mu_fit, var_fit, dn_mask)
    r2, r2_up, r2_dn = _aggregate_branches(r2_up, r2_dn, n_up, n_dn, min_cells, agg)
    return dict(r2=r2, r2_up=r2_up, r2_down=r2_dn, n_up=n_up, n_down=n_dn)


def _r2_meannull(obs, fit, mask=None, eps=1e-8):
    """Per-gene mean-null R2 (obs vs fit), NaN-aware over an optional [N,G] mask.
    The baseline mean is taken over the KEPT cells of each gene."""
    obs = np.asarray(obs, float); fit = np.asarray(fit, float)
    if mask is not None:
        obs = np.where(mask, obs, np.nan); fit = np.where(mask, fit, np.nan)
    ss_res = np.nansum((obs - fit) ** 2, axis=0)
    mean = np.nanmean(obs, axis=0)
    ss_tot = np.nansum((obs - mean[None, :]) ** 2, axis=0) + eps
    return 1.0 - ss_res / ss_tot


def fit_r2_perbranch(mu_obs, var_obs, mu_fit, var_fit, prob_state,
                     up_states=(0, 1), down_states=(2, 3), min_cells=30,
                     agg="max", keep_mask=None, combine="min"):
    """Per-BRANCH MEAN-null R2 (fit_r2 = min of mu/var mean-null R2 within each
    branch), then aggregate up/down. Same split/gate/aggregate machinery as
    geometric_r2_perbranch, but the null is a flat mean rather than a TLS line.
    Returns dict: r2 [G] (aggregated), r2_up, r2_down, n_up, n_down."""
    p_up, p_down, _, _ = branch_posterior(prob_state, up_states, down_states)
    up_mask = p_up >= p_down
    dn_mask = ~up_mask
    if keep_mask is not None:
        up_mask = up_mask & keep_mask
        dn_mask = dn_mask & keep_mask
    comb = np.minimum if combine == "min" else (lambda a, b: 0.5 * (a + b))

    def branch_fit(mask):
        r2 = comb(_r2_meannull(mu_obs, mu_fit, mask),
                  _r2_meannull(var_obs, var_fit, mask))
        return r2, mask.sum(0)

    r2_up, n_up = branch_fit(up_mask)
    r2_dn, n_dn = branch_fit(dn_mask)
    r2, r2_up, r2_dn = _aggregate_branches(r2_up, r2_dn, n_up, n_dn, min_cells, agg)
    return dict(r2=r2, r2_up=r2_up, r2_down=r2_dn, n_up=n_up, n_down=n_dn)


def confidence_report(mu_obs, var_obs, mu_fit, var_fit, prob_state,
                      conf_thresh=0.9, up_states=(0, 1), down_states=(2, 3),
                      max_drop_pct=20.0, combine="min", branch_min_cells=30,
                      branch_agg="max"):
    """Per-gene geometric metrics for ALL cells and for BRANCH-CONFIDENT cells.

    Restriction is per cell-gene: keep cells whose UP/DOWN branch posterior is
    confident (branch_conf >= conf_thresh) for that gene -- an unbiased filter
    (independent of the model-vs-line residual). Flags genes that lose more than
    `max_drop_pct` percent of their cells.

    Returns a dict of per-gene arrays.
    """
    model_d2, line_d2 = _geom_percell_d2(mu_obs, var_obs, mu_fit, var_fit)
    _, _, _, branch_conf = branch_posterior(prob_state, up_states, down_states)
    keep = branch_conf >= conf_thresh                      # [N, G]

    geom_a, frac_a, med_a = _three_metrics(model_d2, line_d2)
    geom_c, frac_c, med_c = _three_metrics(model_d2, line_d2, mask=keep)

    # mean-null fit R2 (min of mu/var), all cells and confident cells
    comb = np.minimum if combine == "min" else (lambda a, b: 0.5 * (a + b))
    fit_a = comb(_r2_meannull(mu_obs, mu_fit), _r2_meannull(var_obs, var_fit))
    fit_c = comb(_r2_meannull(mu_obs, mu_fit, keep), _r2_meannull(var_obs, var_fit, keep))

    # per-branch geometric R2 (split up/down by argmax, separate line per branch);
    # all cells, and restricted to branch-confident cells (robust to weak argmax).
    pb = geometric_r2_perbranch(mu_obs, var_obs, mu_fit, var_fit, prob_state,
                                up_states, down_states, min_cells=branch_min_cells,
                                agg=branch_agg)
    pbc = geometric_r2_perbranch(mu_obs, var_obs, mu_fit, var_fit, prob_state,
                                 up_states, down_states, min_cells=branch_min_cells,
                                 agg=branch_agg, keep_mask=keep)
    # per-branch MEAN-null R2 (all cells + confident)
    fb = fit_r2_perbranch(mu_obs, var_obs, mu_fit, var_fit, prob_state,
                          up_states, down_states, min_cells=branch_min_cells,
                          agg=branch_agg, combine=combine)
    fbc = fit_r2_perbranch(mu_obs, var_obs, mu_fit, var_fit, prob_state,
                           up_states, down_states, min_cells=branch_min_cells,
                           agg=branch_agg, keep_mask=keep, combine=combine)

    pct_ret = 100.0 * keep.mean(0)
    high_loss = (100.0 - pct_ret) > max_drop_pct
    return dict(
        geom_r2_line=geom_a, frac_beats_line=frac_a, median_ratio=med_a,
        fit_r2=fit_a,
        geom_r2_line_conf=geom_c, frac_beats_line_conf=frac_c,
        median_ratio_conf=med_c, fit_r2_conf=fit_c,
        geom_r2_branch=pb["r2"], geom_r2_branch_up=pb["r2_up"],
        geom_r2_branch_down=pb["r2_down"], n_up=pb["n_up"], n_down=pb["n_down"],
        geom_r2_branch_conf=pbc["r2"], geom_r2_branch_up_conf=pbc["r2_up"],
        geom_r2_branch_down_conf=pbc["r2_down"],
        n_up_conf=pbc["n_up"], n_down_conf=pbc["n_down"],
        fit_r2_branch=fb["r2"], fit_r2_branch_up=fb["r2_up"],
        fit_r2_branch_down=fb["r2_down"],
        fit_r2_branch_conf=fbc["r2"], fit_r2_branch_up_conf=fbc["r2_up"],
        fit_r2_branch_down_conf=fbc["r2_down"],
        pct_cells_retained=pct_ret, high_cell_loss=high_loss,
    )


def print_confidence_report(rep, conf_thresh, max_drop_pct):
    def med(x): return float(np.nanmedian(x))
    print(f"\n=== geometric line-null: ALL cells vs branch-confident cells "
          f"(branch_conf >= {conf_thresh}) ===")
    print(f"{'metric (median over genes)':32s}{'all':>10}{'confident':>12}")
    print(f"{'geom_r2_line':32s}{med(rep['geom_r2_line']):>10.3f}"
          f"{med(rep['geom_r2_line_conf']):>12.3f}")
    print(f"{'frac_beats_line':32s}{med(rep['frac_beats_line']):>10.3f}"
          f"{med(rep['frac_beats_line_conf']):>12.3f}")
    print(f"{'median_ratio':32s}{med(rep['median_ratio']):>10.3f}"
          f"{med(rep['median_ratio_conf']):>12.3f}")
    print(f"{'fit_r2 (mean null)':32s}{med(rep['fit_r2']):>10.3f}"
          f"{med(rep['fit_r2_conf']):>12.3f}")
    print(f"\nper-branch geometric R2 (separate line per up/down branch):")
    print(f"  median geom_r2_branch      (all cells): {med(rep['geom_r2_branch']):.3f}"
          f"   [single-line all: {med(rep['geom_r2_line']):.3f}]")
    print(f"  median geom_r2_branch_conf (confident): {med(rep['geom_r2_branch_conf']):.3f}")
    print(f"  median up-branch R2: {med(rep['geom_r2_branch_up']):.3f}   "
          f"down-branch R2: {med(rep['geom_r2_branch_down']):.3f}")
    print(f"  median fit_r2_branch (mean null, all): {med(rep['fit_r2_branch']):.3f}"
          f"   confident: {med(rep['fit_r2_branch_conf']):.3f}")
    nb = int(np.sum(np.isnan(rep['geom_r2_branch'])))
    print(f"  genes with no branch passing the cell gate: {nb}")
    pr = rep['pct_cells_retained']
    print(f"\ncells retained after filter: median {np.median(pr):.1f}%  "
          f"min {np.min(pr):.1f}%  (p5 {np.percentile(pr,5):.1f}%)")
    nfl = int(rep['high_cell_loss'].sum())
    print(f"genes flagged high_cell_loss (lost > {max_drop_pct:.0f}% of cells): "
          f"{nfl} / {len(pr)}")
    # genes rescued: negative -> non-negative geom_r2 after filtering
    rescued = int(np.sum((rep['geom_r2_line'] < 0) & (rep['geom_r2_line_conf'] >= 0)))
    print(f"genes with geom_r2_line < 0 rescued to >= 0 by the filter: {rescued}")


# ---------------------------------------------------------------------------
# plotting: per-branch R2 scatter (up vs down) with KDE density
# ---------------------------------------------------------------------------

def plot_branch_r2_scatter(r2_up, r2_down, n_up, n_down, min_cells, out_png,
                           title="per-branch geometric R2", clip_min=None,
                           xlabel="geom_r2  (up branch)",
                           ylabel="geom_r2  (down branch)"):
    """Scatter of up-branch vs down-branch geometric R2, one point per gene,
    over a KDE density of the genes with both branches present.

    Not-computable (gated) branches are set to 0 (the line is nested under the
    parabola, so 0 is the natural floor). Points are coloured by which branch
    was absent (too few cells): both present / up absent / down absent / both
    absent. Saves to out_png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ru = np.nan_to_num(np.asarray(r2_up, float), nan=0.0)
    rd = np.nan_to_num(np.asarray(r2_down, float), nan=0.0)
    if clip_min is not None:                 # clamp values (not the axes): x,x<clip_min -> clip_min
        ru = np.maximum(ru, clip_min)
        rd = np.maximum(rd, clip_min)
    n_up = np.asarray(n_up); n_down = np.asarray(n_down)
    up_ok = n_up >= min_cells
    dn_ok = n_down >= min_cells
    both = up_ok & dn_ok
    up_abs = (~up_ok) & dn_ok
    dn_abs = up_ok & (~dn_ok)
    both_abs = (~up_ok) & (~dn_ok)

    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    # --- density of the both-present genes ---
    if both.sum() >= 10:
        try:
            import seaborn as sns
            sns.kdeplot(x=ru[both], y=rd[both], fill=True, cmap="Blues",
                        thresh=0.05, levels=15, ax=ax)
        except Exception:
            hb = ax.hexbin(ru[both], rd[both], gridsize=40, cmap="Blues",
                           mincnt=1, alpha=0.9)
            fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04, label="genes")

    # --- points on top, coloured by branch availability ---
    ax.scatter(ru[both], rd[both], s=9, c="#222222", alpha=0.35,
               linewidths=0, label=f"both branches ({both.sum()})")
    ax.scatter(ru[up_abs], rd[up_abs], s=20, c="#d62728", alpha=0.8,
               linewidths=0, label=f"up branch absent ({up_abs.sum()})")
    ax.scatter(ru[dn_abs], rd[dn_abs], s=20, c="#2ca02c", alpha=0.8,
               linewidths=0, label=f"down branch absent ({dn_abs.sum()})")
    if both_abs.any():
        ax.scatter(ru[both_abs], rd[both_abs], s=20, c="#9467bd", alpha=0.8,
                   linewidths=0, label=f"both absent ({both_abs.sum()})")

    ax.axline((0, 0), slope=1, ls="--", c="gray", lw=1)
    ax.axhline(0, c="gray", lw=0.6, ls=":"); ax.axvline(0, c="gray", lw=0.6, ls=":")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # if clip_min is not None:
    #     ax.set_xlim(clip_min, 1.05)
    #     ax.set_ylim(clip_min, 1.05)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_png


# ---------------------------------------------------------------------------
# logging (tee stdout -> console + .log file)
# ---------------------------------------------------------------------------

class _Tee:
    def __init__(self, path):
        self.file = open(path, "w")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.flush()
        self.file.close()


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# h5ad driver
# ---------------------------------------------------------------------------

def run_on_h5ad(path, mu_obs_layer="mu_scvi_smooth", var_obs_layer="var_scvi_smooth",
                mu_fit_layer="mu_fit", var_fit_layer="var_fit", null="line",
                threshold=None, combine="min", known_bad=None, write_h5ad=False,
                out_h5ad=None, out_csv=None, log_file=None, diagnose_only=False,
                prob_state=None, conf_thresh=0.9, up_states=(0, 1),
                down_states=(2, 3), max_drop_pct=20.0, target_error=None,
                branch_min_cells=30, branch_agg="max", out_dir=None):
    out_dir = out_dir or (os.path.dirname(path) or ".")
    os.makedirs(out_dir, exist_ok=True)
    if log_file is None:
        log_file = os.path.join(out_dir, f"gene_fit_{time.strftime('%Y%m%d_%H%M%S')}.log")
    tee = _Tee(log_file)
    old_stdout = sys.stdout
    sys.stdout = tee
    try:
        import anndata as ad
        import pandas as pd

        print(f"[{_now()}] score_gene_fit start")
        print(f"  h5ad        : {os.path.abspath(path)}")
        print(f"  log_file    : {os.path.abspath(log_file)}")
        t0 = time.time()
        adata = ad.read_h5ad(path)
        print(f"  loaded      : {adata.n_obs} cells x {adata.n_vars} genes "
              f"({time.time()-t0:.1f}s)")

        def layer(n):
            if n not in adata.layers:
                raise KeyError(f"layer '{n}' not in {path}; available {list(adata.layers)}")
            return np.asarray(adata.layers[n])

        print(f"  layers      : obs=({mu_obs_layer},{var_obs_layer})  "
              f"fit=({mu_fit_layer},{var_fit_layer})")
        print(f"  null        : {null}  combine={combine}  threshold={threshold}")

        # always run the health check first -- it explains a degenerate R2 result
        report_layer_health(layer(mu_obs_layer), layer(var_obs_layer),
                            layer(mu_fit_layer), layer(var_fit_layer))
        if diagnose_only:
            print("\n[--diagnose] health report only; skipping R2 scoring.")
            return None

        res = score_gene_fit(layer(mu_obs_layer), layer(var_obs_layer),
                             layer(mu_fit_layer), layer(var_fit_layer),
                             null=null, threshold=threshold, combine=combine,
                             gene_names=np.asarray(adata.var_names))
        summarize(res, known_bad=known_bad)

        adata.var["r2_mu"] = res["r2_mu"]
        adata.var["r2_var"] = res["r2_var"]
        adata.var["fit_r2"] = res["fit_r2"]
        adata.var["geom_r2_line"] = res["geom_r2_line"]
        adata.var["scale_ratio_mu"] = res["scale_ratio_mu"]
        adata.var["scale_ratio_var"] = res["scale_ratio_var"]
        adata.var["velocity_reliable"] = res["reliable"]

        csv_cols = {k: res[k] for k in
                    ("gene", "geom_r2_line", "fit_r2", "r2_mu", "r2_var",
                     "scale_ratio_mu", "scale_ratio_var", "reliable")}

        # ---- confidence-restricted geometric report (needs the state posterior) ----
        prob = None
        try:
            if prob_state:
                prob = np.load(prob_state)
            else:
                # try next to the INPUT h5ad (not the output dir)
                in_dir = os.path.dirname(path) or "."
                for fn in ("prob_state_avg_nosplicevelo.npy", "prob_state_avg.npy"):
                    cand = os.path.join(in_dir, fn)
                    if os.path.exists(cand):
                        prob = np.load(cand); print(f"  prob_state  : {cand}"); break
                if prob is None:
                    for key in ("prob_state", "prob_state_avg"):
                        if key in adata.obsm:
                            prob = np.asarray(adata.obsm[key]); break
                        if key in adata.uns:
                            prob = np.asarray(adata.uns[key]); break
        except Exception as e:
            print(f"  [warn] could not load prob_state: {e}")

        if prob is not None and prob.ndim == 3 and prob.shape[0] == adata.n_obs:
            if target_error is not None:
                _, _, _, bc = branch_posterior(prob, up_states, down_states)
                conf_thresh, ret, err = auto_conf_threshold(bc, target_error=target_error)
                print(f"  [auto] target_error<={target_error}  -> conf_thresh="
                      f"{conf_thresh:.3f}  (retain {ret*100:.0f}%, realized "
                      f"misassign {err:.3f})")
            print(f"  prob_state  : shape {prob.shape}  "
                  f"up={up_states} down={down_states} conf_thresh={conf_thresh:.3f}")
            rep = confidence_report(
                layer(mu_obs_layer), layer(var_obs_layer),
                layer(mu_fit_layer), layer(var_fit_layer), prob,
                conf_thresh=conf_thresh, up_states=up_states,
                down_states=down_states, max_drop_pct=max_drop_pct,
                branch_min_cells=branch_min_cells, branch_agg=branch_agg)
            print_confidence_report(rep, conf_thresh, max_drop_pct)
            for k in ("geom_r2_line", "frac_beats_line", "median_ratio", "fit_r2",
                      "geom_r2_line_conf", "frac_beats_line_conf",
                      "median_ratio_conf", "fit_r2_conf",
                      "geom_r2_branch", "geom_r2_branch_up", "geom_r2_branch_down",
                      "n_up", "n_down",
                      "geom_r2_branch_conf", "geom_r2_branch_up_conf",
                      "geom_r2_branch_down_conf", "n_up_conf", "n_down_conf",
                      "fit_r2_branch", "fit_r2_branch_up", "fit_r2_branch_down",
                      "fit_r2_branch_conf", "fit_r2_branch_up_conf",
                      "fit_r2_branch_down_conf",
                      "pct_cells_retained", "high_cell_loss"):
                adata.var[k] = rep[k]
                csv_cols[k] = rep[k]

            # --- branch R2 scatter plots (up vs down) with KDE density ---
            try:
                p1 = plot_branch_r2_scatter(
                    rep["geom_r2_branch_up"], rep["geom_r2_branch_down"],
                    rep["n_up"], rep["n_down"], branch_min_cells,
                    os.path.join(out_dir, "branch_r2_scatter.png"),
                    title="per-branch geometric R2 (all cells)")
                p2 = plot_branch_r2_scatter(
                    rep["geom_r2_branch_up_conf"], rep["geom_r2_branch_down_conf"],
                    rep["n_up_conf"], rep["n_down_conf"], branch_min_cells,
                    os.path.join(out_dir, "branch_r2_scatter_conf.png"),
                    title="per-branch geometric R2 (branch-confident cells)")
                # mean-null per-branch scatters (same gating counts)
                p3 = plot_branch_r2_scatter(
                    rep["fit_r2_branch_up"], rep["fit_r2_branch_down"],
                    rep["n_up"], rep["n_down"], branch_min_cells,
                    os.path.join(out_dir, "branch_fit_r2_scatter.png"),
                    title="per-branch mean-null R2 (all cells)", clip_min=-1.0)
                p4 = plot_branch_r2_scatter(
                    rep["fit_r2_branch_up_conf"], rep["fit_r2_branch_down_conf"],
                    rep["n_up_conf"], rep["n_down_conf"], branch_min_cells,
                    os.path.join(out_dir, "branch_fit_r2_scatter_conf.png"),
                    title="per-branch mean-null R2 (branch-confident cells)", clip_min=-1.0)
                for pp in (p1, p2, p3, p4):
                    print(f"[{_now()}] wrote {pp}")
            except Exception as e:
                print(f"  [warn] branch R2 scatter plot skipped: {e}")
        else:
            print("  [note] no valid prob_state found -> skipping confidence report "
                  "(pass --prob_state prob_state_avg.npy)")

        df = pd.DataFrame(csv_cols)
        out_csv = out_csv or os.path.join(out_dir, "gene_fit_scores.csv")
        df.to_csv(out_csv, index=False)
        print(f"\n[{_now()}] wrote {out_csv}")

        if write_h5ad:
            out_h5ad = out_h5ad or os.path.join(out_dir, os.path.basename(path))
            adata.write_h5ad(out_h5ad)
            print(f"[{_now()}] wrote .var columns (velocity_reliable, geom_r2_line, "
                  f"...) -> {out_h5ad}")
        else:
            print("(.var columns computed; pass --write_h5ad to persist them, or use "
                  "the CSV.  Filter downstream with: --gene_filter velocity_reliable)")
        print(f"[{_now()}] done in {time.time()-t0:.1f}s")
        return res
    finally:
        sys.stdout = old_stdout
        tee.close()
        print(f"log written to {os.path.abspath(log_file)}")


# ---------------------------------------------------------------------------
# self-test: planted good vs scale-broken genes
# ---------------------------------------------------------------------------

def selftest():
    rng = np.random.default_rng(0)
    N, G = 500, 30
    base = np.abs(rng.normal(5, 2, G))[None, :]
    t = np.linspace(0, 1, N)[:, None]
    mu_obs = base * (0.5 + t) + rng.normal(0, 0.1, (N, G))
    # curved var-vs-mu so the TLS line is not a trivial perfect fit
    var_obs = 1.5 * mu_obs - 0.04 * mu_obs ** 2 + rng.normal(0, 0.1, (N, G))
    mu_fit = mu_obs.copy(); var_fit = var_obs.copy()
    mu_fit[:, 0] *= 6.0                      # gene0: mu scale off 6x
    var_fit[:, 1] *= 0.15                    # gene1: var scale collapsed
    mu_fit[:, 2] = mu_obs[:, 2].mean()       # gene2: flat / no dynamics
    for null in ("line", "mean"):
        res = score_gene_fit(mu_obs, var_obs, mu_fit, var_fit, null=null)
        summarize(res, known_bad=["gene_0", "gene_1", "gene_2"], top=5)
        worst3 = set(res["gene"][np.argsort(res["score"])[:3]].tolist())
        ok = worst3 == {"gene_0", "gene_1", "gene_2"}
        print(f"[selftest null={null}] worst-3 = {sorted(worst3)} "
              f"-> {'PASS' if ok else 'FAIL'}\n")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("h5ad", nargs="?")
    p.add_argument("--mu_obs_layer", default="mu_scvi_smooth")
    p.add_argument("--var_obs_layer", default="var_scvi_smooth")
    p.add_argument("--mu_fit_layer", default="mu_fit")
    p.add_argument("--var_fit_layer", default="var_fit")
    p.add_argument("--null", default="line", choices=["line", "mean"],
                   help="baseline: 'line' = TLS straight line in std-scaled "
                        "(mu,var) plane (default); 'mean' = flat-mean reconstruction R2")
    p.add_argument("--threshold", type=float, default=None,
                   help="keep genes with score >= threshold "
                        "(default: 0.0 for line, 0.5 for mean)")
    p.add_argument("--combine", default="min", choices=["min", "mean"])
    p.add_argument("--known_bad", nargs="+", default=None,
                   help="gene ids to locate in the score distribution (calibration)")
    p.add_argument("--write_h5ad", action="store_true",
                   help="persist the .var columns back into the h5ad")
    p.add_argument("--out_h5ad", default=None)
    p.add_argument("--out_csv", default=None)
    p.add_argument("--out_dir", default=None,
                   help="directory for CSV / log / h5ad outputs "
                        "(default: alongside the input h5ad); created if missing")
    p.add_argument("--log_file", default=None,
                   help="detailed run log (default: gene_fit_<timestamp>.log next to h5ad)")
    p.add_argument("--diagnose", action="store_true",
                   help="only run the layer-health / scale check (obs vs fit "
                        "correlation, ranges, NaN/inf); skip R2 scoring")
    # --- branch-confidence restricted geometric report ---
    p.add_argument("--prob_state", default=None,
                   help="path to state posterior .npy [N,G,S] (default: look for "
                        "prob_state_avg.npy next to the h5ad, or adata.obsm/uns)")
    p.add_argument("--conf_thresh", type=float, default=0.9,
                   help="branch (up/down) posterior confidence to keep a cell-gene")
    p.add_argument("--up_states", type=int, nargs="+", default=[0, 1],
                   help="state indices forming the UP branch (default 0 1)")
    p.add_argument("--down_states", type=int, nargs="+", default=[2, 3],
                   help="state indices forming the DOWN branch (default 2 3)")
    p.add_argument("--max_drop_pct", type=float, default=20.0,
                   help="flag a gene when the confidence filter drops more than "
                        "this %% of its cells")
    p.add_argument("--target_error", type=float, default=None,
                   help="auto-pick conf_thresh so retained cells have expected "
                        "branch misassignment <= this (e.g. 0.15); overrides --conf_thresh")
    p.add_argument("--branch_min_cells", type=int, default=30,
                   help="min cells assigned to a branch for its per-branch R2 to count")
    p.add_argument("--branch_agg", default="max", choices=["max", "min", "weighted"],
                   help="aggregate the up/down per-branch R2 (default max)")
    p.add_argument("--selftest", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    a = _parse_args()
    if a.selftest:
        selftest()
    elif a.h5ad:
        run_on_h5ad(a.h5ad, mu_obs_layer=a.mu_obs_layer, var_obs_layer=a.var_obs_layer,
                    mu_fit_layer=a.mu_fit_layer, var_fit_layer=a.var_fit_layer,
                    null=a.null, threshold=a.threshold, combine=a.combine,
                    known_bad=a.known_bad, write_h5ad=a.write_h5ad,
                    out_h5ad=a.out_h5ad, out_csv=a.out_csv, log_file=a.log_file,
                    diagnose_only=a.diagnose, prob_state=a.prob_state,
                    conf_thresh=a.conf_thresh, up_states=tuple(a.up_states),
                    down_states=tuple(a.down_states), max_drop_pct=a.max_drop_pct,
                    target_error=a.target_error, branch_min_cells=a.branch_min_cells,
                    branch_agg=a.branch_agg, out_dir=a.out_dir)
    else:
        raise SystemExit("provide an h5ad path or --selftest")
