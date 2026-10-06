"""
score_gene_fit_v3 -- complete-data meanNull R2 + fair quadratic-vs-line
curvature test (ODR), per branch.

Two independent additions on top of v1/v2 (see SESSION_HANDOFF.md /
OPEN_THREADS.md for the full background; this docstring covers only what v3 adds).

------------------------------------------------------------------------
1. COMPLETE-DATA (whole-cloud, no branch split) mean-null R2
------------------------------------------------------------------------
v2 deliberately dropped the whole-cloud fit metric ("branch-specific only").
v3 brings it back, unchanged in method from v1's `per_gene_r2` (R2 of
mu_fit/var_fit against mu_obs/var_obs across ALL cells, relative to a flat
per-gene mean), because for some questions (e.g. "is this gene's fit usable
at all, before even worrying about branches") the simple whole-population
number is what you want.

  fit_r2_complete_mu, fit_r2_complete_var : per-axis, ALL cells
  fit_r2_complete                         : composite, --combine (min
                                             default | mean)
  reliability_by_mu_meanNull              : fit_r2_complete_mu  >= thresh_mu_meannull
  reliability_by_var_meanNull             : fit_r2_complete_var >= thresh_var_meannull

Scatter: fit_r2_complete_mu (x) vs fit_r2_complete_var (y), one point per
gene -> complete_fit_r2_muvar_scatter.png.

------------------------------------------------------------------------
2. Per-branch QUADRATIC vs LINE, fit directly to the data by ODR
------------------------------------------------------------------------
This is Thread E / OPEN_THREADS.md item 1: v2's lineNull compared the VAE's
PREDICTED (mu,var) to a TLS line fit to the SAME points -- unfair, because
the line is fit to the data and the VAE prediction is not. The fair test
fits BOTH a line and a quadratic DIRECTLY to the observed (mu_obs, var_obs),
by the SAME estimator (orthogonal distance -- errors in both mu and var),
and compares them. The VAE posterior is used ONLY to split cells into
up/down branches (argmax of branch_posterior; all branch-assigned cells are
used, no confidence filtering -- confirmed with the user: task 2 wants the
larger dynamic range, not a cleaner but truncated sample).

Per gene, per branch (up = states in `up_states`, down = states in
`down_states`):
  - gate: branch needs >= `branch_min_cells` cells AND enough mu DYNAMIC
    RANGE ( (max-min)/std >= `branch_min_range_ratio` ) -- curvature is not
    identifiable over a narrow arc. Ungated branches are NaN.
  - both axes are scaled by the BRANCH-LOCAL observed std (mu, var
    separately) so the two noisy axes are comparable, exactly the scaling
    convention used elsewhere in this codebase (v1's geometric_r2_line, v2's
    _linenull_axes).
  - LINE: closed-form total-least-squares (TLS) in the scaled plane (exact
    orthogonal-distance solution for a linear model -- no iteration needed).
  - QUADRATIC (var ~ a*mu^2 + b*mu + c): orthogonal distance regression via
    `scipy.odr` (errors in x AND y, matching the line's estimator). If scipy
    is not installed (this repo's algorithmic sandbox has no scipy -- see
    eval/_compat.py), a pure-numpy alternating projection / Gauss-Newton
    fallback is used instead (`_fit_quadratic_numpy`); this is an
    approximation ONLY used when scipy.odr is unavailable, documented the
    same way _compat.py documents its approximations. The real environment
    (scvelo/anndata/scipy installed) always uses scipy.odr.

For each branch, the quadratic is scored AGAINST THE LINE (not against a
flat mean): both are fit to the identical points by the identical class of
estimator (orthogonal distance), so any improvement is genuine curvature,
not a "curved model beats a null it wasn't fit under" artifact.

  qvl_r2_mu_{up,down}   = 1 - SSE_quad_mu   / SSE_line_mu     (mu-axis residual)
  qvl_r2_var_{up,down}  = 1 - SSE_quad_var  / SSE_line_var    (var-axis residual)
  qvl_r2_geom_{up,down} = 1 - SSE_quad_2D   / SSE_line_2D     (both axes, one
                                                               orthogonal SSE)
  qvl_n_{up,down}       : cells assigned to that branch (argmax)
  qvl_gate_{up,down}    : bool, branch passed the min_cells + dynamic-range gate

Curvature sign (near-free byproduct of the quadratic fit, requested as a
bonus diagnostic): the fitted `a` coefficient's sign is invariant to the
(positive) per-branch std rescaling used here (y_scaled = A*(s_mu^2/s_var) *
x_scaled^2 + ...; s_mu^2/s_var > 0), so checking sign on the scaled fit is
equivalent to checking on raw units.

  qvl_curvature_a_{up,down}                    : fitted a (scaled units)
  qvl_curvature_sign_matches_prior_{up,down}    : 1.0 if (up: a<0) / (down: a>0),
                                                   0.0 if not, NaN if branch gated out

Aggregated (over up/down, via --branch_agg, default 'max', reusing v2's
`_agg_axes` so the selected branch's axis-specific values are carried along):

  qvl_r2_geom, qvl_r2_mu, qvl_r2_var

------------------------------------------------------------------------
2b. Significance of the quadratic-vs-line improvement (is qvl_r2_geom real?)
------------------------------------------------------------------------
A positive `qvl_r2_geom` is not by itself evidence of curvature: the
quadratic has one more free parameter than the line, so its SSE can only
improve, even on purely linear data (it eats some noise). Two ways to tell
"real curvature" from "extra-parameter noise-fitting", both enabled by
default (see --no_ftest / --no_cv to skip either):

  (a) NESTED F-TEST (free -- reuses the SSE already computed above):
      F = ((SSE_line - SSE_quad)/1) / (SSE_quad/(n-3))  ~ F(1, n-3)
      p-value via scipy.stats.f.sf (real env) or a pure-numpy/math
      incomplete-beta fallback (`_f_sf_fallback`) when scipy is unavailable.
      BH-FDR is then applied ACROSS GENES, separately for up and down.

        qvl_fstat_{up,down}, qvl_pvalue_{up,down}, qvl_fdr_{up,down}
        qvl_significant_{up,down}   : qvl_fdr_{up,down} <= --fdr_alpha (0.05)
        qvl_significant_any         : significant in up OR down

      Caveat: treats the orthogonal (errors-in-variables) residuals as if
      they were OLS residuals -- an approximation standard in the
      errors-in-variables literature when scatter is modest relative to the
      curve's range (true here given the branch gate), but not exact.

  (b) K-FOLD CROSS-VALIDATED delta R2 (no distributional assumptions -- the
      recommended cross-check, costs ~cv_folds extra fits per gene/branch):
      fit line + quadratic on (k-1) folds, score both on the held-out fold
      by orthogonal projection onto the FIXED (not refit) curve, sum SSE
      over folds. A truly-linear gene's quadratic overfits in-fold noise and
      generalizes worse, so `qvl_cv_r2_geom` sits near/below 0 for it
      automatically -- no null-distribution calibration needed.

        qvl_cv_r2_geom_{up,down}    : --cv_folds (default 5), --cv_seed

  In short: use `qvl_significant_up/down` (FDR-controlled) or a positive
  `qvl_cv_r2_geom_{up,down}` -- not a raw cutoff on `qvl_r2_geom` -- to select
  genes with real per-branch curvature.

Scatters (all reuse v1/v2's plotting helpers, all clamp at -1 like v2):
  qvl_r2_geom_up_vs_down_scatter.png : geometric (2D) quad-vs-line R2, up vs down
  qvl_r2_mu_up_vs_down_scatter.png   : mu-axis quad-vs-line R2, up vs down
  qvl_r2_var_up_vs_down_scatter.png  : var-axis quad-vs-line R2, up vs down
  qvl_r2_muvar_scatter.png           : mu-axis vs var-axis quad-vs-line R2
                                        (selected/aggregated branch)

Uses the four noSpliceVelo layers:
  observed : mu_scvi_smooth , var_scvi_smooth
  fitted   : mu_fit          , var_fit        (argmax_stable recommended)
and the state posterior (prob_state_avg_nosplicevelo.npy, [N,G,S]) for task 2.
"""

import os
import sys
import time
import math
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# reuse shared, tested helpers from v1 and v2
from score_gene_fit import (branch_posterior, auto_conf_threshold, report_layer_health,
                            per_gene_r2, plot_branch_r2_scatter)
from score_gene_fit_v2 import plot_null_comparison, _agg_axes
from gene_bow_separation import (bow_separation_scores, up_mask_from_posterior,
                                 selftest as bow_selftest)


# ---------------------------------------------------------------------------
# task 1: complete-data (whole-cloud) mean-null R2
# ---------------------------------------------------------------------------

def complete_meannull_r2(mu_obs, var_obs, mu_fit, var_fit, combine="min",
                         thresh_mu_meannull=0.5, thresh_var_meannull=0.5):
    """Whole-cloud (no branch split) per-gene mean-null R2, mu and var
    separately, plus a composite and per-axis reliability flags."""
    r2_mu = per_gene_r2(mu_obs, mu_fit)
    r2_var = per_gene_r2(var_obs, var_fit)
    composite = np.minimum(r2_mu, r2_var) if combine == "min" else 0.5 * (r2_mu + r2_var)
    return dict(
        fit_r2_complete_mu=r2_mu,
        fit_r2_complete_var=r2_var,
        fit_r2_complete=composite,
        reliability_by_mu_meanNull=r2_mu >= thresh_mu_meannull,
        reliability_by_var_meanNull=r2_var >= thresh_var_meannull,
    )


# ---------------------------------------------------------------------------
# task 3: branch separation (polar angle gap in radians) & b_corner normalization
# ---------------------------------------------------------------------------

def compute_b_corner(mu_obs, var_obs, q=0.9):
    """Compute the top-right corner dispersion slope b_corner = var_corner / mu_corner.

    mu_obs, var_obs : [N, G] arrays.
    q               : quantile threshold for top-right corner (default 0.90).
    Returns b_corner [G], mu_corner [G], var_corner [G].
    """
    N, G = mu_obs.shape
    b_corner = np.full(G, np.nan)
    mu_corner = np.full(G, np.nan)
    var_corner = np.full(G, np.nan)

    q_mu = np.quantile(mu_obs, q, axis=0)     # (G,)
    q_var = np.quantile(var_obs, q, axis=0)   # (G,)

    for g in range(G):
        m_g = mu_obs[:, g]
        v_g = var_obs[:, g]
        # Top-right corner mask
        mask = (m_g >= q_mu[g]) & (v_g >= q_var[g])
        if not np.any(mask):
            # Fallback to high var cells
            mask = v_g >= q_var[g]
        if not np.any(mask):
            # Fallback to all cells
            mask = np.ones(N, dtype=bool)

        mc = np.mean(m_g[mask])
        vc = np.mean(v_g[mask])
        mu_corner[g] = mc
        var_corner[g] = vc
        b_corner[g] = vc / (mc + 1e-12)

    return b_corner, mu_corner, var_corner


def compute_branch_separation(mu_obs, var_obs, prob, up_states=(0, 1), down_states=(2, 3),
                               conf_thresh="auto", target_error=0.25, min_conf=0.5,
                               min_cells=30, n_grid=50, bandwidth=0.1):
    """Compute polar angle separation between upper and lower branches in radians.

    prob        : [N, G, S] state probabilities.
    up_states   : indices of up branch states (e.g. 0, 1).
    down_states : indices of down branch states (e.g. 2, 3).
    conf_thresh : confidence threshold ('auto' for data-driven auto_conf_threshold, or float).
    target_error: target expected misassignment error budget for auto threshold (default 0.25, lenient).
    min_cells   : minimum cells required in BOTH branches.

    Returns dict containing:
      separation      : [G] separation in radians (or NaN if gated out)
      sep_gate        : [G] bool mask (True if enough cells in both branches)
      sep_n_up        : [G] count of confident upper branch cells
      sep_n_down      : [G] count of confident lower branch cells
      sep_conf_thresh : [G] threshold used per gene
    """
    N, G = mu_obs.shape
    p_up, p_down, _, branch_conf = branch_posterior(prob, up_states=up_states, down_states=down_states)

    is_auto = conf_thresh is None or conf_thresh == "auto" or (isinstance(conf_thresh, str) and conf_thresh.lower() == "auto")
    used_thresh = np.full(G, 0.7)
    if is_auto:
        for g in range(G):
            t_g, _, _ = auto_conf_threshold(branch_conf[:, g], target_error=target_error, min_conf=min_conf)
            used_thresh[g] = t_g
        print(f"  [branch_conf_thresh] data-driven auto threshold (target_error={target_error:.2f}) mean={used_thresh.mean():.3f} (min={used_thresh.min():.3f}, max={used_thresh.max():.3f})")
    else:
        used_thresh[:] = float(conf_thresh)

    separation = np.full(G, np.nan)
    sep_gate = np.zeros(G, dtype=bool)
    sep_n_up = np.zeros(G, dtype=int)
    sep_n_down = np.zeros(G, dtype=int)

    eps = 1e-8

    for g in range(G):
        thr_g = used_thresh[g]
        # High confidence branch cell assignment
        mask_up = (p_up[:, g] >= thr_g) & (p_up[:, g] > p_down[:, g])
        mask_dn = (p_down[:, g] >= thr_g) & (p_down[:, g] > p_up[:, g])

        n_u = int(np.sum(mask_up))
        n_d = int(np.sum(mask_dn))
        sep_n_up[g] = n_u
        sep_n_down[g] = n_d

        if n_u < min_cells or n_d < min_cells:
            continue

        sep_gate[g] = True

        m_g = mu_obs[:, g]
        v_g = var_obs[:, g]

        # Standardize coordinates by overall gene std to make axes comparable
        s_m = np.std(m_g) + eps
        s_v = np.std(v_g) + eps

        x_u = m_g[mask_up] / s_m
        y_u = v_g[mask_up] / s_v
        x_d = m_g[mask_dn] / s_m
        y_d = v_g[mask_dn] / s_v

        # Polar coordinates (r_norm, theta)
        sr_u = (x_u ** 2 + y_u ** 2) ** 0.25
        th_u = np.arctan2(y_u, x_u)

        sr_d = (x_d ** 2 + y_d ** 2) ** 0.25
        th_d = np.arctan2(y_d, x_d)

        # Shared radius grid over radial overlap
        lo = max(np.min(sr_u), np.min(sr_d))
        hi = min(np.max(sr_u), np.max(sr_d))

        if hi <= lo + eps:
            # If no radial overlap, take direct absolute difference of mean angles
            separation[g] = abs(np.mean(th_u) - np.mean(th_d))
        else:
            rho = np.linspace(lo, hi, n_grid)  # (M,)
            # Nadaraya-Watson kernel smoothing along radial axis
            d_u = rho[:, None] - sr_u[None, :]
            w_u = np.exp(-0.5 * (d_u / bandwidth) ** 2)
            w_u_sum = np.sum(w_u, axis=1, keepdims=True) + eps
            th_u_at = np.sum((w_u / w_u_sum) * th_u[None, :], axis=1)

            d_d = rho[:, None] - sr_d[None, :]
            w_d = np.exp(-0.5 * (d_d / bandwidth) ** 2)
            w_d_sum = np.sum(w_d, axis=1, keepdims=True) + eps
            th_d_at = np.sum((w_d / w_d_sum) * th_d[None, :], axis=1)

            separation[g] = float(np.mean(np.abs(th_u_at - th_d_at)))

    return dict(
        separation=separation,
        sep_gate=sep_gate,
        sep_n_up=sep_n_up,
        sep_n_down=sep_n_down,
        sep_conf_thresh=used_thresh,
    )


def normalize_separation_hvg(separation, b_corner, n_bins=20):
    """Normalize separation against log1p(b_corner) using Scanpy HVG-style mean/std binning.

    separation: [G] separation scores in radians.
    b_corner  : [G] var_corner / mu_corner slope.
    n_bins    : number of bins over log1p(b_corner).

    Returns separation_norm [G].
    """
    G = len(separation)
    separation_norm = np.full(G, np.nan)

    valid = ~np.isnan(separation) & ~np.isnan(b_corner)
    if np.sum(valid) < 2:
        return separation_norm

    sep_valid = separation[valid]
    b_valid = b_corner[valid]
    b_log_valid = np.log1p(np.maximum(0.0, b_valid))

    # Compute correlation for logging / diagnostic
    try:
        from scipy.stats import pearsonr, spearmanr
        pr_lin, _ = pearsonr(sep_valid, b_valid)
        pr_log, _ = pearsonr(sep_valid, b_log_valid)
        sr, _ = spearmanr(sep_valid, b_valid)
        print(f"  [separation scaling] Pearson r(sep, b_corner) = {pr_lin:.3f}, Pearson r(sep, log1p(b_corner)) = {pr_log:.3f}, Spearman rho = {sr:.3f}")
    except Exception:
        pass

    # Quantile/equal-count binning over log1p(b_corner)
    try:
        valid_indices = np.where(valid)[0]
        sorted_order = np.argsort(b_log_valid)
        valid_sorted = valid_indices[sorted_order]

        bin_size = max(1, len(valid_sorted) // n_bins)
        for b_idx in range(n_bins):
            start = b_idx * bin_size
            end = len(valid_sorted) if b_idx == n_bins - 1 else (b_idx + 1) * bin_size
            if start >= end:
                break
            bin_genes = valid_sorted[start:end]
            sep_bin = separation[bin_genes]
            mean_bin = np.mean(sep_bin)
            std_bin = np.std(sep_bin)
            if std_bin < 1e-12 or np.isnan(std_bin):
                std_bin = 1.0
            separation_norm[bin_genes] = (sep_bin - mean_bin) / std_bin
    except Exception as e:
        print(f"  [warn] normalize_separation_hvg binning error: {e}")

    return separation_norm


def compute_cell_separation_obs(mu_obs, var_obs, prob, separation, separation_norm,
                                 up_states=(0, 1), down_states=(2, 3), conf_thresh="auto",
                                 target_error=0.25, min_conf=0.5):
    """Compute per-cell obs annotations for branch call and cell separation metrics.

    Returns dict containing:
      branch_call         : [N] str array ('upper', 'lower', 'unassigned')
      cell_separation     : [N] float array (per-cell average separation of confident genes)
      cell_separation_norm: [N] float array (per-cell average normalized separation)
    """
    N, G = mu_obs.shape
    p_up, p_down, _, branch_conf = branch_posterior(prob, up_states=up_states, down_states=down_states)

    is_auto = conf_thresh is None or conf_thresh == "auto" or (isinstance(conf_thresh, str) and conf_thresh.lower() == "auto")
    used_thresh = np.full(G, 0.7)
    if is_auto:
        for g in range(G):
            t_g, _, _ = auto_conf_threshold(branch_conf[:, g], target_error=target_error, min_conf=min_conf)
            used_thresh[g] = t_g
    else:
        used_thresh[:] = float(conf_thresh)

    mean_p_up = p_up.mean(axis=1)
    mean_p_dn = p_down.mean(axis=1)
    overall_thresh = used_thresh.mean()
    branch_call = np.full(N, "unassigned", dtype=object)
    branch_call[(mean_p_up >= overall_thresh) & (mean_p_up > mean_p_dn)] = "upper"
    branch_call[(mean_p_dn >= overall_thresh) & (mean_p_dn > mean_p_up)] = "lower"

    cell_sep = np.full(N, np.nan)
    cell_sep_norm = np.full(N, np.nan)

    conf_mask = np.maximum(p_up, p_down) >= used_thresh[None, :]
    valid_sep = ~np.isnan(separation)
    valid_sep_norm = ~np.isnan(separation_norm)

    for i in range(N):
        active_sep = conf_mask[i, :] & valid_sep
        if np.any(active_sep):
            cell_sep[i] = np.mean(separation[active_sep])

        active_norm = conf_mask[i, :] & valid_sep_norm
        if np.any(active_norm):
            cell_sep_norm[i] = np.mean(separation_norm[active_norm])

    return dict(
        branch_call=branch_call,
        cell_separation=cell_sep,
        cell_separation_norm=cell_sep_norm,
    )


# ---------------------------------------------------------------------------
# task 2: per-branch quadratic vs line by orthogonal distance regression
# ---------------------------------------------------------------------------

def _tls_fit_line(x, y):
    """Closed-form TLS line fit to 1D arrays (already scaled). Returns the
    fitted line's direction (ux,uy) and centroid (xm,ym) -- the exact
    orthogonal-distance (major-axis) solution, no iteration needed."""
    xm = x.mean(); ym = y.mean()
    xc = x - xm; yc = y - ym
    Sxx = np.sum(xc ** 2); Syy = np.sum(yc ** 2); Sxy = np.sum(xc * yc)
    theta = 0.5 * np.arctan2(2 * Sxy, Sxx - Syy)
    return np.cos(theta), np.sin(theta), xm, ym


def _tls_project_line(x, y, ux, uy, xm, ym):
    """Project (x,y) onto a FIXED TLS line (ux,uy,xm,ym) -- e.g. one fit on a
    separate training set for cross-validation. Returns residuals
    (x - x_hat, y - y_hat) to the orthogonal projection."""
    xc = x - xm; yc = y - ym
    t = xc * ux + yc * uy
    x_hat = t * ux + xm; y_hat = t * uy + ym
    return x - x_hat, y - y_hat


def _tls_line_residuals(x, y):
    """Fit + project in one call (the common in-sample case)."""
    ux, uy, xm, ym = _tls_fit_line(x, y)
    return _tls_project_line(x, y, ux, uy, xm, ym)


def _project_onto_quadratic(x, y, beta, inner_newton=8):
    """Orthogonal foot point of (x,y) on a FIXED quadratic curve
    y=a*u^2+b*u+c (beta=(a,b,c)) by Newton's method on the perpendicularity
    condition (u-x) + (f(u)-y)*f'(u) = 0, started at u=x. Used both inside
    the fallback ODR fit (refit loop) and for cross-validation (projecting
    held-out points onto a curve fit on a separate training fold). Returns
    (x_hat, y_hat)."""
    a, b, c = beta
    u = x.copy()
    for _ in range(inner_newton):
        f = a * u ** 2 + b * u + c
        fp = 2 * a * u + b
        g = (u - x) + (f - y) * fp
        gp = 1.0 + fp ** 2 + (f - y) * 2 * a
        gp = np.where(np.abs(gp) < 1e-10, 1e-10, gp)
        u = u - g / gp
    y_hat = a * u ** 2 + b * u + c
    return u, y_hat


def _fit_quadratic_numpy(x, y, beta0, maxit=60, tol=1e-8, inner_newton=6):
    """FALLBACK quadratic ODR (used only when scipy.odr is unavailable, e.g.
    this repo's algorithmic sandbox -- see eval/_compat.py's fidelity notes
    for the same convention). Alternating projected Gauss-Newton: (1) given
    (a,b,c), Newton-solve each point's orthogonal foot-point u on the curve
    (`_project_onto_quadratic`); (2) refit (a,b,c) by OLS of y on the foot
    points u (treating u as the now-estimated true x). This is a standard
    approximate errors-in-variables scheme; it is NOT ODRPACK and can differ
    slightly from scipy.odr on poorly-conditioned data, but converges well
    for mild quadratic curvature starting from the OLS quadratic fit.
    Returns (beta, x_hat, y_hat)."""
    a, b, c = beta0
    u = x.copy()
    for _ in range(maxit):
        u, _ = _project_onto_quadratic(x, y, (a, b, c), inner_newton)
        beta_new = np.polyfit(u, y, 2)
        if np.max(np.abs(beta_new - np.array([a, b, c]))) < tol:
            a, b, c = beta_new
            break
        a, b, c = beta_new
    u, y_hat = _project_onto_quadratic(x, y, (a, b, c), inner_newton)
    return np.array([a, b, c]), u, y_hat


try:
    from scipy import odr as _spodr
    _HAVE_SCIPY_ODR = True
except ImportError:
    _spodr = None
    _HAVE_SCIPY_ODR = False

try:
    from joblib import Parallel, delayed
    _HAVE_JOBLIB = True
except ImportError:
    Parallel = None
    delayed = None
    _HAVE_JOBLIB = False


def fit_quadratic_odr(x, y, maxit=200):
    """Quadratic (var ~ a*mu^2+b*mu+c) orthogonal distance regression on
    already-scaled 1D arrays. Uses scipy.odr when available (real env);
    otherwise `_fit_quadratic_numpy` (see its docstring). Returns
    (beta=[a,b,c], x_hat, y_hat) where (x_hat,y_hat) is each point's
    orthogonal foot point on the fitted curve."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    beta0 = np.polyfit(x, y, 2)
    if _HAVE_SCIPY_ODR:
        try:
            def _fcn(beta, xx):
                aa, bb, cc = beta
                return aa * xx ** 2 + bb * xx + cc
            data = _spodr.Data(x, y)
            model = _spodr.Model(_fcn)
            out = _spodr.ODR(data, model, beta0=beta0, maxit=maxit).run()
            beta = np.asarray(out.beta, float)
            x_hat = x + out.delta
            y_hat = _fcn(beta, x_hat)
            return beta, x_hat, y_hat
        except Exception:
            pass  # fall through to the numpy fallback on any ODR failure
    return _fit_quadratic_numpy(x, y, beta0, maxit=min(maxit, 60))


# ---------------------------------------------------------------------------
# significance: nested F-test (quadratic vs line) + BH-FDR across genes
# ---------------------------------------------------------------------------
# Line and quadratic are NESTED (line = quadratic with a=0), and both are
# already fit to the branch's points, so SSE_line/SSE_quad give a classical
# nested-model F-test "for free" (no extra fits): a per-gene, per-branch
# p-value for "does the quadratic explain more than you'd expect from one
# extra free parameter fitting noise", which is what raw R2 alone cannot
# tell you (a strictly-more-flexible model's SSE can only improve, so a
# small positive geom R2 is not by itself evidence of curvature).
#
#   F = ((SSE_line - SSE_quad) / 1) / (SSE_quad / (n - 3))     ~ F(1, n-3)
#
# Caveat: this treats the orthogonal (errors-in-variables) residuals as if
# they were ordinary least-squares residuals (classical F-theory assumes
# that), which is an approximation for ODR/TLS fits -- standard practice in
# the errors-in-variables literature when scatter around the curve is modest
# relative to the curve's range (true here given the min_cells + dynamic-
# range gate). It is NOT a substitute for the curvature-sign check.

try:
    from scipy import stats as _spstats
    _HAVE_SCIPY_STATS = True
except ImportError:
    _spstats = None
    _HAVE_SCIPY_STATS = False


def _betacf(x, a, b, maxit=200, eps=3.0e-12):
    """Continued-fraction evaluation for the regularized incomplete beta
    function (Numerical Recipes 6.4, Lentz's algorithm). FALLBACK ONLY --
    used for the F-test p-value when scipy.stats is unavailable (same
    convention as the quadratic-fit fallback above)."""
    qab = a + b; qap = a + 1.0; qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, maxit + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d; d = 1e-30 if abs(d) < 1e-30 else d
        c = 1.0 + aa / c; c = 1e-30 if abs(c) < 1e-30 else c
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d; d = 1e-30 if abs(d) < 1e-30 else d
        c = 1.0 + aa / c; c = 1e-30 if abs(c) < 1e-30 else c
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc_scalar(x, a, b):
    """Regularized incomplete beta I_x(a,b), scalar x (Numerical Recipes 6.4)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(x, a, b) / a
    return 1.0 - front * _betacf(1.0 - x, b, a) / b


def _f_sf_scalar(F, dfd):
    """Scalar P(X >= F) for X ~ F(1, dfd), via the F(1,nu) <-> Student-t
    identity P(F_{1,nu} >= f) = P(|T_nu| >= sqrt(f)) = I_{nu/(nu+f)}(nu/2, 1/2).
    Pure numpy/math FALLBACK used only when scipy.stats is unavailable."""
    if not (np.isfinite(F) and np.isfinite(dfd) and dfd > 0 and F >= 0):
        return float("nan")
    x = dfd / (dfd + F)
    return _betainc_scalar(x, dfd / 2.0, 0.5)


def _f_pvalue(F, dfd):
    """Scalar P(F(1,dfd) >= F); uses scipy.stats.f.sf when available (real
    env), else `_f_sf_scalar`."""
    if _HAVE_SCIPY_STATS:
        with np.errstate(invalid="ignore"):
            return float(_spstats.f.sf(float(F), 1, float(dfd)))
    return _f_sf_scalar(float(F), float(dfd))


def _bh_fdr(pvals):
    """Benjamini-Hochberg FDR q-values, NaN-preserving (NaN entries excluded
    from the correction and left NaN in the output; m = number of non-NaN
    tests)."""
    p = np.asarray(pvals, float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return q
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(p[idx])]
    ranked = p[order]
    ranks = np.arange(1, m + 1)
    raw_q = ranked * m / ranks
    q_sorted = np.clip(np.minimum.accumulate(raw_q[::-1])[::-1], 0, 1)
    q[order] = q_sorted
    return q


# ---------------------------------------------------------------------------
# K-fold cross-validated quadratic-vs-line delta R2 (no distributional
# assumptions -- directly penalizes the quadratic's extra parameter by
# scoring it on held-out cells it was not fit on)
# ---------------------------------------------------------------------------

def _cv_quad_vs_line_geom(x, y, k=5, seed=0, min_fold=5, eps=1e-12):
    """K-fold CV geometric R2 of quadratic vs line: fit both on k-1 folds,
    score both (fixed, no refitting) on the held-out fold by orthogonal
    projection, sum SSE over all folds, then
        cv_r2 = 1 - SSE_quad_heldout / SSE_line_heldout.
    A gene whose true relationship is linear will see its quadratic overfit
    in-fold noise and generalize WORSE, so cv_r2 is near/below 0 for it
    without needing any permutation null. Returns NaN if a fold is too
    small. Uses the SAME branch-wide (mu,var) std scaling as the full-data
    fit (x,y already scaled) -- a minor, harmless simplification versus
    refitting the scale per training fold."""
    n = len(x)
    if n < k * min_fold:
        return np.nan
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    sse_line_ho = 0.0; sse_quad_ho = 0.0
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        if len(train_idx) < min_fold or len(test_idx) < 1:
            return np.nan
        xt, yt = x[train_idx], y[train_idx]
        xh, yh = x[test_idx], y[test_idx]
        ux, uy, xm, ym = _tls_fit_line(xt, yt)
        dxl, dyl = _tls_project_line(xh, yh, ux, uy, xm, ym)
        beta, _, _ = fit_quadratic_odr(xt, yt)
        xhat_q, yhat_q = _project_onto_quadratic(xh, yh, beta)
        dxq = xh - xhat_q; dyq = yh - yhat_q
        sse_line_ho += np.sum(dxl ** 2 + dyl ** 2)
        sse_quad_ho += np.sum(dxq ** 2 + dyq ** 2)
    return 1.0 - sse_quad_ho / (sse_line_ho + eps)


def _process_single_gene_v3(g, mu_g, var_g, mask_up, mask_dn, min_cells,
                            min_range_ratio, run_ftest, run_cv, cv_folds,
                            cv_seed, eps):
    """Compute quadratic vs line ODR metrics for a single gene g across up/down branches."""
    res = {
        'g': g,
        'qvl_r2_mu_up': np.nan, 'qvl_r2_mu_down': np.nan,
        'qvl_r2_var_up': np.nan, 'qvl_r2_var_down': np.nan,
        'qvl_r2_geom_up': np.nan, 'qvl_r2_geom_down': np.nan,
        'qvl_curvature_a_up': np.nan, 'qvl_curvature_a_down': np.nan,
        'qvl_fstat_up': np.nan, 'qvl_fstat_down': np.nan,
        'qvl_pvalue_up': np.nan, 'qvl_pvalue_down': np.nan,
        'qvl_cv_r2_geom_up': np.nan, 'qvl_cv_r2_geom_down': np.nan,
        'qvl_gate_up': False, 'qvl_gate_down': False,
        'fit_fail_count': 0
    }

    for tag, mask in (("up", mask_up), ("down", mask_dn)):
        n = int(mask.sum())
        if n < min_cells:
            continue
        mu_sub = mu_g[mask]
        var_sub = var_g[mask]
        s_mu = mu_sub.std()
        s_var = var_sub.std()
        if s_mu < eps or s_var < eps:
            continue
        dyn_range = (mu_sub.max() - mu_sub.min()) / (s_mu + eps)
        if dyn_range < min_range_ratio:
            continue

        res[f"qvl_gate_{tag}"] = True
        x = mu_sub / s_mu
        y = var_sub / s_var
        try:
            dx_line, dy_line = _tls_line_residuals(x, y)
            beta, x_hat, y_hat = fit_quadratic_odr(x, y)
            dx_quad = x - x_hat
            dy_quad = y - y_hat
            sse_line_mu = np.sum(dx_line ** 2)
            sse_line_var = np.sum(dy_line ** 2)
            sse_quad_mu = np.sum(dx_quad ** 2)
            sse_quad_var = np.sum(dy_quad ** 2)
            sse_line_geom = sse_line_mu + sse_line_var
            sse_quad_geom = sse_quad_mu + sse_quad_var

            res[f"qvl_r2_mu_{tag}"] = 1.0 - sse_quad_mu / (sse_line_mu + eps)
            res[f"qvl_r2_var_{tag}"] = 1.0 - sse_quad_var / (sse_line_var + eps)
            res[f"qvl_r2_geom_{tag}"] = 1.0 - sse_quad_geom / (sse_line_geom + eps)
            res[f"qvl_curvature_a_{tag}"] = beta[0]

            if run_ftest:
                dof = n - 3
                if dof > 0:
                    F = max(0.0, (sse_line_geom - sse_quad_geom) / (sse_quad_geom / dof + eps))
                    res[f"qvl_fstat_{tag}"] = F
                    res[f"qvl_pvalue_{tag}"] = _f_pvalue(F, dof)

            if run_cv:
                res[f"qvl_cv_r2_geom_{tag}"] = _cv_quad_vs_line_geom(
                    x, y, k=cv_folds, seed=cv_seed)
        except Exception:
            res['fit_fail_count'] += 1

    return res


def quad_vs_line_branch(mu_obs, var_obs, prob_state, up_states=(0, 1),
                        down_states=(2, 3), min_cells=30, min_range_ratio=3.0,
                        run_ftest=True, run_cv=True, cv_folds=5, cv_seed=0,
                        fdr_alpha=0.05, eps=1e-12, progress_every=500,
                        verbose=True, n_jobs=-1):
    """Per-gene, per-branch quadratic-vs-line ODR comparison (task 2 core).
    Parallelized over genes across `n_jobs` CPU workers. Returns a dict of
    [G]-length arrays, keyed as documented in the module docstring (qvl_*).

    run_ftest : compute the nested F-test p-value (free, reuses SSE already
                computed for qvl_r2_geom) and BH-FDR across genes per branch.
    run_cv    : compute the K-fold cross-validated qvl_cv_r2_geom (adds
                ~cv_folds extra line+quadratic fits per gene per branch).
    n_jobs    : number of parallel workers (-1 = all available CPU cores)."""
    mu_obs = np.asarray(mu_obs, float); var_obs = np.asarray(var_obs, float)
    N, G = mu_obs.shape
    p_up, p_down, _, _ = branch_posterior(prob_state, up_states, down_states)
    up_mask = p_up >= p_down
    dn_mask = ~up_mask

    keys = ("qvl_r2_mu", "qvl_r2_var", "qvl_r2_geom", "qvl_curvature_a",
            "qvl_fstat", "qvl_pvalue", "qvl_cv_r2_geom")
    out = {f"{k}_{tag}": np.full(G, np.nan) for k in keys for tag in ("up", "down")}
    out["qvl_n_up"] = up_mask.sum(0).astype(float)
    out["qvl_n_down"] = dn_mask.sum(0).astype(float)
    out["qvl_gate_up"] = np.zeros(G, dtype=bool)
    out["qvl_gate_down"] = np.zeros(G, dtype=bool)

    run_cv = run_cv and cv_folds and cv_folds > 1
    t0 = time.time()

    if n_jobs is None or n_jobs == 0:
        n_jobs = 1

    if n_jobs != 1 and _HAVE_JOBLIB:
        if verbose:
            print(f"    [quad-vs-line] parallelizing over {G} genes with joblib (n_jobs={n_jobs})...")
        results = Parallel(n_jobs=n_jobs)(
            delayed(_process_single_gene_v3)(
                g, mu_obs[:, g], var_obs[:, g], up_mask[:, g], dn_mask[:, g],
                min_cells, min_range_ratio, run_ftest, run_cv, cv_folds, cv_seed, eps
            )
            for g in range(G)
        )
    else:
        results = []
        for g in range(G):
            res = _process_single_gene_v3(
                g, mu_obs[:, g], var_obs[:, g], up_mask[:, g], dn_mask[:, g],
                min_cells, min_range_ratio, run_ftest, run_cv, cv_folds, cv_seed, eps
            )
            results.append(res)
            if verbose and progress_every and (g + 1) % progress_every == 0:
                print(f"    [quad-vs-line] {g + 1}/{G} genes  ({time.time() - t0:.1f}s)")

    n_fit_fail = 0
    for res in results:
        g = res['g']
        n_fit_fail += res['fit_fail_count']
        out["qvl_gate_up"][g] = res['qvl_gate_up']
        out["qvl_gate_down"][g] = res['qvl_gate_down']
        for k in keys:
            for tag in ("up", "down"):
                key_tag = f"{k}_{tag}"
                out[key_tag][g] = res[key_tag]

    if verbose:
        print(f"    [quad-vs-line] done: {G} genes in {time.time() - t0:.1f}s"
              f"  (scipy.odr {'available' if _HAVE_SCIPY_ODR else 'NOT available -> numpy fallback used'}"
              f", scipy.stats {'available' if _HAVE_SCIPY_STATS else 'NOT available -> beta-cf fallback used'})")
        if n_fit_fail:
            print(f"    [warn] {n_fit_fail} branch fits raised and were left NaN")

    a_up = out["qvl_curvature_a_up"]; a_dn = out["qvl_curvature_a_down"]
    out["qvl_curvature_sign_matches_prior_up"] = np.where(
        np.isnan(a_up), np.nan, (a_up < 0).astype(float))
    out["qvl_curvature_sign_matches_prior_down"] = np.where(
        np.isnan(a_dn), np.nan, (a_dn > 0).astype(float))

    if run_ftest:
        out["qvl_fdr_up"] = _bh_fdr(out["qvl_pvalue_up"])
        out["qvl_fdr_down"] = _bh_fdr(out["qvl_pvalue_down"])
        out["qvl_significant_up"] = out["qvl_fdr_up"] <= fdr_alpha
        out["qvl_significant_down"] = out["qvl_fdr_down"] <= fdr_alpha
        out["qvl_significant_any"] = out["qvl_significant_up"] | out["qvl_significant_down"]
    return out


def aggregate_quad_vs_line(qvl, min_cells=30, branch_agg="max"):
    """Aggregate the per-branch qvl_* dict over up/down (v2's `_agg_axes`,
    selecting by the geometric score and carrying that branch's axis values).
    Adds qvl_r2_geom, qvl_r2_mu, qvl_r2_var to the dict (in place) and
    returns it."""
    r2, r2_mu, r2_var, _, _ = _agg_axes(
        qvl["qvl_r2_geom_up"], qvl["qvl_r2_geom_down"],
        qvl["qvl_r2_mu_up"], qvl["qvl_r2_mu_down"],
        qvl["qvl_r2_var_up"], qvl["qvl_r2_var_down"],
        qvl["qvl_n_up"], qvl["qvl_n_down"], min_cells, branch_agg)
    qvl["qvl_r2_geom"] = r2
    qvl["qvl_r2_mu"] = r2_mu
    qvl["qvl_r2_var"] = r2_var
    return qvl


def print_v3_report(complete, qvl):
    def med(x):
        return float(np.nanmedian(x))
    print("\n=== task 1: complete-data (whole-cloud) mean-null R2 ===")
    print(f"  fit_r2_complete_mu  : median {med(complete['fit_r2_complete_mu']):.3f}")
    print(f"  fit_r2_complete_var : median {med(complete['fit_r2_complete_var']):.3f}")
    print(f"  fit_r2_complete     : median {med(complete['fit_r2_complete']):.3f}")
    print(f"  reliable (mu)  : {int(np.sum(complete['reliability_by_mu_meanNull']))}"
          f" / {len(complete['fit_r2_complete_mu'])}")
    print(f"  reliable (var) : {int(np.sum(complete['reliability_by_var_meanNull']))}"
          f" / {len(complete['fit_r2_complete_var'])}")

    if qvl is None:
        return
    print("\n=== task 2: per-branch quadratic-vs-line (ODR), scored against the LINE ===")
    print(f"{'metric (median over genes)':30s}{'up':>10}{'down':>10}{'aggregated':>12}")
    print(f"{'qvl_r2_geom':30s}{med(qvl['qvl_r2_geom_up']):>10.3f}"
          f"{med(qvl['qvl_r2_geom_down']):>10.3f}{med(qvl['qvl_r2_geom']):>12.3f}")
    print(f"{'qvl_r2_mu':30s}{med(qvl['qvl_r2_mu_up']):>10.3f}"
          f"{med(qvl['qvl_r2_mu_down']):>10.3f}{med(qvl['qvl_r2_mu']):>12.3f}")
    print(f"{'qvl_r2_var':30s}{med(qvl['qvl_r2_var_up']):>10.3f}"
          f"{med(qvl['qvl_r2_var_down']):>10.3f}{med(qvl['qvl_r2_var']):>12.3f}")
    n_gate_up = int(qvl["qvl_gate_up"].sum()); n_gate_dn = int(qvl["qvl_gate_down"].sum())
    G = len(qvl["qvl_gate_up"])
    print(f"\n  branches passing min_cells+dynamic-range gate: up {n_gate_up}/{G}"
          f"   down {n_gate_dn}/{G}")
    su = qvl["qvl_curvature_sign_matches_prior_up"]
    sd = qvl["qvl_curvature_sign_matches_prior_down"]
    print(f"  curvature sign matches prior (up, a<0)  : {int(np.nansum(su))}"
          f" / {int(np.sum(~np.isnan(su)))} gated genes")
    print(f"  curvature sign matches prior (down, a>0): {int(np.nansum(sd))}"
          f" / {int(np.sum(~np.isnan(sd)))} gated genes")

    if "qvl_pvalue_up" in qvl:
        for tag in ("up", "down"):
            pv = qvl[f"qvl_pvalue_{tag}"]; fd = qvl[f"qvl_fdr_{tag}"]
            sig = qvl[f"qvl_significant_{tag}"]
            n_tested = int(np.sum(~np.isnan(pv)))
            print(f"  nested F-test ({tag}): median p={med(pv):.3g}  "
                  f"significant (FDR<=alpha): {int(sig.sum())} / {n_tested} tested")
    if "qvl_cv_r2_geom_up" in qvl:
        print(f"  qvl_cv_r2_geom (up/down)  : median "
              f"{med(qvl['qvl_cv_r2_geom_up']):.3f} / {med(qvl['qvl_cv_r2_geom_down']):.3f}"
              f"   (near/below 0 for genuinely-linear genes)")


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------

class _Tee:
    def __init__(self, path):
        self.file = open(path, "w"); self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s); self.file.write(s)

    def flush(self):
        self.stdout.flush(); self.file.flush()

    def close(self):
        self.file.flush(); self.file.close()


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# h5ad driver
# ---------------------------------------------------------------------------

def run_on_h5ad(path, mu_obs_layer="mu_scvi_smooth", var_obs_layer="var_scvi_smooth",
                mu_fit_layer="mu_fit", var_fit_layer="var_fit", combine="min",
                thresh_mu_meannull=0.5, thresh_var_meannull=0.5,
                write_h5ad=False, out_h5ad=None, out_csv=None, log_file=None,
                out_dir=None, diagnose_only=False, prob_state=None,
                up_states=(0, 1), down_states=(2, 3), branch_min_cells=30,
                branch_min_range_ratio=3.0, branch_agg="max", progress_every=500,
                run_ftest=True, fdr_alpha=0.05, run_cv=True, cv_folds=5, cv_seed=0,
                n_jobs=-1, branch_conf_thresh="auto", corner_quantile=0.9, sep_n_bins=20,
                bow_scores=True, bow_split="auto", bow_velocity_layer="velocity_mu",
                mu_naive_layer="mu_naive_smooth", var_naive_layer="var_naive_smooth",
                bow_embedding="X_latent", bow_block_size=30, bow_n_boot=100,
                bow_n_bins=8, bow_min_cells=None, bow_min_per_bin=8, bow_z_thresh=2.0,
                signed_sep_n_bins=10, signed_sep_min_per_bin=10, bow_seed=0):
    """Score one noSpliceVelo h5ad.

    Task 4 (bow_scores=True) adds the per-gene bow and signed-separation columns
    from gene_bow_separation.py (see its docstring for definitions):
      bow_split       'posterior' (argmax of the branch posterior, as task 2),
                      'velocity_sign' (up = layers[bow_velocity_layer] > 0), or
                      'auto' = posterior when a valid prob_state exists, else
                      velocity_sign. The split used is recorded in the
                      `bow_branch_split` column.
      bow_embedding   obsm key used to build bootstrap blocks of ~bow_block_size
                      neighbouring cells (the kNN smoothing scale); falls back to
                      X_pca, then to an i.i.d. bootstrap with a warning.
      mu/var_naive_layer  model-free kNN mean/variance for sep_*_naive; skipped
                      with a note when absent.
    """
    out_dir = out_dir or (os.path.dirname(path) or ".")
    os.makedirs(out_dir, exist_ok=True)
    if log_file is None:
        log_file = os.path.join(out_dir, f"gene_fit_v3_{time.strftime('%Y%m%d_%H%M%S')}.log")
    tee = _Tee(log_file); old = sys.stdout; sys.stdout = tee
    try:
        import anndata as ad
        import pandas as pd
        print(f"[{_now()}] score_gene_fit_v3 start\n  h5ad: {os.path.abspath(path)}\n"
              f"  log : {os.path.abspath(log_file)}\n  out_dir: {os.path.abspath(out_dir)}")
        t0 = time.time()
        adata = ad.read_h5ad(path)
        print(f"  loaded {adata.n_obs} cells x {adata.n_vars} genes ({time.time() - t0:.1f}s)")

        def layer(n):
            if n not in adata.layers:
                raise KeyError(f"layer '{n}' not in {path}; have {list(adata.layers)}")
            return np.asarray(adata.layers[n])

        mo, vo = layer(mu_obs_layer), layer(var_obs_layer)
        mf, vf = layer(mu_fit_layer), layer(var_fit_layer)
        report_layer_health(mo, vo, mf, vf)
        if diagnose_only:
            print("\n[--diagnose] health only."); return None

        csv_cols = {"gene": np.asarray(adata.var_names)}

        # --- task 1: complete-data mean-null R2 ---
        complete = complete_meannull_r2(mo, vo, mf, vf, combine=combine,
                                        thresh_mu_meannull=thresh_mu_meannull,
                                        thresh_var_meannull=thresh_var_meannull)
        for k, v in complete.items():
            adata.var[k] = v
            csv_cols[k] = v

        # --- task 3: b_corner ---
        b_corner, mu_corner, var_corner = compute_b_corner(mo, vo, q=corner_quantile)
        adata.var['b_corner'] = b_corner
        csv_cols['b_corner'] = b_corner

        # --- task 2: per-branch quadratic vs line & branch separation ---
        prob = None
        try:
            if prob_state:
                prob = np.load(prob_state)
            else:
                in_dir = os.path.dirname(path) or "."
                for fn in ("prob_state_avg_nosplicevelo.npy", "prob_state_avg.npy"):
                    c = os.path.join(in_dir, fn)
                    if os.path.exists(c):
                        prob = np.load(c); print(f"  prob_state: {c}"); break
        except Exception as e:
            print(f"  [warn] prob_state load: {e}")

        qvl = None
        if prob is None or prob.ndim != 3 or prob.shape[0] != adata.n_obs:
            print("  [note] no valid prob_state -> skipping task-2 quadratic-vs-line & branch separation report.")
        else:
            print(f"  prob_state shape {prob.shape}  up={up_states} down={down_states}"
                  f"  branch_min_cells={branch_min_cells}  branch_conf_thresh={branch_conf_thresh}"
                  f"  branch_min_range_ratio={branch_min_range_ratio}  agg={branch_agg}"
                  f"  run_ftest={run_ftest}  run_cv={run_cv} (cv_folds={cv_folds})")
            qvl = quad_vs_line_branch(mo, vo, prob, up_states=up_states,
                                      down_states=down_states, min_cells=branch_min_cells,
                                      min_range_ratio=branch_min_range_ratio,
                                      run_ftest=run_ftest, run_cv=run_cv,
                                      cv_folds=cv_folds, cv_seed=cv_seed,
                                      fdr_alpha=fdr_alpha, progress_every=progress_every,
                                      n_jobs=n_jobs)
            qvl = aggregate_quad_vs_line(qvl, min_cells=branch_min_cells, branch_agg=branch_agg)
            for k, v in qvl.items():
                adata.var[k] = v
                csv_cols[k] = v

            # --- branch separation ---
            sep_res = compute_branch_separation(
                mo, vo, prob, up_states=up_states, down_states=down_states,
                conf_thresh=branch_conf_thresh, min_cells=branch_min_cells
            )
            sep_norm = normalize_separation_hvg(sep_res['separation'], b_corner, n_bins=sep_n_bins)
            for k, v in sep_res.items():
                adata.var[k] = v
                csv_cols[k] = v
            adata.var['separation_norm'] = sep_norm
            csv_cols['separation_norm'] = sep_norm

            # --- cell obs annotations ---
            cell_res = compute_cell_separation_obs(
                mo, vo, prob, sep_res['separation'], sep_norm,
                up_states=up_states, down_states=down_states, conf_thresh=branch_conf_thresh
            )
            adata.obs['branch_call'] = cell_res['branch_call']
            adata.obs['separation'] = cell_res['cell_separation']
            adata.obs['separation_norm'] = cell_res['cell_separation_norm']

            n_sep_gated = int(np.sum(sep_res['sep_gate']))
            print(f"  [branch separation] {n_sep_gated}/{adata.n_vars} genes passed cell gate "
                  f"(min_cells={branch_min_cells}, conf_thresh={branch_conf_thresh})")

        # --- task 4: bow (per-branch visible curvature) + signed separation ---
        if bow_scores:
            try:
                split = bow_split
                if split == "auto":
                    split = "posterior" if (prob is not None and prob.ndim == 3
                                            and prob.shape[0] == adata.n_obs) else "velocity_sign"
                if split == "posterior":
                    if prob is None or prob.ndim != 3 or prob.shape[0] != adata.n_obs:
                        raise ValueError("bow_split='posterior' needs a valid prob_state")
                    up_mask = up_mask_from_posterior(prob, up_states, down_states)
                elif split == "velocity_sign":
                    up_mask = layer(bow_velocity_layer) > 0
                    print(f"  [bow] branch split from sign of layers['{bow_velocity_layer}'] "
                          f"(no valid state posterior)")
                else:
                    raise ValueError(f"bow_split must be auto|posterior|velocity_sign, got {bow_split!r}")
                mn = vn = None
                if mu_naive_layer in adata.layers and var_naive_layer in adata.layers:
                    mn, vn = layer(mu_naive_layer), layer(var_naive_layer)
                else:
                    print(f"  [bow] note: '{mu_naive_layer}'/'{var_naive_layer}' not in layers; "
                          f"sep_*_naive columns will be NaN")
                emb = None
                for key in (bow_embedding, "X_pca"):
                    if key and key in adata.obsm:
                        emb = np.asarray(adata.obsm[key]); print(f"  [bow] bootstrap blocks from obsm['{key}']")
                        break
                bs = bow_separation_scores(
                    mo, vo, mf, vf, up_mask, mu_naive=mn, var_naive=vn, embedding=emb,
                    n_bins_bow=bow_n_bins,
                    min_cells=branch_min_cells if bow_min_cells is None else bow_min_cells,
                    min_range_ratio=branch_min_range_ratio, min_per_bin=bow_min_per_bin,
                    n_bins_sep=signed_sep_n_bins, sep_min_per_bin=signed_sep_min_per_bin,
                    n_boot=bow_n_boot, block_size=bow_block_size, seed=bow_seed,
                    z_thresh=bow_z_thresh, n_jobs=n_jobs)
                if mn is None:
                    for k in ("sep_up_above_frac_naive", "sep_log2_ratio_naive", "sep_log2_z_naive"):
                        bs[k] = np.full(adata.n_vars, np.nan)
                    bs["sep_signed_nbins_naive"] = np.zeros(adata.n_vars, int)
                bs["bow_branch_split"] = np.array([split] * adata.n_vars, dtype=object)
                for k, v in bs.items():
                    adata.var[k] = v
                    csv_cols[k] = v
            except Exception as e:
                print(f"  [warn] task 4 (bow / signed separation) skipped: {type(e).__name__}: {e}")

        print_v3_report(complete, qvl)

        # --- plots ---
        try:
            paths = []
            paths.append(plot_null_comparison(
                complete["fit_r2_complete_mu"], complete["fit_r2_complete_var"],
                os.path.join(out_dir, "complete_fit_r2_muvar_scatter.png"),
                xlabel="fit_r2_complete_mu", ylabel="fit_r2_complete_var", clip_min=-1.0,
                title="complete-data (whole-cloud) meanNull R2: mu vs var"))
            if qvl is not None:
                paths.append(plot_branch_r2_scatter(
                    qvl["qvl_r2_geom_up"], qvl["qvl_r2_geom_down"],
                    qvl["qvl_n_up"], qvl["qvl_n_down"], branch_min_cells,
                    os.path.join(out_dir, "qvl_r2_geom_up_vs_down_scatter.png"),
                    title="quadratic-vs-line geometric R2  up vs down", clip_min=-1.0,
                    xlabel="qvl_r2_geom (up branch)", ylabel="qvl_r2_geom (down branch)"))
                paths.append(plot_branch_r2_scatter(
                    qvl["qvl_r2_mu_up"], qvl["qvl_r2_mu_down"],
                    qvl["qvl_n_up"], qvl["qvl_n_down"], branch_min_cells,
                    os.path.join(out_dir, "qvl_r2_mu_up_vs_down_scatter.png"),
                    title="quadratic-vs-line R2 (mu axis)  up vs down", clip_min=-1.0,
                    xlabel="qvl_r2_mu (up branch)", ylabel="qvl_r2_mu (down branch)"))
                paths.append(plot_branch_r2_scatter(
                    qvl["qvl_r2_var_up"], qvl["qvl_r2_var_down"],
                    qvl["qvl_n_up"], qvl["qvl_n_down"], branch_min_cells,
                    os.path.join(out_dir, "qvl_r2_var_up_vs_down_scatter.png"),
                    title="quadratic-vs-line R2 (var axis)  up vs down", clip_min=-1.0,
                    xlabel="qvl_r2_var (up branch)", ylabel="qvl_r2_var (down branch)"))
                paths.append(plot_null_comparison(
                    qvl["qvl_r2_mu"], qvl["qvl_r2_var"],
                    os.path.join(out_dir, "qvl_r2_muvar_scatter.png"),
                    xlabel="qvl_r2_mu (selected branch)", ylabel="qvl_r2_var (selected branch)",
                    clip_min=-1.0, title="quadratic-vs-line R2: mu vs var (aggregated branch)"))
            for p in paths:
                print(f"[{_now()}] wrote {p}")
        except Exception as e:
            print(f"  [warn] scatter plots skipped: {e}")

        df = pd.DataFrame(csv_cols)
        out_csv = out_csv or os.path.join(out_dir, "gene_fit_scores_v3.csv")
        df.to_csv(out_csv, index=False)
        print(f"[{_now()}] wrote {out_csv}")
        if write_h5ad:
            out_h5ad = out_h5ad or os.path.join(out_dir, os.path.basename(path))
            adata.write_h5ad(out_h5ad)
            print(f"[{_now()}] wrote {out_h5ad}")
        print(f"[{_now()}] done in {time.time() - t0:.1f}s")
    finally:
        sys.stdout = old; tee.close()
        print(f"log written to {os.path.abspath(log_file)}")


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------

def selftest():
    rng = np.random.default_rng(0)
    N, G, S = 2000, 8, 4

    # --- task 1 data: half the genes fit well, half are scale-broken ---
    mu = np.abs(rng.normal(35, 18, (N, G)))
    branch = rng.integers(0, 2, (N, G))
    var_true = np.where(branch == 0, 60 * mu - 0.35 * mu ** 2, 0.3 * mu ** 2)
    var = var_true + rng.normal(0, 25, (N, G))
    mu_fit = mu.copy(); var_fit = var_true.copy()
    mu_fit[:, -1] *= 5.0            # last gene: bad mu scale -> should fail reliability
    var_fit[:, -2] *= 0.1           # 2nd-last gene: bad var scale

    complete = complete_meannull_r2(mu, var, mu_fit, var_fit)
    ok1 = (complete["reliability_by_mu_meanNull"][-1] == False and
           complete["reliability_by_var_meanNull"][-2] == False and
           bool(np.all(complete["reliability_by_mu_meanNull"][:-2])) and
           bool(np.all(complete["reliability_by_var_meanNull"][:-2])))
    print(f"[selftest task1] mu/var reliability correctly flags the 2 broken genes "
          f"-> {'PASS' if ok1 else 'FAIL'}")

    # --- task 2 data: genuine parabola per branch, correct-sign curvature,
    #     up = concave (a<0), down = convex (a>0); plus 2 "line-truth" genes
    #     (a=0) as a negative control ---
    N2, G2, S2 = 1500, 6, 4
    prob = np.zeros((N2, G2, S2))
    branch2 = rng.integers(0, 2, (N2, G2))       # 0 -> up branch, 1 -> down branch
    prob[..., 0] = np.where(branch2 == 0, 0.85, 0.06)
    prob[..., 1] = np.where(branch2 == 0, 0.07, 0.04)
    prob[..., 2] = np.where(branch2 == 1, 0.85, 0.06)
    prob[..., 3] = np.where(branch2 == 1, 0.07, 0.04)
    prob /= prob.sum(-1, keepdims=True)

    mu2 = np.abs(rng.normal(30, 15, (N2, G2))) + 1.0
    a_true = np.where(branch2 == 0, -0.4, 0.5)   # curvature per cell's branch
    a_true[:, -2:] = 0.0                          # last 2 genes: truly linear
    var2 = 20 * mu2 + a_true * mu2 ** 2 + rng.normal(0, 8, (N2, G2))

    qvl = quad_vs_line_branch(mu2, var2, prob, min_cells=20, min_range_ratio=1.5,
                              run_ftest=True, run_cv=True, cv_folds=5, cv_seed=0,
                              fdr_alpha=0.10, progress_every=0, verbose=True)
    qvl = aggregate_quad_vs_line(qvl, min_cells=20, branch_agg="max")

    curved_genes = slice(0, G2 - 2)
    linear_genes = slice(G2 - 2, G2)
    curved_r2 = np.nanmean(np.concatenate([qvl["qvl_r2_geom_up"][curved_genes],
                                           qvl["qvl_r2_geom_down"][curved_genes]]))
    linear_r2 = np.nanmean(np.concatenate([qvl["qvl_r2_geom_up"][linear_genes],
                                           qvl["qvl_r2_geom_down"][linear_genes]]))
    sign_ok_up = np.nanmean(qvl["qvl_curvature_sign_matches_prior_up"][curved_genes])
    sign_ok_dn = np.nanmean(qvl["qvl_curvature_sign_matches_prior_down"][curved_genes])
    print(f"\n[selftest task2] mean qvl_r2_geom  curved genes={curved_r2:.3f}"
          f"  linear-truth genes={linear_r2:.3f}")
    print(f"[selftest task2] curvature sign matches prior: up={sign_ok_up:.2f}"
          f"  down={sign_ok_dn:.2f}  (expect ~1.0)")
    ok2 = (curved_r2 > linear_r2 > -0.3 and sign_ok_up > 0.8 and sign_ok_dn > 0.8)
    print(f"[selftest task2] quadratic beats line more on curved genes than on "
          f"linear-truth genes, and curvature sign matches theory -> "
          f"{'PASS' if ok2 else 'FAIL'}")

    # --- F-test / FDR: curved genes should be significant, linear-truth genes not ---
    sig_curved = np.concatenate([qvl["qvl_significant_up"][curved_genes],
                                 qvl["qvl_significant_down"][curved_genes]])
    sig_linear = np.concatenate([qvl["qvl_significant_up"][linear_genes],
                                 qvl["qvl_significant_down"][linear_genes]])
    p_curved = np.nanmean(np.concatenate([qvl["qvl_pvalue_up"][curved_genes],
                                          qvl["qvl_pvalue_down"][curved_genes]]))
    p_linear = np.nanmean(np.concatenate([qvl["qvl_pvalue_up"][linear_genes],
                                          qvl["qvl_pvalue_down"][linear_genes]]))
    print(f"\n[selftest F-test] mean p-value  curved={p_curved:.3g}"
          f"  linear-truth={p_linear:.3g}  (expect curved << linear)")
    print(f"[selftest F-test] significant (FDR<=0.10)  curved={int(sig_curved.sum())}/"
          f"{sig_curved.size}   linear-truth={int(sig_linear.sum())}/{sig_linear.size}")
    ok3 = (p_curved < 0.01 and bool(np.all(sig_curved)) and p_linear > p_curved and
           int(sig_linear.sum()) <= 1)
    print(f"[selftest F-test] curved genes flagged significant, "
          f"linear-truth mostly not -> {'PASS' if ok3 else 'FAIL'}")

    # --- CV: curved genes should have a clearly positive CV R2, linear-truth
    #     genes should sit near/below 0 (no overfitting credit) ---
    cv_curved = np.nanmean(np.concatenate([qvl["qvl_cv_r2_geom_up"][curved_genes],
                                           qvl["qvl_cv_r2_geom_down"][curved_genes]]))
    cv_linear = np.nanmean(np.concatenate([qvl["qvl_cv_r2_geom_up"][linear_genes],
                                           qvl["qvl_cv_r2_geom_down"][linear_genes]]))
    print(f"\n[selftest CV] mean qvl_cv_r2_geom  curved={cv_curved:.3f}"
          f"  linear-truth={cv_linear:.3f}  (expect curved >> 0 >= linear-ish)")
    ok4 = (cv_curved > 0.3 and cv_curved > cv_linear)
    print(f"[selftest CV] cross-validated R2 separates curved from linear-truth "
          f"genes -> {'PASS' if ok4 else 'FAIL'}")

    # --- task 3: b_corner & branch separation ---
    b_corner, mu_c, var_c = compute_b_corner(mu2, var2, q=0.9)
    sep_res = compute_branch_separation(mu2, var2, prob, conf_thresh="auto", min_cells=20)
    sep_norm = normalize_separation_hvg(sep_res["separation"], b_corner, n_bins=3)
    cell_res = compute_cell_separation_obs(mu2, var2, prob, sep_res["separation"], sep_norm, conf_thresh="auto")

    ok5 = (len(b_corner) == G2 and not np.isnan(b_corner).any() and
           np.sum(sep_res["sep_gate"]) > 0 and len(sep_norm) == G2 and
           len(cell_res["branch_call"]) == N2)
    print(f"\n[selftest task3] branch separation & b_corner normalization -> "
          f"{'PASS' if ok5 else 'FAIL'}")

    print(f"\n[selftest] overall -> {'PASS' if (ok1 and ok2 and ok3 and ok4 and ok5) else 'FAIL'}")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("h5ad", nargs="?")
    p.add_argument("--config", default=None,
                   help="Path to YAML config file for batch dataset evaluation.")
    p.add_argument("--select", default=None,
                   help="Filter to a single dataset by name when using --config.")
    p.add_argument("--mu_obs_layer", default="mu_scvi_smooth")
    p.add_argument("--var_obs_layer", default="var_scvi_smooth")
    p.add_argument("--mu_fit_layer", default="mu_fit")
    p.add_argument("--var_fit_layer", default="var_fit")
    p.add_argument("--combine", default="min", choices=["min", "mean"],
                   help="task 1 composite of (fit_r2_complete_mu, fit_r2_complete_var)")
    p.add_argument("--thresh_mu_meannull", type=float, default=0.5,
                   help="reliability_by_mu_meanNull threshold")
    p.add_argument("--thresh_var_meannull", type=float, default=0.5,
                   help="reliability_by_var_meanNull threshold")
    p.add_argument("--prob_state", default=None)
    p.add_argument("--up_states", type=int, nargs="+", default=[0, 1])
    p.add_argument("--down_states", type=int, nargs="+", default=[2, 3])
    p.add_argument("--branch_min_cells", type=int, default=30)
    p.add_argument("--branch_conf_thresh", default="auto",
                   help="confidence threshold ('auto' for data-driven auto_conf_threshold, or float e.g. 0.7)")
    p.add_argument("--corner_quantile", type=float, default=0.9,
                   help="quantile for top-right corner definition")
    p.add_argument("--sep_n_bins", type=int, default=20,
                   help="number of bins for b_corner HVG normalization")
    p.add_argument("--branch_min_range_ratio", type=float, default=3.0,
                   help="gate: branch mu (max-min)/std must be >= this for the "
                        "quadratic-vs-line comparison to be attempted")
    p.add_argument("--branch_agg", default="max", choices=["max", "min", "weighted"])
    p.add_argument("--no_ftest", action="store_true",
                   help="skip the nested F-test / BH-FDR significance columns")
    p.add_argument("--fdr_alpha", type=float, default=0.05,
                   help="BH-FDR threshold for qvl_significant_up/down/any")
    p.add_argument("--no_cv", action="store_true",
                   help="skip the K-fold cross-validated qvl_cv_r2_geom (saves "
                        "~cv_folds extra fits per gene per branch)")
    p.add_argument("--cv_folds", type=int, default=5,
                   help="K for the cross-validated quadratic-vs-line R2 (0 or "
                        "--no_cv disables it)")
    p.add_argument("--cv_seed", type=int, default=0)
    p.add_argument("--n_jobs", type=int, default=-1,
                   help="number of parallel workers for quad-vs-line fits (-1 = all cores)")
    p.add_argument("--progress_every", type=int, default=500)
    p.add_argument("--write_h5ad", action="store_true")
    p.add_argument("--out_h5ad", default=None)
    p.add_argument("--out_csv", default=None)
    p.add_argument("--out_dir", default=None)
    p.add_argument("--log_file", default=None)
    p.add_argument("--diagnose", action="store_true")
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--no_bow", action="store_true",
                   help="skip task 4 (bow + signed separation)")
    p.add_argument("--bow_split", default="auto", choices=["auto", "posterior", "velocity_sign"])
    p.add_argument("--bow_velocity_layer", default="velocity_mu")
    p.add_argument("--mu_naive_layer", default="mu_naive_smooth")
    p.add_argument("--var_naive_layer", default="var_naive_smooth")
    p.add_argument("--bow_embedding", default="X_latent")
    p.add_argument("--bow_block_size", type=int, default=30)
    p.add_argument("--bow_n_boot", type=int, default=100)
    p.add_argument("--bow_n_bins", type=int, default=8)
    p.add_argument("--bow_min_cells", type=int, default=None,
                   help="default: --branch_min_cells")
    p.add_argument("--bow_min_per_bin", type=int, default=8)
    p.add_argument("--bow_z_thresh", type=float, default=2.0)
    p.add_argument("--signed_sep_n_bins", type=int, default=10)
    p.add_argument("--signed_sep_min_per_bin", type=int, default=10)
    return p.parse_args(argv)


if __name__ == "__main__":
    a = _parse_args()
    if a.selftest:
        selftest()
        print("\n[selftest] gene_bow_separation")
        bow_selftest()
    elif a.config or (a.h5ad and a.h5ad.endswith((".yaml", ".yml"))):
        cfg_file = a.config or a.h5ad
        from score_gene_fit_v3_run import run_from_config
        run_from_config(cfg_file, select=a.select)
    elif a.h5ad:
        run_on_h5ad(a.h5ad, mu_obs_layer=a.mu_obs_layer, var_obs_layer=a.var_obs_layer,
                    mu_fit_layer=a.mu_fit_layer, var_fit_layer=a.var_fit_layer,
                    combine=a.combine, thresh_mu_meannull=a.thresh_mu_meannull,
                    thresh_var_meannull=a.thresh_var_meannull, write_h5ad=a.write_h5ad,
                    out_h5ad=a.out_h5ad, out_csv=a.out_csv, log_file=a.log_file,
                    out_dir=a.out_dir, diagnose_only=a.diagnose, prob_state=a.prob_state,
                    up_states=tuple(a.up_states), down_states=tuple(a.down_states),
                    branch_min_cells=a.branch_min_cells,
                    branch_conf_thresh=a.branch_conf_thresh,
                    corner_quantile=a.corner_quantile,
                    sep_n_bins=a.sep_n_bins,
                    branch_min_range_ratio=a.branch_min_range_ratio,
                    branch_agg=a.branch_agg, progress_every=a.progress_every,
                    run_ftest=not a.no_ftest, fdr_alpha=a.fdr_alpha,
                    run_cv=not a.no_cv, cv_folds=a.cv_folds, cv_seed=a.cv_seed,
                    n_jobs=a.n_jobs, bow_scores=not a.no_bow, bow_split=a.bow_split,
                    bow_velocity_layer=a.bow_velocity_layer,
                    mu_naive_layer=a.mu_naive_layer, var_naive_layer=a.var_naive_layer,
                    bow_embedding=a.bow_embedding, bow_block_size=a.bow_block_size,
                    bow_n_boot=a.bow_n_boot, bow_n_bins=a.bow_n_bins,
                    bow_min_cells=a.bow_min_cells, bow_min_per_bin=a.bow_min_per_bin,
                    bow_z_thresh=a.bow_z_thresh, signed_sep_n_bins=a.signed_sep_n_bins,
                    signed_sep_min_per_bin=a.signed_sep_min_per_bin)
    else:
        raise SystemExit("provide an h5ad path, a YAML --config, or --selftest")


