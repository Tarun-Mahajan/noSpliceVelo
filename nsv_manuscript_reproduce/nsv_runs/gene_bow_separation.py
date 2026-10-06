"""
gene_bow_separation -- two per-gene tests of the bursting signature that do not
depend on the sign of a fitted quadratic coefficient.

Background: the sign of
a quadratic fitted to one branch's (mu, var) cloud flipped by 24-30 percentage
points between estimators (ODR vs vertical least squares), and a convex var~mu
trend is what ANY overdispersed count data show (Taylor's law), so neither the
coefficient sign nor down-branch convexity is a reliable test. These two are:

------------------------------------------------------------------------
1. BOW  (does the observed branch bend the way the model's branch bends?)
------------------------------------------------------------------------
For one branch of one gene, with both axes standardized by that branch's
observed std (x = mu/s_mu, y = var/s_var):

  * cells are binned into `n_bins_bow` quantile bins of the MODEL's position
    along the branch (x_hat = mu_fit/s_mu), so observed and fitted curves are
    compared at the same places;
  * per bin, medians of observed (X_b, Y_b) and fitted (Xh_b, Yh_b);
  * chord through the first and last bin:  L(X) = Y_1 + (X - X_1)/(X_B - X_1) (Y_B - Y_1)
  * bow = mean over interior bins of  Y_b - L(X_b)          (> 0: above chord = concave)

For y = a x^2 + ..., bow ~= -a * span^2 / 6: curvature times squared span, i.e.
how much bend is VISIBLE, which is what a test on data can see.

  {br}_bow_obs     bow of the observed medians
  {br}_bow_fit     bow of the model's fitted medians (same bins)
  {br}_bow_scatter std over the branch's cells of (y - y_hat)
  {br}_bow_ratio   bow_fit / bow_scatter  (signed; descriptive only)
  {br}_bow_se      block-bootstrap SE of bow_obs  (see below)
  {br}_bow_z_fit   bow_fit / se   -- is the model's predicted bend resolvable?
  {br}_bow_z_obs   bow_obs / se
  {br}_bow_z_diff  (bow_obs - bow_fit) / se
  {br}_bow_misfit  gate & sign(bow_obs) != sign(bow_fit) & |z_diff| > z_thresh
  {br}_bow_gate    enough cells, dynamic range, and every bin populated
  {br}_bow_n       cells on the branch

  The model predicts a concave UP branch and a convex DOWN branch, so the
  "prediction present and resolvable" conditions are  up_bow_z_fit > 2  and
  down_bow_z_fit < -2  respectively.

  Block bootstrap: cells were kNN-smoothed (k ~ 30) before scoring, so
  neighbouring cells share most of their information and an i.i.d. bootstrap
  would understate the SE by up to ~sqrt(k). Cells are grouped into blocks of
  ~`block_size` by k-means on a cell embedding (X_latent by default) and whole
  blocks are resampled. The same resampled cell sets are used for every gene
  (resampling is gene-independent), so genes are comparable.

------------------------------------------------------------------------
2. SIGNED SEPARATION  (is variance higher on the way up, at the same mean?)
------------------------------------------------------------------------
The model's distinguishing prediction. Cells are binned by mean over the range
where BOTH branches have cells (sep_q quantiles of each branch), `n_bins_sep`
equal-width bins, each needing >= sep_min_per_bin cells per branch.

  sep_up_above_frac{_naive}  fraction of valid bins where median var(up) > median var(down)
  sep_log2_ratio{_naive}     mean over valid bins of log2(median var_up / median var_down)
  sep_signed_nbins{_naive}   valid bins used (>= 3 required, else NaN)
  sep_log2_z{_naive}         sep_log2_ratio / its block-bootstrap SE (same resamples
                             as the bow). USE THIS, not the fraction, for decisions:
                             in simulation (NB counts -> kNN k=30 -> this code) the
                             majority rule sep_up_above_frac > 0.5 passed 27% of
                             genes with NO up/down gap, while sep_log2_z > 2 passed
                             4% (and sep_log2_z < -2 flagged 83% of genes with a
                             reversed gap of 20%). Reproduce with calibrate_bow_thresholds.py.

`_naive` uses the model-free kNN variance (mu_naive_smooth / var_naive_smooth)
and is the one to select on: the branch split came from a model fitted to the
scVI variance, so the un-suffixed (scVI) version is partly circular.

Branch split: argmax of the branch posterior (up_states vs down_states), exactly
as quad_vs_line_branch does. `velocity_sign` (up = velocity > 0) is available as
a fallback when no state posterior is stored; it reproduced the posterior-based
curvature numbers within ~3 points on pancreas / neural.
"""

import time
import numpy as np

try:
    from joblib import Parallel, delayed
    _HAVE_JOBLIB = True
except Exception:  # pragma: no cover
    _HAVE_JOBLIB = False


# ---------------------------------------------------------------------------
# blocks for the bootstrap
# ---------------------------------------------------------------------------

def make_blocks(embedding, block_size=30, seed=0):
    """Cell -> block label, blocks of ~block_size cells that are neighbours in
    `embedding` (k-means). Falls back to contiguous chunks of the first
    principal axis when scikit-learn is unavailable."""
    X = np.asarray(embedding, float)
    n = X.shape[0]
    k = max(2, int(round(n / max(1, block_size))))
    try:
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3,
                             batch_size=max(1024, 3 * k))
        return km.fit_predict(X).astype(np.int64)
    except Exception:
        Xc = X - X.mean(0)
        pc1 = Xc @ np.linalg.svd(Xc, full_matrices=False)[2][0]
        order = np.argsort(pc1)
        lab = np.empty(n, np.int64)
        lab[order] = np.arange(n) // max(1, block_size)
        return lab


def bootstrap_indices(blocks, n_boot=100, seed=0):
    """List of resampled cell-index arrays (whole blocks drawn with replacement)."""
    blocks = np.asarray(blocks)
    uniq, inv = np.unique(blocks, return_inverse=True)
    members = [np.where(inv == i)[0] for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(int(n_boot)):
        pick = rng.integers(0, len(members), len(members))
        out.append(np.concatenate([members[i] for i in pick]))
    return out


# ---------------------------------------------------------------------------
# bow
# ---------------------------------------------------------------------------

def _bow_core(x, y, xf, yf, n_bins, min_per_bin):
    """(bow_obs, bow_fit) for one branch; NaNs if any bin is under-populated."""
    edges = np.quantile(xf, np.linspace(0, 1, n_bins + 1))
    b = np.clip(np.searchsorted(edges[1:-1], xf, side="right"), 0, n_bins - 1)
    if np.bincount(b, minlength=n_bins).min() < min_per_bin:
        return np.nan, np.nan
    X = np.empty(n_bins); Y = np.empty(n_bins); XF = np.empty(n_bins); YF = np.empty(n_bins)
    for i in range(n_bins):
        k = b == i
        X[i] = np.median(x[k]); Y[i] = np.median(y[k])
        XF[i] = np.median(xf[k]); YF[i] = np.median(yf[k])

    def bow(xm, ym):
        span = xm[-1] - xm[0]
        if not np.isfinite(span) or abs(span) < 1e-12:
            return np.nan
        t = (xm - xm[0]) / span
        return float(np.mean((ym - (ym[0] + t * (ym[-1] - ym[0])))[1:-1]))
    return bow(X, Y), bow(XF, YF)


def _bow_gene(mu, var, mf, vf, mask, boot, n_bins, min_cells, min_range_ratio,
              min_per_bin, z_thresh, eps=1e-12):
    res = dict(bow_obs=np.nan, bow_fit=np.nan, bow_scatter=np.nan, bow_ratio=np.nan,
               bow_se=np.nan, bow_z_fit=np.nan, bow_z_obs=np.nan, bow_z_diff=np.nan,
               bow_misfit=False, bow_gate=False, bow_n=int(mask.sum()))
    ok = mask & np.isfinite(mu) & np.isfinite(var) & np.isfinite(mf) & np.isfinite(vf)
    if ok.sum() < min_cells:
        return res
    sx, sy = mu[ok].std(), var[ok].std()
    if sx < eps or sy < eps or (mu[ok].max() - mu[ok].min()) / sx < min_range_ratio:
        return res
    x, y, xf, yf = mu / sx, var / sy, mf / sx, vf / sy
    bo, bf = _bow_core(x[ok], y[ok], xf[ok], yf[ok], n_bins, min_per_bin)
    if not (np.isfinite(bo) and np.isfinite(bf)):
        return res
    scatter = float(np.std(y[ok] - yf[ok]))
    draws = []
    for idx in boot:
        idx = idx[ok[idx]]
        if idx.size < min_cells:
            continue
        b_o, _ = _bow_core(x[idx], y[idx], xf[idx], yf[idx], n_bins, min_per_bin)
        if np.isfinite(b_o):
            draws.append(b_o)
    se = float(np.std(draws, ddof=1)) if len(draws) >= max(10, len(boot) // 2) else np.nan
    res.update(bow_obs=bo, bow_fit=bf, bow_scatter=scatter,
               bow_ratio=bf / scatter if scatter > eps else np.nan, bow_gate=True)
    if np.isfinite(se) and se > eps:
        zd = (bo - bf) / se
        res.update(bow_se=se, bow_z_fit=bf / se, bow_z_obs=bo / se, bow_z_diff=zd,
                   bow_misfit=bool(np.sign(bo) != np.sign(bf) and abs(zd) > z_thresh))
    return res


# ---------------------------------------------------------------------------
# signed separation
# ---------------------------------------------------------------------------

def _sep_gene(mu, var, up, n_bins, min_per_bin, q=(0.05, 0.95), min_valid=3):
    ok = np.isfinite(mu) & np.isfinite(var)
    u, d = up & ok, (~up) & ok
    if u.sum() < 2 * min_per_bin or d.sum() < 2 * min_per_bin:
        return np.nan, np.nan, 0
    lo = max(np.quantile(mu[u], q[0]), np.quantile(mu[d], q[0]))
    hi = min(np.quantile(mu[u], q[1]), np.quantile(mu[d], q[1]))
    if not hi > lo:
        return np.nan, np.nan, 0
    edges = np.linspace(lo, hi, n_bins + 1)
    wins, lr = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        iu = u & (mu >= a) & (mu < b)
        idn = d & (mu >= a) & (mu < b)
        if iu.sum() >= min_per_bin and idn.sum() >= min_per_bin:
            vu, vd = np.median(var[iu]), np.median(var[idn])
            wins.append(vu > vd)
            if vu > 0 and vd > 0:
                lr.append(np.log2(vu / vd))
    if len(wins) < min_valid:
        return np.nan, np.nan, len(wins)
    return float(np.mean(wins)), (float(np.mean(lr)) if lr else np.nan), len(wins)


def _sep_gene_z(mu, var, up, n_bins, min_per_bin, q, boot):
    """_sep_gene plus z = log2 ratio / block-bootstrap SE (same resamples as the bow)."""
    frac, lr, nb = _sep_gene(mu, var, up, n_bins, min_per_bin, q)
    z = np.nan
    if np.isfinite(lr) and len(boot):
        d = []
        for idx in boot:
            v = _sep_gene(mu[idx], var[idx], up[idx], n_bins, min_per_bin, q)[1]
            if np.isfinite(v):
                d.append(v)
        if len(d) >= max(10, len(boot) // 2):
            sd = float(np.std(d, ddof=1))
            if sd > 0:
                z = lr / sd
    return frac, lr, nb, z


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def up_mask_from_posterior(prob, up_states=(0, 1), down_states=(2, 3)):
    p = np.asarray(prob, float)
    p_up = p[..., list(up_states)].sum(-1)
    p_dn = p[..., list(down_states)].sum(-1)
    return p_up >= p_dn


def bow_separation_scores(mu_obs, var_obs, mu_fit, var_fit, up_mask,
                          mu_naive=None, var_naive=None, blocks=None,
                          n_bins_bow=8, min_cells=100, min_range_ratio=3.0,
                          min_per_bin=8, n_bins_sep=10, sep_min_per_bin=10,
                          sep_q=(0.05, 0.95), n_boot=100, block_size=30,
                          embedding=None, seed=0, z_thresh=2.0, n_jobs=-1,
                          branches=("up", "down"), verbose=True):
    """All per-gene bow + separation columns, as a dict of [G] arrays.

    up_mask : [N, G] bool, True = cell is on the up branch for that gene.
    blocks  : [N] cell->block labels for the bootstrap; built from `embedding`
              by make_blocks(block_size) when None; i.i.d. cells if both None.
    """
    t0 = time.time()
    mu_obs = np.asarray(mu_obs, float); var_obs = np.asarray(var_obs, float)
    mu_fit = np.asarray(mu_fit, float); var_fit = np.asarray(var_fit, float)
    up_mask = np.asarray(up_mask, bool)
    N, G = mu_obs.shape
    if blocks is None and embedding is not None:
        blocks = make_blocks(embedding, block_size=block_size, seed=seed)
    if blocks is None:
        blocks = np.arange(N)
        if verbose:
            print("  [bow] WARNING: no embedding/blocks given -> i.i.d. cell bootstrap; "
                  "SEs will be too small if the inputs were kNN-smoothed")
    boot = bootstrap_indices(blocks, n_boot=n_boot, seed=seed) if n_boot and n_boot > 1 else []
    if verbose:
        print(f"  [bow] {G} genes, {N} cells, {len(np.unique(blocks))} bootstrap blocks "
              f"(~{N / max(1, len(np.unique(blocks))):.0f} cells each), n_boot={len(boot)}, "
              f"bins={n_bins_bow}, min_cells={min_cells}, z_thresh={z_thresh}")

    keys = ("bow_obs", "bow_fit", "bow_scatter", "bow_ratio", "bow_se", "bow_z_fit",
            "bow_z_obs", "bow_z_diff", "bow_misfit", "bow_gate", "bow_n")
    out = {}

    def run(fn, jobs):
        if n_jobs not in (None, 0, 1) and _HAVE_JOBLIB:
            return Parallel(n_jobs=n_jobs)(delayed(fn)(*a) for a in jobs)
        return [fn(*a) for a in jobs]

    for br in branches:
        m = up_mask if br == "up" else ~up_mask
        r = run(_bow_gene, [(mu_obs[:, g], var_obs[:, g], mu_fit[:, g], var_fit[:, g],
                             m[:, g], boot, n_bins_bow, min_cells, min_range_ratio,
                             min_per_bin, z_thresh) for g in range(G)])
        for k in keys:
            dt = bool if k in ("bow_misfit", "bow_gate") else (int if k == "bow_n" else float)
            out[f"{br}_{k}"] = np.array([x[k] for x in r], dtype=dt)

    variants = [("", mu_obs, var_obs)]
    if mu_naive is not None and var_naive is not None:
        variants.append(("_naive", np.asarray(mu_naive, float), np.asarray(var_naive, float)))
    for suf, mu, var in variants:
        r = run(_sep_gene_z, [(mu[:, g], var[:, g], up_mask[:, g], n_bins_sep,
                               sep_min_per_bin, sep_q, boot) for g in range(G)])
        out[f"sep_up_above_frac{suf}"] = np.array([x[0] for x in r], float)
        out[f"sep_log2_ratio{suf}"] = np.array([x[1] for x in r], float)
        out[f"sep_signed_nbins{suf}"] = np.array([x[2] for x in r], int)
        out[f"sep_log2_z{suf}"] = np.array([x[3] for x in r], float)

    if verbose:
        print(f"  [bow] done in {time.time() - t0:.0f}s")
        report(out)
    return out


def report(out):
    """Short summary of the columns (fractions over genes where defined)."""
    def pct(m, base):
        base = np.asarray(base, bool)
        return f"{100 * np.mean(np.asarray(m)[base]):5.1f}% of {int(base.sum())}" if base.any() else "n/a"
    for br, pred, sgn in (("up", "concave", 1), ("down", "convex", -1)):
        if f"{br}_bow_z_fit" not in out:
            continue
        zf, bo = out[f"{br}_bow_z_fit"], out[f"{br}_bow_obs"]
        g = np.isfinite(zf)
        predicted = g & (sgn * zf > 2)
        print(f"  [bow] {br:4s}: model predicts a resolvable {pred} bend (|z_fit|>2, right sign): "
              f"{pct(predicted, g)}")
        print(f"         ... of those, observed bends the same way: {pct(sgn * bo > 0, predicted)}"
              f" | misfit (opposite, |z_diff|>2): {pct(out[f'{br}_bow_misfit'], g)}")
    for suf, lab in (("", "scVI variance"), ("_naive", "naive variance")):
        k = f"sep_log2_z{suf}"
        if k in out:
            z = out[k]; g = np.isfinite(z)
            print(f"  [sep] {lab:14s}: up significantly ABOVE down (z>2): {pct(z > 2, g)} | "
                  f"significantly BELOW (z<-2): {pct(z < -2, g)} | median log2(var_up/var_down) "
                  f"{np.nanmedian(out[f'sep_log2_ratio{suf}']):+.3f}")


# ---------------------------------------------------------------------------
# self-test on synthetic data with known answers
# ---------------------------------------------------------------------------

def selftest(seed=0):
    """Planted concave / convex-misfit / straight up branches with smoothed
    (block-correlated) noise; checks sign, z and misfit calls and the SE ratio
    between block and i.i.d. bootstraps."""
    rng = np.random.default_rng(seed)
    N, per = 3000, 40                      # 40 "genes" of each kind
    t = np.sort(rng.uniform(0, 1, N))
    up = np.ones((N, 3 * per), bool)       # whole trajectory = up branch
    emb = np.column_stack([t, rng.normal(0, 0.01, N)])
    blocks = make_blocks(emb, block_size=30, seed=seed)
    mu = np.tile(1 + 9 * t[:, None], (1, 3 * per))
    mf = mu.copy()
    base = 1 + 9 * t
    concave = 2 * base - 0.15 * base ** 2          # bows ABOVE its chord
    straight = 1.2 * base
    vf = np.empty_like(mu); vo = np.empty_like(mu)
    # block-shared noise (what kNN smoothing does) + a little cell noise
    for g in range(3 * per):
        kind = g // per
        shared = rng.normal(0, 0.6, blocks.max() + 1)[blocks]
        noise = shared + rng.normal(0, 0.2, N)
        if kind == 0:                          # model concave, data concave
            vf[:, g] = concave; vo[:, g] = concave + noise
        elif kind == 1:                        # model concave, data CONVEX -> misfit
            vf[:, g] = concave
            vo[:, g] = straight + 0.12 * (base - 5.5) ** 2 + noise
        else:                                  # model ~straight, data straight
            vf[:, g] = straight + 0.001 * base ** 2; vo[:, g] = straight + noise
    out = bow_separation_scores(mu, vo, mf, vf, up, blocks=blocks,
                                n_boot=80, n_jobs=1, branches=("up",), verbose=False)
    out_iid = bow_separation_scores(mu, vo, mf, vf, up, blocks=np.arange(N),
                                    n_boot=80, n_jobs=1, branches=("up",), verbose=False)
    zf, zo, mis = out["up_bow_z_fit"], out["up_bow_z_obs"], out["up_bow_misfit"]
    k0, k1, k2 = slice(0, per), slice(per, 2 * per), slice(2 * per, 3 * per)
    checks = {
        "concave genes: model bend resolvable (z_fit>2) in >=90%": np.mean(zf[k0] > 2) >= .9,
        "concave genes: observed concave in >=90%": np.mean(zo[k0] > 0) >= .9,
        "concave genes: misfit in <=10%": np.mean(mis[k0]) <= .1,
        "convex-data genes: flagged misfit in >=90%": np.mean(mis[k1]) >= .9,
        "straight genes: model bend NOT resolvable (|z_fit|<=2) in >=90%": np.mean(np.abs(zf[k2]) <= 2) >= .9,
        "straight genes: misfit in <=10%": np.mean(mis[k2]) <= .1,
        "block SE > 1.5x i.i.d. SE (smoothing respected)":
            np.nanmedian(out["up_bow_se"] / out_iid["up_bow_se"]) > 1.5,
    }
    for k, v in checks.items():
        print(("  PASS  " if v else "  FAIL  ") + k)
    print(f"  (median block/iid SE ratio: {np.nanmedian(out['up_bow_se'] / out_iid['up_bow_se']):.2f})")
    return all(checks.values())


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
