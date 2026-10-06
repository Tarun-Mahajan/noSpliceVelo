"""
Diagnostics for magnitude concentration in a velocity layer.

alpha_sweep   : how per-gene scaling  v / std(v)**alpha  trades off
                effective gene count against expression-rank bias.
gene_autocorr : per-gene Moran's I of velocity over the kNN graph, with an
                optional permutation null. Use it to keep genes whose velocity
                is spatially coherent instead of tuning alpha blindly.
"""
import numpy as np
import pandas as pd


def _dense(M):
    if hasattr(M, "toarray"):
        return M.toarray()
    return np.asarray(M)


def _clean(adata, vkey, xkey=None):
    """Velocity (and optionally expression) as dense float arrays, NaN genes dropped."""
    V = _dense(adata.layers[vkey]).astype(float)
    keep = np.isfinite(V).all(0) & (V.std(0) > 0)
    X = None
    if xkey is not None:
        X = _dense(adata.layers[xkey]).astype(float)
        keep &= np.isfinite(X).all(0) & (X.mean(0) > 0)
    idx = np.flatnonzero(keep)
    return V[:, idx], (X[:, idx] if X is not None else None), idx


def _gene_scale(V, how="mean_abs"):
    """
    Per-gene scale used in  v / scale**alpha.

    "mean_abs" : mean(|v|)          uncentred; makes total |v| equal across genes
                                    at alpha=1, so the expression-neutrality
                                    target is always attainable. Default.
    "rms"      : sqrt(mean(v**2))   uncentred second moment.
    "median_abs": median(|v|)       robust, but 0 if a gene is >50% zeros.
    "std"      : std(v)             CENTRED -- discards a gene's mean velocity,
                                    which is real signal. Reproduces the older
                                    behaviour; can leave the target unreachable.
    """
    if how == "mean_abs":     sc = np.abs(V).mean(0)
    elif how == "rms":        sc = np.sqrt((V ** 2).mean(0))
    elif how == "median_abs": sc = np.median(np.abs(V), 0)
    elif how == "std":        sc = V.std(0)
    else: raise ValueError(how)
    sc = np.asarray(sc, float)
    sc[~np.isfinite(sc) | (sc <= 0)] = 1.0
    return sc


def _eff_genes(X, transform="sqrt"):
    """Effective number of genes carrying each cell's velocity vector."""
    if transform in ("sqrt", "signed_sqrt"):
        p = np.abs(X)
    elif transform == "signed_4throot":
        p = np.abs(X) ** 2
    else:
        p = X ** 2
    denom = (p ** 2).sum(1)
    denom[denom == 0] = np.nan
    return (p.sum(1) ** 2) / denom


def _top_k_idx(X, k):
    k = min(k, X.shape[1])
    return np.argpartition(-np.abs(X), k - 1, axis=1)[:, :k]


def alpha_sweep(adata, vkey, xkey, alphas=None, top_k=16, transform="sqrt",
                scale="mean_abs", verbose=True):
    """
    Sweep per-gene scaling exponent alpha in  v / scale(v)**alpha,
    where scale is chosen by `scale` (see _gene_scale; default mean(|v|)).

    alpha = 0  -> raw velocity            (abundant genes dominate the cosine)
    alpha = 1  -> full standardisation    (low-expression noisy genes dominate)

    Returns a DataFrame with, per alpha:
      eff_genes              median effective genes per cell
      expr_rank_percell      median over cells of (median expression rank of that
                             cell's top-k genes); 0 = most expressed
      expr_rank_union        median expression rank of the union of per-cell top-k
      expr_rank_wtd          |v|-weighted mean expression rank; neutral =
                             (n_genes-1)/2. Free of any top-k choice, so it stays
                             meaningful once eff_genes >> top_k (where the
                             expr_rank_* columns stop being informative).
      energy_top10pct        share of total |v| held by the top-decile most
                             expressed genes; 0.10 = no expression bias
    `neutral` marks the alpha whose energy_top10pct is closest to 0.10 (the
    value expected if velocity magnitude were unrelated to expression level).
    """
    if alphas is None:
        alphas = [0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    V, X, idx = _clean(adata, vkey, xkey)
    n_cells, n_genes = V.shape
    sd = _gene_scale(V, scale)
    mu = X.mean(0)
    erank = np.argsort(np.argsort(-mu))          # 0 = most expressed

    rows = []
    for a in alphas:
        Xa = V / sd ** a
        eff = _eff_genes(Xa, transform)
        top = _top_k_idx(Xa, top_k)
        rows.append(dict(
            alpha=float(a),
            eff_genes=float(np.nanmedian(eff)),
            expr_rank_percell=float(np.median(np.median(erank[top], axis=1))),
            expr_rank_union=float(np.median(erank[np.unique(top)])),
            n_union=int(np.unique(top).size),
            expr_rank_wtd=float(
                (np.abs(Xa).sum(0) * erank).sum() / np.abs(Xa).sum()),
            energy_top10pct=float(
                np.abs(Xa)[:, erank < max(1, n_genes // 10)].sum() / np.abs(Xa).sum()),
        ))

    df = pd.DataFrame(rows)
    df["neutral"] = ""
    df.loc[(df.energy_top10pct - 0.10).abs().idxmin(), "neutral"] = "<-"
    df.attrs.update(n_cells=n_cells, n_genes=n_genes, vkey=vkey, xkey=xkey,
                    scale=scale)

    if verbose:
        print(f"{vkey}: {n_cells} cells x {n_genes} usable genes "
              f"(neutral expression rank = {n_genes/2:.0f})")
        print(df.to_string(index=False,
                           float_format=lambda v: f"{v:9.3f}"))
    return df


def gene_autocorr(adata, vkey, obsp_key="connectivities", n_perm=0, seed=0,
                  decouple=False, verbose=True):
    """
    Per-gene Moran's I of velocity over the kNN graph:
        I_g = sum_i x_ig (W x)_ig / sum_i x_ig**2      (x centred per gene,
                                                        W row-normalised)
    High I  -> velocity varies smoothly over the manifold (signal).
    I near 0 -> velocity is spatially random (noise); these are the genes that
                blow up under full standardisation.

    decouple=True scores on the SECOND neighbour ring (2-hop minus 1-hop)
    instead of the smoothing neighbours. Without it, velocity derived from
    kNN-smoothed moments is smooth over this graph by construction and every
    gene looks significant. Costs one sparse matrix square -- cheap at 4k cells,
    heavy at 35k.

    n_perm > 0 adds a permutation null (cell labels shuffled), giving z and a
    one-sided empirical p per gene.

    Returns a DataFrame indexed by gene name.
    """
    V, _, idx = _clean(adata, vkey)
    W = adata.obsp[obsp_key]
    if decouple:
        A = (W > 0)
        if hasattr(A, "astype") and hasattr(A, "tocsr"):
            A = A.astype(bool).tocsr()
            ring = ((A @ A).astype(bool) > A).tocsr()   # 2-hop not already 1-hop
            ring.setdiag(False); ring.eliminate_zeros()
        else:
            A = np.asarray(A, bool)
            ring = ((A @ A) > 0) & ~A
            np.fill_diagonal(ring, False)
        W = ring.astype(float)
    rs = np.asarray(W.sum(1)).ravel()
    rs[rs == 0] = 1.0
    if hasattr(W, "multiply"):                    # sparse: scale rows
        W = W.multiply((1.0 / rs)[:, None]).tocsr()
    else:
        W = np.asarray(W) / rs[:, None]

    Xc = V - V.mean(0, keepdims=True)
    ss = (Xc ** 2).sum(0)
    obs = (Xc * (W @ Xc)).sum(0) / ss

    out = pd.DataFrame({"moran_I": obs}, index=adata.var_names[idx])

    if n_perm > 0:
        rng = np.random.default_rng(seed)
        null = np.empty((n_perm, V.shape[1]))
        for b in range(n_perm):
            Xp = Xc[rng.permutation(Xc.shape[0])]
            null[b] = (Xp * (W @ Xp)).sum(0) / (Xp ** 2).sum(0)
        m, s = null.mean(0), null.std(0)
        s[s == 0] = np.nan
        out["null_mean"] = m
        out["null_sd"] = s
        out["z"] = (obs - m) / s
        out["pval"] = (null >= obs).sum(0) / n_perm

    out = out.sort_values("moran_I", ascending=False)
    if verbose:
        print(f"{vkey}: Moran's I over {obsp_key} for {V.shape[1]} genes")
        print(f"  median I = {np.median(obs):.3f}   "
              f"frac I > 0.1 = {(obs > 0.1).mean():.2%}")
        if n_perm:
            print(f"  frac z > 3 = {(out['z'] > 3).mean():.2%}  ({n_perm} permutations)")
    return out


# ---------------------------------------------------------------------------
# Automatic, data-dependent choice of alpha
# ---------------------------------------------------------------------------

_CRITERIA = {
    # name              : (column,            target given n_genes)
    "expr_rank_wtd":      ("expr_rank_wtd",   lambda g: (g - 1) / 2.0),
    "energy_top10pct":    ("energy_top10pct", lambda g: 0.10),
}


def pick_alpha_from_sweep(df, n_genes, criterion="expr_rank_wtd",
                          alpha_max=1.0, eff_plateau_tol=0.10):
    """
    Choose alpha from an existing alpha_sweep table by linear interpolation to
    the criterion's neutral target.

    criterion : "expr_rank_wtd"   -> |v|-weighted mean expression rank -> (G-1)/2
                "energy_top10pct" -> top-decile energy share           -> 0.10

    Returns a dict (one row per dataset, ready to concatenate):
      alpha, criterion, target, value_at_alpha, clamped, monotonic,
      eff_genes_at_alpha, eff_genes_max, eff_frac, n_genes
    `clamped` is "low"/"high" when the criterion is never crossed on the grid,
    in which case alpha is pinned to the grid edge -- inspect before trusting.
    """
    col, target_fn = _CRITERIA[criterion]
    d = df.sort_values("alpha")
    x = d["alpha"].to_numpy(float)
    y = d[col].to_numpy(float)
    target = float(target_fn(n_genes))

    # criterion must increase with alpha (energy share decreases -> flip)
    sign = 1.0 if y[-1] >= y[0] else -1.0
    ys, ts = sign * y, sign * target
    monotonic = bool(np.all(np.diff(ys) >= -1e-9))

    if ts <= ys[0]:
        alpha, clamped = float(x[0]), "low"
    elif ts > ys[-1] + 1e-3 * max(abs(ys[-1] - ys[0]), 1e-12):
        alpha, clamped = float(min(x[-1], alpha_max)), "high"
    else:
        alpha, clamped = float(np.interp(ts, ys, x)), ""
    alpha = float(np.clip(alpha, 0.0, alpha_max))

    eff = np.interp(alpha, x, d["eff_genes"].to_numpy(float))
    eff_max = float(d["eff_genes"].max())
    return dict(
        alpha=round(alpha, 4),
        criterion=criterion,
        target=round(target, 4),
        value_at_alpha=round(float(np.interp(alpha, x, y)), 4),
        clamped=clamped,
        monotonic=monotonic,
        eff_genes_at_alpha=round(float(eff), 1),
        eff_genes_max=round(eff_max, 1),
        eff_frac=round(float(eff / eff_max) if eff_max else np.nan, 3),
        near_plateau=bool(eff_max and eff / eff_max >= 1 - eff_plateau_tol),
        n_genes=int(n_genes),
    )


def select_alpha(adata, vkey, xkey, alphas=None, criterion="expr_rank_wtd",
                 refine=True, alpha_max=1.0, top_k=16, transform="sqrt",
                 scale="mean_abs", dataset=None, verbose=True):
    """
    Run alpha_sweep and pick alpha automatically. Intended for pipeline use:
    one call per dataset, returns a dict you can collect into a table.

    refine=True runs a second, finer sweep inside the bracketing interval so the
    answer isn't limited to the coarse grid (costs one extra sweep).

    Example
    -------
    rows = [select_alpha(ad, "velocity_mu_soft", "mu_scvi_smooth", dataset=name)
            for name, ad in datasets.items()]
    pd.DataFrame(rows).to_csv("alpha_per_dataset.csv", index=False)
    """
    df = alpha_sweep(adata, vkey, xkey, alphas=alphas, top_k=top_k,
                     transform=transform, scale=scale, verbose=False)
    n_genes = df.attrs["n_genes"]
    rec = pick_alpha_from_sweep(df, n_genes, criterion, alpha_max)

    if refine and not rec["clamped"]:
        grid = np.sort(df["alpha"].to_numpy(float))
        lo = grid[grid <= rec["alpha"]].max()
        hi = grid[grid >= rec["alpha"]].min()
        if hi > lo:
            fine = alpha_sweep(adata, vkey, xkey,
                               alphas=np.round(np.linspace(lo, hi, 9), 4).tolist(),
                               top_k=top_k, transform=transform, scale=scale,
                               verbose=False)
            rec = pick_alpha_from_sweep(fine, n_genes, criterion, alpha_max)
            df = pd.concat([df, fine]).drop_duplicates("alpha").sort_values("alpha")

    rec["dataset"] = dataset
    rec["vkey"] = vkey
    rec["scale"] = scale
    rec["n_cells"] = df.attrs.get("n_cells", adata.n_obs)

    if verbose:
        flag = ""
        if rec["clamped"]:
            flag += f"  [CLAMPED {rec['clamped']} -- criterion never crossed]"
        if not rec["monotonic"]:
            flag += "  [non-monotonic criterion -- inspect the sweep]"
        if not rec["near_plateau"]:
            flag += "  [eff_genes below plateau]"
        print(f"{dataset or vkey}: alpha = {rec['alpha']:.3f}  "
              f"({criterion} = {rec['value_at_alpha']:.1f} vs target {rec['target']:.1f}; "
              f"eff_genes {rec['eff_genes_at_alpha']:.0f}/{rec['eff_genes_max']:.0f}"
              f" = {rec['eff_frac']:.0%}){flag}")
    return rec


def scale_velocity(adata, vkey, alpha, out_key=None, scale="mean_abs"):
    """Write  v / scale(v)**alpha  to adata.layers[out_key] (default vkey+'_a{alpha}').
    `scale` MUST match what select_alpha/alpha_sweep used."""
    V = _dense(adata.layers[vkey]).astype(float)
    sd = _gene_scale(V, scale)
    out = V / sd ** float(alpha)
    key = out_key or f"{vkey}_a{alpha:g}"
    adata.layers[key] = out
    return key
