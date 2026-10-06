"""Gene-scale-controlled velocity confidence.

Drop-in replacement for ``scvelo.tl.velocity_confidence`` (v0.3.4) that adds
explicit control over per-gene magnitude, for velocity estimates that live on a
raw / corrected-count scale rather than log1p (e.g. noSpliceVelo, where gene
velocities span orders of magnitude and a handful of high-expression genes would
otherwise dominate every cell's correlation).

The stock metric is::

    V = V[:, velocity_genes]        # NaN + velocity_genes + spearmans_score
    V -= V.mean(axis=1)[:, None]    # centre per CELL, across genes
    R[i] = mean_j cos(V[j], V[i])   # j over KNN of i, from adata.obsp distances
    confidence = clip(R, 0, None)

There is no per-gene standardisation anywhere in it, so each cell's correlation
is driven by whichever genes happen to have the largest velocity magnitudes.
This module inserts an optional per-gene rescaling (and optional elementwise
compression) *before* the per-cell centring, leaving everything else identical.

Setting ``gene_scale=None, transform=None, corr="pearson", use_genes=None``
reproduces ``scv.tl.velocity_confidence`` exactly.

Usage
-----
>>> import velocity_confidence_scaled as vcs
>>> vcs.velocity_confidence(adata, vkey="velocity")                      # MAD-scaled
>>> vcs.velocity_confidence(adata, gene_scale=None, key_added="vc_raw")  # scvelo parity
>>> vcs.velocity_confidence(adata, use_genes="all", gene_scale="mad")    # all 2k HVGs

Writes ``{key}``, ``{key}_raw`` (unclipped), ``{vkey}_length`` to ``.obs`` and the
settings to ``adata.uns[{key}_params]``.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.sparse import csr_matrix, issparse

__all__ = [
    "velocity_confidence",
    "velocity_confidence_transition",
    "confidence_report",
]

_MAD_TO_SIGMA = 1.4826  # makes MAD comparable to std under normality


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _to_dense(X) -> np.ndarray:
    """Dense float array; scvelo's ``np.array(layer)`` silently breaks on sparse."""
    if issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float64)


def _signed(fn):
    """Lift a positive-domain transform to signed data: sign(v) * fn(|v|)."""

    def _f(V):
        return np.sign(V) * fn(np.abs(V))

    return _f


_TRANSFORMS = {
    None: lambda V: V,
    "none": lambda V: V,
    "signed_sqrt": _signed(np.sqrt),
    "signed_4throot": _signed(lambda V: np.abs(V) ** 0.25),
    "signed_log1p": _signed(np.log1p),
}


def _gene_scaler(V: np.ndarray, how: str | None) -> np.ndarray:
    """Per-gene divisor, shape (n_genes,). Never centres per gene by default.

    'mad'  robust spread, 1.4826 * median(|v - median(v)|)   [recommended]
    'std'  standard deviation across cells
    'zscore' std, and the gene mean is removed as well (see velocity_confidence)
    'max'  max(|v|) across cells
    'l2'   l2 norm across cells
    None   no rescaling (scvelo behaviour)
    """
    if how in (None, "none"):
        return np.ones(V.shape[1])

    if how == "mad":
        med = np.median(V, axis=0)
        s = _MAD_TO_SIGMA * np.median(np.abs(V - med), axis=0)
    elif how in ("std", "zscore"):
        s = V.std(axis=0)
    elif how == "max":
        s = np.abs(V).max(axis=0)
    elif how == "l2":
        s = np.sqrt((V**2).sum(axis=0))
    else:
        raise ValueError(f"unknown gene_scale {how!r}")

    # MAD collapses to 0 for genes whose velocity is >50% ties (common with
    # sparse / zero-inflated estimates) -> fall back to std, then to 1.
    if how == "mad":
        bad = s <= 0
        if bad.any():
            s = s.copy()
            s[bad] = V[:, bad].std(axis=0)
    s = np.where(s > 0, s, 1.0)
    return s


def _resolve_genes(adata, V, vkey, use_genes):
    """Boolean gene mask. NaN genes are always dropped, as in scvelo."""
    keep = np.invert(np.isnan(np.sum(V, axis=0)))

    if use_genes is None:  # scvelo default: method's own filters
        if f"{vkey}_genes" in adata.var.keys():
            keep &= np.asarray(adata.var[f"{vkey}_genes"], dtype=bool)
        if "spearmans_score" in adata.var.keys():
            keep &= adata.var["spearmans_score"].values > 0.1
        return keep

    if isinstance(use_genes, str):
        if use_genes == "all":
            return keep  # every gene with a finite velocity
        if use_genes in adata.var.keys():
            return keep & np.asarray(adata.var[use_genes], dtype=bool)
        raise ValueError(f"{use_genes!r} is not a column in adata.var")

    arr = np.asarray(use_genes)
    if arr.dtype == bool:
        return keep & arr
    return keep & adata.var_names.isin(arr).values  # list of gene names


def _neighbor_indices(adata, n_neighbors=None, neighbors_key=None):
    from scvelo.preprocessing.neighbors import get_neighs
    from scvelo.tools.utils import get_indices

    if neighbors_key is not None:
        dist = adata.obsp[neighbors_key]
    else:
        dist = get_neighs(adata, "distances")
    return get_indices(dist=dist, n_neighbors=n_neighbors)[0]


def _prepare(adata, vkey, use_genes, transform, gene_scale, corr):
    V = _to_dense(adata.layers[vkey])
    mask = _resolve_genes(adata, V, vkey, use_genes)
    n_used = int(mask.sum())
    if n_used < 10:
        raise ValueError(
            f"only {n_used} usable genes for '{vkey}'. If you meant to score on the "
            "full HVG set, pass use_genes='all' -- note that scvelo applies "
            f"adata.var['{vkey}_genes'] automatically whenever that column exists."
        )
    V = V[:, mask]

    # length is reported on the untouched (centred) velocities, as in scvelo,
    # so it stays comparable to stock output.
    V_centred = V - V.mean(1)[:, None]
    length = np.sqrt((V_centred**2).sum(1))

    if transform not in _TRANSFORMS:
        raise ValueError(f"unknown transform {transform!r}")
    W = _TRANSFORMS[transform](V)

    scaler = _gene_scaler(W, gene_scale)
    W = W / scaler[None, :]
    if gene_scale == "zscore":
        W = W - W.mean(0)[None, :]

    if corr == "spearman":
        from scipy.stats import rankdata

        W = rankdata(W, axis=1)
    elif corr != "pearson":
        raise ValueError(f"unknown corr {corr!r}")

    W = W - W.mean(1)[:, None]  # per-cell centring == correlation across genes
    norms = np.sqrt((W**2).sum(1))
    return W, norms, length, n_used


def _mean_neighbor_cosine(W, norms, indices, drop_self=True):
    """R[i] = mean_j cos(W[j], W[i]); vectorised equivalent of scvelo's loop."""
    n = W.shape[0]
    degenerate = norms <= 0
    safe = np.where(degenerate, 1.0, norms)
    What = W / safe[:, None]

    if drop_self:
        keep = indices != np.arange(n)[:, None]
    else:
        keep = np.ones_like(indices, dtype=bool)
    counts = keep.sum(1)
    rows = np.repeat(np.arange(n), counts)
    cols = indices[keep]
    A = csr_matrix((np.ones(cols.size), (rows, cols)), shape=(n, n))

    R = np.einsum("ij,ij->i", A @ What, What) / np.maximum(counts, 1)
    if degenerate.any():
        warnings.warn(
            f"{int(degenerate.sum())} cells have a zero velocity vector over the "
            "selected genes; their confidence is set to NaN.",
            stacklevel=2,
        )
        R = R.astype(float)
        R[degenerate] = np.nan
    return R


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def velocity_confidence(
    data,
    vkey="velocity",
    copy=False,
    *,
    gene_scale="mad",
    transform=None,
    corr="pearson",
    use_genes=None,
    n_neighbors=None,
    neighbors_key=None,
    drop_self=True,
    clip_negative=True,
    key_added=None,
    compute_transition=False,
):
    """Velocity confidence with explicit per-gene magnitude control.

    Parameters
    ----------
    data, vkey, copy
        As ``scvelo.tl.velocity_confidence``.
    gene_scale
        Per-gene divisor applied before the per-cell centring: ``'mad'``
        (default, robust), ``'std'``, ``'zscore'`` (std + per-gene centring,
        which additionally removes the common-mode drift shared by all cells),
        ``'max'``, ``'l2'``, or ``None`` for stock scvelo behaviour.
    transform
        Optional elementwise compression before scaling: ``'signed_sqrt'`` or
        ``'signed_log1p'``. Both are odd functions, so signs are preserved.
        Skip this if the velocities are already on a log scale.
    corr
        ``'pearson'`` (scvelo) or ``'spearman'`` (rank across genes; discards
        magnitude entirely).
    use_genes
        ``None`` reproduces scvelo's filter chain (``{vkey}_genes`` and
        ``spearmans_score``). ``'all'`` uses every gene with finite velocity.
        Also accepts an ``adata.var`` column name, a boolean mask, or gene names.
    n_neighbors, neighbors_key, drop_self
        Neighbour graph controls. Fix these across methods so the comparison
        isolates the velocities rather than the graphs.
    clip_negative
        Floor at 0 as scvelo does. The unclipped values are always stored under
        ``{key}_raw``.
    """
    adata = data.copy() if copy else data
    if vkey not in adata.layers.keys():
        raise ValueError("You need to run `tl.velocity` first.")

    key = key_added or f"{vkey}_confidence"

    W, norms, length, n_used = _prepare(
        adata, vkey, use_genes, transform, gene_scale, corr
    )
    indices = _neighbor_indices(adata, n_neighbors, neighbors_key)
    R = _mean_neighbor_cosine(W, norms, indices, drop_self=drop_self)

    adata.obs[f"{vkey}_length"] = length.round(2)
    adata.obs[f"{key}_raw"] = R
    adata.obs[key] = np.clip(R, 0, None) if clip_negative else R
    adata.uns[f"{key}_params"] = {
        "vkey": vkey,
        "gene_scale": gene_scale,
        "transform": transform,
        "corr": corr,
        "use_genes": use_genes if isinstance(use_genes, (str, type(None))) else "custom",
        "n_genes": n_used,
        "n_neighbors": int(indices.shape[1]),
        "neighbors_key": neighbors_key,
        "drop_self": drop_self,
        "clip_negative": clip_negative,
    }

    if compute_transition:
        velocity_confidence_transition(
            adata,
            vkey=vkey,
            gene_scale=gene_scale,
            transform=transform,
            use_genes=use_genes,
        )
    return adata if copy else None


def velocity_confidence_transition(
    data,
    vkey="velocity",
    scale=10,
    copy=False,
    *,
    gene_scale="mad",
    transform=None,
    use_genes=None,
    key_added=None,
):
    """Scaled counterpart of ``scvelo.tl.velocity_confidence_transition``.

    Correlates each cell's velocity with the displacement its own transition
    matrix implies. The same per-gene scaling is applied to ``V`` and to ``dX``
    so the two stay in a common geometry.
    """
    from scvelo.core import l2_norm, prod_sum
    from scvelo.tools.transition_matrix import transition_matrix

    adata = data.copy() if copy else data
    if vkey not in adata.layers.keys():
        raise ValueError("You need to run `tl.velocity` first.")

    key = key_added or f"{vkey}_confidence_transition"

    V = _to_dense(adata.layers[vkey])
    mask = _resolve_genes(adata, V, vkey, use_genes)
    X = _to_dense(adata.layers["Ms"])[:, mask]
    V = V[:, mask]

    T = transition_matrix(adata, vkey=vkey, scale=scale)
    dX = T.dot(X) - X

    tf = _TRANSFORMS[transform]
    V, dX = tf(V), tf(dX)
    s = _gene_scaler(V, gene_scale)
    V, dX = V / s[None, :], dX / s[None, :]

    V = V - V.mean(1)[:, None]
    dX = dX - dX.mean(1)[:, None]

    norms = l2_norm(dX, axis=1) * l2_norm(V, axis=1)
    norms += norms == 0
    adata.obs[key] = prod_sum(dX, V, axis=1) / norms
    return adata if copy else None


def confidence_report(data, vkey="velocity", use_genes="all", **kwargs):
    """Run the metric under several scalings; returns a tidy DataFrame.

    A stock-vs-scaled gap is the diagnostic: if confidence drops sharply once
    genes are put on a common footing, the original score was carried by a few
    high-magnitude genes rather than by coherent kinetics.
    """
    import pandas as pd

    settings = {
        "scvelo_stock": dict(gene_scale=None, transform=None, corr="pearson"),
        "mad": dict(gene_scale="mad", transform=None, corr="pearson"),
        "zscore": dict(gene_scale="zscore", transform=None, corr="pearson"),
        "signed_sqrt_mad": dict(gene_scale="mad", transform="signed_sqrt", corr="pearson"),
        "signed_4throot_scvelo": dict(gene_scale=None, transform="signed_4throot", corr="pearson"),
        "spearman": dict(gene_scale=None, transform=None, corr="spearman"),
    }
    out = {}
    for name, cfg in settings.items():
        velocity_confidence(
            data, vkey=vkey, use_genes=use_genes, key_added=f"_tmp_{name}", **cfg, **kwargs
        )
        out[name] = data.obs.pop(f"_tmp_{name}_raw").values
        del data.obs[f"_tmp_{name}"]
        del data.uns[f"_tmp_{name}_params"]
    return pd.DataFrame(out, index=data.obs_names)
