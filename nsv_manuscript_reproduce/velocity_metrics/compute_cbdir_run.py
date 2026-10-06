"""Config-driven wrapper to compute Cross-Boundary Direction (CBDir) correctness
across multiple velocity methods and datasets.

Features:
- Per-cell CBDir scores (cell barcode retained) for every dataset x transition
  edge x method.
- Transition edges (`cluster_edges`) are a dataset-level property declared once
  in the global config; per-method overrides are supported.
- Optional `cluster_map` per dataset: remap the unique levels of the cluster
  column onto custom levels (many-to-one merging supported) before edge
  matching. The original obs column is never modified.
- Embedding modes (`x_source`):
    * "existing"  (DEFAULT) use the stored obsm[emb_key]; nothing recomputed,
                  because the neighbour and velocity graphs in the h5ad were
                  built on that embedding.
    * "layer"     .X = layers[x_layer] -> normalize_total -> log1p -> PCA
    * "layer_sum" .X = layers[a] + layers[b] -> normalize_total -> log1p -> PCA
    * "X"         .X as-is (no normalization, no log) -> PCA
- `vkey` and `xkey` are per-method settings, and every dependent name is composed
  from them: the velocity layer `{vkey}`, the graphs `{vkey}_graph` /
  `{vkey}_graph_neg`, the parameters `{vkey}_params`, and the embedding
  `{vkey}_{basis}` against `X_{basis}` for a configurable `basis` (default pca).
- Velocity is projected with `scv.tl.velocity_embedding(basis=basis, vkey=vkey)`,
  using the velocity graph already present in each method's h5ad. `xkey` names
  the expression layer: it is temporarily swapped into layers['Ms'] for the
  scVelo calls and the original Ms is restored (or removed) afterwards.
  Setting `recompute_velocity_graph: true` rebuilds `{vkey}_graph` with
  `scv.tl.velocity_graph(xkey=xkey, vkey=vkey)` first.
- Three tables per dataset: tidy long, wide pivot on (cell_barcode, edge), and a
  per-edge summary. CSV always; Parquet optional.
- In-cluster velocity consistency (`iccoh`), by default in the ORIGINAL gene
  space (`iccoh_space: gene_confidence`). The vectors are prepared by
  `velocity_confidence_scaled._prepare`, the same code that
  compute_velocity_confidence_run.py uses: shared gene intersection across
  methods, the chosen confidence version's transform / per-gene scaling, and
  Pearson (per-cell centred) correlation across genes, over the kNN
  `distances` graph. The only change is that each cell's neighbours are
  restricted to its own cluster. `velocity_pca` is a transition-probability
  reconstruction, i.e. a smoothing operator, so coherence measured on it is
  largely a property of the projection; `iccoh_space: embedding` restores that
  older behaviour, and `iccoh_embedding` keeps it as a diagnostic column.

Usage:
    python compute_cbdir_run.py cbdir_global_config_4throot.yaml [--str-suffix _v2]
"""

import os
import sys
import gc
import time
import traceback
import argparse
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scvelo as scv
from scipy.sparse import csr_matrix, issparse
from sklearn.metrics.pairwise import cosine_similarity

try:
    import velocity_confidence_scaled as vcs
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import velocity_confidence_scaled as vcs


# Pipeline parameters allowed in the per-method `defaults:` block or per-dataset.
_PIPELINE_PARAMS = (
    "k_cluster",
    "vkey",
    "xkey",
    "basis",
    "emb_key",
    "x_source",
    "x_layer",
    "x_layers",
    "n_comps",
    "n_pcs_metric",
    "target_sum",
    "recompute_neighbors",
    "n_neighbors",
    "recompute_velocity_graph",
    "approx",
    "sqrt_transform",
    "n_jobs",
    "use_negative_cosines",
    "reuse_velocity_embedding",
    "include_skipped",
    "min_target_neighbors",
    "smooth_velocity_rounds",
    "smooth_velocity_lambda",
    "smooth_velocity_space",
    "smooth_graph",
    "smooth_k",
    "compute_iccoh",
    "iccoh_min_neighbors",
    "iccoh_space",
    "iccoh_clusters",
    "iccoh_gene_raw",
    "iccoh_gene_vst",
    "iccoh_gene_power",
    "iccoh_confidence_version",
    "iccoh_neighbor_graph",
    "iccoh_use_genes",
    "iccoh_intersection_mode",
    "iccoh_embedding",
    "gene_names_col",
    "gene_query",
    "apply_gene_query_always",
    "root_power",
    "iccoh_recompute_graph",
    "iccoh_n_neighbors",
    "iccoh_rep",
    "iccoh_gene_set_contrast",
    "iccoh_dimensionality",
    "iccoh_gene_subsample",
    "iccoh_gene_subsample_reps",
    "iccoh_gene_subsample_seed",
    "cluster_edges",
    "cluster_map",
    "cluster_map_drop_unmapped",
)

_DEFAULT_FALLBACKS = {
    "k_cluster": "clusters",
    "vkey": "velocity",
    "xkey": "Ms",
    "basis": "pca",
    "emb_key": None,          # None -> f"X_{basis}"
    "x_source": "existing",
    "x_layer": None,
    "x_layers": None,
    "n_comps": 30,
    "n_pcs_metric": None,
    "target_sum": None,
    "recompute_neighbors": False,
    "n_neighbors": 30,
    "recompute_velocity_graph": False,
    "approx": False,
    "sqrt_transform": True,
    "n_jobs": None,
    "use_negative_cosines": "auto",
    "reuse_velocity_embedding": "auto",
    "include_skipped": False,
    "min_target_neighbors": 3,
    "smooth_velocity_rounds": 0,
    "smooth_velocity_lambda": 0.5,
    "smooth_velocity_space": "gene",
    "smooth_graph": "recompute",
    "smooth_k": 60,
    "compute_iccoh": True,
    "iccoh_min_neighbors": 1,
    "iccoh_space": "gene_confidence",
    "iccoh_clusters": "source",
    "iccoh_gene_raw": True,
    "iccoh_gene_vst": True,
    "iccoh_gene_power": 0.25,
    # gene_confidence settings, mirroring compute_velocity_confidence_run.py
    "iccoh_confidence_version": "signed_4throot_scvelo",
    "iccoh_neighbor_graph": "distances",
    "iccoh_use_genes": "all",
    "iccoh_intersection_mode": True,
    "iccoh_embedding": True,
    "gene_names_col": None,
    # gene selection, exactly as compute_velocity_confidence_run.py applies it
    "gene_query": None,
    "apply_gene_query_always": False,
    "root_power": 2,                 # PR / effective genes only; never enters ICCoh
    # ICCoh's own neighbour graph (confidence's recompute_graph / n_neighbors / rep);
    # it is written under obsp['iccoh_*'] and never replaces the graph CBDir reads
    "iccoh_recompute_graph": False,
    "iccoh_n_neighbors": None,       # None -> n_neighbors
    "iccoh_rep": "X_pca",
    # gene-count bias diagnostics
    "iccoh_gene_set_contrast": True, # ICCoh on BOTH own and shared genes
    "iccoh_dimensionality": True,    # participation ratio + effective genes per set
    "iccoh_gene_subsample": [0.25, 0.5, "min"],
    "iccoh_gene_subsample_reps": 3,
    "iccoh_gene_subsample_seed": 0,
    "cluster_edges": None,
    "cluster_map": None,
    "cluster_map_drop_unmapped": False,
}

# `pca_key` is accepted as a backwards-compatible alias for `emb_key`.
_PARAM_ALIASES = {
    "pca_key": "emb_key",
    # compute_velocity_confidence_run.py spellings, so its `defaults:` block can be
    # pasted in. `n_neighbors` is NOT aliased: here it already names CBDir's own
    # graph size; `iccoh_n_neighbors` falls back to it.
    "use_genes": "iccoh_use_genes",
    "intersection_mode": "iccoh_intersection_mode",
    "recompute_graph": "iccoh_recompute_graph",
    "rep": "iccoh_rep",
}

# Accepted but meaningless for the CBDir pipeline; logged once rather than warned
# as unknown when a confidence config is pasted in.
_CONFIDENCE_ONLY_PARAMS = ("use_confidence_report", "cluster_key", "str_suffix")

_BOOL_PARAMS = ("compute_iccoh", "iccoh_intersection_mode", "iccoh_embedding",
                "iccoh_gene_raw", "iccoh_gene_vst", "apply_gene_query_always",
                "iccoh_recompute_graph", "iccoh_gene_set_contrast",
                "iccoh_dimensionality", "recompute_neighbors",
                "recompute_velocity_graph", "include_skipped")


def _as_bool(name, v):
    """Strict boolean: a string like 'gene_confidence' in a boolean slot is a
    config mistake, not a truthy value."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str) and v.strip().lower() in ("true", "yes", "on", "1"):
        return True
    if isinstance(v, str) and v.strip().lower() in ("false", "no", "off", "0"):
        return False
    raise ValueError(f"'{name}' must be true or false; got {v!r}"
                     + (" (did you mean iccoh_space: gene_confidence?)"
                        if str(v).lower().startswith("gene") else ""))

_VALID_X_SOURCES = ("existing", "layer", "layer_sum", "X")

# Working obs column suffix used for the (optionally) remapped cluster labels.
_MAPPED_SUFFIX = "__cbdir_mapped"

# The confidence versions of `velocity_confidence_scaled.confidence_report`,
# column name -> kwargs. Kept identical to that function's `settings` table so an
# ICCoh version and a confidence column of the same name are the same metric up
# to the neighbour restriction (tests_cbdir checks this numerically).
_CONFIDENCE_VERSIONS = {
    "scvelo_stock": dict(gene_scale=None, transform=None, corr="pearson"),
    "mad": dict(gene_scale="mad", transform=None, corr="pearson"),
    "zscore": dict(gene_scale="zscore", transform=None, corr="pearson"),
    "signed_sqrt_mad": dict(gene_scale="mad", transform="signed_sqrt", corr="pearson"),
    "signed_4throot_scvelo": dict(gene_scale=None, transform="signed_4throot", corr="pearson"),
    "spearman": dict(gene_scale=None, transform=None, corr="spearman"),
}

_ICCOH_SPACE_ALIASES = {
    "gene_confidence": "gene_confidence", "confidence": "gene_confidence",
    "gene_conf": "gene_confidence", "gene_hd": "gene_confidence", "hd": "gene_confidence",
    "gene": "gene", "gene_raw": "gene", "benchmark": "gene",
    "embedding": "embedding", "emb": "embedding",
}


def normalize_iccoh_space(space) -> str:
    """Map an `iccoh_space` spelling onto gene_confidence | gene | embedding."""
    s = str(space).strip().lower()
    if s not in _ICCOH_SPACE_ALIASES:
        raise ValueError(
            f"iccoh_space must be one of gene_confidence (default), gene, embedding; "
            f"got {space!r}")
    return _ICCOH_SPACE_ALIASES[s]


# ---------------------------------------------------------------------------
# Key composition — every scVelo-side name derives from vkey / xkey / basis
# ---------------------------------------------------------------------------

def velocity_graph_key(vkey: str) -> str:
    return f"{vkey}_graph"


def velocity_graph_neg_key(vkey: str) -> str:
    return f"{vkey}_graph_neg"


def velocity_params_key(vkey: str) -> str:
    return f"{vkey}_params"


def velocity_embedding_key(vkey: str, basis: str) -> str:
    return f"{vkey}_{basis}"


def basis_embedding_key(basis: str) -> str:
    return f"X_{basis}"


def _has_key(adata, key: str) -> bool:
    """scVelo stores graphs in .uns (0.3.x) or .obsp depending on version."""
    return (key in adata.uns) or (key in adata.obsp)


# ---------------------------------------------------------------------------
# Core metric (adapted from compute_cbdir.py)
# ---------------------------------------------------------------------------

def keep_type(clusters: np.ndarray, nodes: np.ndarray, target: str):
    """Select the subset of `nodes` whose cluster label equals `target`."""
    if len(nodes) == 0:
        return nodes
    return nodes[clusters[nodes] == target]


def remove_type(clusters: np.ndarray, nodes: np.ndarray, target: str):
    """Exclude nodes of the targeted type."""
    if len(nodes) == 0:
        return nodes
    return nodes[clusters[nodes] != target]


def in_cluster_coherence_percell(
    adata,
    clusters: np.ndarray,
    cluster_list: List[str],
    v_emb_key: str = "velocity_pca",
    n_pcs_metric: Optional[int] = None,
    min_neighbors: int = 1,
    space: str = "embedding",
    v_layer_key: str = "velocity",
    gene_mask: Optional[np.ndarray] = None,
    gene_power: Optional[float] = None,
) -> Dict[str, Dict[str, float]]:
    """Per-cell In-Cluster Coherence (ICCoh), the companion metric to CBDir.

    ICCoh is the mean cosine similarity between a cell's velocity and the
    velocities of its same-cluster neighbours — introduced alongside CBDir in
    veloAE and reported beside it ever since (UniTVelo, veloVI, the 2026
    benchmarks, where it also appears as ICVCoh or "velocity consistency").

    It is NOT a correctness metric and must never be read as one: it is blind to
    orientation, so a wholly reversed velocity field scores as highly as a
    correct one. Its value here is precisely that blindness. ICCoh measures the
    local smoothness of the field, and smoothness is the main thing that can
    inflate CBDir for free, because the graph-derived arrow that CBDir scores is
    a transition-weighted average over the same neighbourhood. Carrying ICCoh
    alongside CBDir is what lets a reader see whether a method's boundary
    correctness is bought by coherence or earned on top of it.

    This function serves the two LEGACY spaces. The pipeline default is
    `iccoh_space: gene_confidence`, handled by
    `in_cluster_consistency_gene_percell`.

    `space` picks where the cosines are taken, and the choice matters:

    - "embedding": the same embedding, graph and dimensions as CBDir. It shares
      CBDir's geometry, but `velocity_pca` is reconstructed from the velocity
      graph's transition probabilities, so a method whose gene-space velocities
      are mutually incoherent still gets a smooth arrow field; coherence
      measured here is substantially a property of the projection.
    - "gene": `adata.layers[v_layer_key]` restricted to genes with a non-NaN
      velocity, which is what the 2026 benchmark's `inner_cluster_coh` computes.
      Use it to reproduce published numbers; note that the same benchmark scores
      CBDir in UMAP space, so its two metrics do not share a geometry.

    `gene_power` (gene space only) applies sign(v) * |v|^p elementwise before the
    cosines. It exists for two reasons, and the second is the binding one:

    * Variance stabilisation. For var ~ mean^q the stabilising transform is
      x^(1 - q/2): q = 1 gives sqrt (scVelo's `sqrt_transform`, the Poisson case)
      and q = 1.5 gives the fourth root. Without it a handful of very highly
      expressed genes dominate the dot product and the cosine stops reflecting
      the velocity gene set.
    * Consistency with CBDir. When the velocity graph was built on a transformed
      velocity, `{vkey}_{basis}` — and therefore CBDir — scores the TRANSFORMED
      field. Gene-space ICCoh on the raw layer then measures the smoothness of a
      different field from the one being controlled for, which is no control at
      all. Embedding-space ICCoh inherits the transform through the graph and
      needs no correction; gene space does.

    Two honest caveats. Velocity is signed, so this is the same heuristic
    extension of a count-data transform that scVelo already makes. And the map is
    per coordinate and nonlinear, so it does NOT preserve the direction of the
    velocity vector: what comes back is the coherence of the transformed field,
    not of the velocity field itself. Leave it None to reproduce published
    numbers.

    `min_neighbors` defaults to 1, matching the reference implementation's
    `len(same_cat_nodes) > 0`. Raising it to 3 aligns the rule with CBDir's own
    skip of cells with <= 2 target neighbours, at the cost of departing from the
    published definition. The cell's own velocity is excluded from its neighbour
    set either way.
    """
    space = str(space).lower()
    if space.startswith("g"):
        if v_layer_key not in adata.layers:
            raise KeyError(f"v_layer_key '{v_layer_key}' not in adata.layers "
                           f"(have: {list(adata.layers)})")
        v_emb = np.asarray(adata.layers[v_layer_key], dtype=np.float64)
        mask = (~np.isnan(v_emb[0]) if gene_mask is None
                else np.asarray(gene_mask, dtype=bool))
        v_emb = v_emb[:, mask]
        if gene_power is not None:
            p = float(gene_power)
            if not (0.0 < p <= 1.0):
                raise ValueError(f"gene_power must be in (0, 1]; got {p}.")
            # sign-preserving: |v|^p on the magnitude, direction of each
            # coordinate untouched. p = 1 is the identity.
            v_emb = np.sign(v_emb) * np.power(np.abs(v_emb), p)
    else:
        if v_emb_key not in adata.obsm:
            raise KeyError(f"v_emb_key '{v_emb_key}' not found in adata.obsm "
                           f"(have: {list(adata.obsm)})")
        v_emb = np.asarray(adata.obsm[v_emb_key], dtype=np.float64)
        if n_pcs_metric is not None and int(n_pcs_metric) <= v_emb.shape[1]:
            v_emb = v_emb[:, :int(n_pcs_metric)]

    if "neighbors" not in adata.uns:
        raise KeyError("adata.uns['neighbors'] not found; the h5ad carries no neighbour graph.")
    conn_key = adata.uns["neighbors"].get("connectivities_key", "connectivities")
    if conn_key not in adata.obsp:
        raise KeyError(f"adata.obsp['{conn_key}'] not found (connectivities_key from uns['neighbors']).")
    connectivities = adata.obsp[conn_key]
    if not isinstance(connectivities, csr_matrix):
        connectivities = csr_matrix(connectivities)

    min_nb_c = 1 if min_neighbors is None else int(min_neighbors)
    obs_names = np.asarray(adata.obs_names, dtype=object)
    out: Dict[str, Dict[str, float]] = {}
    for c_name in cluster_list:
        sel_indices = np.where(clusters == c_name)[0]
        if sel_indices.size == 0:
            continue
        per_cell = {}
        for idx in sel_indices:
            nbs = connectivities[idx].indices
            nodes = keep_type(clusters, nbs, c_name)
            nodes = nodes[nodes != idx]                 # a cell is trivially coherent with itself
            if nodes.size < min_nb_c:
                per_cell[obs_names[idx]] = np.nan
                continue
            v_i = v_emb[idx].reshape(1, -1)
            if not np.any(np.isfinite(v_i)) or np.allclose(v_i, 0):
                per_cell[obs_names[idx]] = np.nan
                continue
            sims = cosine_similarity(v_emb[nodes], v_i).flatten()
            per_cell[obs_names[idx]] = float(np.nanmean(sims))
        out[c_name] = per_cell
    return out


def _same_cluster_neighbors(adata, clusters, cluster_list, neighbor_graph="distances",
                            neighbors_key=None):
    """0/1 matrix of each selected cell's same-cluster neighbours (self excluded).

    Built once per (method, dataset) and reused for every gene set, because the
    neighbourhoods do not depend on the genes.

    neighbor_graph  distances       the kNN index matrix velocity confidence uses
                                    (scVelo `get_neighs` -> `get_indices`).
                    connectivities  the symmetrised graph CBDir uses.
    neighbors_key   an obsp key to read instead of the default graph (set when
                    ICCoh recomputes its own graph, `iccoh_recompute_graph`).

    Returns (S, counts, sel, info): S is n x n csr, counts the row sums, sel the
    indices of the cells in `cluster_list`.
    """
    n = adata.n_obs
    clusters = np.asarray(clusters, dtype=object)
    wanted = set(map(str, cluster_list))
    sel = np.array([i for i, c in enumerate(clusters)
                    if isinstance(c, str) and c in wanted], dtype=np.int64)
    graph = str(neighbor_graph).lower()
    if graph.startswith("d"):
        idx_mat = (vcs._neighbor_indices(adata, neighbors_key=neighbors_key)
                   if neighbors_key else vcs._neighbor_indices(adata))
        idx_mat = np.asarray(idx_mat)
        k = idx_mat.shape[1]
        rows = np.repeat(sel, k)
        cols = idx_mat[sel].ravel() if sel.size else np.array([], dtype=np.int64)
        graph, k_info = "distances", int(k)
    elif graph.startswith("c"):
        if neighbors_key:
            conn_key = neighbors_key
        else:
            if "neighbors" not in adata.uns:
                raise KeyError("adata.uns['neighbors'] not found; the h5ad carries no neighbour graph.")
            conn_key = adata.uns["neighbors"].get("connectivities_key", "connectivities")
        conn = adata.obsp[conn_key]
        conn = conn if isinstance(conn, csr_matrix) else csr_matrix(conn)
        sub = conn[sel]
        rows = sel[np.repeat(np.arange(sel.size), np.diff(sub.indptr))]
        cols = sub.indices
        graph, k_info = "connectivities", np.nan
    else:
        raise ValueError(f"iccoh_neighbor_graph must be 'distances' or 'connectivities'; "
                         f"got {neighbor_graph!r}")
    keep = (cols != rows) & (clusters[cols] == clusters[rows]) if rows.size else \
        np.zeros(0, dtype=bool)
    S = csr_matrix((np.ones(int(keep.sum())), (rows[keep], cols[keep])), shape=(n, n))
    S.sum_duplicates()
    S.data[:] = 1.0
    counts = np.diff(S.indptr)
    return S, counts, sel, dict(neighbor_graph=graph, n_neighbors=k_info)


def _prepare_unit(adata, vkey, use_genes, version):
    """Unit-norm rows of the confidence-prepared velocity; the confidence code path."""
    if version not in _CONFIDENCE_VERSIONS:
        raise ValueError(f"iccoh_confidence_version must be one of "
                         f"{list(_CONFIDENCE_VERSIONS)}; got {version!r}")
    if vkey not in adata.layers:
        raise KeyError(f"velocity layer '{vkey}' not in adata.layers (have: {list(adata.layers)})")
    W, norms, _length, n_used = vcs._prepare(
        adata, vkey, use_genes, **_CONFIDENCE_VERSIONS[version])
    degenerate = ~(norms > 0)
    What = W / np.where(degenerate, 1.0, norms)[:, None]
    return What, degenerate, int(n_used)


def _iccoh_values(What, degenerate, S, counts, sel, min_neighbors=1):
    """ICCoh_i = mean over same-cluster neighbours j of <What_j, What_i>, for i in sel.

    A zero-vector neighbour has a zero row and contributes a cosine of 0, as in
    the confidence metric; a zero-vector cell, or one with fewer than
    `min_neighbors` neighbours, is NaN.
    """
    if sel.size == 0:
        return np.zeros(0)
    SW = S[sel] @ What
    num = np.einsum("ij,ij->i", SW, What[sel])
    c = counts[sel].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / c
    min_nb = 1 if min_neighbors is None else int(min_neighbors)
    r[(c < max(min_nb, 1)) | degenerate[sel]] = np.nan
    return r


def in_cluster_consistency_gene_percell(
    adata,
    clusters: np.ndarray,
    cluster_list: List[str],
    vkey: str = "velocity",
    use_genes="all",
    version: str = "signed_4throot_scvelo",
    neighbor_graph: str = "distances",
    min_neighbors: int = 1,
    neighbors_key: Optional[str] = None,
    nbr=None,
) -> Tuple[Dict[str, Dict[str, float]], dict]:
    """Per-cell in-cluster velocity consistency in the original gene space.

    Velocity confidence as compute_velocity_confidence_run.py computes it, with
    the neighbours restricted to the cell's own cluster:

        W = velocity_confidence_scaled._prepare(layers[vkey], use_genes, version)
            # NaN genes dropped, transform, per-gene scale, per-cell centring
        ICCoh_i = mean_{j in N(i), cluster(j) == cluster(i), j != i} cos(W_j, W_i)

    Everything except the cluster restriction is the confidence metric's own
    code, so on a single cluster covering every cell this reproduces the
    `{version}` confidence column exactly. Unlike the "embedding" space it never
    touches `velocity_pca`, which is built from the velocity graph's transition
    probabilities and so smooths the field before coherence is measured.

    `neighbor_graph` / `neighbors_key`: see `_same_cluster_neighbors`. `nbr` is
    that function's output, passed in to reuse one neighbour matrix across gene
    sets. Values are left unclipped, like the `confidence_report` columns.

    Returns ({cluster: {barcode: value}}, info) with info = {n_genes, n_neighbors,
    neighbor_graph, version}.
    """
    What, degenerate, n_used = _prepare_unit(adata, vkey, use_genes, version)
    if nbr is None:
        nbr = _same_cluster_neighbors(adata, clusters, cluster_list, neighbor_graph,
                                      neighbors_key)
    S, counts, sel, ninfo = nbr
    r = _iccoh_values(What, degenerate, S, counts, sel, min_neighbors)
    clusters = np.asarray(clusters, dtype=object)
    obs_names = np.asarray(adata.obs_names, dtype=object)
    out: Dict[str, Dict[str, float]] = {}
    for c_name in cluster_list:
        m = clusters[sel] == c_name
        if m.any():
            out[c_name] = dict(zip(obs_names[sel[m]], map(float, r[m])))
    info = dict(n_genes=n_used, version=version, **ninfo)
    return out, info


def valid_velocity_gene_symbols(adata_path, vkey, gene_names_col=None):
    """Gene symbols whose velocity is finite in every cell (None if unreadable).

    Same rule as compute_velocity_confidence_run._get_valid_velocity_genes.
    """
    if not os.path.exists(adata_path):
        return None
    try:
        a = ad.read_h5ad(adata_path)
        if vkey not in a.layers:
            return None
        V = a.layers[vkey]
        V = V.toarray() if issparse(V) else V
        ok = ~np.isnan(np.asarray(V, dtype=np.float64)).any(axis=0)
        if gene_names_col and gene_names_col in a.var.columns:
            names = a.var[gene_names_col].astype(str).to_numpy()[ok]
        else:
            names = np.asarray(a.var_names.astype(str))[ok]
        del a
        gc.collect()
        return set(names)
    except Exception as e:
        _log(f"WARNING: failed to read valid velocity genes from {adata_path}: {e}")
        return None


def iccoh_gene_sets(adata, vkey, use_genes="all", intersected_genes=None,
                    gene_names_col=None, gene_query=None, apply_gene_query_always=False):
    """Boolean gene masks {'own': ..., 'shared': ... or None}.

    A line-for-line port of the gene selection in
    compute_velocity_confidence_run.compute_confidence_for_dataset, so the two
    pipelines pick the same genes from the same config:

      own     finite-velocity genes, narrowed by `use_genes` ('all', a list of
              names — matched on var[gene_names_col] when given — or a boolean
              var column). This is the set used with intersection_mode off.
      shared  finite-velocity genes whose symbol is in the dataset's
              cross-method intersection; `use_genes` is ignored, as there.
      gene_query (a pandas query on adata.var) narrows BOTH, but only when
              `apply_gene_query_always` is true — exactly as in the confidence
              pipeline, where the query otherwise reaches only the velocity graph,
              which confidence never reads.
    """
    V = adata.layers[vkey]
    V = V.toarray() if issparse(V) else V
    finite = ~np.isnan(np.asarray(V, dtype=np.float64)).any(axis=0)
    has_col = bool(gene_names_col) and gene_names_col in adata.var.columns
    symbols = (adata.var[gene_names_col].astype(str) if has_col
               else pd.Series(adata.var_names.astype(str), index=adata.var_names))

    if isinstance(use_genes, str) and use_genes == "all":
        own = finite.copy()
    elif isinstance(use_genes, (list, tuple, set, np.ndarray)) and \
            np.asarray(list(use_genes)).dtype != bool:
        own = np.asarray(symbols.isin(set(map(str, use_genes))), dtype=bool) & finite
    elif isinstance(use_genes, np.ndarray) and use_genes.dtype == bool:
        own = use_genes & finite
    elif isinstance(use_genes, str) and use_genes in adata.var.columns:
        own = np.asarray(adata.var[use_genes], dtype=bool) & finite
    else:
        own = finite.copy()

    shared = None
    if intersected_genes is not None:
        shared = np.asarray(symbols.isin(set(intersected_genes)), dtype=bool) & finite

    if gene_query and apply_gene_query_always:
        try:
            q = np.asarray(adata.var_names.isin(adata.var.query(gene_query).index), dtype=bool)
            n0 = int(own.sum())
            own = own & q
            if shared is not None:
                shared = shared & q
            _log(f"  apply_gene_query_always: gene_query '{gene_query}' -> own genes "
                 f"{n0} -> {int(own.sum())}"
                 + (f", shared genes {int(shared.sum())}" if shared is not None else ""))
        except Exception as e:
            _log(f"  WARNING: failed to apply gene_query '{gene_query}' on adata.var: {e}")
    elif gene_query:
        _log(f"  NOTE: gene_query '{gene_query}' is set but apply_gene_query_always is "
             f"false, so (as in compute_velocity_confidence_run.py) it does not narrow "
             f"the ICCoh genes. CBDir's velocity graph is never touched by it here.")
    return {"own": own, "shared": shared}


def gene_set_dimensionality(adata, vkey, mask, root_power=2):
    """(participation ratio, median effective genes per cell) on one gene set.

    The functions are imported from compute_velocity_confidence_run.py, so
    `root_power` means there exactly what it means here: the variance-
    stabilising sign(v)|v|^(1/root_power) applied before both statistics. It
    does not enter ICCoh or velocity confidence themselves.
    """
    try:
        try:
            from compute_velocity_confidence_run import (compute_participation_ratio,
                                                         compute_eff_genes)
        except ImportError:
            sys.path.append(os.path.dirname(os.path.abspath(__file__)))
            from compute_velocity_confidence_run import (compute_participation_ratio,
                                                         compute_eff_genes)
        V = adata.layers[vkey][:, np.asarray(mask, dtype=bool)]
        return (float(compute_participation_ratio(V, root_power=root_power)),
                float(compute_eff_genes(V, root_power=root_power)))
    except Exception as e:
        _log(f"  WARNING: participation ratio / effective genes failed: "
             f"{type(e).__name__}: {e}")
        return np.nan, np.nan


def _subsample_sizes(spec, n_own, n_min=None):
    """Resolve `iccoh_gene_subsample` into [(kind, k)], k < n_own and >= 10.

    Entries: a float in (0, 1) is a fraction of the method's own genes, an int
    > 1 an absolute count, and "min" the smallest own-gene count among the
    dataset's methods (count-matched ICCoh).
    """
    if not spec:
        return []
    out = []
    for s in (spec if isinstance(spec, (list, tuple)) else [spec]):
        if isinstance(s, str):
            if s.strip().lower() != "min":
                raise ValueError(f"iccoh_gene_subsample entry {s!r}: use a fraction, "
                                 f"a count, or 'min'")
            if n_min is None or not np.isfinite(n_min):
                continue
            kind, k = "min", int(n_min)
        elif isinstance(s, float) and 0 < s < 1:
            kind, k = f"frac{s:g}", int(round(s * n_own))
        elif float(s) > 1 and float(s) == int(s):
            kind, k = f"n{int(s)}", int(s)
        else:
            raise ValueError(f"iccoh_gene_subsample entry {s!r} is neither a fraction "
                             f"in (0, 1), a count > 1, nor 'min'")
        if 10 <= k < n_own and (kind, k) not in out:
            out.append((kind, k))
    return out


def _cluster_summaries(values, clusters_sel, **meta):
    """Per-cluster median / mean / n of one per-cell ICCoh vector."""
    rows = []
    df = pd.DataFrame({"source": clusters_sel, "v": values})
    for c, g in df.groupby("source", sort=True):
        v = g["v"].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        rows.append(dict(meta, source=c, n_cells=int(v.size),
                         median_iccoh=float(np.median(v)) if v.size else np.nan,
                         mean_iccoh=float(np.mean(v)) if v.size else np.nan))
    return rows


def _recompute_iccoh_graph(adata, rep="X_pca", n_neighbors=30, neighbor_graph="distances"):
    """ICCoh's own kNN, as compute_velocity_confidence_run.py builds it with
    recompute_graph: sc.pp.neighbors(n_pcs=all of rep, use_rep=rep). Written under
    key_added='iccoh', so the graph CBDir reads is left exactly as it was.
    Returns the obsp key to read.
    """
    if rep not in adata.obsm:
        if rep == "X_pca":
            _log("  iccoh_recompute_graph: X_pca not in obsm; running PCA first ...")
            sc.pp.pca(adata)
        else:
            raise KeyError(f"iccoh_rep '{rep}' not found in adata.obsm")
    n_pcs = adata.obsm[rep].shape[1]
    _log(f"  iccoh_recompute_graph: sc.pp.neighbors(n_neighbors={n_neighbors}, "
         f"use_rep='{rep}', n_pcs={n_pcs}, key_added='iccoh') ...")
    sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors, use_rep=rep,
                    key_added="iccoh")
    return ("iccoh_distances" if str(neighbor_graph).lower().startswith("d")
            else "iccoh_connectivities")


def compute_gene_confidence_iccoh(adata, clusters, cluster_list, vkey, sets,
                                  primary="shared", version="signed_4throot_scvelo",
                                  neighbor_graph="distances", neighbors_key=None,
                                  min_neighbors=1, contrast=True, subsample=None,
                                  subsample_reps=3, subsample_seed=0, n_min=None,
                                  dimensionality=True, root_power=2,
                                  dataset="", method=""):
    """Every gene-space ICCoh quantity for one (method, dataset), on one graph.

    Returns a dict:
      per_cell   {gene_set: {barcode: value}} for 'own' / 'shared' (as computed)
      meta       gene counts, PR / effective genes, graph info, primary set
      clusters   per-cluster summaries for every gene set and every subsample
                 draw (the table the gene-count bias analysis reads)
    """
    nbr = _same_cluster_neighbors(adata, clusters, cluster_list, neighbor_graph,
                                  neighbors_key)
    S, counts, sel, ninfo = nbr
    cl_sel = np.asarray(clusters, dtype=object)[sel]
    bcs = np.asarray(adata.obs_names, dtype=object)[sel]
    n_own = int(sets["own"].sum())
    n_shared = int(sets["shared"].sum()) if sets["shared"] is not None else np.nan
    if primary == "shared" and sets["shared"] is None:
        primary = "own"
    todo = [primary] + ([k for k in ("own", "shared")
                         if k != primary and sets[k] is not None] if contrast else [])

    per_cell, rows = {}, []
    base = dict(dataset=dataset, method=method, version=version,
                neighbor_graph=ninfo["neighbor_graph"], primary=primary,
                n_genes_own=n_own, n_genes_shared=n_shared)
    for key in todo:
        n_key = int(sets[key].sum())
        if n_key < 10:
            msg = (f"the {key} gene set has {n_key} gene(s) (< 10). If that is the shared "
                   f"set, the methods' var_names probably differ (Ensembl IDs vs symbols, "
                   f"case, version suffixes): set gene_names_col, or check the per-method "
                   f"counts in the 'Shared velocity genes' block of the log")
            if key == primary:
                raise ValueError(msg)
            _log(f"    NOTE: skipping the {key}-gene contrast: {msg}")
            continue
        try:
            What, deg, n_used = _prepare_unit(adata, vkey, sets[key], version)
        except Exception as e:
            if key == primary:
                raise
            # a failing CONTRAST set must never cost the primary score
            _log(f"    WARNING: {key}-gene contrast failed ({type(e).__name__}: {e}); "
                 f"primary ICCoh kept")
            continue
        r = _iccoh_values(What, deg, S, counts, sel, min_neighbors)
        per_cell[key] = dict(zip(bcs, map(float, r)))
        rows += _cluster_summaries(r, cl_sel, **base, gene_set=key, kind="full",
                                   k=n_used, rep=0)
        del What

    draws = _subsample_sizes(subsample, n_own, n_min)
    if draws:
        own_idx = np.where(sets["own"])[0]
        import zlib
        seed = [int(subsample_seed), zlib.crc32(str(dataset).encode()),
                zlib.crc32(str(method).encode())]
        rng = np.random.default_rng(seed)
        for kind, k in draws:
            for rep in range(int(subsample_reps or 1)):
                m = np.zeros(adata.n_vars, dtype=bool)
                m[rng.choice(own_idx, size=k, replace=False)] = True
                What, deg, n_used = _prepare_unit(adata, vkey, m, version)
                r = _iccoh_values(What, deg, S, counts, sel, min_neighbors)
                rows += _cluster_summaries(r, cl_sel, **base, gene_set="own_subsample",
                                           kind=kind, k=n_used, rep=rep)
                del What
        _log(f"    gene subsampling: {len(draws)} size(s) x {int(subsample_reps or 1)} "
             f"draw(s) from {n_own} own genes: {[k for _, k in draws]}")

    meta = dict(n_genes=(n_own if primary == "own" else n_shared),
                n_genes_own=n_own, n_genes_shared=n_shared, primary=primary, **ninfo)
    if dimensionality:
        meta["pr_own"], meta["eff_genes_own"] = gene_set_dimensionality(
            adata, vkey, sets["own"], root_power)
        if sets["shared"] is not None and int(sets["shared"].sum()) >= 10:
            meta["pr_shared"], meta["eff_genes_shared"] = gene_set_dimensionality(
                adata, vkey, sets["shared"], root_power)
    return dict(per_cell=per_cell, meta=meta, clusters=pd.DataFrame(rows))


def smooth_velocity_field(adata, vkey="velocity", rounds=0, lam=0.5, space="gene",
                          graph="recompute", k=60, emb_key="X_pca", n_pcs=None,
                          v_emb_key=None):
    """Average each cell's velocity with its neighbours, `rounds` times.

    The experiment this exists for: hold the manifold, the neighbour graph, the
    cluster labels and the underlying velocity estimates fixed, and vary ONLY the
    spatial smoothness of the field. Coherence rises by construction; whether
    CBDir improves, and what happens to its spread across transitions, is the
    question. It is the causal version of the coherence correlation — smoothness
    is manipulated rather than observed.

    One update is  V <- (1 - lam) V + lam * mean(V over neighbours).

    `graph` decides which neighbours do the averaging, and the choice is not a
    detail:

      "recompute" (default) builds a FRESH kNN with `k` neighbours on `emb_key`.
      "metric" reuses the stored graph — the very graph CBDir scores against.
      Smoothing over the scoring neighbourhood drags each velocity toward the
      local manifold flow, which is correlated with the displacement vectors
      CBDir measures, so CBDir can rise for a purely mechanical reason. Use
      "metric" only to demonstrate that artefact, never to argue from it.

    `space`:
      "gene"      smooths layers[vkey], so the velocity graph and the embedding
                  are rebuilt from the smoothed field — the faithful analogue of
                  a method that produces a smoother field. Requires the caller to
                  recompute the graph (compute_cbdir_for_dataset forces it).
      "embedding" smooths obsm[v_emb_key] directly. Cheap, but it bypasses the
                  graph, so it answers a narrower question.
    """
    rounds = int(rounds or 0)
    if rounds <= 0:
        return adata, {}
    lam = float(lam)
    space = str(space).lower()

    if str(graph).lower().startswith("m"):
        if "neighbors" not in adata.uns:
            raise KeyError("smooth_graph='metric' needs adata.uns['neighbors'].")
        conn_key = adata.uns["neighbors"].get("connectivities_key", "connectivities")
        A = adata.obsp[conn_key]
        A = A if isinstance(A, csr_matrix) else csr_matrix(A)
        A = (A > 0).astype(np.float64)
        used = f"stored graph ({conn_key})"
    else:
        from sklearn.neighbors import NearestNeighbors
        if emb_key not in adata.obsm:
            raise KeyError(f"smooth_graph='recompute' needs obsm['{emb_key}'].")
        X = np.asarray(adata.obsm[emb_key], dtype=np.float64)
        if n_pcs:
            X = X[:, :int(n_pcs)]
        kk = int(k)
        nn = NearestNeighbors(n_neighbors=min(kk + 1, X.shape[0])).fit(X)
        _, idx = nn.kneighbors(X)
        idx = idx[:, 1:]
        n = X.shape[0]
        A = csr_matrix((np.ones(idx.size), (np.repeat(np.arange(n), idx.shape[1]),
                                            idx.ravel())), shape=(n, n))
        used = f"fresh kNN (k={kk}) on {emb_key}"

    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0

    if space.startswith("e"):
        key = v_emb_key or f"{vkey}_pca"
        if key not in adata.obsm:
            raise KeyError(f"smooth space='embedding' needs obsm['{key}'].")
        M = np.asarray(adata.obsm[key], dtype=np.float64)
        for _ in range(rounds):
            M = (1 - lam) * M + lam * (A @ M) / deg[:, None]
        adata.obsm[key] = M
        target = f"obsm['{key}']"
    else:
        if vkey not in adata.layers:
            raise KeyError(f"velocity layer '{vkey}' not in adata.layers.")
        V = np.asarray(adata.layers[vkey], dtype=np.float64)
        # Velocity genes only: a NaN column must stay NaN, or the smoothing
        # would quietly invent velocities for genes the model never fitted.
        # A column is smoothable only if it is finite for EVERY cell: one NaN in
        # the column would spread to k neighbours on the first round.
        finite = np.isfinite(V).all(axis=0)
        sub = V[:, finite]
        for _ in range(rounds):
            sub = (1 - lam) * sub + lam * (A @ sub) / deg[:, None]
        V[:, finite] = sub
        adata.layers[vkey] = V
        target = f"layers['{vkey}'] ({int(finite.sum())} velocity genes)"

    info = dict(smooth_rounds=rounds, smooth_lambda=lam, smooth_space=space,
                smooth_graph=("metric" if str(graph).lower().startswith("m")
                              else "recompute"),
                smooth_k=(np.nan if str(graph).lower().startswith("m") else int(k)))
    _log(f"  Smoothed {target}: {rounds} round(s), lambda={lam}, over the {used}")
    return adata, info


def cross_boundary_direction_percell(
    adata,
    clusters: np.ndarray,
    cluster_edges: List[Tuple[str, str]],
    x_emb_key: str = "X_pca",
    v_emb_key: str = "velocity_pca",
    n_pcs_metric: Optional[int] = None,
    include_skipped: bool = False,
    min_target_neighbors: int = 3,
    edge_stats: Optional[dict] = None,
) -> Dict[Tuple[str, str], List[dict]]:
    """Per-cell Cross-Boundary Direction Correctness score (A -> B).

    Identical in substance to `cross_boundary_correctness` in compute_cbdir.py,
    with two changes: cell identities are retained, and the embedding keys are
    explicit rather than inferred by prefix matching over adata.obsm.

    `min_target_neighbors` is the one place this pipeline departs from the
    published reference implementation, so it is a parameter rather than a
    hard-coded rule. The 2026 benchmark's `cbdir.py` scores every source cell
    with at least ONE neighbour in the target cluster (`len(nodes) == 0` skips);
    the default of 3 here reproduces the `<= 2` skip inherited from the snippet
    this pipeline was built from. Set it to 1 to match the published metric
    exactly. It is not a cosmetic choice: a cell with a single target neighbour
    contributes the cosine to one displacement vector rather than a mean over
    several, so the excluded cells are the noisiest ones and dropping them
    trims the tails of the per-transition distribution.

    Returns
    -------
    dict
        Keyed by (source, target); each value is a list of records
        {cell_barcode, cbdir, n_target_neighbors, resultant_length, alignment}.
        When `edge_stats` is given it is filled in place with per-edge quantities
        that cannot be recovered from per-cell scalars — currently the coherent
        ceiling ||mean_i m_i||, which needs the resultant VECTORS.
    """
    if x_emb_key not in adata.obsm:
        raise KeyError(f"x_emb_key '{x_emb_key}' not found in adata.obsm (have: {list(adata.obsm)})")
    if v_emb_key not in adata.obsm:
        raise KeyError(f"v_emb_key '{v_emb_key}' not found in adata.obsm (have: {list(adata.obsm)})")

    x_emb = np.asarray(adata.obsm[x_emb_key], dtype=np.float64)
    v_emb = np.asarray(adata.obsm[v_emb_key], dtype=np.float64)

    if x_emb.shape != v_emb.shape:
        raise ValueError(
            f"Embedding shape mismatch: {x_emb_key}{x_emb.shape} vs {v_emb_key}{v_emb.shape}"
        )

    if n_pcs_metric is not None:
        n = int(n_pcs_metric)
        if n < 2:
            raise ValueError("n_pcs_metric must be >= 2")
        if n > x_emb.shape[1]:
            _log(f"  WARNING: n_pcs_metric={n} exceeds available dimensions ({x_emb.shape[1]}); using all.")
        else:
            x_emb = x_emb[:, :n]
            v_emb = v_emb[:, :n]

    # Neighbour graph exactly as stored in the h5ad.
    if "neighbors" not in adata.uns:
        raise KeyError("adata.uns['neighbors'] not found; the h5ad carries no neighbour graph.")
    conn_key = adata.uns["neighbors"].get("connectivities_key", "connectivities")
    if conn_key not in adata.obsp:
        raise KeyError(f"adata.obsp['{conn_key}'] not found (connectivities_key from uns['neighbors']).")
    connectivities = adata.obsp[conn_key]
    if not isinstance(connectivities, csr_matrix):
        connectivities = csr_matrix(connectivities)

    def get_neighbors(idx):
        return connectivities[idx].indices

    # An explicit `null` in the YAML arrives as None and must fall back to the
    # documented default rather than raising inside the loop.
    min_nb = 3 if min_target_neighbors is None else int(min_target_neighbors)

    obs_names = np.asarray(adata.obs_names, dtype=object)
    results: Dict[Tuple[str, str], List[dict]] = {}

    for u, v in cluster_edges:
        sel = clusters == u
        sel_indices = np.where(sel)[0]
        if sel_indices.size == 0:
            _log(f"  WARNING: source cluster '{u}' has no cells; edge ({u} -> {v}) ignored.")
            continue

        records = []
        n_scored = 0
        resultants = []                       # the m_i vectors, for the coherent ceiling
        for idx in sel_indices:
            nbs = get_neighbors(idx)
            nodes = keep_type(clusters, nbs, v)
            n_nb = int(len(nodes))
            if n_nb < min_nb:
                if include_skipped:
                    records.append({
                        "cell_barcode": obs_names[idx],
                        "cbdir": np.nan,
                        "n_target_neighbors": n_nb,
                        "resultant_length": np.nan,
                        "alignment": np.nan,
                    })
                continue
            position_dif = x_emb[nodes] - x_emb[idx]
            dir_scores = cosine_similarity(position_dif, v_emb[idx].reshape(1, -1)).flatten()
            cb = float(np.nanmean(dir_scores))

            # CBDir_i = <v_hat, m_i> with m_i the mean UNIT displacement, so
            # |CBDir_i| <= ||m_i|| =: Rbar_i, attained when the velocity points
            # along the resultant. Rbar is the ceiling the geometry allows; the
            # alignment CBDir/Rbar = cos(angle to that best direction) is the part
            # the method is answerable for. Zero-length displacements (duplicate
            # coordinates) carry no direction and are dropped from m_i; they are
            # counted so the drop is visible rather than silent.
            norms = np.linalg.norm(position_dif, axis=1)
            good = norms > 0
            r_bar, align = np.nan, np.nan
            if good.any():
                m_i = (position_dif[good] / norms[good, None]).mean(axis=0)
                r_bar = float(np.linalg.norm(m_i))
                resultants.append(m_i)
                if r_bar > 0:
                    align = float(cb / r_bar)
            records.append({
                "cell_barcode": obs_names[idx],
                "cbdir": cb,
                "n_target_neighbors": n_nb,
                "resultant_length": r_bar,
                "alignment": align,
                **({"n_zero_length_displacements": int((~good).sum())}
                   if not good.all() else {}),
            })
            n_scored += 1

        # A single shared direction cannot beat ||mean_i m_i||, which is at most
        # the mean of the per-cell ceilings (triangle inequality). The gap is what
        # a perfectly coherent field would have to give up at this boundary.
        if edge_stats is not None and resultants:
            M = np.mean(np.asarray(resultants, dtype=np.float64), axis=0)
            edge_stats[(u, v)] = {
                "ceiling_coherent": float(np.linalg.norm(M)),
                "n_cells_with_resultant": int(len(resultants)),
            }

        if n_scored == 0:
            _log(f"  WARNING: cell type transition pair ({u},{v}) does not exist in the KNN graph. Ignored.")
            if not include_skipped:
                continue
        results[(u, v)] = records

    return results


# ---------------------------------------------------------------------------
# Cluster labels and edges
# ---------------------------------------------------------------------------

def _parse_edges(raw_edges) -> List[Tuple[str, str]]:
    """Accept [[u, v], ...] or ['u -> v', ...] and normalise to [(u, v), ...]."""
    if raw_edges is None:
        return []
    edges = []
    for e in raw_edges:
        if isinstance(e, str):
            if "->" not in e:
                _log(f"  WARNING: cannot parse edge string '{e}' (expected 'u -> v'); skipping.")
                continue
            u, v = e.split("->", 1)
            edges.append((u.strip(), v.strip()))
        elif isinstance(e, (list, tuple)) and len(e) == 2:
            edges.append((str(e[0]), str(e[1])))
        else:
            _log(f"  WARNING: cannot parse edge entry {e!r}; skipping.")
    return edges


def apply_cluster_map(
    adata,
    k_cluster: str,
    cluster_map: Optional[dict],
    drop_unmapped: bool = False,
) -> Tuple[str, np.ndarray, Dict[str, List[str]]]:
    """Apply an optional level -> level mapping to the cluster column.

    Writes the result to a working column `f"{k_cluster}{_MAPPED_SUFFIX}"` and
    leaves the original column untouched. Returns the working column name, the
    label array, and a {mapped_label: [raw_labels]} reverse index.
    """
    if k_cluster not in adata.obs.columns:
        raise KeyError(f"k_cluster '{k_cluster}' not found in adata.obs (have: {list(adata.obs.columns)})")

    raw = adata.obs[k_cluster].astype(str).values
    raw_levels = sorted(pd.unique(raw).tolist())

    if not cluster_map:
        _log(f"  Cluster column '{k_cluster}': {len(raw_levels)} levels, no cluster_map applied.")
        adata.obs[k_cluster + _MAPPED_SUFFIX] = raw
        return k_cluster + _MAPPED_SUFFIX, raw, {lvl: [lvl] for lvl in raw_levels}

    cmap = {str(k): (None if v is None else str(v)) for k, v in dict(cluster_map).items()}
    unknown_keys = [k for k in cmap if k not in set(raw_levels)]
    if unknown_keys:
        _log(f"  WARNING: cluster_map keys not present in '{k_cluster}': {unknown_keys}")

    unmapped_levels = [lvl for lvl in raw_levels if lvl not in cmap]
    if unmapped_levels:
        if drop_unmapped:
            _log(f"  cluster_map_drop_unmapped=True: {len(unmapped_levels)} level(s) set to NaN: {unmapped_levels}")
        else:
            _log(f"  cluster_map: {len(unmapped_levels)} level(s) kept as-is: {unmapped_levels}")

    def _map_one(lvl):
        if lvl in cmap:
            return cmap[lvl]
        return None if drop_unmapped else lvl

    mapped = np.array([_map_one(lvl) for lvl in raw], dtype=object)
    mapped = np.array([np.nan if (m is None) else m for m in mapped], dtype=object)

    reverse: Dict[str, List[str]] = defaultdict(list)
    for lvl in raw_levels:
        m = _map_one(lvl)
        if m is not None:
            reverse[m].append(lvl)

    new_levels = sorted([lv for lv in pd.unique(mapped).tolist() if isinstance(lv, str)])
    _log(f"  cluster_map applied to '{k_cluster}': {len(raw_levels)} raw level(s) -> {len(new_levels)} mapped level(s).")
    for lv in new_levels:
        n_cells = int(np.sum(mapped == lv))
        srcs = reverse.get(lv, [])
        note = f" <- {srcs}" if len(srcs) > 1 else ""
        _log(f"    '{lv}': {n_cells} cells{note}")
    n_nan = int(np.sum([not isinstance(m, str) for m in mapped]))
    if n_nan:
        _log(f"    (NaN / dropped: {n_nan} cells)")

    adata.obs[k_cluster + _MAPPED_SUFFIX] = pd.Categorical(
        [m if isinstance(m, str) else None for m in mapped]
    )
    return k_cluster + _MAPPED_SUFFIX, mapped, dict(reverse)


def _validate_edges(edges, clusters, reverse_map):
    """Drop edges whose endpoints are absent from the (mapped) cluster labels."""
    present = set([c for c in pd.unique(clusters).tolist() if isinstance(c, str)])
    kept = []
    for u, v in edges:
        missing = [lbl for lbl in (u, v) if lbl not in present]
        if missing:
            _log(f"  WARNING: edge ({u} -> {v}) references label(s) not present in the cluster column: {missing}; skipping.")
            _log(f"           available labels: {sorted(present)}")
            continue
        kept.append((u, v))
    return kept


# ---------------------------------------------------------------------------
# Embedding preparation
# ---------------------------------------------------------------------------

def _set_X_from_layers(adata, layer_names):
    """Assign the (sum of) named layer(s) to adata.X."""
    missing = [ln for ln in layer_names if ln not in adata.layers]
    if missing:
        raise KeyError(f"layer(s) {missing} not found in adata.layers (have: {list(adata.layers)})")
    acc = adata.layers[layer_names[0]].copy()
    for ln in layer_names[1:]:
        acc = acc + adata.layers[ln]
    adata.X = acc
    return adata


def _looks_log_transformed(X) -> bool:
    try:
        sub = X[: min(500, X.shape[0])]
        if issparse(sub):
            mx = sub.max() if sub.nnz else 0.0
        else:
            mx = np.nanmax(np.asarray(sub))
        return bool(mx) and float(mx) < 50.0
    except Exception:
        return False


def prepare_embedding(
    adata,
    x_source: str = "existing",
    x_layer: Optional[str] = None,
    x_layers: Optional[list] = None,
    emb_key: str = "X_pca",
    basis: str = "pca",
    n_comps: int = 30,
    target_sum=None,
    recompute_neighbors: bool = False,
    n_neighbors: int = 30,
) -> str:
    """Prepare adata.obsm[emb_key]. Returns the obsm key holding the embedding."""
    if x_source not in _VALID_X_SOURCES:
        raise ValueError(f"x_source must be one of {_VALID_X_SOURCES}, got '{x_source}'")

    if x_source == "existing":
        if emb_key not in adata.obsm:
            raise KeyError(
                f"x_source='existing' but obsm['{emb_key}'] not found (have: {list(adata.obsm)}). "
                f"Set x_source to 'layer', 'layer_sum' or 'X' to compute a PCA, or point "
                f"emb_key/basis at an embedding the h5ad actually carries."
            )
        _log(f"  x_source='existing': using stored obsm['{emb_key}'] "
             f"{adata.obsm[emb_key].shape} — no .X change, no PCA, graphs untouched.")
        return emb_key

    if basis != "pca":
        raise ValueError(
            f"x_source='{x_source}' recomputes a PCA, which only makes sense with basis='pca' "
            f"(got basis='{basis}'). Use x_source='existing' for a non-PCA basis."
        )

    _log("  " + "!" * 72)
    _log(f"  WARNING: x_source='{x_source}' recomputes the PCA, but the neighbour graph and "
         f"velocity graph are taken from the h5ad as-is. The graph was built on the ORIGINAL "
         f"embedding while CBDir geometry will be measured in the NEW one.")
    _log("  " + "!" * 72)

    if x_source == "layer":
        if not x_layer:
            raise ValueError("x_source='layer' requires 'x_layer'")
        _log(f"  Assigning layers['{x_layer}'] to .X ...")
        _set_X_from_layers(adata, [x_layer])
        do_norm = True
    elif x_source == "layer_sum":
        if not x_layers or len(x_layers) < 2:
            raise ValueError("x_source='layer_sum' requires 'x_layers' with at least two layer names")
        _log(f"  Assigning sum of layers {list(x_layers)} to .X ...")
        _set_X_from_layers(adata, list(x_layers))
        do_norm = True
    else:  # "X"
        _log("  x_source='X': using .X as-is (no normalization, no log).")
        do_norm = False

    if do_norm:
        if _looks_log_transformed(adata.X):
            _log("  WARNING: the assigned .X looks already log-transformed (max < 50); "
                 "normalize_total + log1p will be applied anyway as configured.")
        _log(f"  sc.pp.normalize_total(target_sum={target_sum}) ...")
        sc.pp.normalize_total(adata, target_sum=target_sum)
        _log("  sc.pp.log1p ...")
        sc.pp.log1p(adata)

    n_comps_eff = int(min(n_comps, min(adata.n_obs, adata.n_vars) - 1))
    if n_comps_eff != int(n_comps):
        _log(f"  WARNING: n_comps reduced from {n_comps} to {n_comps_eff} to fit the data shape.")
    _log(f"  sc.pp.pca(n_comps={n_comps_eff}, svd_solver='arpack') -> obsm['{emb_key}'] ...")
    sc.pp.pca(adata, n_comps=n_comps_eff, svd_solver="arpack")
    if emb_key != "X_pca":
        adata.obsm[emb_key] = adata.obsm["X_pca"]

    if recompute_neighbors:
        _log(f"  recompute_neighbors=True: sc.pp.neighbors(n_neighbors={n_neighbors}, use_rep='{emb_key}') ...")
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=emb_key)

    return emb_key


class _MsSwap:
    """Temporarily expose layers[xkey] as layers['Ms'] for scVelo calls.

    scVelo's velocity_graph / velocity_embedding read the expression layer under
    a fixed name. Any pre-existing 'Ms' layer is stashed and restored on exit; if
    there was none, the temporary 'Ms' is removed again. xkey='Ms' is a no-op,
    and xkey='X' swaps in adata.X.
    """

    def __init__(self, adata, xkey: str = "Ms"):
        self.adata = adata
        self.xkey = xkey
        self._stashed = None
        self._had_ms = False
        self._active = False

    def __enter__(self):
        a, xkey = self.adata, self.xkey
        if xkey in (None, "Ms"):
            _log(f"  xkey='{xkey}': layers['Ms'] used as-is (no swap).")
            return self
        if xkey == "X":
            source = a.X.copy() if hasattr(a.X, "copy") else a.X
            _log("  xkey='X': temporarily swapping .X into layers['Ms'] ...")
        elif xkey in a.layers:
            source = a.layers[xkey]
            _log(f"  xkey='{xkey}': temporarily swapping layers['{xkey}'] into layers['Ms'] ...")
        else:
            raise KeyError(f"xkey '{xkey}' not found in adata.layers (have: {list(a.layers)})")
        self._had_ms = "Ms" in a.layers
        if self._had_ms:
            self._stashed = a.layers["Ms"].copy()
        a.layers["Ms"] = source
        self._active = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self._active:
            return False
        a = self.adata
        if self._had_ms:
            a.layers["Ms"] = self._stashed
            _log("  Restored the original layers['Ms'].")
        elif "Ms" in a.layers:
            del a.layers["Ms"]
            _log("  Removed the temporary layers['Ms'] (none existed before).")
        self._stashed = None
        self._active = False
        return False


def compute_velocity_embedding(
    adata,
    vkey: str = "velocity",
    xkey: str = "Ms",
    basis: str = "pca",
    emb_key: str = "X_pca",
    reuse: str = "auto",
    x_source: str = "existing",
    recompute_velocity_graph: bool = False,
    approx: bool = False,
    sqrt_transform: bool = True,
    n_jobs=None,
    use_negative_cosines="auto",
) -> str:
    """Ensure obsm[f'{vkey}_{basis}'] exists via scv.tl.velocity_embedding."""
    v_emb_key = velocity_embedding_key(vkey, basis)
    x_basis_key = basis_embedding_key(basis)
    graph_key = velocity_graph_key(vkey)
    graph_neg_key = velocity_graph_neg_key(vkey)
    params_key = velocity_params_key(vkey)

    if isinstance(reuse, str):
        reuse_eff = (x_source == "existing" and not recompute_velocity_graph) \
            if reuse.lower() == "auto" else (reuse.lower() == "true")
    else:
        reuse_eff = bool(reuse)

    if reuse_eff and v_emb_key in adata.obsm:
        _log(f"  Reusing existing obsm['{v_emb_key}'] (reuse={reuse}, x_source='{x_source}').")
        return v_emb_key

    if "neighbors" not in adata.uns:
        raise KeyError("adata.uns['neighbors'] not found; velocity_embedding needs a neighbour graph.")
    if vkey not in adata.layers:
        raise KeyError(f"velocity layer '{vkey}' not found in adata.layers (have: {list(adata.layers)})")

    # The embedding scVelo reads is named X_{basis}; expose emb_key under it.
    restore_x_basis = None
    had_x_basis = x_basis_key in adata.obsm
    if emb_key != x_basis_key:
        restore_x_basis = adata.obsm[x_basis_key].copy() if had_x_basis else None
        adata.obsm[x_basis_key] = adata.obsm[emb_key]
        _log(f"  Exposing obsm['{emb_key}'] as obsm['{x_basis_key}'] for the scVelo call.")

    try:
        with _MsSwap(adata, xkey):
            if recompute_velocity_graph:
                _log(f"  recompute_velocity_graph=True: scv.tl.velocity_graph(xkey='{xkey}', "
                     f"vkey='{vkey}', approx={approx}, sqrt_transform={sqrt_transform}) "
                     f"-> '{graph_key}' ...")
                scv.tl.velocity_graph(
                    adata, xkey="Ms" if xkey not in (None, "Ms") else xkey,
                    vkey=vkey, approx=approx, sqrt_transform=sqrt_transform,
                    n_jobs=n_jobs,
                )
            elif not _has_key(adata, graph_key):
                raise KeyError(
                    f"velocity graph '{graph_key}' not found in adata.uns or adata.obsp. "
                    f"The h5ad is expected to already carry the velocity graph for vkey='{vkey}'. "
                    f"Set recompute_velocity_graph: true to build it here."
                )

            if params_key in adata.uns:
                _log(f"  uns['{params_key}']: {adata.uns[params_key]}")

            has_neg = _has_key(adata, graph_neg_key)
            if isinstance(use_negative_cosines, str) and use_negative_cosines.lower() == "auto":
                neg = bool(has_neg)
                if not has_neg:
                    _log(f"  WARNING: '{graph_neg_key}' not present; calling velocity_embedding "
                         f"with use_negative_cosines=False.")
            else:
                neg = bool(use_negative_cosines) if not isinstance(use_negative_cosines, str) \
                    else (use_negative_cosines.lower() == "true")
                if neg and not has_neg:
                    raise KeyError(
                        f"use_negative_cosines is on but '{graph_neg_key}' is not in adata.uns/.obsp."
                    )

            _log(f"  scv.tl.velocity_embedding(basis='{basis}', vkey='{vkey}', "
                 f"use_negative_cosines={neg}) -> obsm['{v_emb_key}'] ...")
            scv.tl.velocity_embedding(adata, basis=basis, vkey=vkey, use_negative_cosines=neg)
    finally:
        if emb_key != x_basis_key:
            if restore_x_basis is not None:
                adata.obsm[x_basis_key] = restore_x_basis
            elif not had_x_basis and x_basis_key in adata.obsm:
                del adata.obsm[x_basis_key]

    if v_emb_key not in adata.obsm:
        raise KeyError(f"velocity_embedding did not produce obsm['{v_emb_key}'].")
    return v_emb_key


# ---------------------------------------------------------------------------
# Per (method, dataset) driver
# ---------------------------------------------------------------------------

def compute_cbdir_for_dataset(
    name,
    adata_path,
    method_name,
    cluster_edges=None,
    k_cluster="clusters",
    vkey="velocity",
    xkey="Ms",
    basis="pca",
    emb_key=None,
    x_source="existing",
    x_layer=None,
    x_layers=None,
    n_comps=30,
    n_pcs_metric=None,
    target_sum=None,
    recompute_neighbors=False,
    n_neighbors=30,
    recompute_velocity_graph=False,
    approx=False,
    sqrt_transform=True,
    n_jobs=None,
    use_negative_cosines="auto",
    reuse_velocity_embedding="auto",
    include_skipped=False,
    min_target_neighbors=3,
    smooth_velocity_rounds=0,
    smooth_velocity_lambda=0.5,
    smooth_velocity_space="gene",
    smooth_graph="recompute",
    smooth_k=60,
    compute_iccoh=True,
    iccoh_min_neighbors=1,
    iccoh_space="gene_confidence",
    iccoh_clusters="source",
    iccoh_gene_raw=True,
    iccoh_gene_vst=True,
    iccoh_gene_power=0.25,
    iccoh_confidence_version="signed_4throot_scvelo",
    iccoh_neighbor_graph="distances",
    iccoh_use_genes="all",
    iccoh_intersection_mode=True,
    iccoh_embedding=True,
    gene_names_col=None,
    gene_query=None,
    apply_gene_query_always=False,
    root_power=2,
    iccoh_recompute_graph=False,
    iccoh_n_neighbors=None,
    iccoh_rep="X_pca",
    iccoh_gene_set_contrast=True,
    iccoh_dimensionality=True,
    iccoh_gene_subsample=(0.25, 0.5, "min"),
    iccoh_gene_subsample_reps=3,
    iccoh_gene_subsample_seed=0,
    cluster_map=None,
    cluster_map_drop_unmapped=False,
    intersected_genes=None,
    dataset_min_genes=None,
):
    """Load one method's AnnData for one dataset and return tidy CBDir records.

    `intersected_genes` is the dataset's shared velocity-gene set and
    `dataset_min_genes` the smallest per-method count of finite-velocity genes,
    both computed by run_from_config across methods. They are used only by
    gene_confidence ICCoh: the shared set is the primary gene set when
    `iccoh_intersection_mode` is on and the contrast set otherwise.

    Returns (long_df, n_source_total). For gene_confidence ICCoh, long_df.attrs
    carries 'iccoh_genes' (one row: gene counts and dimensionality for this
    method x dataset) and 'iccoh_gene_clusters' (per-cluster ICCoh for every gene
    set and subsample draw); save_aggregated_results writes both out.
    """
    iccoh_space = normalize_iccoh_space(iccoh_space)
    _log(f"Loading AnnData from {adata_path} ...")
    if not os.path.exists(adata_path):
        raise FileNotFoundError(f"adata_path not found: {adata_path}")

    adata = ad.read_h5ad(adata_path)
    _log(f"AnnData loaded. Shape: {adata.shape}")

    if emb_key is None:
        emb_key = basis_embedding_key(basis)
    _log(f"  Keys for this method: vkey='{vkey}', xkey='{xkey}', basis='{basis}', "
         f"embedding='{emb_key}', velocity graph='{velocity_graph_key(vkey)}', "
         f"velocity embedding='{velocity_embedding_key(vkey, basis)}'")

    try:
        # 1. Cluster labels (optionally remapped)
        _, clusters, reverse_map = apply_cluster_map(
            adata, k_cluster, cluster_map, drop_unmapped=cluster_map_drop_unmapped
        )

        # 2. Edges
        edges = _parse_edges(cluster_edges)
        if not edges:
            raise ValueError(
                f"No cluster_edges resolved for dataset '{name}'. Declare them in the global "
                f"config under datasets: {name}: cluster_edges:, or per method."
            )
        edges = _validate_edges(edges, clusters, reverse_map)
        if not edges:
            raise ValueError(f"All cluster_edges for dataset '{name}' were dropped during validation.")
        _log(f"  {len(edges)} transition edge(s) to score: {[f'{u} -> {v}' for u, v in edges]}")

        # 3. Embedding
        emb_key = prepare_embedding(
            adata,
            x_source=x_source,
            x_layer=x_layer,
            x_layers=x_layers,
            emb_key=emb_key,
            basis=basis,
            n_comps=n_comps,
            target_sum=target_sum,
            recompute_neighbors=recompute_neighbors,
            n_neighbors=n_neighbors,
        )

        # 3b. Optional smoothing of the velocity field (robustness sweep).
        #     Gene-space smoothing only reaches CBDir through the velocity graph
        #     and the embedded velocity, so both MUST be rebuilt from the smoothed
        #     field; reusing a stored graph or embedding would make the sweep a
        #     no-op that silently looks like "smoothing does nothing".
        smooth_info = {}
        smooth_on = int(smooth_velocity_rounds or 0) > 0
        smooth_in_gene_space = str(smooth_velocity_space).lower().startswith("g")
        if smooth_on and smooth_in_gene_space:
            if not recompute_velocity_graph:
                _log("  Smoothing in gene space: forcing recompute_velocity_graph=True.")
                recompute_velocity_graph = True
            if str(reuse_velocity_embedding).lower() != "false":
                _log("  Smoothing in gene space: forcing reuse_velocity_embedding=False.")
                reuse_velocity_embedding = False
            adata, smooth_info = smooth_velocity_field(
                adata,
                vkey=vkey,
                rounds=int(smooth_velocity_rounds),
                lam=float(smooth_velocity_lambda),
                space=smooth_velocity_space,
                graph=smooth_graph,
                k=int(smooth_k),
                emb_key=emb_key,
                n_pcs=n_pcs_metric,
                v_emb_key=velocity_embedding_key(vkey, basis),
            )

        # 4. Velocity in the embedding space
        v_emb_key = compute_velocity_embedding(
            adata, vkey=vkey, xkey=xkey, basis=basis, emb_key=emb_key,
            reuse=reuse_velocity_embedding, x_source=x_source,
            recompute_velocity_graph=recompute_velocity_graph,
            approx=approx, sqrt_transform=sqrt_transform, n_jobs=n_jobs,
            use_negative_cosines=use_negative_cosines,
        )

        # 4b. Embedding-space smoothing acts on the field CBDir actually reads,
        #     so it has to come after the embedded velocity exists.
        if smooth_on and not smooth_in_gene_space:
            adata, smooth_info = smooth_velocity_field(
                adata,
                vkey=vkey,
                rounds=int(smooth_velocity_rounds),
                lam=float(smooth_velocity_lambda),
                space=smooth_velocity_space,
                graph=smooth_graph,
                k=int(smooth_k),
                emb_key=emb_key,
                n_pcs=n_pcs_metric,
                v_emb_key=v_emb_key,
            )

        # 5. CBDir
        _log(f"  Computing per-cell CBDir (x_emb='{emb_key}', v_emb='{v_emb_key}', "
             f"n_pcs_metric={n_pcs_metric}, include_skipped={include_skipped}, "
             f"min_target_neighbors={min_target_neighbors}) ...")
        edge_stats = {}
        per_edge = cross_boundary_direction_percell(
            adata,
            clusters=clusters,
            cluster_edges=edges,
            x_emb_key=emb_key,
            v_emb_key=v_emb_key,
            n_pcs_metric=n_pcs_metric,
            include_skipped=include_skipped,
            min_target_neighbors=min_target_neighbors,
            edge_stats=edge_stats,
        )

        # 5b. ICCoh on the source clusters. Up to four variants, because one
        #     number cannot serve every purpose:
        #       iccoh           - the configured space. Default gene_confidence:
        #                         velocity consistency exactly as
        #                         compute_velocity_confidence_run.py computes it
        #                         in the original gene space, with neighbours
        #                         restricted to the cell's own cluster.
        #       iccoh_embedding - embedding space (velocity_pca): CBDir's own
        #                         geometry, but post-projection, so partly a
        #                         property of the transition-probability smoothing.
        #                         A copy of `iccoh` when iccoh_space is embedding.
        #       iccoh_gene_raw  - gene space, untransformed raw cosine: reproduces
        #                         the published benchmark number exactly.
        #       iccoh_gene_vst  - gene space, sign(v)|v|^p raw cosine: the
        #                         readout for whether gene-space smoothing took.
        #     With gene_confidence, `iccoh_owngenes` / `iccoh_sharedgenes` carry the
        #     score on each gene set (one of them equals `iccoh`), for the
        #     gene-count bias analysis in plot_cbdir.py.
        iccoh_variants = {}
        gc_meta = {}
        gc_clusters = pd.DataFrame()
        if compute_iccoh:
            src_clusters = (sorted(set(map(str, [c for c in np.unique(clusters.astype(str))
                                                 if c != "nan"])))
                            if str(iccoh_clusters).lower().startswith("a")
                            else sorted({u for u, _ in edges}))
            wanted = [("iccoh", iccoh_space, None)]
            if iccoh_embedding and iccoh_space != "embedding":
                wanted.append(("iccoh_embedding", "embedding", None))
            if iccoh_gene_raw and iccoh_space != "gene":
                wanted.append(("iccoh_gene_raw", "gene", None))
            if iccoh_gene_vst:
                wanted.append(("iccoh_gene_vst", "gene", float(iccoh_gene_power)))

            for col, sp, pw in wanted:
                if sp == "gene_confidence":
                    where = (f"layer={vkey}, version={iccoh_confidence_version}, "
                             f"graph={iccoh_neighbor_graph}")
                elif sp == "gene":
                    where = f"layer={vkey}, raw cosine, graph=connectivities"
                else:
                    where = f"v_emb={v_emb_key}, graph=connectivities"
                pw_txt = "" if pw is None else f", power={pw}"
                _log(f"  Computing per-cell {col} for {len(src_clusters)} "
                     f"{'cluster(s)' if str(iccoh_clusters).lower().startswith('a') else 'source cluster(s)'} "
                     f"(space='{sp}', {where}, "
                     f"min_neighbors={iccoh_min_neighbors}{pw_txt}) ...")
                cells_all = {}
                try:
                    if sp == "gene_confidence":
                        nb_key = None
                        if iccoh_recompute_graph:
                            nb_key = _recompute_iccoh_graph(
                                adata, iccoh_rep,
                                int(iccoh_n_neighbors or n_neighbors or 30),
                                iccoh_neighbor_graph)
                        sets = iccoh_gene_sets(
                            adata, vkey, use_genes=iccoh_use_genes,
                            intersected_genes=intersected_genes,
                            gene_names_col=gene_names_col, gene_query=gene_query,
                            apply_gene_query_always=apply_gene_query_always)
                        if iccoh_intersection_mode and intersected_genes is None:
                            _log("  NOTE: iccoh_intersection_mode is on but no shared gene "
                                 "set was available; scoring on this method's own genes.")
                        res = compute_gene_confidence_iccoh(
                            adata, clusters, src_clusters, vkey, sets,
                            primary=("shared" if iccoh_intersection_mode else "own"),
                            version=iccoh_confidence_version,
                            neighbor_graph=iccoh_neighbor_graph, neighbors_key=nb_key,
                            min_neighbors=iccoh_min_neighbors,
                            contrast=iccoh_gene_set_contrast,
                            subsample=iccoh_gene_subsample,
                            subsample_reps=iccoh_gene_subsample_reps,
                            subsample_seed=iccoh_gene_subsample_seed,
                            n_min=dataset_min_genes,
                            dimensionality=iccoh_dimensionality, root_power=root_power,
                            dataset=name, method=method_name)
                        gc_meta, gc_clusters = res["meta"], res["clusters"]
                        prim = gc_meta["primary"]
                        cells_all = dict(res["per_cell"][prim])
                        iccoh_variants["iccoh_owngenes"] = res["per_cell"].get("own", {})
                        iccoh_variants["iccoh_sharedgenes"] = res["per_cell"].get("shared", {})
                        _log(f"    primary gene set: {prim} ({gc_meta['n_genes']} genes; "
                             f"own {gc_meta['n_genes_own']}, shared "
                             f"{gc_meta['n_genes_shared']}), {gc_meta['neighbor_graph']} graph"
                             + (f" (k={gc_meta['n_neighbors']})"
                                if np.isfinite(gc_meta["n_neighbors"]) else ""))
                        per_cluster = {}
                        cl_arr = np.asarray(clusters, dtype=object)
                        bc_to_cl = dict(zip(np.asarray(adata.obs_names, dtype=object), cl_arr))
                        for bc, v in cells_all.items():
                            per_cluster.setdefault(bc_to_cl[bc], {})[bc] = v
                    else:
                        per_cluster = in_cluster_coherence_percell(
                            adata, clusters=clusters, cluster_list=src_clusters,
                            v_emb_key=v_emb_key, n_pcs_metric=n_pcs_metric,
                            min_neighbors=iccoh_min_neighbors,
                            space=sp, v_layer_key=vkey, gene_power=pw,
                        )
                        cells_all = {}
                    for c_name, cells in per_cluster.items():
                        vals = np.array([v for v in cells.values() if np.isfinite(v)])
                        _log(f"    {c_name:<40s} {col} mean = "
                             f"{vals.mean():+.4f} over {vals.size} cells" if vals.size else
                             f"    {c_name:<40s} {col} not computable")
                        cells_all.update(cells)
                except Exception as e:
                    # One variant failing must not cost the others, nor CBDir.
                    _log(f"  WARNING: {col} failed ({type(e).__name__}: {e}); "
                         f"continuing without it.")
                    cells_all = {}
                iccoh_variants[col] = cells_all
            if iccoh_embedding and iccoh_space == "embedding":
                iccoh_variants["iccoh_embedding"] = iccoh_variants.get("iccoh", {})
        iccoh_by_cell = iccoh_variants.get("iccoh", {})
        gc_on = compute_iccoh and iccoh_space == "gene_confidence"
        iccoh_version_col = iccoh_confidence_version if gc_on else ""
        iccoh_graph_col = ((gc_meta.get("neighbor_graph", str(iccoh_neighbor_graph))
                            if gc_on else "connectivities") if compute_iccoh else "")
        _g = lambda k: float(gc_meta.get(k, np.nan)) if gc_on else np.nan
        iccoh_ngenes_col = _g("n_genes")

        # 6. Tidy records
        n_source_total = {}
        for u, v in edges:
            n_source_total[(u, v)] = int(np.sum(clusters == u))

        def _raw_label(mapped_label):
            srcs = reverse_map.get(mapped_label, [])
            return srcs[0] if len(srcs) == 1 else ""

        n_dim_used = adata.obsm[emb_key].shape[1] if n_pcs_metric is None \
            else int(min(n_pcs_metric, adata.obsm[emb_key].shape[1]))

        params_key = velocity_params_key(vkey)
        velo_params = str(adata.uns[params_key]) if params_key in adata.uns else ""

        rows = []
        for (u, v), recs in per_edge.items():
            edge_label = f"{u} -> {v}"
            for r in recs:
                rows.append({
                    "dataset": name,
                    "method": method_name,
                    "edge": edge_label,
                    "source": u,
                    "target": v,
                    "source_raw": _raw_label(u),
                    "target_raw": _raw_label(v),
                    "cell_barcode": r["cell_barcode"],
                    "cbdir": r["cbdir"],
                    "n_target_neighbors": r["n_target_neighbors"],
                    "resultant_length": r.get("resultant_length", np.nan),
                    "alignment": r.get("alignment", np.nan),
                    # constant within the edge: a vector mean cannot be rebuilt
                    # downstream from per-cell scalars
                    "ceiling_coherent": float(
                        edge_stats.get((u, v), {}).get("ceiling_coherent", np.nan)),
                    "iccoh": float(iccoh_by_cell.get(r["cell_barcode"], np.nan)),
                    "iccoh_gene_raw": float(iccoh_variants.get("iccoh_gene_raw", {})
                                            .get(r["cell_barcode"], np.nan)),
                    "iccoh_gene_vst": float(iccoh_variants.get("iccoh_gene_vst", {})
                                            .get(r["cell_barcode"], np.nan)),
                    "iccoh_embedding": float(iccoh_variants.get("iccoh_embedding", {})
                                             .get(r["cell_barcode"], np.nan)),
                    # provenance of the `iccoh` column
                    "iccoh_space": iccoh_space if compute_iccoh else "",
                    "iccoh_version": iccoh_version_col,
                    "iccoh_neighbor_graph": iccoh_graph_col,
                    "iccoh_n_genes": iccoh_ngenes_col,
                    "iccoh_gene_set": (gc_meta.get("primary", "") if gc_on else ""),
                    "iccoh_owngenes": float(iccoh_variants.get("iccoh_owngenes", {})
                                            .get(r["cell_barcode"], np.nan)),
                    "iccoh_sharedgenes": float(iccoh_variants.get("iccoh_sharedgenes", {})
                                               .get(r["cell_barcode"], np.nan)),
                    "iccoh_n_genes_own": _g("n_genes_own"),
                    "iccoh_n_genes_shared": _g("n_genes_shared"),
                    "iccoh_pr_own": _g("pr_own"),
                    "iccoh_eff_genes_own": _g("eff_genes_own"),
                    "iccoh_pr_shared": _g("pr_shared"),
                    "iccoh_eff_genes_shared": _g("eff_genes_shared"),
                    "vkey": vkey,
                    "xkey": xkey,
                    "basis": basis,
                    "emb_key": emb_key,
                    "v_emb_key": v_emb_key,
                    "velocity_graph_recomputed": bool(recompute_velocity_graph),
                    "velocity_params": velo_params,
                    "x_source": x_source,
                    "x_layer": x_layer if x_source == "layer" else (
                        "+".join(map(str, x_layers)) if x_source == "layer_sum" else ""),
                    "n_comps": n_comps if x_source != "existing" else np.nan,
                    "n_pcs_metric": n_dim_used,
                    "smooth_rounds": int(smooth_velocity_rounds or 0),
                    "smooth_lambda": float(smooth_info.get(
                        "smooth_lambda", smooth_velocity_lambda)) if smooth_on else np.nan,
                    "smooth_space": str(smooth_info.get("smooth_space", "")) if smooth_on else "",
                    "smooth_graph": str(smooth_info.get("smooth_graph", "")) if smooth_on else "",
                    "smooth_k": float(smooth_info.get("smooth_k", np.nan)) if smooth_on else np.nan,
                })

        long_df = pd.DataFrame(rows, columns=[
            "dataset", "method", "edge", "source", "target", "source_raw", "target_raw",
            "cell_barcode", "cbdir", "n_target_neighbors",
            "resultant_length", "alignment", "ceiling_coherent",
            "iccoh", "iccoh_gene_raw", "iccoh_gene_vst", "iccoh_embedding",
            "iccoh_space", "iccoh_version", "iccoh_neighbor_graph", "iccoh_n_genes",
            "iccoh_gene_set", "iccoh_owngenes", "iccoh_sharedgenes",
            "iccoh_n_genes_own", "iccoh_n_genes_shared",
            "iccoh_pr_own", "iccoh_eff_genes_own", "iccoh_pr_shared", "iccoh_eff_genes_shared",
            "vkey", "xkey", "basis", "emb_key", "v_emb_key",
            "velocity_graph_recomputed", "velocity_params",
            "x_source", "x_layer", "n_comps", "n_pcs_metric",
            "smooth_rounds", "smooth_lambda", "smooth_space", "smooth_graph", "smooth_k",
        ])

        for (u, v) in edges:
            edge_label = f"{u} -> {v}"
            n_scored = int(long_df.loc[long_df["edge"] == edge_label, "cbdir"].notna().sum())
            mean_val = long_df.loc[long_df["edge"] == edge_label, "cbdir"].mean()
            _log(f"    {edge_label:<40s} scored {n_scored}/{n_source_total[(u, v)]} source cells; "
                 f"mean CBDir = {mean_val:.4f}" if n_scored else
                 f"    {edge_label:<40s} scored 0/{n_source_total[(u, v)]} source cells")

        if gc_on and gc_meta:
            long_df.attrs["iccoh_genes"] = dict(
                dataset=name, method=method_name, vkey=vkey,
                version=iccoh_confidence_version, primary=gc_meta.get("primary"),
                n_genes=gc_meta.get("n_genes"), n_genes_own=gc_meta.get("n_genes_own"),
                n_genes_shared=gc_meta.get("n_genes_shared"),
                pr_own=gc_meta.get("pr_own", np.nan),
                eff_genes_own=gc_meta.get("eff_genes_own", np.nan),
                pr_shared=gc_meta.get("pr_shared", np.nan),
                eff_genes_shared=gc_meta.get("eff_genes_shared", np.nan),
                root_power=root_power, gene_query=gene_query or "",
                apply_gene_query_always=bool(apply_gene_query_always),
                use_genes=(iccoh_use_genes if isinstance(iccoh_use_genes, str) else "custom"),
                neighbor_graph=gc_meta.get("neighbor_graph"),
                n_neighbors=gc_meta.get("n_neighbors"),
                iccoh_graph_recomputed=bool(iccoh_recompute_graph))
            long_df.attrs["iccoh_gene_clusters"] = gc_clusters
        return long_df, n_source_total

    finally:
        del adata
        gc.collect()


# ---------------------------------------------------------------------------
# Aggregation and IO
# ---------------------------------------------------------------------------

def _ceiling_stats(grp):
    """Attainable CBDir for a set of scored cells, and the fraction achieved.

    CBDir_i = Rbar_i * cos(theta_i): the cone geometry sets a hard ceiling that no
    velocity method can move, and only the alignment cos(theta_i) is the method's
    doing. Three numbers follow.

    ceiling_free      mean_i Rbar_i — attained when every cell's velocity points
                      along its own resultant. CBDir imposes no coupling between
                      cells, so this bound is attainable and tight.
    ceiling_coherent  ||mean_i m_i||, the best a single shared direction can do,
                      computed upstream because it needs the resultant vectors.
                      It is <= ceiling_free by the triangle inequality, and the
                      gap is what a perfectly coherent field gives up here.
    alignment_weighted
                      sum_i Rbar_i cos(theta_i) / sum_i Rbar_i = CBDir / ceiling,
                      the Rbar-weighted mean alignment. Weighted, not a plain mean
                      of per-cell ratios: a cell whose cone is nearly isotropic
                      (Rbar ~ 0) knows nothing about direction and its ratio is
                      numerically unstable, so it must not carry equal weight.

    Two corrections that matter:

    * The Fisher-z ceiling is mean_i atanh(Rbar_i), NOT atanh of the mean. atanh
      is convex, so the second is smaller and the observed z-mean could appear to
      breach its own ceiling.
    * Rbar is inflated at small n: for n directions with true concentration rho,
      E[Rbar^2] = rho^2 + (1 - rho^2)/n, so an isotropic cone of n = 3 still
      reports Rbar ~ 0.58. For the realised neighbour set the raw Rbar IS the
      exact ceiling and needs no correction; rho2_corrected = (n Rbar^2 - 1)/(n - 1)
      is for comparing cone tightness ACROSS transitions with different n.
    """
    out = {}
    if "resultant_length" not in grp.columns:
        return out
    r = grp["resultant_length"].to_numpy(dtype=np.float64)
    cb = grp["cbdir"].to_numpy(dtype=np.float64)
    ok = np.isfinite(r) & np.isfinite(cb)
    if not ok.any():
        return out
    r_ok, cb_ok = r[ok], cb[ok]
    out["ceiling_free"] = float(np.mean(r_ok))
    out["ceiling_free_z"] = float(np.mean(np.arctanh(np.clip(r_ok, -0.999999, 0.999999))))
    denom = float(np.sum(r_ok))
    out["alignment_weighted"] = float(np.sum(cb_ok) / denom) if denom > 0 else np.nan
    out["frac_of_ceiling"] = out["alignment_weighted"]
    if "alignment" in grp.columns:
        a = grp["alignment"].to_numpy(dtype=np.float64)
        a = a[np.isfinite(a)]
        out["alignment_mean_unweighted"] = float(np.mean(a)) if a.size else np.nan
    if "ceiling_coherent" in grp.columns:
        cc = grp["ceiling_coherent"].to_numpy(dtype=np.float64)
        cc = cc[np.isfinite(cc)]
        out["ceiling_coherent"] = float(cc[0]) if cc.size else np.nan
        out["coherence_cost"] = (out["ceiling_free"] - out["ceiling_coherent"]
                                 if cc.size else np.nan)
    if "n_target_neighbors" in grp.columns:
        n = grp["n_target_neighbors"].to_numpy(dtype=np.float64)[ok]
        with np.errstate(invalid="ignore", divide="ignore"):
            rho2 = np.where(n > 1, (n * r_ok ** 2 - 1.0) / (n - 1.0), np.nan)
        rho2 = np.clip(rho2, 0.0, 1.0)
        good = np.isfinite(rho2)
        out["rho2_corrected"] = float(np.mean(rho2[good])) if good.any() else np.nan
        out["ceiling_free_corrected"] = (float(np.mean(np.sqrt(rho2[good])))
                                         if good.any() else np.nan)
        out["median_n_target_neighbors"] = float(np.median(n)) if n.size else np.nan
    return out


def _edge_summary(long_df, n_source_total_by_method):
    """Per (edge, method) summary plus an ALL_EDGES row per method."""
    out = []
    for (edge, method), grp in long_df.groupby(["edge", "method"], sort=False):
        vals = grp["cbdir"].to_numpy(dtype=np.float64)
        finite = vals[~np.isnan(vals)]
        src = grp["source"].iloc[0]
        tgt = grp["target"].iloc[0]
        n_src_total = n_source_total_by_method.get(method, {}).get((src, tgt), np.nan)
        out.append({
            "edge": edge,
            "method": method,
            "source": src,
            "target": tgt,
            "n_cells": int(finite.size),
            "n_source_cells_total": n_src_total,
            "n_skipped": (int(n_src_total - finite.size) if not pd.isna(n_src_total) else np.nan),
            "mean": float(np.mean(finite)) if finite.size else np.nan,
            "median": float(np.median(finite)) if finite.size else np.nan,
            "std": float(np.std(finite, ddof=1)) if finite.size > 1 else np.nan,
            "q25": float(np.percentile(finite, 25)) if finite.size else np.nan,
            "q75": float(np.percentile(finite, 75)) if finite.size else np.nan,
            "frac_positive": float(np.mean(finite > 0)) if finite.size else np.nan,
            **_ceiling_stats(grp),
            # ICCoh of the SOURCE cells scored on this edge: the coherence covariate
            # CBDir has to be read against, never a second correctness number.
            "mean_iccoh": (float(np.nanmean(grp["iccoh"].to_numpy(dtype=np.float64)))
                           if "iccoh" in grp.columns
                           and np.isfinite(grp["iccoh"].to_numpy(dtype=np.float64)).any()
                           else np.nan),
        })

    summary = pd.DataFrame(out)
    if summary.empty:
        return summary

    all_rows = []
    for method, grp in summary.groupby("method", sort=False):
        pooled = long_df.loc[long_df["method"] == method, "cbdir"].to_numpy(dtype=np.float64)
        pooled = pooled[~np.isnan(pooled)]
        all_rows.append({
            "edge": "ALL_EDGES",
            "method": method,
            "source": "",
            "target": "",
            "n_cells": int(pooled.size),
            "n_source_cells_total": float(grp["n_source_cells_total"].sum(skipna=True)),
            "n_skipped": float(grp["n_skipped"].sum(skipna=True)),
            "mean": float(grp["mean"].mean(skipna=True)),          # edge-mean-of-means (snippet scalar)
            "median": float(np.median(pooled)) if pooled.size else np.nan,
            "std": float(np.std(pooled, ddof=1)) if pooled.size > 1 else np.nan,
            "q25": float(np.percentile(pooled, 25)) if pooled.size else np.nan,
            "q75": float(np.percentile(pooled, 75)) if pooled.size else np.nan,
            "frac_positive": float(np.mean(pooled > 0)) if pooled.size else np.nan,
            **_ceiling_stats(long_df[long_df["method"] == method]),
            "mean_iccoh": float(grp["mean_iccoh"].mean(skipna=True))
            if "mean_iccoh" in grp.columns else np.nan,
        })
        all_rows[-1]["pooled_cell_mean"] = float(np.mean(pooled)) if pooled.size else np.nan

    summary["pooled_cell_mean"] = np.nan
    return pd.concat([summary, pd.DataFrame(all_rows)], ignore_index=True)


def save_aggregated_results(collector, dataset_dir_paths, method_order,
                            n_source_totals, save_parquet=False, str_suffix=None,
                            gene_collector=None):
    """Write the long, wide and edge-summary tables for each dataset.

    With gene_confidence ICCoh, also:
      {dataset}_iccoh_genes{suffix}.csv          one row per method: the genes
          used (own / shared / primary), participation ratio and effective genes
      {dataset}_iccoh_gene_clusters{suffix}.csv  per source cluster: ICCoh on the
          own set, the shared set and every random own-gene subsample
    """
    _log("")
    _log("=== Saving Aggregated Results ===")

    suffix = ""
    if str_suffix is not None and str(str_suffix).strip() != "":
        s = str(str_suffix).strip()
        suffix = s if (s.startswith("_") or s.startswith("-")) else f"_{s}"

    for dataset_name, method_frames in collector.items():
        if not method_frames:
            _log(f"No results collected for dataset '{dataset_name}'; skipping.")
            continue

        dir_path = dataset_dir_paths.get(dataset_name, f"./results/{dataset_name}")
        if not os.path.exists(dir_path):
            _log(f"Creating output directory: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)

        ordered = [m for m in method_order if m in method_frames]
        for m in method_frames:
            if m not in ordered:
                ordered.append(m)

        long_df = pd.concat([method_frames[m] for m in ordered], ignore_index=True)

        # Wide pivot on (cell_barcode, edge)
        wide = long_df.pivot_table(
            index=["cell_barcode", "edge"],
            columns="method",
            values="cbdir",
            aggfunc="first",
        )
        wide.columns = [f"{c}_cbdir" for c in wide.columns]
        meta = (long_df.drop_duplicates(subset=["cell_barcode", "edge"])
                       .set_index(["cell_barcode", "edge"])[["source", "target"]])
        wide = meta.join(wide, how="right").reset_index()
        ordered_cols = ["cell_barcode", "edge", "source", "target"] + \
                       [f"{m}_cbdir" for m in ordered if f"{m}_cbdir" in wide.columns]
        wide = wide[ordered_cols]

        summary = _edge_summary(long_df, n_source_totals.get(dataset_name, {}))

        base = os.path.join(dir_path, f"{dataset_name}_cbdir")
        targets = [
            (long_df, f"{base}_long{suffix}"),
            (wide, f"{base}_wide{suffix}"),
            (summary, f"{base}_edge_summary{suffix}"),
        ]
        extras = (gene_collector or {}).get(dataset_name, {})
        g_rows = [extras[m]["iccoh_genes"] for m in ordered
                  if m in extras and extras[m].get("iccoh_genes")]
        g_clus = [extras[m]["iccoh_gene_clusters"] for m in ordered
                  if m in extras and isinstance(extras[m].get("iccoh_gene_clusters"), pd.DataFrame)
                  and not extras[m]["iccoh_gene_clusters"].empty]
        if g_rows:
            targets.append((pd.DataFrame(g_rows),
                            os.path.join(dir_path, f"{dataset_name}_iccoh_genes{suffix}")))
        if g_clus:
            targets.append((pd.concat(g_clus, ignore_index=True),
                            os.path.join(dir_path, f"{dataset_name}_iccoh_gene_clusters{suffix}")))
        for df, out_base in targets:
            csv_path = f"{out_base}.csv"
            df.to_csv(csv_path, index=False)
            _log(f"  Saved CSV    : {csv_path}  shape={df.shape}")
            if save_parquet:
                pq_path = f"{out_base}.parquet"
                df.to_parquet(pq_path, index=False)
                _log(f"  Saved Parquet: {pq_path}")


# ---------------------------------------------------------------------------
# Config handling
# ---------------------------------------------------------------------------

def _resolve_params(defaults, entry):
    merged = {}
    for src in (defaults or {}, ):
        for k, v in src.items():
            k = _PARAM_ALIASES.get(k, k)
            if k in _PIPELINE_PARAMS:
                merged[k] = v
            elif k in _CONFIDENCE_ONLY_PARAMS:
                pass                    # velocity-confidence setting with no role here
            else:
                _log(f"  WARNING: ignoring unknown parameter '{k}' in defaults")

    for k, v in entry.items():
        if k in ("name", "adata_path", "dir_path"):
            continue
        k = _PARAM_ALIASES.get(k, k)
        if k in _PIPELINE_PARAMS:
            merged[k] = v
        elif k in _CONFIDENCE_ONLY_PARAMS:
            pass
        else:
            _log(f"  WARNING: ignoring unknown parameter '{k}'")

    resolved = {}
    for k in _PIPELINE_PARAMS:
        resolved[k] = merged[k] if k in merged else _DEFAULT_FALLBACKS[k]
    return resolved


def _build_params(defaults, entry, g_entry, overrides=None):
    """Resolved parameters for one (method, dataset) pair, overrides applied."""
    params = _resolve_params(defaults, entry)

    # Dataset-level edges / map from the global config fill in when the
    # method config does not override them.
    if params.get("cluster_edges") is None:
        params["cluster_edges"] = g_entry.get("cluster_edges")
    if params.get("cluster_map") is None:
        params["cluster_map"] = g_entry.get("cluster_map")
    if "cluster_map_drop_unmapped" not in (defaults or {}) and \
            "cluster_map_drop_unmapped" not in entry and \
            "cluster_map_drop_unmapped" in g_entry:
        params["cluster_map_drop_unmapped"] = g_entry["cluster_map_drop_unmapped"]

    if overrides:
        unknown = [k for k in overrides if k not in _PIPELINE_PARAMS]
        if unknown:
            raise ValueError(
                f"run_from_config overrides contain unknown parameter(s): {unknown}"
            )
        params.update(dict(overrides))

    for k in _BOOL_PARAMS:
        if k in params and params[k] is not None:
            params[k] = _as_bool(k, params[k])
    params["iccoh_space"] = normalize_iccoh_space(params.get("iccoh_space"))
    return params


def _dataset_gene_intersections(method_cfgs, global_datasets, overrides=None):
    """Validate every (method, dataset) config, then build the shared gene sets.

    Returns ({dataset: sorted shared symbols}, {dataset: smallest per-method count
    of finite-velocity genes}). Mirrors
    compute_velocity_confidence_run.compute_dataset_gene_intersection: a gene is
    kept when every method that could be read has a finite velocity for it.

    Datasets are read only where some pair computes gene_confidence ICCoh and
    needs the shared set: for the primary score (iccoh_intersection_mode), for
    the own-vs-shared contrast, or for count-matched ("min") subsampling.

    A malformed config (non-boolean in a boolean slot, unknown iccoh_space,
    unknown override) raises here, before any heavy compute.
    """
    by_dataset, errors = defaultdict(list), []
    for method_name, m_cfg in method_cfgs.items():
        defaults = m_cfg.get("defaults", {})
        for entry in m_cfg.get("datasets", []) or []:
            d_name = entry.get("name")
            if not d_name or "adata_path" not in entry:
                continue
            try:
                p = _build_params(defaults, entry, global_datasets.get(d_name, {}) or {},
                                  overrides)
            except ValueError as e:
                errors.append(f"{method_name}/{d_name}: {e}")
                continue
            by_dataset[d_name].append((method_name, entry["adata_path"], p))
    if errors:
        raise ValueError("Invalid CBDir configuration:\n  " + "\n  ".join(errors))

    def _needs(p):
        sub = p.get("iccoh_gene_subsample") or []
        sub = sub if isinstance(sub, (list, tuple)) else [sub]
        return (p.get("compute_iccoh") and p.get("iccoh_space") == "gene_confidence"
                and (p.get("iccoh_intersection_mode") or p.get("iccoh_gene_set_contrast")
                     or any(isinstance(x, str) for x in sub)))

    out, mins = {}, {}
    for d_name, pairs in by_dataset.items():
        wanted = [(m, path, p) for m, path, p in pairs if _needs(p)]
        if not wanted:
            continue
        _log(f"Shared velocity genes for dataset '{d_name}' across {len(wanted)} method(s) ...")
        sets = []
        for m, path, p in wanted:
            genes = valid_velocity_gene_symbols(path, p.get("vkey", "velocity"),
                                                p.get("gene_names_col"))
            if genes is None:
                _log(f"  {m}: could not read valid genes (missing file or layer); not constraining.")
                continue
            _log(f"  {m}: {len(genes)} genes with finite velocity "
                 f"(gene_names_col={p.get('gene_names_col')})")
            sets.append(genes)
        if sets:
            out[d_name] = sorted(set.intersection(*sets))
            mins[d_name] = int(min(len(g) for g in sets))
            _log(f"  -> {len(out[d_name])} shared gene symbols for '{d_name}' "
                 f"(smallest per-method set: {mins[d_name]}).")
            if len(out[d_name]) < 10:
                _log(f"  WARNING: fewer than 10 shared genes for '{d_name}'; shared-gene "
                     f"ICCoh will fail there. Check gene_names_col / vkey across methods.")
        else:
            _log(f"  WARNING: no gene sets read for '{d_name}'; each method uses its own genes.")
    return out, mins


def run_from_config(config_path, str_suffix=None, overrides=None, out_subdir=None,
                    log_file=None):
    """Run the CBDir pipeline across methods and datasets defined in YAML configs.

    overrides   mapping of compute_cbdir_for_dataset parameters forced on every
                (method, dataset) pair, after the method/entry configs resolve.
                This is how a sweep driver varies one knob (e.g. the smoothing
                rounds) without editing or duplicating any config file.
    out_subdir  results go to <dir_path>/<out_subdir>/ instead of <dir_path>/,
                so each rung of a sweep keeps its own tables.
    log_file    overrides the config's log_file.
    """
    try:
        import yaml
    except ImportError as e:
        raise SystemExit(
            "PyYAML is required to read config files. Install it with `pip install pyyaml`."
        ) from e

    with open(config_path, "r") as fh:
        global_cfg = yaml.safe_load(fh)

    if not isinstance(global_cfg, dict) or "methods" not in global_cfg:
        raise ValueError("Global config YAML must be a mapping containing a 'methods:' key.")

    if str_suffix is None:
        str_suffix = global_cfg.get("str_suffix", None)

    methods_dict = global_cfg["methods"]
    if not isinstance(methods_dict, dict) or not methods_dict:
        raise ValueError("'methods' must be a non-empty mapping of method_name -> config_path.")

    global_datasets = global_cfg.get("datasets", {}) or {}

    default_log = f"cbdir_{time.strftime('%Y%m%d_%H%M%S')}.log"
    if log_file is None:
        log_file = global_cfg.get("log_file", default_log)
    log_dir = os.path.dirname(os.path.abspath(log_file))
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    log_fh = _FilteredFile(open(log_file, "a"))
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    sys.stderr = _Tee(sys.__stderr__, log_fh)

    try:
        _log("=== CBDir Batch Run Started ===")
        _log(f"Global config file : {os.path.abspath(config_path)}")
        _log(f"Log file           : {os.path.abspath(log_file)}")
        _log(f"Methods            : {list(methods_dict.keys())}")
        if str_suffix:
            _log(f"str_suffix         : {str_suffix}")
        if overrides:
            _log(f"param overrides    : {dict(overrides)}")
        if out_subdir:
            _log(f"output subdir      : {out_subdir}")

        method_cfgs = {}
        for method_name, method_cfg_path in methods_dict.items():
            if not os.path.exists(method_cfg_path):
                _log(f"WARNING: Config file for method '{method_name}' not found: {method_cfg_path}")
                continue
            with open(method_cfg_path, "r") as fh:
                m_cfg = yaml.safe_load(fh)
            if not isinstance(m_cfg, dict) or "datasets" not in m_cfg:
                _log(f"WARNING: Config for method '{method_name}' is not a dict or lacks 'datasets': {method_cfg_path}")
                continue
            method_cfgs[method_name] = m_cfg

        # Shared velocity-gene set per dataset for gene_confidence ICCoh, as
        # compute_velocity_confidence_run.py does for velocity confidence: each
        # method's velocity is scored on the same genes, so a method is not
        # rewarded or penalised for which genes it happened to fit.
        dataset_gene_intersections, dataset_min_genes = _dataset_gene_intersections(
            method_cfgs, global_datasets, overrides)

        collector = defaultdict(dict)
        gene_collector = defaultdict(dict)
        dataset_dir_paths = {}
        n_source_totals = defaultdict(dict)
        summary_rows = []
        method_order = list(methods_dict.keys())

        for method_name, method_cfg_path in methods_dict.items():
            _log("")
            _log("=" * 50)
            _log(f" Processing Method: {method_name}")
            _log(f" Config path: {os.path.abspath(method_cfg_path)}")
            _log("=" * 50)

            if method_name not in method_cfgs:
                _log(f"  ERROR: Method '{method_name}' config not loaded; skipping.")
                summary_rows.append((method_name, "ALL", "SKIPPED", 0.0))
                continue

            m_cfg = method_cfgs[method_name]
            defaults = m_cfg.get("defaults", {})
            datasets = m_cfg.get("datasets", [])

            for i, entry in enumerate(datasets, start=1):
                d_name = entry.get("name", f"dataset_{i}")
                _log("")
                _log(f"----- [{method_name}] dataset {i}/{len(datasets)}: {d_name} -----")

                if "adata_path" not in entry or "dir_path" not in entry:
                    _log(f"  ERROR: entry '{d_name}' missing adata_path/dir_path; skipping")
                    summary_rows.append((method_name, d_name, "SKIPPED (missing paths)", 0.0))
                    continue

                params = _build_params(defaults, entry,
                                       global_datasets.get(d_name, {}) or {}, overrides)

                out_dir = entry["dir_path"]
                if out_subdir:
                    out_dir = os.path.join(out_dir, str(out_subdir))
                dataset_dir_paths.setdefault(d_name, out_dir)

                _log(f"  adata_path : {entry['adata_path']}")
                _log(f"  dir_path   : {entry['dir_path']}")
                loggable = {k: v for k, v in params.items() if k != "cluster_map"}
                _log(f"  params     : {loggable}")
                if params.get("cluster_map"):
                    _log(f"  cluster_map: {params['cluster_map']}")

                t0 = time.time()
                try:
                    long_df, n_src = compute_cbdir_for_dataset(
                        name=d_name,
                        adata_path=entry["adata_path"],
                        method_name=method_name,
                        intersected_genes=dataset_gene_intersections.get(d_name),
                        dataset_min_genes=dataset_min_genes.get(d_name),
                        **params,
                    )
                    dt = time.time() - t0
                    # side tables travel on attrs out of the worker; take them off
                    # before the frame meets pd.concat, which compares attrs
                    gene_collector[d_name][method_name] = {
                        k: long_df.attrs.pop(k) for k in list(long_df.attrs)}
                    if long_df.empty:
                        _log(f"  WARNING: no CBDir records produced for '{d_name}' / '{method_name}'.")
                    collector[d_name][method_name] = long_df
                    n_source_totals[d_name][method_name] = n_src
                    n_scored = int(long_df["cbdir"].notna().sum())
                    _log(f"  DONE in {dt/60:.2f} min ({len(long_df)} rows; {n_scored} scored cells; "
                         f"{long_df['edge'].nunique()} edges)")
                    summary_rows.append((method_name, d_name, "OK", dt))
                except Exception as e:
                    dt = time.time() - t0
                    _log(f"  FAILED after {dt/60:.2f} min: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    summary_rows.append((method_name, d_name, f"FAILED ({type(e).__name__})", dt))

        save_aggregated_results(
            collector,
            dataset_dir_paths,
            method_order,
            n_source_totals,
            save_parquet=global_cfg.get("save_parquet", False),
            str_suffix=str_suffix,
            gene_collector=gene_collector,
        )

        _log("")
        _log("=== Batch Run Summary ===")
        for method_name, d_name, status, dt in summary_rows:
            _log(f"  {method_name:<15s} {d_name:<25s} {status:<30s} {dt/60:6.2f} min")
        _log("=== CBDir Batch Run Finished ===")

    finally:
        log_fh.flush()
        log_fh.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__


# ---------------------------------------------------------------------------
# Output capture and filtering helpers (shared with compute_velocity_confidence_run.py)
# ---------------------------------------------------------------------------

class _Tee:
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

    def close(self):
        self.flush()


class _FilteredFile:
    """Log-file writer that drops progress-bar noise.

    Lines starting with '[' are treated as progress bars EXCEPT this module's own
    '[YYYY-MM-DD ...]' timestamped lines; before that exception every _log line
    was filtered out and reached only the console.
    """

    def __init__(self, fh):
        self._fh = fh
        self._buf = ""

    def write(self, data):
        self._buf += data
        import re
        _PROGRESS_RE = re.compile(r"^\s*(loss|epoch|\d+%|\[(?!\d{4}-\d\d-\d\d )|\.|\*|-)", re.IGNORECASE)
        while "\n" in self._buf:
            nl = self._buf.find("\n")
            cr = self._buf.rfind("\r", 0, nl)
            if cr != -1:
                line = self._buf[cr + 1:nl]
                self._buf = self._buf[nl + 1:]
                eff = line.rsplit("\r", 1)[-1]
                if eff.strip() and _PROGRESS_RE.search(eff):
                    continue
                self._fh.write(eff + "\n")
                self._fh.flush()
                continue
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
        import re
        _PROGRESS_RE = re.compile(r"^\s*(loss|epoch|\d+%|\[(?!\d{4}-\d\d-\d\d )|\.|\*|-)", re.IGNORECASE)
        if rem.strip() and not _PROGRESS_RE.search(rem):
            self._fh.write(rem)
        self._buf = ""
        self._fh.flush()
        self._fh.close()


def _timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg):
    print(f"[{_timestamp()}] {msg}", flush=True)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute per-cell Cross-Boundary Direction (CBDir) correctness "
                    "across multiple velocity methods and datasets."
    )
    parser.add_argument(
        "config",
        help="Path to the global YAML config file (see cbdir_global_config_4throot.yaml).",
    )
    parser.add_argument(
        "--str-suffix", "--suffix",
        dest="str_suffix",
        default=None,
        help="Optional string suffix to append to all output filenames (e.g. '_v2').",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_from_config(args.config, str_suffix=args.str_suffix)
    sys.exit(0)
