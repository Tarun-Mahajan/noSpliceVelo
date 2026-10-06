#!/usr/bin/env python
"""Per-gene direction benchmark on the CBDir transitions.

Answers two questions with the datasets, edges and method h5ads that the
CBDir pipeline already uses (it reads the same global + per-method configs):

  (1) On transitions whose direction is known, what fraction of genes does each
      method give the correct direction?
  (2) Gene by gene, how often do noSpliceVelo and the splicing-based methods agree,
      and when they disagree, which one is right and what distinguishes those genes?

Ground truth (method-independent)
---------------------------------
All methods are scored on ONE reference h5ad per dataset (`reference_method`'s h5ad,
or `reference_adata`), which supplies cells, graph, clusters and the truth layer.
Agreement tables compare `compare_to` (default nSV) with every other method. Its kNN graph defines the boundary
exactly as CBDir does:

  S  source cells of A with >= min_target_neighbors neighbours in B
  T  the B cells that neighbour those source cells

The truth layer is model-free expression (auto: log-normalised raw counts, smoothed
once over the kNN graph like scVelo's Ms). For every gene

  d_g = (mean_T x_g - mean_S x_g) / sd_g          (sd over all cells)

and the true direction across A -> B is sign(d_g) when |d_g| >= d_min, otherwise the
gene is "ambiguous" and not scored. Results are reported for a sweep of d_min.

Predictions
-----------
For each method, the velocity of gene g in the source cells S (cells and genes matched
to the reference by barcode and gene name):

  pred_sign        sign of the median velocity over S (0 = abstain)
  frac_cells_agree fraction of S cells whose velocity sign equals the true sign
  contribution     the gene's share of the gene-space boundary cosine,
                     c_g = mean_i  dx_ig t(v_ig) / (|dx_i| |t(v_i)|),
                   dx_i = mean over i's target neighbours of x_j - x_i (truth layer),
                   t(v) = sign(v)|v|^(1/root_power). Sum_g c_g is CBDir computed in
                   gene space on that gene set, so c_g says how much each gene
                   pushes the field forward (> 0) or backward (< 0).

Gene sets (each method is scored on each)
  own       the method's velocity genes: finite velocity in every matched cell,
            narrowed by `own_gene_query` (string = all methods; dict = per method), else the CBDir gene_query
            when it evaluates on that h5ad)
  shared    genes in the own set of EVERY method in the dataset (head-to-head)
  pair      own(compare_to) & own(method): used for the agreement tables

Outputs (out_dir/<dataset>/, plus out_dir/all_datasets_*.csv)
  gd_long.csv        one row per (edge, gene, method): truth, prediction, contribution,
                     gene-set membership, flags, splice features, method var features
  gd_summary.csv     per (edge | ALL, method, gene_set, d_min): accuracy with bootstrap
                     CI over genes, abstain rate, |d|-weighted accuracy, mean per-cell
                     agreement, gene-space CBDir, forward share and the share of the
                     boundary signal carried by correctly-signed genes, and the
                     majority-sign baseline
  gd_agreement.csv   reference vs each other method on the pair set: sign agreement,
                     Cohen's kappa, and among genes with a true direction the four
                     classes both-correct / reference-only / other-only / both-wrong
  gd_features.csv    feature medians by agreement class; gd_feature_tests.csv tests what
                     separates correct from wrong reference calls (Mann-Whitney per
                     feature, and a standardised logistic regression when sklearn is
                     available)
  figures: acc_vs_effect_<ds>.png, acc_heatmap_<ds>.png, agreement_<ds>.png

Usage:
  python gene_direction_benchmark.py gene_direction_config.yaml
"""
import argparse
import fnmatch
import os
import re
import sys
import time
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy import sparse as sp

EPS = 1e-12


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# configuration (mirrors compute_cbdir_run semantics for the keys used here)
# ---------------------------------------------------------------------------

_KEYS = ("k_cluster", "vkey", "gene_names_col", "gene_query", "apply_gene_query_always",
         "cluster_edges", "cluster_map", "cluster_map_drop_unmapped", "min_target_neighbors",
         "root_power")
_FALLBACK = dict(k_cluster="clusters", vkey="velocity", gene_names_col=None, gene_query=None,
                 apply_gene_query_always=False, cluster_edges=None, cluster_map=None,
                 cluster_map_drop_unmapped=False, min_target_neighbors=3, root_power=2)


def read_yaml(path):
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def resolve_jobs(gcfg_path, methods_sel=None, datasets_sel=None):
    """{dataset: {'global': g_entry, 'methods': OrderedDict(method -> (adata_path, params))}}"""
    g = read_yaml(gcfg_path)
    base = os.path.dirname(os.path.abspath(gcfg_path))
    gdef = g.get("defaults", {}) or {}
    gds = g.get("datasets", {}) or {}
    top = {k: g[k] for k in _KEYS if k in g}               # e.g. min_target_neighbors at top level
    jobs = OrderedDict()
    for m, mpath in (g.get("methods") or {}).items():
        if methods_sel and m not in methods_sel:
            continue
        mp = mpath if os.path.isabs(mpath) else os.path.join(base, mpath)
        if not os.path.exists(mp):
            log(f"WARNING: config for method '{m}' not found: {mp}; skipped")
            continue
        mc = read_yaml(mp)
        mdef = mc.get("defaults", {}) or {}
        for e in mc.get("datasets", []) or []:
            d = e.get("name")
            if not d or "adata_path" not in e:
                continue
            if datasets_sel and d not in datasets_sel:
                continue
            ge = gds.get(d, {}) or {}
            p = dict(_FALLBACK)
            for src in (top, gdef, mdef, e):
                for k in _KEYS:
                    if k in src and src[k] is not None:
                        p[k] = src[k]
            if not e.get("cluster_edges") and not mdef.get("cluster_edges"):
                p["cluster_edges"] = ge.get("cluster_edges", p["cluster_edges"])
            if not e.get("cluster_map") and not mdef.get("cluster_map"):
                p["cluster_map"] = ge.get("cluster_map", p["cluster_map"])
                p["cluster_map_drop_unmapped"] = ge.get("cluster_map_drop_unmapped",
                                                        p["cluster_map_drop_unmapped"])
            ap = e["adata_path"]
            ap = ap if os.path.isabs(ap) else os.path.join(base, ap)
            jobs.setdefault(d, {"global": ge, "methods": OrderedDict()})["methods"][m] = (ap, p)
    return jobs


def parse_edges(raw):
    out = []
    for e in raw or []:
        if isinstance(e, str) and "->" in e:
            u, v = e.split("->", 1)
            out.append((u.strip(), v.strip()))
        elif isinstance(e, (list, tuple)) and len(e) == 2:
            out.append((str(e[0]), str(e[1])))
    return out


def mapped_clusters(adata, key, cmap, drop):
    raw = adata.obs[key].astype(str).to_numpy()
    if not cmap:
        return raw
    cm = {str(k): (None if v is None else str(v)) for k, v in dict(cmap).items()}
    return np.array([cm[x] if x in cm else (None if drop else x) for x in raw], dtype=object)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def dense(x, dtype=np.float32):
    return np.asarray(x.toarray() if sp.issparse(x) else x, dtype=dtype)


def norm_barcode(b):
    b = str(b)
    b = b.split(":")[-1]
    b = re.sub(r"x$", "", b)
    b = re.sub(r"-\d+$", "", b)
    return b


def _first_indexer(keys_other, keys_ref):
    """get_indexer that tolerates duplicated keys in `keys_other` (first occurrence wins)."""
    o = pd.Series(np.arange(len(keys_other)), index=pd.Index(keys_other))
    o = o[~o.index.duplicated(keep="first")]
    return o.reindex(pd.Index(keys_ref)).fillna(-1).astype(int).to_numpy()


def match_index(ref_names, other_names, what, label=""):
    """Positions of ref_names in other_names (-1 if absent).

    Duplicated names on the other side match their first occurrence (logged).
    Cells fall back to normalised barcodes, genes to case-insensitive names, when the
    exact match rate is low."""
    other_names = np.asarray(other_names, dtype=object).astype(str)
    ref_names = np.asarray(ref_names, dtype=object).astype(str)
    n_dup = int(pd.Index(other_names).duplicated().sum())
    if n_dup:
        log(f"    {label}: {n_dup} duplicated name(s) among {what}; the first occurrence of each is used")
    idx = _first_indexer(other_names, ref_names)
    rate = float((idx >= 0).mean()) if len(idx) else 0.0
    if rate < 0.5:
        if what == "cells":
            f = norm_barcode
            note = "normalising barcodes"
        else:
            f = str.lower
            note = "case-insensitive gene names"
        idx2 = _first_indexer([f(x) for x in other_names], [f(x) for x in ref_names])
        if (idx2 >= 0).mean() > rate:
            log(f"    {label}: {what} matched after {note}: {rate:.1%} -> {(idx2 >= 0).mean():.1%}")
            idx, rate = idx2, float((idx2 >= 0).mean())
    return idx, rate


def gene_names(adata, col, warn_label=None):
    if col and col in adata.var.columns:
        return adata.var[col].astype(str).to_numpy()
    if col and warn_label:
        log(f"  WARNING: {warn_label}: var column '{col}' not found; matching on var_names")
    return np.asarray(adata.var_names.astype(str))


def gene_col_for(cfg, dname, m, p):
    """per_dataset.<d>.gene_names_col.<m>  >  gene_names_col_override.<m>  >  CBDir gene_names_col."""
    d = ((cfg.get("per_dataset") or {}).get(dname) or {}).get("gene_names_col") or {}
    if isinstance(d, str):
        return d
    if m in d:
        return d[m]
    return (cfg.get("gene_names_col_override") or {}).get(m, p.get("gene_names_col"))


def own_query_for(cfg, m, p):
    q = cfg.get("own_gene_query")
    if isinstance(q, str):
        return q
    if isinstance(q, dict) and m in q:
        return q[m]
    return p.get("gene_query")


def lognorm(C):
    C = sp.csr_matrix(C, dtype=np.float32) if sp.issparse(C) else np.asarray(C, np.float32)
    tot = np.asarray(C.sum(1)).ravel()
    tgt = float(np.median(tot[tot > 0])) if (tot > 0).any() else 1.0
    s = tgt / np.maximum(tot, EPS)
    X = (sp.diags(s) @ C) if sp.issparse(C) else C * s[:, None]
    X = X.toarray() if sp.issparse(X) else X
    return np.log1p(X).astype(np.float32)


def row_norm_graph(C):
    C = sp.csr_matrix(C, dtype=np.float64) + sp.identity(C.shape[0], format="csr")
    rs = np.asarray(C.sum(1)).ravel()
    return sp.diags(1.0 / np.maximum(rs, EPS)) @ C


def read_flag_genes(spec):
    """'file.txt' (one gene per line) or 'file.gmt[:SET1,SET2]' -> lower-case set."""
    path, sets = (spec.split(".gmt:", 1)[0] + ".gmt", spec.split(".gmt:", 1)[1].split(",")) \
        if ".gmt:" in spec else (spec, None)
    out = set()
    with open(path) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if path.endswith(".gmt"):
                if sets is None or f[0] in sets:
                    out.update(x.strip().lower() for x in f[2:] if x.strip())
            elif f[0].strip() and not f[0].startswith("#"):
                out.add(f[0].strip().lower())
    return out


def cohen_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0:
        return np.nan
    po = np.mean(a == b)
    pe = sum(np.mean(a == k) * np.mean(b == k) for k in np.union1d(a, b))
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def boot_ci(correct, n_boot, rng, alpha=0.05):
    correct = np.asarray(correct, float)
    if len(correct) < 2 or n_boot <= 0:
        return np.nan, np.nan
    idx = rng.integers(0, len(correct), size=(n_boot, len(correct)))
    m = correct[idx].mean(1)
    return float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


# ---------------------------------------------------------------------------
# reference: boundaries and truth
# ---------------------------------------------------------------------------

def truth_matrix(ref, cfg_truth, raw_out):
    layer = cfg_truth.get("layer", "auto")
    transform = cfg_truth.get("transform", "auto")
    if layer == "auto":
        if "counts" in ref.layers:
            layer = "counts"
        elif "spliced" in ref.layers and "unspliced" in ref.layers:
            layer = "spliced+unspliced"
        elif "spliced" in ref.layers:
            layer = "spliced"
        else:
            layer = "X"
    if layer == "spliced+unspliced":
        R = ref.layers["spliced"] + ref.layers["unspliced"]
    elif layer == "X":
        R = ref.X
    else:
        R = ref.layers[layer]
    Rd = R.toarray() if sp.issparse(R) else np.asarray(R)
    is_counts = bool(np.all(np.isfinite(Rd[: min(500, Rd.shape[0])])) and
                     np.allclose(Rd[: min(500, Rd.shape[0])], np.round(Rd[: min(500, Rd.shape[0])])) and
                     np.nanmax(Rd) > 20)
    if transform == "auto":
        if is_counts:
            transform = "lognorm"
        elif np.nanmin(Rd) >= 0 and np.nanmax(Rd) > 50:
            transform = "log1p"       # e.g. scVelo's size-normalised spliced/unspliced layers
        else:
            transform = "none"
    raw_out["detect"] = (Rd > 0)
    if transform == "lognorm":
        X = lognorm(Rd)
    elif transform == "log1p":
        X = np.log1p(np.clip(Rd, 0, None)).astype(np.float32)
    else:
        X = Rd.astype(np.float32)
    return X, f"layer={layer}, transform={transform}"


def boundaries(clusters, C, edges, min_nb):
    out = OrderedDict()
    for u, v in edges:
        src = np.where(clusters == u)[0]
        S, T, nbrs = [], set(), []
        for i in src:
            nb = C.indices[C.indptr[i]:C.indptr[i + 1]]
            tn = nb[clusters[nb] == v]
            if len(tn) >= min_nb:
                S.append(i)
                nbrs.append(tn)
                T.update(tn.tolist())
        if S:
            out[(u, v)] = dict(S=np.array(S), T=np.array(sorted(T)), nbrs=nbrs)
        else:
            log(f"  WARNING: edge {u} -> {v}: no source cell with >= {min_nb} target neighbours; skipped")
    return out


# ---------------------------------------------------------------------------
# main per-dataset routine
# ---------------------------------------------------------------------------

def run_dataset(dname, job, cfg, out_root, rng):
    meths = job["methods"]
    ref_m = cfg.get("reference_method") or next(iter(meths))
    if ref_m not in meths:
        log(f"[{dname}] reference method '{ref_m}' has no entry for this dataset; using '{next(iter(meths))}'")
        ref_m = next(iter(meths))
    dcfg = (cfg.get("per_dataset") or {}).get(dname, {}) or {}
    ref_path, ref_p = meths[ref_m]
    ref_path = dcfg.get("reference_adata") or ref_path
    import anndata as ad
    log(f"[{dname}] reference: {ref_m} ({ref_path})")
    ref = ad.read_h5ad(ref_path)
    k_cluster = dcfg.get("k_cluster") or ref_p["k_cluster"]
    clusters = mapped_clusters(ref, k_cluster, ref_p["cluster_map"], ref_p["cluster_map_drop_unmapped"])
    edges = parse_edges(job["global"].get("cluster_edges") or ref_p["cluster_edges"])
    present = {c for c in clusters if isinstance(c, str)}
    for u, v in list(edges):
        if u not in present or v not in present:
            log(f"  WARNING: edge {u} -> {v} references labels absent from '{k_cluster}': "
                f"{[x for x in (u, v) if x not in present]}; skipped. Labels: {sorted(present)}")
            edges.remove((u, v))
    if not edges:
        log(f"[{dname}] no valid edges; skipped")
        return None
    if "neighbors" not in ref.uns:
        raise SystemExit(f"[{dname}] reference h5ad has no neighbour graph")
    ck = ref.uns["neighbors"].get("connectivities_key", "connectivities")
    C = sp.csr_matrix(ref.obsp[ck])
    min_nb = cfg.get("min_target_neighbors") or ref_p["min_target_neighbors"] or 3
    B = boundaries(clusters, C, edges, int(min_nb))
    min_src = int(cfg.get("truth", {}).get("min_source_cells", 10))
    for e in list(B):
        if len(B[e]["S"]) < min_src:
            log(f"  WARNING: edge {e[0]} -> {e[1]}: only {len(B[e]['S'])} boundary source cells (< {min_src}); skipped")
            B.pop(e)
    if not B:
        return None

    # restrict the reference to genes that at least one method has (names only, backed reads)
    ref_gcol = dcfg.get("reference_gene_names_col") or gene_col_for(cfg, dname, ref_m, ref_p)
    log(f"  reference gene names: {ref_gcol or 'var_names'}")
    ref_names_all = gene_names(ref, ref_gcol, f"reference ({ref_m})")
    universe = set()
    for m, (path, p) in meths.items():
        try:
            h = ad.read_h5ad(path, backed="r")
            universe |= set(gene_names(h, gene_col_for(cfg, dname, m, p)))
            h.file.close()
        except Exception as ex:                                                    # noqa: BLE001
            log(f"  {m}: could not read gene names ({ex})")
    keep_u = np.isin(ref_names_all, list(universe)) if universe else np.ones(ref.n_vars, bool)
    if keep_u.sum() < ref.n_vars:
        log(f"  reference genes: {ref.n_vars} -> {int(keep_u.sum())} present in at least one method")
        ref = ref[:, keep_u].copy()

    # truth
    tcfg = cfg.get("truth", {}) or {}
    raw = {}
    X, tdesc = truth_matrix(ref, tcfg, raw)
    if tcfg.get("smooth", True):
        X = (row_norm_graph(C) @ X).astype(np.float32)
    sd = X.std(0) + 1e-6
    ref_genes = gene_names(ref, ref_gcol)
    if pd.Index(ref_genes).has_duplicates:
        log(f"  WARNING: {pd.Index(ref_genes).duplicated().sum()} duplicated reference gene names; keeping first")
    keep_g = ~pd.Index(ref_genes).duplicated()
    X, sd, ref_genes, detect = X[:, keep_g], sd[keep_g], ref_genes[keep_g], raw["detect"][:, keep_g]
    G = len(ref_genes)
    log(f"  truth: {tdesc}, smoothed={tcfg.get('smooth', True)}; {ref.n_obs} cells x {G} genes; "
        f"{len(B)} edges; boundary source cells: "
        + ", ".join(f"{u}->{v}: {len(b['S'])}" for (u, v), b in B.items()))
    truth = {}
    scope = tcfg.get("scope", "cluster")
    for e, b in B.items():
        mS, mT = X[b["S"]].mean(0), X[b["T"]].mean(0)
        det = detect[np.r_[b["S"], b["T"]]].mean(0)
        dx = np.vstack([X[nb].mean(0) - X[i] for i, nb in zip(b["S"], b["nbrs"])])     # |S| x G
        cA, cB = X[clusters == e[0]].mean(0), X[clusters == e[1]].mean(0)
        d_b, d_c = (mT - mS) / sd, (cB - cA) / sd
        prim = d_c if scope == "cluster" else d_b
        truth[e] = dict(d=prim, d_boundary=d_b, d_cluster=d_c, det=det, mean_expr=(mS + mT) / 2, dx=dx)
    agree_t = np.mean([np.mean(np.sign(t_["d_boundary"]) == np.sign(t_["d_cluster"])) for t_ in truth.values()])
    log(f"  truth scope '{scope}'; boundary and cluster-mean directions agree for {agree_t:.1%} of genes")

    # flags
    flags = {}
    for fname, spec in (dcfg.get("flag_genes") or {}).items():
        try:
            if not os.path.isabs(spec):
                spec = os.path.join(cfg.get("_base", "."), spec)
            fs = read_flag_genes(spec)
            alt = gene_names(ref, None)[keep_g]
            flags[fname] = np.array([g.lower() in fs or a.lower() in fs for g, a in zip(ref_genes, alt)])
            log(f"  flag '{fname}': {int(flags[fname].sum())} genes from {spec}")
        except Exception as ex:                                                    # noqa: BLE001
            log(f"  WARNING: flag '{fname}' from {spec} failed: {ex}")

    # splice features (u/s) on the reference cells
    splice = splice_features(ref, ref_genes, keep_g, B, dcfg, meths, ref_gcol,
                             {m: gene_col_for(cfg, dname, m, p) for m, (_, p) in meths.items()})

    # methods
    per_m, own = OrderedDict(), OrderedDict()
    feat_cols = cfg.get("feature_cols") or {}
    var_feats = {}
    for m, (path, p) in meths.items():
        if m == ref_m and path == ref_path:
            a = ref
        else:
            if not os.path.exists(path):
                log(f"  {m}: h5ad not found ({path}); skipped")
                continue
            a = ad.read_h5ad(path)
        vkey = (cfg.get("vkey_override") or {}).get(m) or p["vkey"]
        if vkey not in a.layers:
            log(f"  {m}: no layer '{vkey}' (layers: {sorted(a.layers.keys())}); skipped")
            continue
        ci, crate = match_index(ref.obs_names.astype(str), a.obs_names.astype(str), "cells", m)
        gcol = gene_col_for(cfg, dname, m, p)
        mg = gene_names(a, gcol, m)
        gi, grate = match_index(ref_genes, mg, "genes", m)
        if (gi >= 0).sum() < 10:
            log(f"    {m}: example reference genes {list(ref_genes[:5])}; example {m} names "
                f"{list(mg[:5])}; available var columns: {list(a.var.columns)[:15]}")
        log(f"  {m}: vkey={vkey}, genes by {gcol or 'var_names'}; matched {crate:.1%} of reference cells, {int((gi >= 0).sum())}/{G} genes")
        if (gi >= 0).sum() < 10 or crate < 0.2:
            log(f"  {m}: too few matched cells/genes (check gene_names_col / barcodes); skipped")
            continue
        V = np.full((ref.n_obs, G), np.nan, dtype=np.float32)
        Va = dense(a.layers[vkey])
        rows, cols = np.where(ci >= 0)[0], np.where(gi >= 0)[0]
        V[np.ix_(rows, cols)] = Va[np.ix_(ci[rows], gi[cols])]
        del Va
        finite = np.zeros(G, bool)
        finite[cols] = np.isfinite(V[rows][:, cols]).all(0)
        q = own_query_for(cfg, m, p)
        own_m = finite.copy()
        if q:
            try:
                ok = a.var.index.isin(a.var.query(q).index)
                qm = np.zeros(G, bool)
                qm[cols] = ok[gi[cols]]
                own_m &= qm
                log(f"    own set: {int(finite.sum())} finite-velocity genes -> {int(own_m.sum())} after '{q}'")
            except Exception as ex:                                                # noqa: BLE001
                if "velocity_genes" in a.var.columns:
                    vg = np.zeros(G, bool)
                    vg[cols] = a.var["velocity_genes"].fillna(False).astype(bool).to_numpy()[gi[cols]]
                    own_m &= vg
                    log(f"    own set: gene query '{q}' does not evaluate on this h5ad ({ex}); using "
                        f"var['velocity_genes']: {int(own_m.sum())} genes")
                else:
                    log(f"    own set: gene query '{q}' does not evaluate on this h5ad ({ex}); "
                        f"using the {int(finite.sum())} finite-velocity genes")
        else:
            log(f"    own set: {int(finite.sum())} finite-velocity genes")
        wanted = []
        for pat in feat_cols.get(m, []) or []:
            wanted += [c for c in a.var.columns if fnmatch.fnmatch(c, pat)]
        for c in dict.fromkeys(wanted):
            col = np.full(G, np.nan)
            try:
                col[cols] = pd.to_numeric(a.var[c], errors="coerce").to_numpy()[gi[cols]]
                var_feats[f"{m}:{c}"] = col
            except Exception:                                                      # noqa: BLE001
                pass
        per_m[m] = dict(V=V, root=float(p.get("root_power") or 2), cell_rate=crate)
        own[m] = own_m
        if a is not ref:
            del a
    cmp_m = cfg.get("compare_to") or ("nSV" if "nSV" in per_m else ref_m)
    if cmp_m not in per_m:
        log(f"[{dname}] compare_to method '{cmp_m}' could not be scored; using '{ref_m}'")
        cmp_m = ref_m
    if cmp_m not in per_m:
        log(f"[{dname}] no method to compare against; skipped")
        return None
    log(f"  truth/cells/graph from {ref_m}; agreement tables are {cmp_m} vs each other method")
    shared = np.logical_and.reduce([own[m] for m in per_m])
    log(f"  shared set (own genes of all {len(per_m)} methods): {int(shared.sum())} genes")
    if shared.sum() < 10:
        log("  WARNING: fewer than 10 shared genes; check gene_names_col across methods")

    # long table
    recs = []
    sets_of = {}
    for m in per_m:
        sets_of[m] = {"own": own[m], "shared": shared, "pair": own[m] & own[cmp_m]}
    for m, mm in per_m.items():
        V, root = mm["V"], mm["root"]
        for e, b in B.items():
            S = b["S"]
            Vs = V[S]
            tv = np.sign(Vs) * np.abs(Vs) ** (1.0 / root)
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(Vs, 0)
            pred = np.sign(np.nan_to_num(med, nan=0.0))
            td = np.sign(truth[e]["d"])
            sgn = np.sign(Vs)
            valid = np.isfinite(Vs) & (sgn != 0)
            agree = np.where(valid.sum(0) > 0,
                             ((sgn == td[None, :]) & valid).sum(0) / np.maximum(valid.sum(0), 1), np.nan)
            contrib = {}
            for sname, mask in sets_of[m].items():
                c = np.full(G, np.nan)
                if mask.sum() >= 2:
                    dx = truth[e]["dx"][:, mask]
                    vv = np.nan_to_num(tv[:, mask])
                    nd = np.linalg.norm(dx, axis=1)
                    nv = np.linalg.norm(vv, axis=1)
                    okc = (nd > 0) & (nv > 0)
                    c[mask] = (dx[okc] * vv[okc] / (nd[okc] * nv[okc])[:, None]).mean(0) if okc.any() else np.nan
                contrib[sname] = c
            for g in np.where(own[m] | shared)[0]:
                r = dict(dataset=dname, edge=f"{e[0]} -> {e[1]}", source=e[0], target=e[1],
                         gene=ref_genes[g], method=m, is_comparator=(m == cmp_m),
                         d=truth[e]["d"][g], d_boundary=truth[e]["d_boundary"][g],
                         d_cluster=truth[e]["d_cluster"][g], detect_frac=truth[e]["det"][g],
                         mean_expr=truth[e]["mean_expr"][g], n_source_cells=len(S),
                         median_velocity=med[g], pred_sign=pred[g], frac_cells_agree=agree[g],
                         in_own=bool(own[m][g]), in_shared=bool(shared[g]),
                         in_pair=bool(sets_of[m]["pair"][g]),
                         contribution_own=contrib["own"][g], contribution_shared=contrib["shared"][g],
                         contribution_pair=contrib["pair"][g])
                for fname, fl in flags.items():
                    r[f"flag_{fname}"] = bool(fl[g])
                for k, arr in splice.get(e, {}).items():
                    r[k] = arr[g]
                for k, arr in var_feats.items():
                    r[k] = arr[g]
                recs.append(r)
    L = pd.DataFrame.from_records(recs)
    out = os.path.join(out_root, dname)
    os.makedirs(out, exist_ok=True)
    L.to_csv(os.path.join(out, "gd_long.csv"), index=False)
    log(f"  wrote {len(L)} rows to {out}/gd_long.csv")
    return L, cmp_m, list(B.keys()), out


def splice_features(ref, ref_genes, keep_g, B, dcfg, meths, ref_gcol, gcols):
    """unspliced fraction and the change of log2(u/s) across each edge, on reference cells."""
    import anndata as ad
    src = dcfg.get("splice_adata")
    a, gcol = None, None
    if src:
        a = ad.read_h5ad(src)
        gcol = dcfg.get("splice_gene_names_col")
    elif "spliced" in ref.layers and "unspliced" in ref.layers:
        a, gcol = ref, ref_gcol
    else:
        for m, (path, p) in meths.items():
            try:
                h = ad.read_h5ad(path, backed="r")
                ok = "spliced" in h.layers and "unspliced" in h.layers
                h.file.close()
            except Exception:                                                      # noqa: BLE001
                ok = False
            if ok:
                a, gcol = ad.read_h5ad(path), gcols.get(m)
                log(f"  splice features from {m} ({path})")
                break
    if a is None:
        log("  no spliced/unspliced layers found; splice features skipped")
        return {}
    ci, crate = match_index(ref.obs_names.astype(str), a.obs_names.astype(str), "cells")
    gi, _ = match_index(ref_genes, gene_names(a, gcol), "genes")
    G = len(ref_genes)
    cols = np.where(gi >= 0)[0]
    Sx, Ux = dense(a.layers["spliced"]), dense(a.layers["unspliced"])
    tot = Sx.sum(1) + Ux.sum(1)
    sf = np.median(tot[tot > 0]) / np.maximum(tot, EPS)
    feats = {}
    for e, b in B.items():
        def mean_of(M, idx):
            rr = ci[idx]
            rr = rr[rr >= 0]
            out = np.full(G, np.nan)
            if len(rr):
                out[cols] = (M[rr][:, gi[cols]] * sf[rr, None]).mean(0)
            return out
        uS, sS = mean_of(Ux, b["S"]), mean_of(Sx, b["S"])
        uT, sT = mean_of(Ux, b["T"]), mean_of(Sx, b["T"])
        ps = 0.01
        feats[e] = {"unspliced_frac": (uS + uT) / np.maximum(uS + uT + sS + sT, EPS),
                    "log2_us_ratio_change": np.log2((uT + ps) / (sT + ps)) - np.log2((uS + ps) / (sS + ps)),
                    "log2_unspliced_change": np.log2((uT + ps) / (uS + ps)),
                    "log2_spliced_change": np.log2((sT + ps) / (sS + ps))}
    log(f"  splice features: matched {crate:.1%} of cells, {len(cols)} genes")
    return feats


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------

def scored_rows(L, cfg):
    """Rows eligible for scoring: detected, not flagged (if excluded); with
    require_consistent_truth, genes whose boundary and cluster-mean directions disagree
    get d = 0, i.e. no true direction."""
    t = cfg.get("truth", {}) or {}
    base = L[L.detect_frac >= float(t.get("min_detect_frac", 0.05))].copy()
    if cfg.get("exclude_flagged", False):
        for c in [c for c in L.columns if c.startswith("flag_")]:
            base = base[~base[c].astype(bool)]
    if t.get("require_consistent_truth", True):
        bad = np.sign(base.d_boundary) != np.sign(base.d_cluster)
        base.loc[bad, "d"] = 0.0
    return base


def summarise(L, ref_m, edges, cfg, rng):
    d_sweep = cfg.get("truth", {}).get("d_sweep", [0.1, 0.25, 0.5, 1.0])
    n_boot = int(cfg.get("n_boot", 1000))
    base = scored_rows(L, cfg)
    rows = []
    edge_labels = [f"{u} -> {v}" for u, v in edges] + ["ALL"]
    for gset in ("own", "shared", "pair"):
        sub_g = base[base[f"in_{gset}"]]
        for m in sub_g.method.unique():
            Lm = sub_g[sub_g.method == m]
            for el in edge_labels:
                Le = Lm if el == "ALL" else Lm[Lm.edge == el]
                if Le.empty:
                    continue
                Lfull = L[(L.method == m) & L[f"in_{gset}"]]
                Lfull = Lfull if el == "ALL" else Lfull[Lfull.edge == el]
                c = Lfull[f"contribution_{gset}"].to_numpy()
                for dmin in d_sweep:
                    lab = Le[np.abs(Le.d) >= dmin]
                    ts = np.sign(lab.d.to_numpy())
                    called = lab.pred_sign.to_numpy() != 0
                    corr = (lab.pred_sign.to_numpy() == ts)
                    lo, hi = boot_ci(corr[called], n_boot, rng)
                    w = np.abs(lab.d.to_numpy())
                    cl = lab[f"contribution_{gset}"].to_numpy()
                    rows.append(dict(
                        edge=el, method=m, gene_set=gset, d_min=dmin,
                        n_genes_in_set=int(Le.gene.nunique()), n_with_direction=len(lab),
                        n_called=int(called.sum()), abstain_rate=1 - called.mean() if len(lab) else np.nan,
                        accuracy=corr[called].mean() if called.any() else np.nan, ci_low=lo, ci_high=hi,
                        accuracy_weighted_by_effect=(w[called] * corr[called]).sum() / max(w[called].sum(), EPS)
                        if called.any() else np.nan,
                        mean_frac_cells_agree=np.nanmean(lab.frac_cells_agree) if len(lab) else np.nan,
                        majority_baseline=max(np.mean(ts > 0), np.mean(ts < 0)) if len(lab) else np.nan,
                        gene_space_cbdir=np.nansum(c) if el != "ALL" else np.nan,
                        forward_share=np.nansum(np.clip(c, 0, None)) / max(np.nansum(np.abs(c)), EPS),
                        signal_share_correct_genes=np.nansum(np.abs(cl[corr])) / max(np.nansum(np.abs(cl)), EPS)
                        if len(lab) else np.nan))
    S = pd.DataFrame(rows)

    # agreement: reference vs each other method on the pair set
    dprim = float(cfg.get("truth", {}).get("d_min", 0.25))
    A, F = [], []
    ref_t = base[(base.method == ref_m)].set_index(["edge", "gene"])
    for m in [x for x in base.method.unique() if x != ref_m]:
        oth = base[(base.method == m) & base.in_pair].set_index(["edge", "gene"])
        J = ref_t[["pred_sign", "d"]].join(oth[["pred_sign"]], rsuffix="_other", how="inner")
        J = J[(J.pred_sign != 0) & (J.pred_sign_other != 0)]
        for el in edge_labels:
            Je = J if el == "ALL" else J[J.index.get_level_values(0) == el]
            if Je.empty:
                continue
            ts = np.sign(Je.d.to_numpy())
            lab = np.abs(Je.d.to_numpy()) >= dprim
            rc = (Je.pred_sign.to_numpy() == ts)
            oc = (Je.pred_sign_other.to_numpy() == ts)
            n = lab.sum()
            A.append(dict(edge=el, comparator=ref_m, other=m, n_both_called=len(Je),
                          sign_agreement=np.mean(Je.pred_sign.to_numpy() == Je.pred_sign_other.to_numpy()),
                          cohen_kappa=cohen_kappa(Je.pred_sign.to_numpy(), Je.pred_sign_other.to_numpy()),
                          n_with_direction=int(n),
                          both_correct=np.mean(rc[lab] & oc[lab]) if n else np.nan,
                          comparator_only_correct=np.mean(rc[lab] & ~oc[lab]) if n else np.nan,
                          other_only_correct=np.mean(~rc[lab] & oc[lab]) if n else np.nan,
                          both_wrong=np.mean(~rc[lab] & ~oc[lab]) if n else np.nan))
        # features by class (pooled over edges)
        lab = np.abs(J.d.to_numpy()) >= dprim
        Jl = J[lab].copy()
        ts = np.sign(Jl.d.to_numpy())
        rc, oc = Jl.pred_sign.to_numpy() == ts, Jl.pred_sign_other.to_numpy() == ts
        Jl["class"] = np.select([rc & oc, rc & ~oc, ~rc & oc], ["both_correct", "comparator_only", "other_only"],
                                "both_wrong")
        feats = feature_columns(base)
        R = base[base.method == ref_m].set_index(["edge", "gene"])[feats]
        Jl = Jl.join(R, how="left")
        Jl["abs_d"] = np.abs(Jl.d)
        g = Jl.groupby("class")[["abs_d"] + feats].median()
        g.insert(0, "n", Jl.groupby("class").size())
        g.insert(0, "other", m)
        F.append(g.reset_index())
    A = pd.DataFrame(A)
    F = pd.concat(F, ignore_index=True) if F else pd.DataFrame()
    T = feature_tests(base, ref_m, dprim)
    return S, A, F, T


def feature_columns(L):
    skip = {"d", "d_boundary", "d_cluster", "pred_sign", "median_velocity", "frac_cells_agree", "n_source_cells",
            "contribution_own", "contribution_shared", "contribution_pair"}
    cols = []
    for c in L.columns:
        if c in skip or c.startswith(("in_", "flag_", "is_")):
            continue
        if pd.api.types.is_numeric_dtype(L[c]) and not pd.api.types.is_bool_dtype(L[c]):
            cols.append(c)
    return cols


def feature_tests(base, ref_m, dprim):
    """What separates correct from wrong reference calls (own set, pooled over edges)."""
    from scipy.stats import mannwhitneyu
    R = base[(base.method == ref_m) & base.in_own & (np.abs(base.d) >= dprim) & (base.pred_sign != 0)].copy()
    if R.empty:
        return pd.DataFrame()
    R["correct"] = R.pred_sign == np.sign(R.d)
    R["abs_d"] = np.abs(R.d)
    R["log_mean_expr"] = np.log1p(np.clip(R.mean_expr, 0, None))
    feats = [c for c in ["abs_d", "log_mean_expr"] + feature_columns(R) if c not in ("mean_expr",)]
    feats = list(dict.fromkeys(f for f in feats if f != "correct"))
    rows = []
    for f in feats:
        x = pd.to_numeric(R[f], errors="coerce")
        a, b = x[R.correct].dropna(), x[~R.correct].dropna()
        if len(a) >= 5 and len(b) >= 5:
            p = mannwhitneyu(a, b).pvalue
            rows.append(dict(feature=f, test="mann_whitney", median_correct=a.median(), median_wrong=b.median(),
                             n_correct=len(a), n_wrong=len(b), p_value=p, coef=np.nan))
    try:
        from sklearn.linear_model import LogisticRegression
        use = [f for f in feats if R[f].notna().mean() > 0.8 and R[f].nunique() > 2]
        Z = R[use].apply(pd.to_numeric, errors="coerce")
        ok = Z.notna().all(1)
        Z, y = Z[ok], R.correct[ok]
        if len(y) >= 30 and y.nunique() == 2 and use:
            Zs = (Z - Z.mean()) / (Z.std() + EPS)
            lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Zs, y)
            for f, cf in zip(use, lr.coef_[0]):
                rows.append(dict(feature=f, test="logistic_standardised", coef=cf, n_correct=int(y.sum()),
                                 n_wrong=int((~y).sum())))
    except Exception as ex:                                                        # noqa: BLE001
        log(f"  logistic regression skipped: {ex}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def figures(L, S, A, ref_m, dname, out, cfg):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as ex:                                                        # noqa: BLE001
        log(f"  figures skipped: {ex}")
        return
    dprim = float(cfg.get("truth", {}).get("d_min", 0.25))
    bins = cfg.get("effect_bins", [0.1, 0.25, 0.5, 1.0, 2.0, np.inf])
    base = scored_rows(L, cfg)
    base = base[base.pred_sign != 0]
    meths = list(dict.fromkeys(L.method))
    # 1. accuracy vs effect size (own set)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for k, gset in enumerate(("own", "shared")):
        for m in meths:
            x = base[(base.method == m) & base[f"in_{gset}"]]
            ad_ = np.abs(x.d)
            acc, ns = [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                s = x[(ad_ >= lo) & (ad_ < hi)]
                acc.append((s.pred_sign == np.sign(s.d)).mean() if len(s) >= 5 else np.nan)
                ns.append(len(s))
            lw = 2.5 if m == ref_m else 1.2
            ax[k].plot(range(len(acc)), acc, marker="o", lw=lw, label=m)
        ax[k].axhline(0.5, color="grey", ls=":", lw=1)
        ax[k].set_xticks(range(len(bins) - 1))
        ax[k].set_xticklabels([f"{lo:g}-{hi:g}" for lo, hi in zip(bins[:-1], bins[1:])], rotation=30)
        ax[k].set_xlabel("|d| (expression change across the edge, SD units)")
        ax[k].set_ylabel("fraction of genes with correct sign")
        ax[k].set_title(f"{dname}: {gset} genes, pooled over edges")
        ax[k].set_ylim(0, 1)
    ax[1].legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"acc_vs_effect_{dname}.png"), dpi=140)
    plt.close(fig)
    # 2. heatmap method x edge (shared set, primary d_min)
    H = S[(S.gene_set == "shared") & (np.isclose(S.d_min, dprim))].pivot(index="method", columns="edge",
                                                                         values="accuracy")
    if not H.empty:
        cols = [c for c in H.columns if c != "ALL"] + (["ALL"] if "ALL" in H.columns else [])
        H = H.reindex(index=[m for m in meths if m in H.index], columns=cols)
        fig, ax = plt.subplots(figsize=(1.1 * H.shape[1] + 3, 0.45 * H.shape[0] + 1.8))
        im = ax.imshow(H.to_numpy(float), cmap="RdBu", vmin=0, vmax=1, aspect="auto")
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                v = H.iat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
        ax.set_xticks(range(H.shape[1]))
        ax.set_xticklabels(H.columns, rotation=40, ha="right", fontsize=7)
        ax.set_yticks(range(H.shape[0]))
        ax.set_yticklabels(H.index, fontsize=8)
        ax.set_title(f"{dname}: fraction of shared genes with correct sign (|d| >= {dprim:g})", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.03)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"acc_heatmap_{dname}.png"), dpi=140)
        plt.close(fig)
    # 3. agreement classes
    if A is not None and not A.empty:
        others = list(dict.fromkeys(A.other))
        fig, axes = plt.subplots(len(others), 1, figsize=(max(6, 0.9 * A.edge.nunique() + 3), 2.3 * len(others)),
                                 squeeze=False)
        cls = ["both_correct", "comparator_only_correct", "other_only_correct", "both_wrong"]
        colors = ["#4c9a6a", "#4a78b5", "#d9893b", "#b8433f"]
        for ax, m in zip(axes[:, 0], others):
            sub = A[A.other == m].set_index("edge")
            order = [e for e in sub.index if e != "ALL"] + (["ALL"] if "ALL" in sub.index else [])
            sub = sub.reindex(order)
            bottom = np.zeros(len(sub))
            for c, col in zip(cls, colors):
                v = sub[c].fillna(0).to_numpy()
                ax.bar(range(len(sub)), v, bottom=bottom, color=col,
                       label=c.replace("comparator", ref_m).replace("other", m).replace("_", " "))
                bottom += v
            ax.set_xticks(range(len(sub)))
            ax.set_xticklabels(sub.index, rotation=30, ha="right", fontsize=7)
            ax.set_ylim(0, 1)
            ax.set_ylabel("fraction of genes")
            ax.set_title(f"{ref_m} vs {m} (kappa ALL = {sub.cohen_kappa.get('ALL', np.nan):.2f})", fontsize=9)
            ax.legend(fontsize=6, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"agreement_{dname}.png"), dpi=140)
        plt.close(fig)


# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    a = ap.parse_args(argv)
    cfg = read_yaml(a.config)
    base = os.path.dirname(os.path.abspath(a.config))
    cfg["_base"] = base
    for d in (cfg.get("per_dataset") or {}).values():
        for k in ("reference_adata", "splice_adata"):
            if d and d.get(k) and not os.path.isabs(d[k]):
                d[k] = os.path.join(base, d[k])
    gpath = cfg["cbdir_global_config"]
    gpath = gpath if os.path.isabs(gpath) else os.path.join(base, gpath)
    out_root = cfg.get("out_dir", "./results/gene_direction")
    out_root = out_root if os.path.isabs(out_root) else os.path.join(base, out_root)
    os.makedirs(out_root, exist_ok=True)
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    jobs = resolve_jobs(gpath, a.methods or cfg.get("methods"), a.datasets or cfg.get("datasets"))
    log(f"{len(jobs)} dataset(s): {list(jobs)}")
    allS, allA = [], []
    for dname, job in jobs.items():
        try:
            r = run_dataset(dname, job, cfg, out_root, rng)
        except Exception as ex:                                                    # noqa: BLE001
            import traceback
            log(f"[{dname}] FAILED: {ex}")
            traceback.print_exc()
            continue
        if r is None:
            continue
        L, ref_m, edges, out = r
        S, A, F, T = summarise(L, ref_m, edges, cfg, rng)
        S.insert(0, "dataset", dname)
        S.to_csv(os.path.join(out, "gd_summary.csv"), index=False)
        if not A.empty:
            A.insert(0, "dataset", dname)
            A.to_csv(os.path.join(out, "gd_agreement.csv"), index=False)
        if not F.empty:
            F.insert(0, "dataset", dname)
            F.to_csv(os.path.join(out, "gd_features.csv"), index=False)
        if not T.empty:
            T.insert(0, "dataset", dname)
            T.to_csv(os.path.join(out, "gd_feature_tests.csv"), index=False)
        figures(L, S, A, ref_m, dname, out, cfg)
        dprim = float(cfg.get("truth", {}).get("d_min", 0.25))
        P = S[(S.edge == "ALL") & np.isclose(S.d_min, dprim)].pivot(index="method", columns="gene_set",
                                                                    values="accuracy")
        with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200):
            log(f"[{dname}] accuracy pooled over edges (|d| >= {dprim:g}):\n{P}")
            if not A.empty:
                log(f"[{dname}] agreement with {ref_m} (ALL edges; comparator = {ref_m}):\n"
                    f"{A[A.edge == 'ALL'].drop(columns=['dataset', 'edge', 'comparator']).to_string(index=False)}")
        allS.append(S)
        if not A.empty:
            allA.append(A)
    if allS:
        pd.concat(allS).to_csv(os.path.join(out_root, "all_datasets_summary.csv"), index=False)
    if allA:
        pd.concat(allA).to_csv(os.path.join(out_root, "all_datasets_agreement.csv"), index=False)
    log(f"done; outputs under {out_root}")


if __name__ == "__main__":
    sys.exit(main())
