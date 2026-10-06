"""
make_capture_bias_datasets -- capture-efficiency robustness experiment.

Builds one input AnnData per condition by BINOMIAL THINNING of the raw counts,
X'_ig ~ Binomial(X_ig, f_i), which is exactly what a cell-specific capture
efficiency f_i does to molecule counts (each molecule captured independently).
Every condition has the SAME mean f, so conditions differ only in how capture
is distributed across cells, not in how much data is lost.

Why this can bias noSpliceVelo: after rescaling by f,
    E[X'/f] = mu,    Var[X'/f] = sigma^2 + (1 - f)/f * mu
so low-capture cells gain variance (1/f - 1) mu at the same mean -- nSV's
signature of UPREGULATION. A pseudotime-ordered capture gradient can therefore
flip late, down-regulating cells to "up". The first VAE (SCVIModified) infers a
per-cell capture latent, so its likelihood should absorb this term; the
experiment tests whether it does.

Conditions (all at mean f = f_bar, default = mean of the reviewer's rule):
  reviewer_s{k}          f = 1/(1+sqrt(pt))/2.5, pt = ORIGINAL nSV pseudotime
                         (pseudotime-dependent capture; f from ~0.37 to ~0.21)
  uniform_s{k}           f = f_bar for every cell   (depth loss, no bias: CONTROL)
  gradient_down_r{R}     log f linear in pt, ratio R between pt=0 and pt=1,
                         capture FALLING along the lineage (same direction as reviewer_*)
  gradient_up_r{R}       same, capture RISING along the lineage (sign control)
  gradient_down_r4_lineage  as gradient_down_r4 but ordered by cluster lineage
                         rank (Ductal < Ngn3 low < Ngn3 high < Pre-endocrine <
                         endocrine) -- independent of nSV's own pseudotime
  ngn3high_half          f_bar everywhere except Ngn3 high EP at f_bar/2
                         (targets the population most sensitive to reversal)
  cluster_random_s{k}    per-cluster log-normal f, sd 0.5 on the log scale
                         (cell-type-specific permeabilisation)
  cell_random            per-cell log-normal f, sd 0.5 (cell size; state-independent)

  --set minimal : reviewer_s0, reviewer_s1, uniform_s0, uniform_s1,
                  gradient_down_r4, gradient_up_r4, ngn3high_half
  --set full    : everything above (17 conditions)

What is written per condition, under <out_root>/<condition>/ :
  adata_pan.h5ad   thinned count layers; X = log1p(normalize_total(counts));
                   derived layers (naive/scVI moments, ...) DROPPED so the
                   pipeline recomputes them from the thinned counts;
                   obs['capture_bias_f'] (planted f), obs['capture_bias_condition'];
                   obsm['X_pca_orig'], obsm['X_umap_orig'] (ORIGINAL embedding, so
                   CBDir compares velocity fields on identical geometry);
                   var['orig_reliable_velo_gene'] (original gene set, for the
                   primary fixed-gene-set analysis)
And under <out_root>/ :
  manifest.csv, nsv_config_capture_bias.yaml, compute_velo_config_capture_bias.yaml,
  score_gene_fit_v3_config_capture_bias.yaml, velo_stream_config_capture_bias.yaml,
  cbdir_capture_bias/ (one CBDir method config per condition + global config)

Usage (from nsv_runs/; run_capture_bias.sh does this and launches the training):
  python ../analyses/capture_bias/make_capture_bias_datasets.py \\
      --adata ../data/Pancreas/adata_pan.h5ad \\
      --reference ../data/Pancreas_studentT_time/nosplicevelo_score_gene_fit_v3_4throot_type1_vote/adata_nosplicevelo_stream.h5ad \\
      --out_root ../data/Pancreas_capture_bias --set minimal

  After the pipeline has run, if the embedding did not survive into the stream
  outputs:  python make_capture_bias_datasets.py --attach --out_root ... --reference ...
"""

import os
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp

PANCREAS_LINEAGE = {"Ductal": 0, "Ngn3 low EP": 1, "Ngn3 high EP": 2, "Pre-endocrine": 3,
                    "Alpha": 4, "Beta": 4, "Delta": 4, "Epsilon": 4}
PANCREAS_EDGES = [["Ngn3 low EP", "Ngn3 high EP"], ["Ngn3 high EP", "Pre-endocrine"],
                  ["Pre-endocrine", "Delta"], ["Pre-endocrine", "Beta"],
                  ["Pre-endocrine", "Epsilon"], ["Pre-endocrine", "Alpha"]]
COUNT_LAYERS = ("counts", "spliced", "unspliced")
MINIMAL = ["reviewer_s0", "reviewer_s1", "uniform_s0", "uniform_s1",
           "gradient_down_r4", "gradient_up_r4", "ngn3high_half"]
FULL = ["reviewer_s0", "reviewer_s1", "reviewer_s2", "uniform_s0", "uniform_s1", "uniform_s2",
        "gradient_down_r2", "gradient_down_r4", "gradient_up_r2", "gradient_up_r4",
        "gradient_down_r4_lineage", "ngn3high_half", "cluster_random_s0", "cluster_random_s1",
        "cell_random"]


# ---------------------------------------------------------------------------
# capture-efficiency vectors
# ---------------------------------------------------------------------------

def reviewer_f(pt):
    return 1.0 / (1.0 + np.sqrt(np.clip(pt, 0, 1))) / 2.5


def _rescale(f, f_bar, lo=0.01, hi=1.0):
    """Scale f to mean f_bar (iterating because of the clip)."""
    f = np.asarray(f, float)
    for _ in range(50):
        f = np.clip(f * f_bar / f.mean(), lo, hi)
        if abs(f.mean() - f_bar) < 1e-6:
            break
    return f


def lineage_order(clusters, rng, lineage=PANCREAS_LINEAGE):
    """Pseudotime-like ordering from cluster lineage rank only (nSV-independent):
    rank / max rank, jittered uniformly within each rank so gradients are smooth."""
    r = np.array([lineage.get(str(c), np.nan) for c in clusters], float)
    if np.isnan(r).any():
        bad = sorted({str(c) for c, x in zip(clusters, r) if np.isnan(x)})
        raise ValueError(f"clusters missing from the lineage map: {bad}")
    top = np.nanmax(r)
    return np.clip((r + rng.uniform(0, 1, r.size)) / (top + 1), 0, 1)


def capture_vector(condition, pt, clusters, f_bar, seed=0):
    rng = np.random.default_rng(seed + 12345)
    name = condition
    if name.startswith("reviewer"):
        return reviewer_f(pt)                                      # the reviewer's own rule
    if name.startswith("uniform"):
        return np.full(pt.size, f_bar)
    if name.startswith("gradient"):
        parts = name.split("_")                                    # gradient_down_r4[_lineage]
        direction, R = parts[1], float(parts[2][1:])
        order = lineage_order(clusters, rng) if name.endswith("_lineage") else pt
        s = 1.0 if direction == "down" else -1.0
        return _rescale(np.exp(-s * np.log(R) * (order - 0.5)), f_bar)
    if name == "ngn3high_half":
        f = np.full(pt.size, 1.0)
        f[np.asarray(clusters).astype(str) == "Ngn3 high EP"] = 0.5
        return _rescale(f, f_bar)
    if name.startswith("cluster_random"):
        cl = np.asarray(clusters).astype(str)
        lev = {c: np.exp(rng.normal(0, 0.5)) for c in sorted(set(cl))}
        return _rescale(np.array([lev[c] for c in cl]), f_bar)
    if name == "cell_random":
        return _rescale(np.exp(rng.normal(0, 0.5, pt.size)), f_bar)
    raise ValueError(f"unknown condition {condition!r}")


def condition_seed(condition):
    tail = condition.rsplit("_s", 1)
    return int(tail[1]) if len(tail) == 2 and tail[1].isdigit() else 0


# ---------------------------------------------------------------------------
# thinning
# ---------------------------------------------------------------------------

def thin(X, f, rng):
    """Binomial thinning of a (cells x genes) count matrix by per-cell f."""
    if sp.issparse(X):
        X = sp.csr_matrix(X, copy=True)
        rows = np.repeat(np.arange(X.shape[0]), np.diff(X.indptr))
        vals = np.rint(X.data).astype(np.int64)
        X.data = rng.binomial(vals, f[rows]).astype(np.float32)
        X.eliminate_zeros()
        return X
    X = np.rint(np.asarray(X)).astype(np.int64)
    return rng.binomial(X, f[:, None]).astype(np.float32)


def build_condition(adata, condition, pt, clusters, f_bar, ref_genes=None, ref_obsm=None):
    import anndata as ad
    seed = condition_seed(condition)
    rng = np.random.default_rng(1000 + 17 * seed + sum(map(ord, condition)))
    f = capture_vector(condition, pt, clusters, f_bar, seed=seed)
    layers = {k: adata.layers[k] for k in COUNT_LAYERS if k in adata.layers}
    if "counts" not in layers:
        raise KeyError("input AnnData has no 'counts' layer")
    su_sum = ("spliced" in layers and "unspliced" in layers and
              _same(layers["counts"], layers["spliced"] + layers["unspliced"]))
    new = {}
    if su_sum:
        # thin spliced and unspliced molecules independently; counts = their sum
        new["spliced"] = thin(layers["spliced"], f, rng)
        new["unspliced"] = thin(layers["unspliced"], f, rng)
        new["counts"] = new["spliced"] + new["unspliced"]
    else:
        for k, X in layers.items():
            new[k] = thin(X, f, rng)
    out = ad.AnnData(X=new["counts"].copy(), obs=adata.obs.copy(), var=adata.var.copy())
    for k, X in new.items():
        out.layers[k] = X
    # X = log1p(normalize_total(counts)) so any PCA the pipeline runs sees thinned data
    tot = np.asarray(out.layers["counts"].sum(1)).ravel()
    scale = np.median(tot[tot > 0]) / np.maximum(tot, 1)
    Xn = sp.diags(scale) @ sp.csr_matrix(out.layers["counts"])
    Xn.data = np.log1p(Xn.data)
    out.X = Xn.astype(np.float32)
    out.obs["capture_bias_f"] = f
    out.obs["capture_bias_condition"] = condition
    out.obs["total_counts"] = tot
    out.obs["n_genes_by_counts"] = np.asarray((out.layers["counts"] > 0).sum(1)).ravel()
    for key in ("clusters_colors",):
        if key in adata.uns:
            out.uns[key] = adata.uns[key]
    emb = ref_obsm if ref_obsm is not None else {k: adata.obsm[k] for k in ("X_pca", "X_umap") if k in adata.obsm}
    for k, v in emb.items():
        out.obsm[f"{k}_orig"] = np.asarray(v)
    if ref_genes is not None:
        out.var["orig_reliable_velo_gene"] = out.var_names.isin(ref_genes)
    return out, f


def _same(A, B):
    A = sp.csr_matrix(A); B = sp.csr_matrix(B)
    return A.shape == B.shape and (A != B).nnz == 0


# ---------------------------------------------------------------------------
# configs
# ---------------------------------------------------------------------------

def write_configs(out_root, conditions, adata_rel, cluster_key, cbdir_vkey, cbdir_xkey):
    import yaml
    ds = [dict(name=f"pancreas_{c}", adata_path=os.path.join(out_root, c, "adata_pan.h5ad"),
               dir_path=os.path.join(out_root, c)) for c in conditions]
    with open(os.path.join(out_root, "nsv_config_capture_bias.yaml"), "w") as fh:
        yaml.safe_dump(dict(log_file="./logs/nsv_pipeline_capture_bias.log",
                            defaults=dict(continue_from_prev=False, batch_size=256),
                            datasets=ds), fh, sort_keys=False)
    with open(os.path.join(out_root, "compute_velo_config_capture_bias.yaml"), "w") as fh:
        yaml.safe_dump(dict(log_file="./logs/compute_velo_capture_bias.log",
                            defaults=dict(state_reduction="both"),
                            datasets=[dict(name=d["name"], dir_path=d["dir_path"]) for d in ds]),
                       fh, sort_keys=False)
    with open(os.path.join(out_root, "score_gene_fit_v3_config_capture_bias.yaml"), "w") as fh:
        yaml.safe_dump(dict(log_file="./logs/score_gene_fit_v3_capture_bias.log",
                            defaults=dict(branch_min_cells=100, write_h5ad=True, n_jobs=6,
                                          thresh_mu_meannull=0.01, thresh_var_meannull=0.01),
                            datasets=[dict(name=d["name"], dir_path=d["dir_path"],
                                           out_dir=os.path.join(d["dir_path"], "nosplicevelo_score_gene_fit_v3"),
                                           prob_state=os.path.join(d["dir_path"], "prob_state_avg_nosplicevelo.npy"))
                                      for d in ds]), fh, sort_keys=False)
    # velocity streams on the ORIGINAL gene set (primary analysis: isolates capture)
    with open(os.path.join(out_root, "velo_stream_config_capture_bias.yaml"), "w") as fh:
        yaml.safe_dump(dict(
            log_file="./logs/velo_stream_capture_bias.log",
            defaults=dict(xkey="mu_scvi_smooth", vkey="velocity_mu_soft", basis="umap_orig",
                          n_neighbors=30, rep="X_pca", label_col=cluster_key, compute_pseudotime=True,
                          save_adata=True, output_filename="adata_nosplicevelo_stream.h5ad",
                          gene_query="orig_reliable_velo_gene == True", quant_filter_col=None,
                          filter_by_latent_time=False, sqrt_transform_func="x**(1/4)"),
            datasets=[dict(name=d["name"],
                           adata_path=os.path.join(d["dir_path"], "nosplicevelo_score_gene_fit_v3",
                                                   "adata_nosplicevelo.h5ad"),
                           dir_path=os.path.join(d["dir_path"], "stream_orig_genes")) for d in ds]),
            fh, sort_keys=False)
    # CBDir: one "method" per condition, all on the ORIGINAL PCA
    cdir = os.path.join(out_root, "cbdir_capture_bias"); os.makedirs(cdir, exist_ok=True)
    methods = {}
    for c, d in zip(conditions, ds):
        p = os.path.join(cdir, f"cbdir_{c}.yaml"); methods[c] = p
        yaml.safe_dump(dict(defaults=dict(k_cluster=cluster_key, vkey=cbdir_vkey, xkey=cbdir_xkey,
                                          basis="pca_orig", x_source="existing",
                                          # use the stream step's own graph (4th-root
                                          # transform); recomputing would switch to
                                          # scVelo's default sqrt transform
                                          recompute_velocity_graph=False),
                            datasets=[dict(name="mouse_pancreas",
                                           adata_path=os.path.join(d["dir_path"], "stream_orig_genes",
                                                                   "adata_nosplicevelo_stream.h5ad"),
                                           dir_path=os.path.join(out_root, "cbdir_results"))]),
                       open(p, "w"), sort_keys=False)
    yaml.safe_dump(dict(log_file="./logs/cbdir_capture_bias.log", str_suffix="_capture_bias",
                        methods=methods,
                        datasets={"mouse_pancreas": {"cluster_edges": PANCREAS_EDGES}}),
                   open(os.path.join(cdir, "cbdir_global_capture_bias.yaml"), "w"), sort_keys=False)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _read_reference(path, pt_key, n_expected_names):
    import anndata as ad
    ref = ad.read_h5ad(path, backed="r")
    pt = ref.obs[pt_key] if pt_key in ref.obs else None
    genes = (ref.var_names[np.asarray(ref.var["reliable_velo_gene"], bool)]
             if "reliable_velo_gene" in ref.var else None)
    obsm = {k: np.asarray(ref.obsm[k]) for k in ("X_pca", "X_umap") if k in ref.obsm}
    names = ref.obs_names.copy()
    ref.file.close()
    return pt, genes, obsm, names


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adata", help="input AnnData with a raw 'counts' layer (e.g. adata_pan.h5ad)")
    ap.add_argument("--reference", required=True,
                    help="ORIGINAL nSV stream output: pseudotime, reliable genes, embedding")
    ap.add_argument("--pt_key", default="velocity_mu_soft_pseudotime")
    ap.add_argument("--cluster_key", default="clusters")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--set", default="minimal", choices=["minimal", "full"])
    ap.add_argument("--conditions", nargs="*", default=None, help="explicit list (overrides --set)")
    ap.add_argument("--skip_existing", action="store_true",
                    help="do not rewrite <out_root>/<condition>/adata_pan.h5ad if it already exists "
                         "(the manifest and configs still cover every listed condition). Seeds are "
                         "fixed by the condition name, so a skipped file is identical to a rebuilt one.")
    ap.add_argument("--f_bar", type=float, default=None,
                    help="mean capture for every condition (default: mean of the reviewer's rule)")
    ap.add_argument("--cbdir_vkey", default="velocity_mu_soft")
    ap.add_argument("--cbdir_xkey", default="mu_scvi_smooth")
    ap.add_argument("--attach", action="store_true",
                    help="after the pipeline: copy the original embedding into each condition's stream output")
    a = ap.parse_args(argv)
    import anndata as ad

    pt_ref, genes, obsm, ref_names = _read_reference(a.reference, a.pt_key, None)

    if a.attach:
        for c in sorted(os.listdir(a.out_root)):
            p = os.path.join(a.out_root, c, "stream_orig_genes", "adata_nosplicevelo_stream.h5ad")
            if not os.path.exists(p):
                continue
            s = ad.read_h5ad(p)
            idx = pd.Index(ref_names).get_indexer(s.obs_names)
            if (idx < 0).any():
                print(f"  {c}: {int((idx < 0).sum())} cells not in the reference; skipped"); continue
            for k, v in obsm.items():
                s.obsm[f"{k}_orig"] = v[idx]
            s.write_h5ad(p); print(f"  {c}: attached {list(obsm)} as *_orig")
        return

    adata = ad.read_h5ad(a.adata)
    if pt_ref is None:
        raise KeyError(f"'{a.pt_key}' not in the reference obs")
    idx = pd.Index(ref_names).get_indexer(adata.obs_names)
    if (idx < 0).any():
        raise ValueError(f"{int((idx < 0).sum())} input cells are missing from the reference; "
                         f"the reviewer's rule needs a pseudotime for every cell")
    pt = np.asarray(pt_ref)[idx].astype(float)
    pt = (pt - pt.min()) / max(pt.max() - pt.min(), 1e-12)
    obsm_aligned = {k: v[idx] for k, v in obsm.items()}
    clusters = np.asarray(adata.obs[a.cluster_key]).astype(str)
    f_bar = a.f_bar if a.f_bar is not None else float(reviewer_f(pt).mean())
    conds = a.conditions or (MINIMAL if a.set == "minimal" else FULL)
    os.makedirs(a.out_root, exist_ok=True)
    print(f"input {a.adata}: {adata.n_obs} cells x {adata.n_vars} genes; f_bar = {f_bar:.3f}; "
          f"{len(conds)} conditions; original gene set: {0 if genes is None else len(genes)} genes")

    rows = []
    for c in conds:
        out, f = build_condition(adata, c, pt, clusters, f_bar, ref_genes=genes, ref_obsm=obsm_aligned)
        d = os.path.join(a.out_root, c); os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, "adata_pan.h5ad")
        if a.skip_existing and os.path.exists(dst):
            print(f"  {c}: {dst} exists; not rewritten (--skip_existing)")
        else:
            out.write_h5ad(dst)
        by = pd.Series(f).groupby(clusters).median()
        early = by.reindex(["Ductal", "Ngn3 low EP"]).mean()
        late = by.reindex(["Pre-endocrine", "Alpha", "Beta", "Delta", "Epsilon"]).mean()
        tot0 = np.asarray(adata.layers["counts"].sum(1)).ravel()
        rows.append(dict(condition=c, f_mean=f.mean(), f_min=f.min(), f_max=f.max(),
                         early_over_late=early / late,
                         ngn3high_f=by.get("Ngn3 high EP", np.nan),
                         depth_kept=float(out.obs["total_counts"].sum() / tot0.sum()),
                         spearman_f_pt=(pd.Series(f).rank().corr(pd.Series(pt).rank())
                                        if np.ptp(f) > 0 else np.nan)))
        print(f"  {c:28s} mean f {f.mean():.3f}  range {f.min():.3f}-{f.max():.3f}  "
              f"early/late {early / late:5.2f}  kept {rows[-1]['depth_kept']:.3f} of UMIs")
    pd.DataFrame(rows).to_csv(os.path.join(a.out_root, "manifest.csv"), index=False)
    write_configs(a.out_root, conds, a.adata, a.cluster_key, a.cbdir_vkey, a.cbdir_xkey)
    print(f"wrote manifest + configs under {a.out_root}")


if __name__ == "__main__":
    main()
