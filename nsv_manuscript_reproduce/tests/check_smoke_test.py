"""Check the outputs of run_smoke_test.sh. Exits non-zero if anything is missing."""

import glob
import os
import sys

import anndata as ad
import numpy as np
import pandas as pd
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
D = os.path.join(ROOT, "data")
STREAM = "nosplicevelo_score_gene_fit_v3_4throot_type1_vote"
fail = []


def need(path):
    if not os.path.exists(path):
        fail.append(f"missing: {os.path.relpath(path, ROOT)}")
        return False
    return True


for ds in ("dummy_a", "dummy_b"):
    for run, flag in (("time", True), ("no_time", False)):
        d = os.path.join(D, f"{ds}_studentT_{run}")
        # 1. checkpoints, and the use_time_dependence flag stored in them
        ck = os.path.join(d, "model_nosplicevelo.pt", "model.pt")
        if need(ck) and need(os.path.join(d, "model_scvi_modified.pt", "model.pt")):
            sd = torch.load(ck, map_location="cpu")
            ip = sd["attr_dict"]["init_params_"]
            mk = ip.get("kwargs", ip).get("model_kwargs", {})
            has_bt = any("b_t" in k for k in sd["model_state_dict"])
            if mk.get("use_time_dependence") is not flag or has_bt is not flag:
                fail.append(f"{ds}_{run}: use_time_dependence={mk.get('use_time_dependence')}, "
                            f"b_t layer present={has_bt}, expected {flag}")
        # 2. velocities
        f = os.path.join(d, "adata_nosplicevelo_vote.h5ad")
        if need(f):
            a = ad.read_h5ad(f, backed="r")
            miss = {"velocity_mu", "velocity_mu_vote", "mu_fit", "var_fit", "time_latent",
                    "mu_scvi_smooth", "var_scvi_smooth", "mu_naive_smooth"} - set(a.layers.keys())
            if miss:
                fail.append(f"{ds}_{run}: compute_velo layers missing {sorted(miss)}")
            nbest = a.var["best_fit"].value_counts().to_dict()
            a.file.close()
        # 3. gene scores
        f = os.path.join(d, "nosplicevelo_score_gene_fit_v3_vote", "gene_fit_scores_v3.csv")
        if need(f):
            s = pd.read_csv(f, index_col=0)
            miss = {"fit_r2_complete_mu", "fit_r2_complete_var", "sep_log2_z_naive",
                    "up_bow_z_fit", "up_bow_obs", "up_bow_misfit"} - set(s.columns)
            if miss:
                fail.append(f"{ds}_{run}: score columns missing {sorted(miss)}")
        # 4. streams + gene filter
        f = os.path.join(d, STREAM, "adata_nosplicevelo_stream.h5ad")
        if need(f):
            a = ad.read_h5ad(f, backed="r")
            n_rel = int(np.asarray(a.var["reliable_velo_gene"]).sum())
            pt = "velocity_mu_vote_pseudotime" in a.obs
            a.file.close()
            note = " (query matched no gene -> all genes used)" if n_rel == a.n_vars else ""
            print(f"{ds}_{run:8s} model selection {nbest}; velocity genes {n_rel}/{a.n_vars}{note}; "
                  f"pseudotime={'yes' if pt else 'NO'}")
        need(os.path.join(d, STREAM, f"{ds}_{run}_velocity_stream.png"))

need(os.path.join(D, "dummy_a_studentT_time", STREAM + "_without_murk", "adata_nosplicevelo_stream.h5ad"))

# 5. CBDir
for ds in ("dummy_a", "dummy_b"):
    f = os.path.join(D, f"{ds}_studentT_time", STREAM, f"{ds}_cbdir_edge_summary_4throot.csv")
    if need(f):
        e = pd.read_csv(f)
        allr = e[e["edge"] == "ALL_EDGES"].set_index("method")["mean"]
        print(f"{ds} CBDir (mean over cells, all transitions): " +
              ", ".join(f"{m}={v:+.3f}" for m, v in allr.items()))
        if set(allr.index) != {"nSV", "nSV_no_time", "shuffled"}:
            fail.append(f"{ds}: CBDir methods {sorted(allr.index)}")
# 6. plots
pngs = glob.glob(os.path.join(ROOT, "tests", "results_cbdir", "**", "*.png"), recursive=True) + \
       glob.glob(os.path.join(ROOT, "tests", "results_cbdir", "*.png"))
print(f"plot_cbdir figures: {len(set(pngs))}")
if not pngs:
    fail.append("plot_cbdir wrote no figures")

# 6b. Table S2
t2 = os.path.join(ROOT, "tests", "results_table_s2", "table_s2_mean_cbdir.csv")
if need(t2) and need(os.path.join(ROOT, "tests", "results_table_s2", "table_s2_pairwise_wilcoxon.csv")):
    print("Table S2 (dummy):\n" + pd.read_csv(t2, index_col=0).round(3).to_string())

# 7. analyses
gd = glob.glob(os.path.join(ROOT, "tests", "results_gene_direction", "*", "gd_summary.csv"))
print(f"gene_direction: {len(gd)} dataset summaries")
if len(gd) < 2:
    fail.append("gene_direction: expected gd_summary.csv for dummy_a and dummy_b")
mv = os.path.join(ROOT, "tests", "results_mu_var_corr", "mu_var_corr_dummy.h5ad")
if need(mv):
    m = ad.read_h5ad(mv)
    summ = m.uns.get("summary")
    print(f"mu_var_corr: {m.n_obs} (dataset, gene) rows x {m.n_vars} metrics")

if fail:
    print("\nSMOKE TEST FAILED:\n  " + "\n  ".join(fail))
    sys.exit(1)
print("\nSMOKE TEST PASSED")
