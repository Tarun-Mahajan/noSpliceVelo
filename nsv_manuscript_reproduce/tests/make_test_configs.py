"""Write the smoke-test configs under tests/configs/ from the manuscript configs.

Each test config is the manuscript config with the dataset list replaced by the
synthetic datasets and (for the training step only) small epoch caps, so the
smoke test runs the same code paths and settings as the manuscript runs.

Synthetic datasets: dummy_a (seed 0) and dummy_b (seed 1), each trained with
use_time_dependence true and false. CBDir/plotting compare three "methods":
  nSV            use_time_dependence: true  (the manuscript setting)
  nSV_no_time    use_time_dependence: false
  shuffled       nSV velocities permuted across cells (negative control)
"""

import copy
import os
import re

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CFG = os.path.join(HERE, "configs")
os.makedirs(CFG, exist_ok=True)

DATASETS = ["dummy_a", "dummy_b"]
RUNS = {"time": True, "no_time": False}
SCORE_DIR = "nosplicevelo_score_gene_fit_v3_vote"
STREAM_DIR = "nosplicevelo_score_gene_fit_v3_4throot_type1_vote"


def load(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def dump(obj, name, header):
    path = os.path.join(CFG, name)
    with open(path, "w") as fh:
        fh.write(header.rstrip() + "\n")
        yaml.safe_dump(obj, fh, sort_keys=False, width=1000)
    print("wrote", os.path.relpath(path, ROOT))


def run_dir(ds, run):
    return f"../data/{ds}_studentT_{run}"


# ---- 1. nSV training (run from nsv_runs/) -----------------------------------
c = load(os.path.join(ROOT, "nsv_runs", "nsv_config_runs_studentT_time_template.yaml"))
c["log_file"] = "../tests/logs/nsv_pipeline_dummy.log"
c["defaults"].update(batch_size=128, n_epochs_kl_warmup=100,
                     max_epochs_scvi=150, max_epochs_nsv=600)
c["datasets"] = [dict(name=f"{ds}_{run}", adata_path=f"../data/{ds}/adata_pan.h5ad",
                      dir_path=run_dir(ds, run), use_time_dependence=flag)
                 for ds in DATASETS for run, flag in RUNS.items()]
dump(c, "nsv_dummy.yaml", "# smoke test: nsv_run_pipeline.py (run from nsv_runs/); epoch caps lowered")

# ---- 2. compute_velo (run from nsv_runs/) ------------------------------------
c = load(os.path.join(ROOT, "nsv_runs", "compute_velo_config_template_.yaml"))
c["log_file"] = "../tests/logs/compute_velo_dummy.log"
c["datasets"] = [dict(name=f"{ds}_{run}", dir_path=run_dir(ds, run))
                 for ds in DATASETS for run in RUNS]
dump(c, "compute_velo_dummy.yaml", "# smoke test: compute_velo_run.py (run from nsv_runs/)")

# ---- 3. gene scoring (run from nsv_runs/) ------------------------------------
c = load(os.path.join(ROOT, "nsv_runs", "score_gene_fit_v3_config_template_.yaml"))
c["log_file"] = "../tests/logs/score_gene_fit_v3_dummy.log"
c["defaults"]["n_jobs"] = 2
c["datasets"] = [dict(name=f"{ds}_{run}", dir_path=run_dir(ds, run),
                      out_dir=f"{run_dir(ds, run)}/{SCORE_DIR}")
                 for ds in DATASETS for run in RUNS]
dump(c, "score_gene_fit_v3_dummy.yaml", "# smoke test: score_gene_fit_v3_run.py (run from nsv_runs/)")

# ---- 4. velocity graph / streams / gene filter (run from nsv_runs/) ----------
c = load(os.path.join(ROOT, "nsv_runs", "velo_stream_config_template_nsv.yaml"))
ref = c["datasets"][0]                               # bone marrow entry: gene_query etc.
c["log_file"] = "../tests/logs/velo_stream_dummy.log"
# The manuscript query also requires fit R2 >= 0.01 and dispersions_norm >= 1.
# With the short training used here the toy fits rarely reach that R2, and with
# only 100 genes few have dispersions_norm >= 1, so the query would leave 0-1
# genes. The test keeps every other criterion so the same columns and the
# @quant_ machinery are exercised.
TEST_QUERY = ("(sep_log2_z_naive.isna() | sep_log2_z_naive > -2.0) & "
              "(up_bow_z_fit.isna() | up_bow_z_fit <= 2.0 | up_bow_obs > 0.0) & "
              "(up_bow_misfit == False) & (dispersions_norm >= @quant_)")
assert "fit_r2_complete_mu >= 0.01" in ref["gene_query"]
ref["gene_query"] = TEST_QUERY
c["datasets"] = []
for ds in DATASETS:
    for run in RUNS:
        e = copy.deepcopy(ref)
        e.update(name=f"{ds}_{run}",
                 adata_path=f"{run_dir(ds, run)}/{SCORE_DIR}/adata_nosplicevelo_vote.h5ad",
                 dir_path=f"{run_dir(ds, run)}/{STREAM_DIR}")
        c["datasets"].append(e)
murk = copy.deepcopy(c["datasets"][0])               # exercise the erythroid MURK filter
murk.update(name="dummy_a_time_murk_filter", dir_path=murk["dir_path"] + "_without_murk",
            filter_by_latent_time=True, latent_time_col="time_latent",
            boolean_gene_col="MURK_gene", leiden_res=1.2)
c["datasets"].append(murk)
dump(c, "velo_stream_dummy.yaml", "# smoke test: generate_velo_stream_run.py (run from nsv_runs/)")

# ---- 5. CBDir / ICCoh (run from velocity_metrics/) ---------------------------
src = open(os.path.join(ROOT, "velocity_metrics", "cbdir_config_nsv_cell_4throot.yaml")).read()
head = src[:re.search(r"^datasets:", src, re.M).start()]


def method_cfg(name, run, extra=""):
    body = "datasets:\n"
    for ds in DATASETS:
        sdir = f"{run_dir(ds, run)}/{STREAM_DIR}"
        fn = "adata_shuffled_control.h5ad" if name == "shuffled" else "adata_nosplicevelo_stream.h5ad"
        # absolute paths: compute_cbdir_run.py resolves them from its working
        # directory, gene_direction_benchmark.py from the config's folder
        sdir = os.path.normpath(os.path.join(ROOT, "nsv_runs", sdir))
        body += f"  - name: {ds}\n    adata_path: {sdir}/{fn}\n    dir_path: {sdir}\n{extra}"
    path = os.path.join(CFG, f"cbdir_{name}_dummy.yaml")
    with open(path, "w") as fh:
        fh.write(f"# smoke test, method {name} (defaults copied from cbdir_config_nsv_cell_4throot.yaml)\n")
        fh.write(head + body)
    print("wrote", os.path.relpath(path, ROOT))
    return path


methods = {
    "nSV": method_cfg("nSV", "time"),
    "nSV_no_time": method_cfg("nSV_no_time", "no_time"),
    # permuted velocities need a fresh velocity graph
    "shuffled": method_cfg("shuffled", "time", "    recompute_velocity_graph: true\n"),
}
g = load(os.path.join(ROOT, "velocity_metrics", "cbdir_global_config_4throot.yaml"))
g["log_file"] = "../tests/logs/cbdir_dummy.log"
g["datasets"] = {ds: {"cluster_edges": [["A", "B"], ["B", "C"], ["C", "D"]]} for ds in DATASETS}
g["methods"] = methods
dump(g, "cbdir_global_dummy.yaml", "# smoke test: compute_cbdir_run.py / plot_cbdir.py (run from velocity_metrics/)")

# ---- 6. CBDir plots (run from velocity_metrics/) -----------------------------
p = load(os.path.join(ROOT, "velocity_metrics", "plot_cbdir_config_nsv_comparison.yaml"))
p["save_path"] = "../tests/results_cbdir/cbdir_plot"
p["log_file"] = "../tests/logs/plot_cbdir_dummy.log"
p["method_groups"] = {"group1": ["shuffled"], "group2": ["nSV_no_time", "nSV"]}
p["method_colors"] = {"shuffled": "#888888", "nSV_no_time": "#8C1515", "nSV": "#F5B041"}
p["head_to_head"] = ["nSV", "nSV_no_time"]
p["stability_summary_dataset_labels"] = {"dummy_a": "Dummy A", "dummy_b": "Dummy B"}
for k, v in list(p.items()):
    if isinstance(v, list) and set(map(str, v)) & {"TFvelo", "veloVI", "uniTVelo"}:
        p[k] = [m for m in ("shuffled", "nSV_no_time", "nSV")]
    if isinstance(v, dict) and set(v) & {"mouse_pancreas", "human_bonemarrow"}:
        p[k] = {"dummy_a": "Dummy A", "dummy_b": "Dummy B"}
dump(p, "plot_cbdir_dummy.yaml", "# smoke test: plot_cbdir.py (run from velocity_metrics/)")

# ---- 7. analyses: gene direction and mu/var correlation ---------------------
gd = load(os.path.join(ROOT, "analyses", "gene_direction", "gene_direction_config_template.yaml"))
gd.update(cbdir_global_config=os.path.join(CFG, "cbdir_global_dummy.yaml"),
          out_dir="../results_gene_direction", reference_method="nSV", compare_to="nSV",
          per_dataset={}, feature_cols={"nSV": gd["feature_cols"]["nSV"]}, n_boot=100)
dump(gd, "gene_direction_dummy.yaml", "# smoke test: gene_direction_benchmark.py (paths relative to this file)")

mv = load(os.path.join(ROOT, "analyses", "mu_var_corr", "mu_var_corr_config_template.yaml"))
mv["out_h5ad"] = "../results_mu_var_corr/mu_var_corr_dummy.h5ad"
mv["datasets"] = {f"{ds}_{run}": f"../../data/{ds}_studentT_{run}/adata_nosplicevelo_vote.h5ad"
                  for ds in DATASETS for run in RUNS}
dump(mv, "mu_var_corr_dummy.yaml", "# smoke test: mu_var_corr.py (paths relative to this file)")
