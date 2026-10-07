# noSpliceVelo: code for the manuscript

[![DOI](https://zenodo.org/badge/776338571.svg)](https://doi.org/10.5281/zenodo.23222847)

This folder, `nsv_manuscript_reproduce/`, holds the code behind the analyses in

> T. Mahajan and S. Maslov. noSpliceVelo infers gene expression dynamics without separating unspliced and spliced transcripts. *bioRxiv* 2024.08.08.607261 (2024). https://doi.org/10.1101/2024.08.08.607261

noSpliceVelo (nSV) infers RNA velocity from total counts alone. It reads the
direction of change of each gene from where cells sit in the variance-vs-mean
plane of transcriptional bursting. It has two variational autoencoders:

1. The first VAE (`SCVIModified`) jointly infers each cell's capture efficiency
   and the mean and variance of each gene's expression.
2. The second VAE (`noSpliceVelo`) fits a bursting model with an up branch and a
   down branch to those (mean, variance) pairs. From that fit it infers latent
   time, kinetic states and velocities.

Everything was run with the environment in `environment_gpu_final.yml` on one
NVIDIA GPU. `tests/` runs the whole pipeline on synthetic data on a CPU.

## Folder layout

| Folder | What it contains | README |
|---|---|---|
| `nsv/` | Model code: the first VAE (`scvi_modified_capture_efficiency_*`), the second VAE (`nosplicevelo_model_v5_polar.py`, `nosplicevelo_module_v5_polar.py`), model selection (`ellipse_fit.py`), posterior-predictive velocities (`nosplicevelo_ll_params.py`, `state_reductions.py`) | [nsv/README.md](nsv/README.md) |
| `preprocessing/` | Builds the nSV input object (`adata_pan.h5ad`) from a published dataset | [preprocessing/README.md](preprocessing/README.md) |
| `other_methods/` | Wrappers used to run scVelo (dynamical, stochastic), veloVI, VeloVAE, UniTVelo and cellDancer on the seven datasets | [other_methods/README.md](other_methods/README.md) |
| `nsv_runs/` | Config-driven drivers for the nSV pipeline: training, velocities, gene scoring, gene filter and velocity streams | [nsv_runs/README.md](nsv_runs/README.md) |
| `velocity_metrics/` | Benchmark metrics: CBDir and ICCoh for all eight method configurations, the benchmark figures and statistics, and Supplementary Table S2 | [velocity_metrics/README.md](velocity_metrics/README.md) |
| `analyses/` | Additional analyses: gene-level direction (Supplementary Data 1), correlation of VAE moments with naive moments, capture-efficiency robustness | [analyses/README.md](analyses/README.md) |
| `tests/` | End-to-end smoke test on synthetic data | [tests/README.md](tests/README.md) |
| `external_licenses/` | Licenses of third-party code included here | – |
| `_not_used/` | Files from the original code folders that no step imports; kept for reference only | [_not_used/README.md](_not_used/README.md) |

## Installation

```bash
conda env create -f environment_gpu_final.yml     # Python 3.10, torch 2.3.1 (CUDA 12.1), scvi-tools 1.0.4
conda activate nsv_env
```

Main versions: scvi-tools 1.0.4, torch 2.3.1, lightning 2.0.9, scanpy 1.9.3,
anndata 0.8.0, scvelo 0.3.4, numpy 1.24.3, scipy 1.10.1, pandas 2.0.1,
statsmodels 0.14.4, scikit-learn 1.7.2, jax 0.4.23 (pinned so that `import scvi`
works with scvi-tools 1.0.4).

The scripts are run directly from their folders; nothing needs to be installed
as a package. Each driver adds `../nsv` to `sys.path` itself. Set `NSV_SRC` to
use another copy of the model code.

## Quick check (no real data needed)

```bash
cd nsv_manuscript_reproduce
bash tests/run_smoke_test.sh     # about 45 min on 2 CPU cores (mostly training); uses a GPU if available
```

The script simulates two small bursty trajectories and runs every stage below
with the manuscript settings (only the epoch caps are lowered). It runs each
dataset with `use_time_dependence: true` and `false`. It ends with
`SMOKE TEST PASSED`. See [tests/README.md](tests/README.md).

## Pipeline

Paths in all configs are relative to the folder the script runs from, and all
data live in a `data/` folder inside `nsv_manuscript_reproduce/` (`../data/...`
from the script folders). `data/` is not part of the repository: put the
published objects there, or edit
`adata_path` / `dir_path` in the configs.

| Step | Methods section | Script (run from) | Config | Main output |
|---|---|---|---|---|
| 0 | Data preprocessing (cell and gene filters, 2,000 highly variable genes) | `preprocessing/prepare_adata.py` | command line | `data/<ds>/adata_pan.h5ad` |
| 0b | Compared methods | `other_methods/<method>_runs/run.sh` | one YAML per method | each method's output h5ad in `data/<ds>/` |
| 1 | First VAE, smoothing, model selection, second VAE (steps 3–5) | `nsv_runs/nsv_run_pipeline.py` | `nsv_config_runs_studentT_time_template.yaml` | `model_scvi_modified.pt/`, `model_nosplicevelo.pt/` |
| 2 | Downstream tasks: posterior-predictive moments, latent time and velocity (10 samples) | `nsv_runs/compute_velo_run.py` | `compute_velo_config_template_.yaml` | `adata_nosplicevelo_vote.h5ad`, `prob_state_avg_nosplicevelo_vote.npy` |
| 3 | Gene-level goodness of fit | `nsv_runs/score_gene_fit_v3_run.py` | `score_gene_fit_v3_config_template_.yaml` | `nosplicevelo_score_gene_fit_v3_vote/` (scores CSV + h5ad) |
| 4 | Gene selection (step 6, including the MURK filter), velocity graph, streams, pseudotime | `nsv_runs/generate_velo_stream_run.py` | `velo_stream_config_template_nsv.yaml` (+ one per compared method) | `.../nosplicevelo_score_gene_fit_v3_4throot_type1_vote/adata_nosplicevelo_stream.h5ad`, stream and pseudotime PNGs |
| 5 | Cross-boundary direction correctness and ICCoh | `velocity_metrics/compute_cbdir_run.py` | `cbdir_global_config_4throot.yaml` (+ one per method) | `<ds>_cbdir_{long,wide,edge_summary}_4throot.csv` |
| 6 | Statistical analysis of the benchmark (benchmark figure and its supplementary figures) | `velocity_metrics/plot_cbdir.py` | `plot_cbdir_config_nsv_comparison.yaml` | figures and tables under `results_cbdir_nsv_comparison/` |
| 6b | Supplementary Table S2 (mean CBDir per dataset, pairwise Wilcoxon tests) and the nSV vs TFvelo sign-flip test | `velocity_metrics/benchmark_table_s2.py` | the CBDir global config | `results_table_s2/` |
| 7 | Gene-level direction, Supplementary Data 1 | `analyses/gene_direction/gene_direction_benchmark.py` | `gene_direction_config_template.yaml` | `results/gene_direction/` |
| 8 | VAE vs naive moments | `analyses/mu_var_corr/mu_var_corr.py` | `mu_var_corr_config_template.yaml` | one h5ad of correlations |
| 9 | Robustness to cell-specific capture efficiency | `analyses/capture_bias/run_capture_bias.sh` | generated | thinned datasets and CBDir per condition |

To reproduce the benchmark from the published objects:

```bash
for m in scvelo velovi velovae unitvelo celldancer; do bash other_methods/${m}_runs/run.sh; done   # see other_methods/README.md for environments
cd nsv_runs
python nsv_run_pipeline.py          nsv_config_runs_studentT_time_template.yaml   # GPU; 2-19 h per dataset
python compute_velo_run.py          compute_velo_config_template_.yaml
python score_gene_fit_v3_run.py     score_gene_fit_v3_config_template_.yaml
bash   run_generate_velo_stream.sh  all          # nSV + the seven compared methods
cd ../velocity_metrics
python compute_cbdir_run.py cbdir_global_config_4throot.yaml
python plot_cbdir.py --global-config cbdir_global_config_4throot.yaml \
                     --plot-config plot_cbdir_config_nsv_comparison.yaml
python benchmark_table_s2.py --global-config cbdir_global_config_4throot.yaml
```

Step 4 for the compared methods (`velo_stream_config_template_<method>_4throot.yaml`)
reads each method's output h5ad, for example `data/Pancreas/adata_scvelo.h5ad`,
written by the wrappers in `other_methods/`. TFvelo 1.0 was run from its GitHub
repository; its wrapper is not included, and its output is expected at
`data/<ds>/adata_tfvelo.h5ad`. For the two erythroid datasets, all eight method
configurations exclude the MURK genes: nSV through the latent-time filter, the
other methods by dropping the annotated MURK genes (`MURK_gene == False`).

Each method selects its own 2,000 highly variable genes on its own
preprocessed object, so the gene sets of the methods overlap but are not
identical; [other_methods/README.md](other_methods/README.md#gene-sets) lists
the filters of each method.

## Datasets

### Original objects

Download these into the `data/` folders below. Zip archives contain one h5ad
each; the file names are the ones used in the configs of `other_methods/`.

| Dataset (config name) | Download | File | `data/` folder |
|---|---|---|---|
| Mouse pancreas, E15.5 (`mouse_pancreas`) | https://github.com/theislab/scvelo_notebooks/raw/master/data/Pancreas/endocrinogenesis_day15.h5ad | `endocrinogenesis_day15.h5ad` | `Pancreas` |
| Human bone marrow CD34+ (`human_bonemarrow`) | https://ndownloader.figshare.com/files/27686835 | `human_cd34_bone_marrow.h5ad` | `BoneMarrow` |
| Mouse embryonic cortex, stimulated neurons (`mouse_neural`) | https://ccsm.uth.edu/Benchmarking/VelocityBenchmarking/RealData/16_mouse_NeuronGenesis_GSE141851_with_time.zip | `16_mouse_NeuronGenesis_GSE141851_with_time.h5ad` | `dynamo_mouse_neural` |
| Mouse gastrulation erythropoiesis (`mouse_erythroid`) | https://ndownloader.figshare.com/files/27686871 | `erythroid_lineage.h5ad` | `mouse_erythroid` |
| Mouse intestinal organoid (`mouse_organoid`) | https://ccsm.uth.edu/Benchmarking/VelocityBenchmarking/RealData/14_mouse_IntestinalOrganoids_GSE128365.zip | `14_mouse_IntestinalOrganoids_GSE128365.h5ad` | `unitvelo_mouse_organoid` |
| Mouse hippocampus (`mouse_dentategyrus`) | https://ccsm.uth.edu/Benchmarking/VelocityBenchmarking/RealData/1_mouse_DentateGyrusP0P5_GSE104323_addUmap.zip | `1_mouse_DentateGyrusP0P5_GSE104323_addUmap.h5ad` | `dentategyrus_new` |
| Human fetal-liver erythroid (`human_erythroid`) | https://ccsm.uth.edu/Benchmarking/VelocityBenchmarking/RealData/13_human_erythroid_E-MTAB-7407.zip | `13_human_erythroid_E-MTAB-7407.h5ad` | `human_erythroid` |

Primary accessions: pancreas GEO GSE132188; bone marrow Human Cell Atlas
091cf39b-01bc-42e5-9437-f419a66c8a45; cortex GEO GSE141851; mouse erythroid
ArrayExpress E-MTAB-6967; organoid GEO GSE128365; hippocampus GEO GSE104323;
human erythroid ArrayExpress E-MTAB-7407.

### Processed noSpliceVelo objects (figshare)

The noSpliceVelo results for all seven datasets are deposited on figshare as one
archive, `adata_nosplicevelo_streams.tar.gz` (about 18 GB):

**figshare:** https://doi.org/10.6084/m9.figshare.34112379

The archive holds one h5ad per dataset. Each is the object written by step 4
(`generate_velo_stream_run.py`), so it already contains steps 1–4: both VAEs,
the velocities, the gene scores, the velocity-gene filter, the velocity graph
and the pseudotime. Steps 5–8 and the plots can be run on it without a GPU and
without retraining.

| File in the archive | Dataset | Cells | Genes (after model selection) | Velocity genes (`reliable_velo_gene`) | Place at (to run steps 5–8 with the template configs) |
|---|---|---|---|---|---|
| `adata_nosplicevelo_stream_Pancreas.h5ad` | Mouse pancreas | 3,696 | 1,994 | 1,034 | `data/Pancreas_studentT_time/nosplicevelo_score_gene_fit_v3_4throot_type1_vote/` |
| `adata_nosplicevelo_stream_BoneMarrow.h5ad` | Human bone marrow | 5,719 | 1,813 | 862 | `data/BoneMarrow_studentT_time/nosplicevelo_score_gene_fit_v3_4throot_type1_vote/` |
| `adata_nosplicevelo_stream_dynamo.h5ad` | Mouse embryonic cortex | 3,060 | 1,975 | 482 | `data/dynamo_mouse_neural_studentT_time/nosplicevelo_score_gene_fit_v3_4throot_type1_vote/` |
| `adata_nosplicevelo_stream_mouse.h5ad` | Mouse gastrulation erythropoiesis | 9,815 | 1,973 | 303 | `data/mouse_erythroid_studentT_time/nosplicevelo_score_gene_fit_v3_4throot_type1_vote_without_murk/` |
| `adata_nosplicevelo_stream_human.h5ad` | Human fetal-liver erythroid | 35,877 | 1,991 | 884 | `data/human_erythroid_studentT_time/nosplicevelo_score_gene_fit_v3_4throot_type1_vote_without_murk/` |
| `adata_nosplicevelo_stream_unitvelo_organoid.h5ad` | Mouse intestinal organoid | 3,831 | 1,840 | 300 | `data/unitvelo_mouse_organoid_studentT_time/nosplicevelo_score_gene_fit_v3_4throot_type1_vote/` |
| `adata_nosplicevelo_stream_dentategyrus.h5ad` | Mouse hippocampus | 18,213 | 1,997 | 1,287 | `data/dentategyrus_new_studentT_time/nosplicevelo_score_gene_fit_v3_4throot_type1_vote/` |

The configs read each file as `adata_nosplicevelo_stream.h5ad` in the folder
shown, so rename it when placing it:

```bash
tar -xzf adata_nosplicevelo_streams.tar.gz          # from nsv_manuscript_reproduce/
V=nosplicevelo_score_gene_fit_v3_4throot_type1_vote
place() { mkdir -p "data/$2" && mv "$1" "data/$2/adata_nosplicevelo_stream.h5ad"; }
place adata_nosplicevelo_stream_Pancreas.h5ad          Pancreas_studentT_time/$V
place adata_nosplicevelo_stream_BoneMarrow.h5ad        BoneMarrow_studentT_time/$V
place adata_nosplicevelo_stream_dynamo.h5ad            dynamo_mouse_neural_studentT_time/$V
place adata_nosplicevelo_stream_mouse.h5ad             mouse_erythroid_studentT_time/${V}_without_murk
place adata_nosplicevelo_stream_human.h5ad             human_erythroid_studentT_time/${V}_without_murk
place adata_nosplicevelo_stream_unitvelo_organoid.h5ad unitvelo_mouse_organoid_studentT_time/$V
place adata_nosplicevelo_stream_dentategyrus.h5ad      dentategyrus_new_studentT_time/$V
```

What each object contains, as written by the scripts of steps 1–4:

| Slot | Fields | Written by |
|---|---|---|
| `X`, `layers['counts']`, `layers['log_counts']` | Raw total counts (`counts`) and log-normalised expression; `spliced`/`unspliced` too where the published object has them (not used by nSV) | step 0 |
| `layers['mu_naive_smooth']`, `layers['var_naive_smooth']` | Naive kNN mean and variance of the raw counts | step 1 |
| `layers['mu_scvi']`, `layers['var_scvi']`, `layers['mu_scvi_smooth']`, `layers['var_scvi_smooth']` | Mean and variance from the first VAE (10 posterior samples), and after kNN smoothing (k = 30, first-VAE latent space): the data the second VAE fits | step 1 |
| `layers['branch_assignment']`, `layers['prior_pi_up']` | Empirical up/down branch assignment and branch prior | step 1 |
| `layers['velocity_mu_vote']` | **The velocity used in the manuscript**: per-sample most probable state, its velocity of the mean, averaged over 10 samples | step 2 |
| `layers['velocity_mu']`, `layers['velocity_var']`, `layers['mu_fit']`, `layers['var_fit']` | Velocities and fitted moments of the most probable state of the sample-averaged posterior | step 2 |
| `layers['velocity_*_soft']`, `layers['*_tempered2']`, `layers['velocity_*_vote_knn']`, `layers['velocity_var_vote']` | Other state reductions (diagnostics) | step 2 |
| `layers['time_latent']` | Latent time per cell and gene | step 2 |
| `var` | `dispersions_norm` and the other HVG columns; `best_fit` (line, parabola, ellipse) and fit statistics of the model selection; kinetic parameters (`mu_0_gene`, `mu_up_f_gene`, `mu_down_f_gene`, their variances, `time_switch`, `gamma_mRNA`); the goodness-of-fit scores (`fit_r2_complete_*`, `qvl_*`, `up_bow_*`, `down_bow_*`, `sep_*`); `reliable_velo_gene` (the velocity genes); `MURK_gene` for the erythroid datasets | steps 0–4 |
| `obs` | The published annotations (cell type or time point), and `velocity_mu_vote_pseudotime` | published, step 4 |
| `obsm` | `X_pca`, the published `X_umap`, `X_latent` (first-VAE latent space), `velocity_mu_vote_umap` | steps 0–4 |
| `obsp`, `uns` | kNN graph (k = 30 on `X_pca`), velocity graph `velocity_mu_vote_graph` / `_graph_neg` (4th-root transform, velocity genes only), `velocity_mu_vote_params` | step 4 |

For example, the velocity stream:

```python
import anndata as ad, scvelo as scv
adata = ad.read_h5ad("adata_nosplicevelo_stream_Pancreas.h5ad")
scv.pl.velocity_embedding_stream(adata, vkey="velocity_mu_vote", basis="umap", color="clusters")
```

The trained models are not in the archive. To retrain from scratch, rebuild
`adata_pan.h5ad` from the original objects (step 0) and start at step 1. For the
cortex, nSV used the total counts of the metabolic-labelling object of the same
experiment (3,060 cells), and the other methods used the splicing object above
(3,066 cells).

The annotated transitions are listed once, in `velocity_metrics/cbdir_global_config_4throot.yaml`.

## Settings used for the manuscript

| Setting | Value | Where |
|---|---|---|
| Preprocessing | genes in ≥ 3 cells; cells with ≥ 200 genes; genes with ≥ 20 counts; top 2,000 highly variable genes (scanpy, Seurat flavour); compositional distorters and mitochondrial genes (`MT-` human, `mt-` mouse) removed | `preprocessing/prepare_adata.py` |
| First VAE | 64 hidden, 10 latent, 1 layer; correlation loss weight 0.01; ≤ 10,000 epochs, early stopping (patience 100) | `nsv_run_pipeline.py` |
| Posterior samples of the first VAE | 10 | `nsv_run_pipeline.py` (`nrepeats`) |
| Smoothing of mean and variance | kNN, k = 30, in the first VAE's latent space | `nsv_run_pipeline.py` |
| Model selection | BIC margins 2 (parabola vs line) and 5 (ellipse); noise if adj. R² < 0.1 or slope < 1 | `nsv/ellipse_fit.py` (`thresh_p=2, thresh_e=5`) |
| Second VAE | 4 states, t_max = 24, 64 hidden, 10 latent; Student-t likelihood, df annealed 1000 → 4; KL warm-up 1,000 epochs; early stopping (patience 1,000) | `nsv_run_pipeline.py`, `nsv/nosplicevelo_module_v5_polar.py` |
| Shared time-dependence factor ω on the final up-branch burst frequency and size | on (`use_time_dependence: true`), penalty weight 0.1 | config key `use_time_dependence`, `time_loss` |
| Velocity | per-sample argmax state, averaged over 10 samples (`velocity_mu_vote`) | `compute_velo_run.py` |
| Velocity genes (step 6) | `gene_query` in `velo_stream_config_template_nsv.yaml`; dispersion ≥ max(1, 20th percentile); MURK filter for the two erythroid datasets (Leiden resolution 1.2, clusters with > 10 % of MURK genes removed; the compared methods drop the annotated MURK genes) | `generate_velo_stream_run.py` |
| Velocity graph transform | sign(v)·\|v\|^(1/4) for all methods | `sqrt_transform_func: "x**(1/4)"` |
| CBDir | k = 30 stored kNN graph, ≥ 3 target neighbours, stored PCA, Fisher z aggregation | `velocity_metrics/` configs |

### `use_time_dependence`

The final up-branch steady state of each gene has burst frequency `f` and burst
size `B`. With `use_time_dependence: true`, both are multiplied by one per-cell
factor ω ∈ (0, 1] (`b_t` in `nosplicevelo_module_v5_polar.py`), so the target of
upregulation can drift along the trajectory. A penalty with weight `time_loss`
keeps ω near 1. The module keeps a commented line that would give `f` its own
factor. For the manuscript, `f` and `B` share ω, which limits the extra freedom
of the model. With `use_time_dependence: false`, ω = 1 and the penalty is off.
The option can be set in the `defaults:` block or for each dataset in
`nsv_config_runs_studentT_time_template.yaml`; the default is `true`.

## Hardware and runtime

The manuscript runs used a single NVIDIA A10 GPU. One full nSV run (step 1)
took 2.1 h to 19.2 h per dataset (Supplementary Table S1). Steps 2–6 take
minutes to about an hour per dataset.

## Citation

```bibtex
@article{Mahajan2024.08.08.607261,
  author    = {Mahajan, Tarun and Maslov, Sergei},
  title     = {noSpliceVelo infers gene expression dynamics without separating unspliced and spliced transcripts},
  journal   = {bioRxiv},
  elocation-id = {2024.08.08.607261},
  year      = {2024},
  doi       = {10.1101/2024.08.08.607261},
  publisher = {Cold Spring Harbor Laboratory},
  URL       = {https://www.biorxiv.org/content/10.1101/2024.08.08.607261v1}
}
```

## License

BSD 3-Clause; see [LICENSE](LICENSE). Parts of the code are adapted from
scvi-tools and scVelo (BSD 3-Clause) and VeloAE (MIT); the licenses of these and
of the other packages used are in [external_licenses/](external_licenses/README.md).

## Contact

Tarun Mahajan and Sergei Maslov.
