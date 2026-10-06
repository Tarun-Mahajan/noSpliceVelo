# `nsv_runs/`: the noSpliceVelo pipeline

Four config-driven steps, run in order from this folder. Each driver reads a
YAML file with an optional `log_file`, a `defaults:` block and a `datasets:`
list. Every key under `defaults:` can be overridden for each dataset. Datasets
run one after another. A failure is logged with its traceback and the run moves
on to the next dataset. Each run ends with a summary table in the log.
Progress bars are kept out of the log files.

Relative paths are resolved from this folder. The templates expect the data in
`../data/` (see the top-level README).

```
data/<ds>/adata_pan.h5ad                                     <- preprocessing/prepare_adata.py
  │ step 1  nsv_run_pipeline.py
  ▼
data/<ds>_studentT_time/model_scvi_modified.pt/, model_nosplicevelo.pt/
  │ step 2  compute_velo_run.py
  ▼
data/<ds>_studentT_time/adata_nosplicevelo_vote.h5ad, prob_state_avg_nosplicevelo_vote.npy
  │ step 3  score_gene_fit_v3_run.py
  ▼
data/<ds>_studentT_time/nosplicevelo_score_gene_fit_v3_vote/adata_nosplicevelo_vote.h5ad, gene_fit_scores_v3.csv
  │ step 4  generate_velo_stream_run.py
  ▼
data/<ds>_studentT_time/nosplicevelo_score_gene_fit_v3_4throot_type1_vote[_without_murk]/adata_nosplicevelo_stream.h5ad
  │ steps 5-7  ../velocity_metrics/, ../analyses/
```

The shell wrappers `run_nsv_time.sh`, `run_compute_velo.sh`,
`run_score_gene_fit_v3.sh` and `run_generate_velo_stream.sh` run the four steps
with the template configs. Each takes an optional config path.

---

## Step 1: train both VAEs (`nsv_run_pipeline.py`)

```bash
python nsv_run_pipeline.py nsv_config_runs_studentT_time_template.yaml [--no-smoothing]
```

For each dataset, `run_nsv_pipeline(adata, dir_path, **params)` does the following:

1. If missing, computes the naive moments `mu_naive_smooth` / `var_naive_smooth`
   from the raw `counts` layer: 50-PC PCA, kNN graph with k = 30, then two rounds
   of neighbour averaging. These are the targets of the correlation loss.
2. Trains `SCVIModified`, the first VAE (Methods, preprocessing step 3). It
   draws 10 posterior samples and averages them into `mu_scvi`, `var_scvi`.
3. Smooths these over a k = 30 graph in the first VAE's latent space to give
   `mu_scvi_smooth`, `var_scvi_smooth` (step 4).
4. Runs model selection with `ellipse_fit.VelocityModelSelector`
   (`thresh_p=2, thresh_e=5`) and drops genes labelled noise (step 5).
5. Computes the empirical corners and branch prior, then trains `noSpliceVelo`,
   the second VAE.
6. Saves both models, each with its AnnData, to `dir_path`.

Inputs (`adata_path`): `layers['counts']` (raw total counts), `X` (log-normalised),
`var['dispersions_norm']`, and `obsm['X_umap']` for the later stream plots.

Outputs (`dir_path`): `model_scvi_modified.pt/` and `model_nosplicevelo.pt/`.

| Key | Default (= manuscript) | Meaning |
|---|---|---|
| `use_time_dependence` | `true` | Shared per-cell factor ω on burst frequency and burst size of the final up-branch state. `false` fixes ω = 1. Strings such as `"false"` are parsed. |
| `time_loss` | `0.1` | Weight of the penalty keeping ω near 1. Ignored when `use_time_dependence` is false. |
| `continue_from_prev` | `false` | Reuse `model_scvi_modified.pt` if present. Resume `model_nosplicevelo.pt` if present, with KL warm-up set to 0. |
| `n_hidden_scvi`, `n_latent_scvi` | 64, 10 | Width and latent size of the first VAE |
| `n_hidden_nsv`, `n_latent_nsv` | 64, 10 | Width and latent size of the second VAE |
| `n_layers`, `n_layers_scvi`, `n_layers_nsv` | 1 | Hidden layers (shared, or per model) |
| `n_epochs_kl_warmup` | 1000 | KL warm-up of the second VAE |
| `batch_size` | 512 | Minibatch size for both models. The manuscript used 256 for pancreas, cortex and organoid. |
| `use_emp_total_count` | `false` | Total-count prior of the first VAE: fixed 1e4, or median library size / 0.1 |
| `max_epochs_scvi`, `max_epochs_nsv` | 10000, 300000 | Epoch caps. Early stopping on the validation ELBO (patience 100 / 1,000) normally ends training first. Lower them only for tests. |
| `w_sep`, `match_bottom_left` | 0.01, `false` | Branch-separation loss weight; corner-matching variant (not used) |
| `no_smoothing` (or `--no-smoothing`) | `false` | Skip step 3 and use `mu_scvi`/`var_scvi` directly (ablation only) |

Fixed in the code, as in the manuscript: correlation-loss weight 0.01 and 10
posterior samples for the first VAE. For the second VAE: 4 states, t_max = 24,
loss weights λ = 0.1 and validation split 0.9/0.1. Seeds are fixed (scvi, numpy
and torch seed 0). Training uses a GPU when one is available and the CPU
otherwise.

Reference configs: `nsv_config_runs_studentT_time_template.yaml` (all seven
datasets).

## Step 2: velocities (`compute_velo_run.py`, `compute_velo.py`)

```bash
python compute_velo_run.py compute_velo_config_template_.yaml
```

This step loads `model_nosplicevelo.pt` (and `model_scvi_modified.pt`) from
each `dir_path`. It then runs posterior-predictive inference with `nboot: 10`
samples (Methods, Downstream tasks). The annotated AnnData and the per-state
probabilities are written back to `dir_path`.

| Key | Manuscript | Meaning |
|---|---|---|
| `nboot` | 10 | Posterior samples |
| `state_reduction` | `both` | Primary `mu_fit`/`var_fit`/`velocity_mu`: argmax of the sample-averaged state posterior; also writes the soft mixture (`*_soft`) |
| `extra_reductions` | `[vote, vote_knn, tempered]` | `vote` = argmax state in each sample, its velocity averaged over samples: **`velocity_mu_vote`, the velocity used everywhere downstream** |
| `output_filename`, `probs_filename` | `adata_nosplicevelo_vote.h5ad`, `prob_state_avg_nosplicevelo_vote.npy` | Outputs in `dir_path` |
| `cell_batch_size`, `gpu_batch_size` | 4096 (1024 for the cortex), 1024 | Memory bounds only |

The layer names are listed in [`../nsv/README.md`](../nsv/README.md#velocity-output-names).

## Step 3: gene-level goodness of fit (`score_gene_fit_v3_run.py`)

```bash
python score_gene_fit_v3_run.py score_gene_fit_v3_config_template_.yaml [--select mouse_pancreas_studentT_time]
```

`score_gene_fit_v3.py` holds the scoring code. It uses helpers from
`score_gene_fit.py` and `score_gene_fit_v2.py`, and the bow and separation code
in `gene_bow_separation.py`. Every gene retained by model selection is scored
on the smoothed estimates of all cells (Methods, Gene-level goodness of fit):

| Diagnostic (Methods) | Columns | Used for the velocity genes (step 4)? |
|---|---|---|
| (i) R² of the fitted mean and variance against a gene-wise constant | `fit_r2_complete_mu`, `fit_r2_complete_var` | yes: both ≥ 0.01 |
| (ii) Per branch, quadratic vs straight line by orthogonal distance regression; nested F test (BH) and 5-fold CV | `qvl_*` | no (reported only) |
| (iii) Bow of the up and down branches, observed vs fitted, block-bootstrap z | `up_bow_obs`, `up_bow_z_fit`, `up_bow_misfit`, `down_bow_*` | yes |
| (iv) Signed separation of the branches on the naive variance, block-bootstrap z | `sep_log2_ratio_naive`, `sep_log2_z_naive` | yes: z > −2 (NaN kept) |

Outputs in `out_dir`: `gene_fit_scores_v3.csv`, the input AnnData with the
scores added to `var` (same file name), scatter plots and a log. The thresholds
were calibrated on simulated negative binomial counts.

## Step 4: velocity genes, velocity graph, streams, pseudotime (`generate_velo_stream_run.py`)

```bash
python generate_velo_stream_run.py velo_stream_config_template_nsv.yaml
bash run_generate_velo_stream.sh all      # also the seven compared methods
```

For each dataset, in order:

1. **Velocity genes (Methods, step 6).** `quant_` is the `q_` = 0.2 quantile of
   `quant_filter_col` = `dispersions_norm`. The pandas `gene_query` is then
   evaluated on `var`, and matching genes get `var['reliable_velo_gene'] = True`.
   The nSV query is

   ```
   (fit_r2_complete_mu >= 0.01 & fit_r2_complete_var >= 0.01)
   & (sep_log2_z_naive.isna() | sep_log2_z_naive > -2.0)
   & (up_bow_z_fit.isna() | up_bow_z_fit <= 2.0 | up_bow_obs > 0.0)
   & (up_bow_misfit == False)
   & ((dispersions_norm >= @quant_) & (dispersions_norm >= 1.0))
   ```

   If the query matches no gene, all genes are used and a warning is logged.
2. **MURK filter** (`filter_by_latent_time: true`; the two erythroid datasets).
   Genes are clustered by their latent-time profiles across cells (`time_latent`,
   PCA, kNN, Leiden at `leiden_res: 1.2`). Clusters holding more than 10 % of the
   genes flagged in `var['MURK_gene']` are removed.
3. **Velocity graph.** A kNN graph (`n_neighbors: 30` on `rep: X_pca`) and
   `scv_velocity_graph_new.velocity_graph`, a copy of scVelo's function that
   accepts `gene_subset` and a custom transform. Here
   `sqrt_transform_func: "x**(1/4)"`, i.e. sign(v)·|v|^(1/4), for all methods.
4. **Outputs.** The stream plot on `basis: umap`, velocity pseudotime
   (`compute_pseudotime`), and the AnnData (`output_filename`) for the metrics.
   For pseudotime, `xkey` is swapped into `layers['Ms']` during the call and the
   original `Ms` is restored afterwards.

| Key | nSV value | Meaning |
|---|---|---|
| `xkey`, `vkey` | `mu_scvi_smooth`, `velocity_mu_vote` | Expression and velocity layers |
| `gene_query`, `quant_filter_col`, `q_` | see above | Velocity-gene filter |
| `filter_by_latent_time`, `latent_time_col`, `boolean_gene_col`, `leiden_res` | `true`, `time_latent`, `MURK_gene`, 1.2 (erythroid only) | MURK filter |
| `scale_velo`, `scale_mu` | `false` | Optional per-gene velocity rescaling (`velocity_scale_diagnostics.py`; not used) |
| `label_col` | per dataset | `obs` column for colouring |

The compared methods have one config each (`velo_stream_config_template_<method>_4throot.yaml`).
Each reads the method's own output h5ad, keeps that method's own velocity genes
(e.g. `velocity_genes == True` for scVelo, veloVI and UniTVelo,
`~fit_scaling_y.isna()` for TFvelo, all genes for cellDancer and VeloVAE), and
applies the same graph and transform. For the two erythroid datasets, the
entries (`<dataset>_<method>_without_murk`) also drop the annotated MURK genes
(`MURK_gene == False`) and write to `<method>_4throot_without_murk/`, the
folder the CBDir configs in `../velocity_metrics/` read. The method outputs
come from the wrappers in `../other_methods/`.

## Files

| File | Used by |
|---|---|
| `nsv_run_pipeline.py` | step 1 |
| `compute_velo_run.py`, `compute_velo.py` | step 2 |
| `score_gene_fit_v3_run.py`, `score_gene_fit_v3.py`, `score_gene_fit_v2.py`, `score_gene_fit.py`, `gene_bow_separation.py` | step 3 |
| `generate_velo_stream_run.py`, `scv_velocity_graph_new.py`, `velocity_scale_diagnostics.py` | step 4 |
| `*_template*.yaml` | reference configs for the seven datasets |
| `run_*.sh` | thin wrappers |
