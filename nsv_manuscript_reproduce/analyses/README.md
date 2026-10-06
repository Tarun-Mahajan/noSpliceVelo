# `analyses/`: additional analyses

These three analyses build on the outputs of `../nsv_runs/` and `../velocity_metrics/`.

## `gene_direction/`: gene-level direction and Supplementary Data 1

```bash
cd analyses/gene_direction
python gene_direction_benchmark.py gene_direction_config_template.yaml [--datasets mouse_pancreas] [--methods nSV TFvelo]
```

The script reads the CBDir global config
(`../../velocity_metrics/cbdir_global_config_4throot.yaml`) and every method
config listed in it. Genes are therefore scored on exactly the transitions,
cells and velocity genes of the CBDir benchmark (Methods, Gene-level direction):

- **Boundary cells.** Source cells of A with at least 3 of their k = 30
  neighbours in B. The targets are those neighbours. The graph and cells come
  from `reference_method` (scVelo-dynamical).
- **Truth.** d_g is the difference between target and source means of
  log-normalised, once-smoothed expression, divided by the gene's standard
  deviation over all cells. A gene is scored if |d_g| ≥ 0.25, it is detected in
  ≥ 5 % of boundary cells, and the boundary and cluster-mean directions agree.
- **Prediction.** The sign of the median velocity over the source cells, for
  each method's own velocity genes (finite velocity in every cell and
  `reliable_velo_gene` / the method's gene query). Methods are compared on the
  genes they share, with McNemar's test.
- **Contribution.** c_g is each gene's share of the gene-space boundary cosine,
  with the 4th-root transform. The ten genes with the highest c_g per transition,
  method and dataset form Supplementary Data 1.
- **Cell-cycle flag.** Cell-cycle genes are flagged from
  `cc_genes/cell_cycle_{mouse,human}.gmt` (Tirosh S and G2M sets). The flag is
  recorded but not used to exclude genes (`exclude_flagged: false`).

Outputs go to `results/gene_direction/<dataset>/`: `gd_long.csv`, `gd_summary.csv`,
`gd_agreement.csv`, `gd_features.csv`, `gd_feature_tests.csv` and figures, plus
`all_datasets_*.csv` across datasets. The pooled n of gene-transition pairs in
the supplementary gene-direction figure counts each scored (transition, gene)
pair once across all datasets.

## `mu_var_corr/`: VAE moments vs naive moments

```bash
cd analyses/mu_var_corr
python mu_var_corr.py mu_var_corr_config_template.yaml [--datasets mouse_pancreas]
```

This compares the smoothed first-VAE estimates (`mu_scvi_smooth`,
`var_scvi_smooth`) with the naive kNN moments of the raw counts
(`mu_naive_smooth`, `var_naive_smooth`). Variance is compared on the
standard-deviation scale (`pre_transform: var: sqrt`). Pearson (raw and log1p)
and Spearman correlations are computed per gene (over cells), per cell (over
genes) and over all entries. The input is the `compute_velo_run.py` output
(`adata_nosplicevelo_vote.h5ad`). The output is one h5ad: `X` = (dataset::gene)
× metric, with `uns['per_cell']` and `uns['summary']`.

## `capture_bias/`: robustness to cell-specific capture efficiency

```bash
bash analyses/capture_bias/run_capture_bias.sh      # everything runs from nsv_runs/
```

`make_capture_bias_datasets.py` thins the raw pancreas counts binomially,
X' ~ Binomial(X, f_n). It writes one `adata_pan.h5ad` per condition, plus
ready-made configs for steps 1–5 (`nsv_config_capture_bias.yaml`,
`compute_velo_config_capture_bias.yaml`, ...). The conditions differ only in how
capture is distributed across cells; all have the same mean capture f̄ = 0.27.

- `reviewer_s0`, `reviewer_s1`: f_n = 1/(1+√τ_n)/2.5, where τ_n is nSV's velocity
  pseudotime on the full data. Capture falls from about 0.37 to about 0.21.
- `uniform_s0`, `uniform_s1`: f_n = f̄ for every cell. This is the control: depth
  loss without bias.
- `gradient_down_r4(_s1)` and others (`--set full`): additional stress tests.

`run_capture_bias.sh` builds the inputs and writes one nSV config per GPU. It
then starts the training in a tmux session, one window per GPU, and prints the
commands for the remaining steps. In those steps the velocity streams and CBDir
are computed on the original velocity-gene set and the original embedding
(`velocity_mu_soft`, `basis: umap_orig` / `pca_orig`). Velocity fields are
therefore compared on identical geometry. The manuscript reports `reviewer_*`
and `uniform_*` (Supplementary Fig. on capture efficiency).
