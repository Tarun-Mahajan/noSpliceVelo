# `velocity_metrics/`: CBDir, ICCoh and the benchmark statistics

Run from this folder. Paths in the configs are relative to it, and the data are
expected in `../data/`.

```bash
python compute_cbdir_run.py cbdir_global_config_4throot.yaml          # or: bash run_cbdir.sh
python plot_cbdir.py --global-config cbdir_global_config_4throot.yaml \
                     --plot-config plot_cbdir_config_nsv_comparison.yaml   # or: bash run_plot_cbdir.sh
```

The inputs are the stream objects written by `../nsv_runs/generate_velo_stream_run.py`
for nSV and for each compared method. Each one holds the velocity layer, the
method's kNN graph, its velocity graph (4th-root transform) and its PCA.

## Configs

| File | Content |
|---|---|
| `cbdir_global_config_4throot.yaml` | Shared settings (`str_suffix: _4throot`, `min_target_neighbors: 3`, ICCoh in gene space with `signed_4throot_scvelo`). Also the transitions of each dataset (`cluster_edges`), optional relabelling (`cluster_map`), and the list of method configs. |
| `cbdir_config_<method>_4throot.yaml` | One per method: `vkey`/`xkey`, the cluster column, and for each dataset `adata_path` (the stream object) and `dir_path` (where the tables go). nSV: `cbdir_config_nsv_cell_4throot.yaml` (`vkey: velocity_mu_vote`, `xkey: mu_scvi_smooth`). |
| `plot_cbdir_config_nsv_comparison.yaml` | Figure layout, method order and colours, reference method (nSV), tests (Wilcoxon, sign-flip permutation, logistic), head-to-head nSV vs TFvelo |

Method names used in all configs and tables: `scVelo-dynamical`,
`scVelo-stochastic`, `veloVI`, `veloVAE`, `celldancer`, `uniTVelo`, `TFvelo`, `nSV`.

## `compute_cbdir_run.py`

For each dataset, method and annotated transition A → B (Methods,
Cross-boundary direction correctness):

- **Embedding.** By default (`x_source: existing`) it uses the PCA (`basis: pca`)
  and kNN graph stored in each method's object: 30 PCs for the splicing-based
  methods, 50 for nSV. Velocities are projected with
  `scv.tl.velocity_embedding` from the stored velocity graph. `xkey` is swapped
  into `layers['Ms']` during the call and restored afterwards.
- **CBDir per source cell.** A cell of A is scored if at least 3 of its k = 30
  neighbours are in B. Its score is the mean cosine between its projected
  velocity and the displacements to those neighbours.
- **ICCoh per cell.** The mean cosine similarity between the cell's velocity and
  those of its same-cluster neighbours, computed in gene space on the velocity
  genes (`iccoh_space: gene_confidence`).
- **Restrictions.** `gene_query: "reliable_velo_gene == True"` limits the genes
  to the ones chosen in step 4. `cluster_map` relabels clusters before edges are
  matched (for example, Mono_1/Mono_2 → Mono in bone marrow).

Outputs per dataset, in each method's `dir_path`, with the `_4throot` suffix:

| File | Content |
|---|---|
| `<ds>_cbdir_long_4throot.csv` | One row per (method, transition, source cell): CBDir, number of target neighbours, ICCoh |
| `<ds>_cbdir_wide_4throot.csv` | (cell, transition) × method |
| `<ds>_cbdir_edge_summary_4throot.csv` | Per (transition, method): n cells, mean, median, quartiles, fraction positive, ICCoh; plus an `ALL_EDGES` row |
| `<ds>_iccoh_genes_4throot.csv`, `<ds>_iccoh_gene_clusters_4throot.csv` | ICCoh gene sets and per-cluster summaries |

`velocity_confidence_scaled.py` provides the gene-space vector preparation for
ICCoh. `compute_velocity_confidence_run.py` and `eff_gene_report.py` are only
imported for the participation-ratio and effective-gene diagnostics; those
columns are NaN if the import fails.

## `plot_cbdir.py`

`plot_cbdir.py` reads the long tables of every dataset listed in the CBDir
configs. It then draws the benchmark figures and runs the statistics (Methods,
Statistical analysis of the benchmark):

- Per-cell CBDir is Fisher z-transformed and averaged within each transition.
  Transitions are averaged within each dataset, and the dataset scores are
  back-transformed.
- Comparisons against the reference method (`reference_method: nSV`): paired
  Wilcoxon tests per dataset and per transition, with BH adjustment.
- The pre-specified head-to-head comparison nSV vs TFvelo
  (`head_to_head: [nSV, TFvelo]`): paired per-transition differences with a
  dataset-level sign-flip permutation test. The test is exact, so its smallest
  one-sided p is 1/2^7.
- Boxplots per dataset and per transition, per-method panels, CBDir vs ICCoh,
  stability summaries, and logistic and offset models with transitions as the
  replication unit (`cluster_unit: edge`).

Figures are saved as PNG and PDF, with their tables as CSV, under `save_path`
(`./results_cbdir_nsv_comparison/`).

## `benchmark_table_s2.py`: Supplementary Table S2

```bash
python benchmark_table_s2.py --global-config cbdir_global_config_4throot.yaml     # from the CBDir long tables
python benchmark_table_s2.py --means my_dataset_by_method_means.csv               # tests only, from a matrix of mean CBDir
```

The script finds every `<ds>_cbdir_long_4throot.csv` through the CBDir configs.
For each dataset it takes per-cell CBDir to Fisher z (arctanh), averages within
each transition and then over transitions, and back-transforms with tanh. These
are the values of the benchmark heatmap. On the Fisher-z scale it then runs two
sets of tests:

- **Pairwise.** Two-sided exact Wilcoxon signed-rank tests across the seven
  datasets for all 28 pairs of methods, Benjamini-Hochberg adjusted.
- **Primary.** nSV vs TFvelo, a one-sided sign-flip permutation test of the
  mean paired difference over all 2^7 sign assignments.

Outputs go to `results_table_s2/`:

| File | Content |
|---|---|
| `table_s2_mean_cbdir.csv` | Dataset × method means, the mean and s.d. across datasets, and the number of datasets with CBDir > 0 |
| `table_s2_per_transition.csv` | Per-transition scores |
| `table_s2_pairwise_wilcoxon.csv`, `table_s2_pairwise_matrix.csv` | The pairwise tests |
| `table_s2_primary_signflip.csv` | nSV vs TFvelo |
| `table_s2.tex` | The two tabulars |

