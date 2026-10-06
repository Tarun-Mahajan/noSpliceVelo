# `tests/`: end-to-end smoke test on synthetic data

The real datasets are large and take hours per dataset on a GPU. This test runs
every step of the pipeline on two small synthetic datasets in well under an
hour on a CPU. It uses the manuscript configs; only the dataset lists and the
epoch caps are changed.

```bash
bash tests/run_smoke_test.sh            # from nsv_manuscript_reproduce/, in the nsv_env environment
PY=/path/to/python bash tests/run_smoke_test.sh
```

## What it does

| Step | Script | Checks |
|---|---|---|
| 0 | `make_dummy_adata.py` (seeds 0 and 1 → `data/dummy_a`, `data/dummy_b`), `make_test_configs.py` | Builds `adata_pan.h5ad` through `preprocessing/prepare_adata.py` |
| 1 | `nsv_runs/nsv_run_pipeline.py` | Both VAEs train and are saved, for `use_time_dependence: true` and `false`. The flag reaches the saved model: the `b_t` layer exists only when it is true. |
| 2 | `nsv_runs/compute_velo_run.py` | Velocity, fitted moments, latent time, vote reductions |
| 3 | `nsv_runs/score_gene_fit_v3_run.py` | All goodness-of-fit columns used by the gene query |
| 4 | `nsv_runs/generate_velo_stream_run.py` | Gene query, MURK filter (one gene flagged as a fake MURK gene), velocity graph, streams, pseudotime |
| 5 | `velocity_metrics/compute_cbdir_run.py` | CBDir and ICCoh for three "methods": `nSV`, `nSV_no_time`, and `shuffled` (nSV velocities permuted across cells; `make_shuffled_control.py`) |
| 6 | `velocity_metrics/plot_cbdir.py`, `velocity_metrics/benchmark_table_s2.py` | Benchmark figures and statistics, Table S2 |
| 7 | `analyses/gene_direction/gene_direction_benchmark.py`, `analyses/mu_var_corr/mu_var_corr.py` | Gene-level direction and moment correlations |
| 8 | `check_smoke_test.py` | Every expected output exists. Prints a short summary and `SMOKE TEST PASSED`. |

Logs are written to `tests/logs/`. Data and results go to `data/dummy_*` and
`tests/results_*` (both can be deleted).

## Synthetic data

`make_dummy_adata.py` simulates one linear trajectory: 800 cells, 80 kinetic
genes and 20 constant genes. Each kinetic gene is switched on at t = 0 and off
at a gene-specific time. Its mean and variance follow the moment equations of
the bursty model (degradation rate 1):

    dμ/dt  = f·B − μ
    dσ²/dt = f·B·(1 + 2B) + μ − 2σ²

At the same mean, the variance is therefore higher on the way up than on the
way down. Counts are negative binomial with that mean and variance, thinned
with a per-cell capture efficiency between 0.25 and 0.45, and split into
spliced and unspliced counts. Clusters A–D are quartiles of time.

With the epoch caps used here the fits are rough. Some datasets may end up with
few velocity genes, and CBDir values are not meaningful. The test checks that
the code runs and writes the expected outputs, not that the results are
accurate.

## Configs

`make_test_configs.py` writes `configs/*.yaml` from the manuscript templates in
`nsv_runs/` and `velocity_metrics/`. It replaces the dataset lists and sets
`max_epochs_scvi: 150`, `max_epochs_nsv: 600`, `n_epochs_kl_warmup: 100` and
`batch_size: 128`. All other settings come from the templates.
