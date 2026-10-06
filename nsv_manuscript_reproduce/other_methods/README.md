# `other_methods/`: runs of the compared methods

These are the wrappers used to run the six published methods (seven
configurations) on the seven benchmark datasets. Each folder has a `run.sh`, a
batch script and a YAML config listing the seven datasets. Inputs are the
published objects in `nsv_manuscript_reproduce/data/<dataset>/` (see
"Datasets" in [../README.md](../README.md)). Paths in the configs are relative to
the method's folder (`../../data/...`).

```bash
bash other_methods/scvelo_runs/run.sh                                  # any folder; PY=/path/to/python to choose the interpreter
PY=/path/to/celldancer_env/bin/python bash other_methods/celldancer_runs/run.sh
```

| Folder | Method and version | Environment | Output read by `nsv_runs/velo_stream_config_template_<method>_4throot.yaml` |
|---|---|---|---|
| `scvelo_runs/` | scVelo 0.3.4, dynamical mode (`recover_dynamics`) and stochastic mode (entries with `mode: stochastic`) | main (`environment_gpu_final.yml`) | `<ds>/adata_scvelo.h5ad`, `<ds>/scvelo_stochastic/adata_scvelo.h5ad` |
| `velovi_runs/` | veloVI 0.3.1: `velovi_run_pipeline.py` trains, then `compute_velovi_velocity_run.py` computes velocities (25 posterior samples) | main | `<ds>/adata_velovi.h5ad` |
| `velovae_runs/` | VeloVAE 0.1.2 | main | `<ds>/<name>_velovae_out.h5ad` |
| `unitvelo_runs/` | UniTVelo 0.2.5.2 (`FIT_OPTION: '1'`, unified time; `'2'` for the dentate gyrus) | separate: Python 3.10.20, tensorflow 2.21.0, scvelo 0.3.4, scanpy 1.11.5, anndata 0.11.4, numpy 1.26.4, pandas 2.3.3, scikit-learn 1.7.2 (full list: `unitvelo_runs/conda_list_unitvelo_env.txt`) | `<ds>/<name>_unitvelo_out.h5ad` |
| `celldancer_runs/` | cellDancer 1.1.7 | separate: Python 3.7.6, torch 1.10.0, scvelo 0.2.5, scanpy 1.9.3, pytorch-lightning 1.5.2 | `<ds>/<name>_celldancer_out.h5ad` |
| – | TFvelo 1.0 (run from its GitHub repository; wrapper not included) | main | `<ds>/adata_tfvelo.h5ad` |

The main environment is Python 3.10.13 with torch 2.3.1, scvi-tools 1.0.4,
scanpy 1.9.3 and anndata 0.8.0.

Each wrapper preprocesses the published object itself and selects its own
2,000 highly variable genes, so the gene sets of the methods overlap but are
not identical (see "Gene sets" below). In every wrapper `X` is first set to the
total counts (`layers['matrix']` or `layers['total']` if present, else spliced +
unspliced), and the moments use 30 PCs and 30 neighbours. Each method's own outputs
then go to step 4 of the nSV pipeline (`../nsv_runs/run_generate_velo_stream.sh all`).
That step applies the method's velocity genes and, for the two erythroid
datasets, removes the annotated MURK genes (`MURK_gene == False`).

## Gene sets

Every method, nSV included, keeps 2,000 highly variable genes (scanpy
`highly_variable_genes`, Seurat flavour), but each selects them on its own
object after its own filters, so the selected genes differ between methods:

| Method | Filters and normalisation before HVG selection | After HVG selection |
|---|---|---|
| nSV (`preprocessing/prepare_adata.py`) | genes in ≥ 3 cells; cells with ≥ 200 genes; genes with ≥ 20 counts; `normalize_total(1e4)`, `log1p` | compositional distorters and mitochondrial genes removed; then model selection (step 1) and the velocity-gene filter (step 4) |
| scVelo (both modes), veloVI | `scv.pp.filter_and_normalize(min_shared_counts=20)`, `log1p` | veloVI: `velovi.preprocess_data` (min-max scaling, drops genes with a poor steady-state fit) |
| UniTVelo, VeloVAE, cellDancer | `scv.pp.filter_genes(min_shared_counts=20)`, `scv.pp.normalize_per_cell`, `log1p` | each method's own velocity genes |
| TFvelo | its own preprocessing (wrapper not included) | genes with a fitted model (`~fit_scaling_y.isna()`) |

For the cortex, nSV used the labelling object and the other methods the
splicing object, so the HVGs were also selected on different objects. The
genes each method finally uses for its velocity field are those flagged
`reliable_velo_gene` by step 4 (`nsv_runs/velo_stream_config_template_<method>_4throot.yaml`).

## Consistency with the rest of the pipeline

Checked without running the methods (the published objects are not in this
repository):

- Every config lists the seven benchmark datasets. scVelo has 14 entries: the
  seven in dynamical mode, and the seven in stochastic mode writing to
  `<ds>/scvelo_stochastic/`.
- For every method and dataset, the output file a wrapper writes is the file
  that `nsv_runs/velo_stream_config_template_<method>_4throot.yaml` reads, and
  the stream outputs are the files the CBDir configs in `velocity_metrics/` read.
- All wrappers parse; all configs load. Paths in the configs are relative to
  each method's folder (`../../data/...`).
