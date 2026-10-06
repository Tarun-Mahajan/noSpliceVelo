# `preprocessing/`: building the input object

`prepare_adata.py` turns a published AnnData (see "Datasets" in the top-level
README) into the `adata_pan.h5ad` that `../nsv_runs/nsv_run_pipeline.py` reads.
It is a command-line version of the steps used for the manuscript.
`prepare_adata_temp.py` holds the original notebook cells.

```bash
python prepare_adata.py --in ../data/Pancreas/endocrinogenesis_day15.h5ad --out ../data/Pancreas/adata_pan.h5ad --organism mouse
python prepare_adata.py --in ../data/BoneMarrow/human_cd34_bone_marrow.h5ad --out ../data/BoneMarrow/adata_pan.h5ad --organism human
# var_names are Ensembl IDs: read the gene symbols from a var column
python prepare_adata.py --in in.h5ad --out out.h5ad --organism mouse --gene-names-col Gene_Symbol
```

`--organism` is `human` for bone marrow and human erythroid, and `mouse` for
pancreas, cortex, mouse erythroid, organoid and dentate gyrus.

| Step | Call | Default |
|---|---|---|
| 1. Gene filter | `sc.pp.filter_genes(min_cells=3)` | |
| 2. Cell filter (empty droplets) | `sc.pp.filter_cells(min_genes=200)` | `--min-genes` |
| 3. QC metrics | mitochondrial flag `var['mito']`: gene names starting with `MT-` (human) or `mt-` (mouse), taken from `var_names` or from the `var` column given with `--gene-names-col`; `sc.pp.calculate_qc_metrics` | `--organism`, `--mito-prefix`, `--gene-names-col` |
| 4. Gene filter | `sc.pp.filter_genes(min_counts=20)` | |
| 5. Normalisation | `normalize_total(target_sum=1e4)`, `log1p`; result in `X` and `layers['log_counts']` | |
| 6. Highly variable genes | `highly_variable_genes(flavor='seurat', n_top_genes=2000)`; gives `var['dispersions_norm']`, used by the burstiness criterion of step 6 of the Methods | `--n-top-genes` |
| 7. Embedding | PCA (50 PCs) and kNN graph on all genes | |
| 8. Compositional distorters | Genes whose mean share of a cell's total counts exceeds 0.04 in any Leiden cluster, scanned over resolutions 0.1–1.0 and 2.0 (`identify_compositional_distorters_multires`); stored in `uns['compositional_distorters']` | `--fraction-threshold` |
| 9. Subset | Highly variable genes, minus distorters, minus mitochondrial genes | |

The raw total counts are read from `layers['counts']`. If that layer is
missing, the script uses `layers['matrix']`, then `layers['total']`, then
spliced + unspliced, then `.X`; this is the same order the wrappers in
`../other_methods/` use. `--counts-layer` overrides the choice. Published UMAPs and cell annotations are kept.

Notes:

- Erythroid datasets: the velocity-gene filter of all methods uses a boolean
  `var['MURK_gene']` (MURK genes of Barile et al., 2021). It must be present in
  `adata_pan.h5ad` and in the objects given to the compared methods.
- Mitochondrial genes: the prefix is case-sensitive, so set `--organism` (or
  `--mito-prefix`) to match the dataset. If `var_names` are Ensembl IDs, pass the
  symbol column with `--gene-names-col`; otherwise no gene is flagged.
- Cortex: nSV used the total counts (new + pre-existing) of the
  metabolic-labelling object of the same experiment.
- Human erythroid: all 35,877 cells.

`../tests/make_dummy_adata.py` calls `prepare_adata()` on synthetic counts.
