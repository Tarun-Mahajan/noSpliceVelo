"""Build the noSpliceVelo input object (`adata_pan.h5ad`) from a published AnnData.

These are the preprocessing steps used for the manuscript (originally run
interactively; see prepare_adata_temp.py for the notebook cells):

  1. gene filter      sc.pp.filter_genes(min_cells=3)
  2. cell filter      sc.pp.filter_cells(min_genes=200)          (empty droplets)
  3. QC metrics       mitochondrial flag var['mito'] (gene-name prefix 'MT-' for human,
                      'mt-' for mouse; names from var_names or --gene-names-col)
  4. gene filter      sc.pp.filter_genes(min_counts=20)
  5. normalisation    normalize_total(target_sum=1e4) + log1p  ->  X, layers['log_counts']
  6. HVG              highly_variable_genes(flavor='seurat', n_top_genes=2000)
                      (adds var['dispersions_norm'], used by the step-6 burstiness filter)
  7. embedding        PCA (50 PCs) + kNN graph on all genes
  8. distorters       genes whose mean fraction of a cell's total counts exceeds
                      0.04 in any Leiden cluster, scanned over resolutions
                      0.1-2.0 (identify_compositional_distorters_multires)
  9. subset           highly variable genes, minus distorters, minus mitochondrial genes

The raw total counts are taken from layers['counts']; if absent, from
layers['matrix'] or layers['total'], else spliced + unspliced, else .X. Published UMAPs and cell annotations are kept.

Usage
-----
    python prepare_adata.py --in endocrinogenesis_day15.h5ad --out ../data/Pancreas/adata_pan.h5ad --organism mouse
    python prepare_adata.py --in human_cd34_bone_marrow.h5ad --out ../data/BoneMarrow/adata_pan.h5ad --organism human
    # var_names are Ensembl IDs: take the gene symbols from a var column
    python prepare_adata.py --in in.h5ad --out out.h5ad --organism mouse --gene-names-col Gene_Symbol

Organisms of the seven datasets: human for bone marrow and human erythroid;
mouse for pancreas, cortex, mouse erythroid, organoid and dentate gyrus.

For the erythroid datasets, add a boolean `var['MURK_gene']` (genes with
multiple rate kinetics, Barile et al. 2021) before the velocity-stream step.
"""

import argparse
import gc

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

MITO_PREFIX = {"human": "MT-", "mouse": "mt-"}
DEFAULT_RESOLUTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0]


def identify_compositional_distorters_multires(adata, resolutions=DEFAULT_RESOLUTIONS,
                                               layer=None, fraction_threshold=0.05):
    """Genes that dominate the counts of some cell population.

    For every Leiden resolution, the mean fraction of each cell's total counts
    taken by each gene is computed per cluster; a gene is a distorter if its
    largest cluster mean over all resolutions exceeds `fraction_threshold`.
    Needs a neighbour graph in adata (sc.pp.neighbors).

    Returns (summary DataFrame sorted by max fraction, list of distorter genes).
    """
    if "neighbors" not in adata.uns:
        raise ValueError("Neighbor graph not found; run sc.pp.neighbors first.")
    X_raw = adata.layers[layer] if layer is not None else adata.X
    if not isinstance(X_raw, np.ndarray):
        X_raw = X_raw.toarray()
    cell_totals = X_raw.sum(axis=1, keepdims=True)
    cell_totals[cell_totals == 0] = 1
    df_fractions = pd.DataFrame(X_raw / cell_totals, columns=adata.var_names)

    global_max = {g: 0.0 for g in adata.var_names}
    peak_res = {g: None for g in adata.var_names}
    peak_cluster = {g: None for g in adata.var_names}
    print(f"Starting compositional scan across {len(resolutions)} Leiden resolutions...")
    for res in resolutions:
        key = f"leiden_res_{res}"
        sc.tl.leiden(adata, resolution=res, key_added=key)
        df_fractions["temp_cluster"] = adata.obs[key].values
        group_means = df_fractions.groupby("temp_cluster").mean()
        res_max = group_means.max(axis=0)
        res_arg = group_means.idxmax(axis=0)
        for g in adata.var_names:
            if res_max[g] > global_max[g]:
                global_max[g] = res_max[g]
                peak_res[g] = res
                peak_cluster[g] = res_arg[g]
        print(f" -> Resolution {res} completed: {len(group_means.index)} clusters evaluated.")
    summary = pd.DataFrame({"max_localized_fraction": pd.Series(global_max),
                            "detecting_resolution": pd.Series(peak_res),
                            "detecting_cluster": pd.Series(peak_cluster)})
    summary["is_distorter"] = summary["max_localized_fraction"] > fraction_threshold
    distorters = summary.index[summary["is_distorter"]].tolist()
    print(f"\nScan complete. Flagged {len(distorters)} compositional distorters.")
    return summary.sort_values("max_localized_fraction", ascending=False), distorters


def prepare_adata(adata, counts_layer=None, min_cells=3, min_genes=200, min_counts=20,
                  n_top_genes=2000, organism="human", mito_prefix=None, gene_names_col=None,
                  fraction_threshold=0.04, resolutions=DEFAULT_RESOLUTIONS):
    """See the module docstring. `mito_prefix` overrides the organism default
    (MITO_PREFIX); `gene_names_col` names a var column with gene symbols, used for
    the mitochondrial flag when var_names are Ensembl IDs."""
    if mito_prefix is None:
        if organism not in MITO_PREFIX:
            raise ValueError(f"organism must be one of {sorted(MITO_PREFIX)} (or pass mito_prefix)")
        mito_prefix = MITO_PREFIX[organism]
    adata = adata.copy()

    # raw total counts in layers['counts']
    if counts_layer is not None:
        adata.layers["counts"] = adata.layers[counts_layer].copy()
    elif "counts" not in adata.layers:
        # same order as the wrappers of the compared methods (other_methods/)
        for key in ("matrix", "total"):
            if key in adata.layers:
                adata.layers["counts"] = adata.layers[key].copy()
                break
        else:
            if "spliced" in adata.layers and "unspliced" in adata.layers:
                adata.layers["counts"] = adata.layers["spliced"] + adata.layers["unspliced"]
            else:
                adata.layers["counts"] = adata.X.copy()
    adata.layers["counts"] = sp.csr_matrix(adata.layers["counts"])
    # the filters below act on .X, which must hold the raw counts here
    adata.X = adata.layers["counts"].copy()

    # 1-2. basic gene and cell filters
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.filter_cells(adata, min_genes=min_genes)

    # 3. mitochondrial genes and QC metrics
    if gene_names_col is not None:
        if gene_names_col not in adata.var:
            raise KeyError(f"gene_names_col '{gene_names_col}' not in adata.var")
        names = adata.var[gene_names_col].astype(str)
    else:
        names = adata.var_names.to_series()
    adata.var["mito"] = names.str.startswith(mito_prefix).values
    print(f"{int(adata.var['mito'].sum())} mitochondrial genes (prefix '{mito_prefix}').")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mito"], inplace=True, percent_top=None, log1p=False)

    # 4-6. gene filter, normalisation, highly variable genes
    adata_pan = adata.copy()
    sc.pp.filter_genes(adata_pan, min_counts=min_counts)
    sc.pp.normalize_total(adata_pan, target_sum=1e4)
    sc.pp.log1p(adata_pan)
    adata_pan.layers["log_counts"] = adata_pan.X.copy()
    sc.pp.highly_variable_genes(adata_pan, flavor="seurat",
                                n_top_genes=min(n_top_genes, adata_pan.n_vars), subset=False)
    gc.collect()

    # 7-8. embedding and compositional distorters (all genes)
    sc.tl.pca(adata_pan)
    sc.pp.neighbors(adata_pan, n_pcs=adata_pan.obsm["X_pca"].shape[1])
    summary, distorters = identify_compositional_distorters_multires(
        adata_pan, resolutions=resolutions, layer="counts", fraction_threshold=fraction_threshold)
    adata_pan.uns["compositional_distorters"] = list(distorters)

    # 9. HVG, minus distorters, minus mitochondrial genes
    adata_pan = adata_pan[:, adata_pan.var["highly_variable"]].copy()
    adata_pan = adata_pan[:, ~adata_pan.var_names.isin(distorters)].copy()
    adata_pan = adata_pan[:, ~adata_pan.var["mito"]].copy()
    print(f"final object: {adata_pan.n_obs} cells x {adata_pan.n_vars} genes")
    return adata_pan


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build adata_pan.h5ad for noSpliceVelo")
    ap.add_argument("--in", dest="inp", required=True, help="input .h5ad")
    ap.add_argument("--out", required=True, help="output adata_pan.h5ad")
    ap.add_argument("--counts-layer", default=None,
                    help="layer with total raw counts (default: layers['counts'], else "
                         "'matrix', else 'total', else spliced+unspliced, else .X)")
    ap.add_argument("--min-genes", type=int, default=200)
    ap.add_argument("--n-top-genes", type=int, default=2000)
    ap.add_argument("--organism", choices=sorted(MITO_PREFIX), default="human",
                    help="sets the mitochondrial gene prefix: human 'MT-', mouse 'mt-'")
    ap.add_argument("--mito-prefix", default=None, help="override the organism's prefix")
    ap.add_argument("--gene-names-col", default=None,
                    help="var column with gene symbols (when var_names are Ensembl IDs)")
    ap.add_argument("--fraction-threshold", type=float, default=0.04)
    a = ap.parse_args(argv)
    adata = sc.read_h5ad(a.inp)
    out = prepare_adata(adata, counts_layer=a.counts_layer, min_genes=a.min_genes,
                        n_top_genes=a.n_top_genes, organism=a.organism,
                        mito_prefix=a.mito_prefix, gene_names_col=a.gene_names_col,
                        fraction_threshold=a.fraction_threshold)
    out.write_h5ad(a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
