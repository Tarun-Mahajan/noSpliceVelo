# 1. Basic gene filter
sc.pp.filter_genes(adata, min_cells=3)

# 2. Basic cell filter (to remove empty droplets)
sc.pp.filter_cells(adata, min_genes=200)


# remove mitochondrial genes
genes_mt = np.where(adata.var_names.str.startswith('MT-') == True)[0]
adata.var['mito'] = adata.var_names.str.startswith('MT-')
print(f'{len(genes_mt)} mitochondrial genes.')
# adata = adata[:, genes_not_mt].copy()

sc.pp.calculate_qc_metrics(adata, qc_vars=['mito'], inplace=True, percent_top=None, log1p=False)


sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mito'],
             jitter=0.4, multi_panel=True)

# adata_pan = adata_filtered.copy()
adata_pan = adata.copy()
sc.pp.filter_genes(adata_pan, min_counts=20)
sc.pp.normalize_total(adata_pan, target_sum=1e4)
sc.pp.log1p(adata_pan)
adata_pan.layers['log_counts'] = adata_pan.X.copy()
# adata_pan_all = adata_pan.copy()


sc.pp.highly_variable_genes(
    adata_pan,
    flavor="seurat",
    n_top_genes=2000,
    subset=False,
)

import gc
gc.collect()


import numpy as np
import pandas as pd
import scanpy as sc

def identify_compositional_distorters_multires(
    adata, 
    resolutions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0], 
    layer=None, 
    fraction_threshold=0.05
):
    """
    Unsupervised identification of compositional distorters across multiple 
    Leiden clustering granularities. Bypasses the need for a priori pseudotime 
    or cell-state annotations.
    
    Parameters:
    -----------
    adata : AnnData
        The annotated data matrix containing RAW counts. 
        Assumes sc.pp.neighbors() has already been run on a standard embedding (e.g. PCA).
    resolutions : list of float
        The list of Leiden clustering resolutions to evaluate.
    layer : str, optional
        Layer containing raw counts. If None, uses adata.X.
    fraction_threshold : float
        The maximum average fraction of total cellular counts a gene is allowed 
        to occupy within any cluster before being flagged as a distorter.
        
    Returns:
    --------
    df_summary : pd.DataFrame
        Summary dataframe containing the peak fraction and the resolution/cluster where it occurred.
    all_distorters : list
        Deduplicated list of genes flagged as compositional distorters.
    """
    # 1. Ensure neighborhood graph exists for clustering
    if 'neighbors' not in adata.uns:
        raise ValueError(
            "Neighbor graph not found in adata.uns. Please run sc.pp.neighbors(adata) "
            "on your standard preprocessed/PCA representation before running this function."
        )

    # 2. Extract raw counts and compute cell-specific fractions (denominators)
    X_raw = adata.layers[layer] if layer is not None else adata.X
    if not isinstance(X_raw, np.ndarray):
        X_raw = X_raw.toarray()  # Handle sparse matrices safely
        
    cell_totals = X_raw.sum(axis=1, keepdims=True)
    cell_totals[cell_totals == 0] = 1  # Avoid division by zero
    fraction_matrix = X_raw / cell_totals
    
    # Create baseline dataframe to store results per gene
    df_fractions = pd.DataFrame(fraction_matrix, columns=adata.var_names)
    
    # Dictionaries to track the maximum footprint of each gene across all sweeps
    global_max_fraction = {gene: 0.0 for gene in adata.var_names}
    global_peak_resolution = {gene: None for gene in adata.var_names}
    global_peak_cluster = {gene: None for gene in adata.var_names}
    
    print(f"Starting compositional scan across {len(resolutions)} Leiden resolutions...")
    
    # 3. Loop through granularities
    for res in resolutions:
        key_name = f'leiden_res_{res}'
        
        # Run unsupervised clustering at current resolution
        sc.tl.leiden(adata, resolution=res, key_added=key_name)
        
        # Group computed fractions by these temporary cluster assignments
        df_fractions['temp_cluster'] = adata.obs[key_name].values
        group_means = df_fractions.groupby('temp_cluster').mean()
        
        # Find the max fraction and corresponding cluster for every gene at this resolution
        res_max_fractions = group_means.max(axis=0)
        res_peak_clusters = group_means.idxmax(axis=0)
        
        # Update global tracker if this resolution uncovers a higher localized dominance
        for gene in adata.var_names:
            current_res_fraction = res_max_fractions[gene]
            if current_res_fraction > global_max_fraction[gene]:
                global_max_fraction[gene] = current_res_fraction
                global_peak_resolution[gene] = res
                global_peak_cluster[gene] = res_peak_clusters[gene]
                
        print(f" -> Resolution {res} completed: {len(group_means.index)} clusters evaluated.")
        
    # 4. Compile compiled metrics into a final summary DataFrame
    df_summary = pd.DataFrame({
        'max_localized_fraction': pd.Series(global_max_fraction),
        'detecting_resolution': pd.Series(global_peak_resolution),
        'detecting_cluster': pd.Series(global_peak_cluster)
    })
    
    # 5. Flag outliers exceeding the safety threshold
    df_summary['is_distorter'] = df_summary['max_localized_fraction'] > fraction_threshold
    all_distorters = df_summary[df_summary['is_distorter']].index.tolist()
    
    print(f"\nScan complete. Flagged {len(all_distorters)} generalizable compositional distorters.")
    
    return df_summary.sort_values(by='max_localized_fraction', ascending=False), all_distorters

sc.tl.pca(adata_pan)
sc.pp.neighbors(adata_pan, n_pcs=adata_pan.obsm['X_pca'].shape[1]) # Graph is ready

# 3. Find distorters across a wide resolution spectrum without cell type labels
summary_df, distorter_genes = identify_compositional_distorters_multires(
    adata_pan, 
    # resolutions=[0.1, 0.4, 1.0, 2.0], 
    layer="counts", 
    fraction_threshold=0.040
)

adata_pan_filter = adata_pan[
    :, adata_pan.var['highly_variable']
].copy()
adata_pan_filter


genes_keep = adata_pan_filter.var_names[
    ~adata_pan_filter.var_names.isin(distorter_genes)
]
adata_pan = adata_pan_filter[:, genes_keep].copy()


adata_pan = adata_pan[:, ~adata_pan.var['mito']].copy()