"""Config-driven wrapper to compute velocity confidence across multiple methods and datasets.

Features:
- Uses `velocity_confidence_scaled.py` for scaled and stock velocity confidence calculations.
- By default runs `confidence_report` to generate multiple confidence versions (scvelo_stock, mad, zscore, signed_sqrt_mad, spearman).
- Pre-parses h5ad files across all methods for each dataset to find the intersection of valid (non-NaN) velocity genes (`intersection_mode=True` by default).
- Supports optional `gene_names_col` per dataset/method to map gene symbols from `adata.var[gene_names_col]` without modifying `var_names`.
- Uses `use_genes="all"` by default when intersection mode is disabled.
- Saves aggregated results to CSV and Parquet per dataset.

Usage:
    python compute_velocity_confidence_run.py velo_confidence_global_config_template.yaml
"""

import os
import sys
import gc
import time
import traceback
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scvelo as scv

import velocity_confidence_scaled as vcs

try:
    from eff_gene_report import eff_genes_report
except ImportError:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from eff_gene_report import eff_genes_report

# Pipeline parameters allowed in the per-method `defaults:` block or per-dataset.
_PIPELINE_PARAMS = (
    "xkey",
    "vkey",
    "recompute_graph",
    "n_neighbors",
    "rep",
    "approx",
    "sqrt_transform",
    "gene_query",
    "apply_gene_query_always",
    "use_confidence_report",
    "use_genes",
    "intersection_mode",
    "gene_names_col",
    "root_power",
    "cluster_key",
    "cluster_map",
    "cluster_map_drop_unmapped",
)

_DEFAULT_FALLBACKS = {
    "xkey": "Ms",
    "vkey": "velocity",
    "recompute_graph": False,
    "n_neighbors": 30,
    "rep": "X_pca",
    "approx": False,
    "sqrt_transform": True,
    "gene_query": None,
    "apply_gene_query_always": False,
    "use_confidence_report": True,
    "use_genes": "all",
    "intersection_mode": True,
    "gene_names_col": None,
    "root_power": 2,
    # Cell-type / cluster labels. REQUIRED: they give the aggregated table a
    # within-dataset unit of replication, without which every downstream test has
    # to treat the dataset as the only unit (7 of them) or the cell (which is
    # pseudoreplicated). Set explicitly to the string "none" to opt out.
    "cluster_key": None,
    "cluster_map": None,
    "cluster_map_drop_unmapped": False,
}


def apply_cluster_map(labels, cluster_map, drop_unmapped=False, context=""):
    """Remap cluster levels, accepting either level names or categorical codes as keys.

    A pandas Categorical stores labels, not codes, so a map written as {0: 'HSC'}
    silently matches nothing if the column's levels are strings. Both spellings are
    resolved here, and a map that matches no level at all raises rather than
    quietly NaN-ing the column — that failure mode cost a debugging session in the
    CBDir pipeline and is not worth repeating.
    """
    if not cluster_map:
        return labels
    labels = pd.Series(labels).astype(object)
    present = pd.unique(labels.dropna())
    by_name = {str(k): v for k, v in cluster_map.items()}
    codes = {str(i): lv for i, lv in enumerate(sorted(map(str, present)))}
    resolved = {}
    for k, v in cluster_map.items():
        ks = str(k)
        if ks in {str(p) for p in present}:
            resolved[ks] = v
        elif ks in codes:
            resolved[codes[ks]] = v
    if not resolved:
        raise ValueError(
            f"cluster_map{context} matched none of the {len(present)} cluster levels "
            f"present ({list(present)[:8]}...). Keys given: {list(by_name)[:8]}"
        )
    out = labels.map(lambda v: resolved.get(str(v), np.nan if drop_unmapped else v))
    if out.notna().sum() == 0:
        raise ValueError(f"cluster_map{context} left every cell unmapped.")
    n_un = int(labels.notna().sum() - out.notna().sum())
    if n_un:
        _log(f"  cluster_map{context}: {n_un} cells dropped as unmapped")
    return out.to_numpy()


def extract_clusters(adata, cluster_key, cluster_map=None, drop_unmapped=False,
                     context=""):
    """Per-cell cluster labels as a Series indexed by obs_names."""
    if cluster_key is None or str(cluster_key).lower() == "none":
        return None
    if cluster_key not in adata.obs:
        raise KeyError(
            f"cluster_key '{cluster_key}'{context} not found in adata.obs. "
            f"Available: {list(adata.obs.columns)[:20]}"
        )
    labels = adata.obs[cluster_key].astype(object).to_numpy()
    labels = apply_cluster_map(labels, cluster_map, drop_unmapped, context)
    s = pd.Series(labels, index=adata.obs_names, name="cluster")
    _log(f"  cluster_key '{cluster_key}': {s.nunique(dropna=True)} levels over "
         f"{int(s.notna().sum())}/{len(s)} cells")
    return s


def _get_valid_velocity_genes(adata_path, vkey, gene_names_col=None):
    """Extract gene symbols that have no NaN velocity values across cells."""
    if not os.path.exists(adata_path):
        return None
    try:
        adata = ad.read_h5ad(adata_path)
        if vkey not in adata.layers:
            return None
        V = adata.layers[vkey]
        if hasattr(V, "toarray"):
            V = V.toarray()
        V = np.asarray(V, dtype=np.float64)
        valid_mask = ~np.isnan(V).any(axis=0)

        if gene_names_col and gene_names_col in adata.var.columns:
            gene_symbols = set(adata.var[gene_names_col].iloc[valid_mask].astype(str))
        else:
            gene_symbols = set(adata.var_names[valid_mask].astype(str))

        del adata
        gc.collect()
        return gene_symbols
    except Exception as e:
        _log(f"WARNING: Failed to parse valid genes from {adata_path}: {e}")
        return None


def compute_dataset_gene_intersection(dataset_name, method_dataset_entries):
    """Find the intersection of valid (non-NaN) velocity gene symbols across all methods for a given dataset."""
    valid_gene_sets = []
    _log(f"Pre-parsing gene intersection for dataset '{dataset_name}' across {len(method_dataset_entries)} method(s)...")

    for m_name, entry in method_dataset_entries:
        adata_path = entry.get("adata_path")
        vkey = entry.get("vkey", "velocity")
        gene_names_col = entry.get("gene_names_col", None)
        if not adata_path:
            continue
        genes = _get_valid_velocity_genes(adata_path, vkey, gene_names_col=gene_names_col)
        if genes is not None:
            valid_gene_sets.append(genes)
            _log(f"  Method '{m_name}': {len(genes)} valid non-NaN gene symbols found (gene_names_col={gene_names_col}).")
        else:
            _log(f"  Method '{m_name}': could not load valid genes (missing file or layer).")

    if valid_gene_sets:
        common_genes = sorted(set.intersection(*valid_gene_sets))
        _log(f"Intersection complete for '{dataset_name}': {len(common_genes)} common gene symbols retained.")
        return common_genes
    else:
        _log(f"WARNING: No valid gene sets obtained for dataset '{dataset_name}'.")
        return None


def compute_participation_ratio(V, root_power=2):
    """Compute participation ratio for velocity matrix V (n_cells, n_genes) on gene_subset."""
    if hasattr(V, "toarray"):
        V = V.toarray()
    V = np.asarray(V, dtype=np.float64)
    valid_cols = ~np.isnan(V).any(axis=0)
    V = V[:, valid_cols]
    if V.shape[1] == 0:
        return np.nan
    # Variance stabilization transform: sign(V) * |V|**(1 / root_power)
    p_exp = 1.0 / float(root_power)
    V = np.sign(V) * (np.abs(V) ** p_exp)
    # Center per gene across cells
    V = V - V.mean(axis=0, keepdims=True)
    n_cells, n_genes = V.shape
    if n_genes <= n_cells:
        G = V.T @ V
    else:
        G = V @ V.T
    lam = np.linalg.eigvalsh(G)
    lam = np.clip(lam, 0, None)
    lam_sum = lam.sum()
    lam_sq_sum = (lam**2).sum()
    if lam_sq_sum <= 0:
        return 0.0
    return float((lam_sum**2) / lam_sq_sum)


def compute_eff_genes(V, root_power=2):
    """Compute median effective genes per cell for velocity matrix V (n_cells, n_genes) on gene_subset."""
    if hasattr(V, "toarray"):
        V = V.toarray()
    V = np.asarray(V, dtype=np.float64)
    valid_cols = ~np.isnan(V).any(axis=0)
    V = V[:, valid_cols]
    if V.shape[1] == 0:
        return np.nan
    p_exp = 1.0 / float(root_power)
    V_stab = np.sign(V) * (np.abs(V) ** p_exp)
    p = np.abs(V_stab) ** float(root_power)
    denom = (p ** 2).sum(axis=1)
    denom[denom == 0] = np.nan
    eff_per_cell = (p.sum(axis=1) ** 2) / denom
    return float(np.nanmedian(eff_per_cell))


def compute_confidence_for_dataset(
    name,
    adata_path,
    xkey="Ms",
    vkey="velocity",
    recompute_graph=False,
    n_neighbors=30,
    rep="X_pca",
    approx=False,
    sqrt_transform=True,
    gene_query=None,
    apply_gene_query_always=False,
    use_confidence_report=True,
    use_genes="all",
    intersection_mode=True,
    intersected_genes=None,
    gene_names_col=None,
    root_power=2,
    cluster_key=None,
    cluster_map=None,
    cluster_map_drop_unmapped=False,
):
    """Load AnnData, compute velocity confidence, and return (versions, clusters, ...).

    `cluster_key` names the .obs column holding cell-type labels; those labels are
    carried into the aggregated table so downstream analyses have a within-dataset
    unit of replication rather than only the dataset and the cell.
    """
    _log(f"Loading AnnData from {adata_path}...")
    if not os.path.exists(adata_path):
        raise FileNotFoundError(f"adata_path not found: {adata_path}")

    adata = ad.read_h5ad(adata_path)
    _log(f"AnnData loaded successfully. Shape: {adata.shape}")

    clusters = extract_clusters(adata, cluster_key, cluster_map,
                                cluster_map_drop_unmapped, context=f" for '{name}'")

    # Verify layer existence
    if xkey not in adata.layers and xkey != "X":
        raise KeyError(f"xkey '{xkey}' not found in adata.layers")
    if vkey not in adata.layers:
        raise KeyError(f"vkey '{vkey}' not found in adata.layers")

    # Determine gene set / mask to use, ensuring NaN velocity genes are ALWAYS excluded
    V_layer = adata.layers[vkey]
    if hasattr(V_layer, "toarray"):
        V_dense = V_layer.toarray()
    else:
        V_dense = V_layer
    valid_non_nan_mask = ~np.isnan(np.asarray(V_dense, dtype=np.float64)).any(axis=0)
    n_valid_non_nan = int(valid_non_nan_mask.sum())
    _log(f"Method velocity layer '{vkey}': {n_valid_non_nan}/{adata.n_vars} genes have non-NaN velocity values across all cells.")

    if intersection_mode and intersected_genes is not None:
        intersected_set = set(intersected_genes)
        if gene_names_col and gene_names_col in adata.var.columns:
            _log(f"Filtering dataset '{name}' using var['{gene_names_col}'] against {len(intersected_set)} intersected gene symbols (preserving original var_names)...")
            symbol_mask = np.asarray(adata.var[gene_names_col].astype(str).isin(intersected_set), dtype=bool)
        else:
            _log(f"Filtering dataset '{name}' using var_names against {len(intersected_set)} intersected gene symbols...")
            symbol_mask = np.asarray(adata.var_names.astype(str).isin(intersected_set), dtype=bool)
        effective_genes = symbol_mask & valid_non_nan_mask
    else:
        if use_genes == "all":
            _log(f"intersection_mode=False: using all {n_valid_non_nan} valid non-NaN velocity genes for this method.")
            effective_genes = valid_non_nan_mask
        elif gene_names_col and gene_names_col in adata.var.columns and isinstance(use_genes, (list, set, np.ndarray)):
            _log(f"intersection_mode=False: filtering using var['{gene_names_col}'] and excluding NaN velocity genes...")
            symbol_mask = np.asarray(adata.var[gene_names_col].astype(str).isin(set(use_genes)), dtype=bool)
            effective_genes = symbol_mask & valid_non_nan_mask
        elif isinstance(use_genes, (list, set, np.ndarray)):
            _log(f"intersection_mode=False: filtering using var_names and excluding NaN velocity genes...")
            symbol_mask = np.asarray(adata.var_names.astype(str).isin(set(use_genes)), dtype=bool)
            effective_genes = symbol_mask & valid_non_nan_mask
        elif isinstance(use_genes, str) and use_genes in adata.var.columns:
            var_col_mask = np.asarray(adata.var[use_genes], dtype=bool)
            effective_genes = var_col_mask & valid_non_nan_mask
        else:
            effective_genes = valid_non_nan_mask

    # Apply gene_query filter if apply_gene_query_always is True (even when recompute_graph is False)
    if gene_query and apply_gene_query_always:
        try:
            query_genes = adata.var.query(gene_query).index
            query_mask = np.asarray(adata.var_names.isin(query_genes), dtype=bool)
            prev_cnt = int(effective_genes.sum())
            effective_genes = effective_genes & query_mask
            _log(f"apply_gene_query_always=True: applied gene_query '{gene_query}', reducing gene count from {prev_cnt} to {int(effective_genes.sum())}.")
        except Exception as e:
            _log(f"WARNING: Failed to apply gene_query '{gene_query}' on adata.var: {e}")

    _log(f"Final effective gene set count for calculation: {int(effective_genes.sum())} genes.")

    if recompute_graph:
        _log("recompute_graph=True: recomputing neighbor and velocity graph...")
        if rep not in adata.obsm:
            if rep == "X_pca":
                _log("X_pca not found in obsm. Running PCA first...")
                sc.pp.pca(adata)
            else:
                raise KeyError(f"Specified representation rep='{rep}' not found in adata.obsm")

        n_pcs = adata.obsm[rep].shape[1]
        _log(f"Computing neighbor graph (n_neighbors={n_neighbors}, use_rep={rep}, n_pcs={n_pcs})...")
        sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors, use_rep=rep)

        gene_subset = None
        if gene_query:
            _log(f"Applying gene query filter: {gene_query}")
            filtered_var = adata.var.query(gene_query)
            selected_genes = filtered_var.index.tolist()
            if selected_genes:
                _log(f"Selected {len(selected_genes)} genes matching query.")
                gene_subset = selected_genes

        _log(f"Computing velocity graph (xkey={xkey}, vkey={vkey}, approx={approx}, sqrt_transform={sqrt_transform})...")
        scv.tl.velocity_graph(
            adata,
            xkey=xkey,
            vkey=vkey,
            approx=approx,
            sqrt_transform=sqrt_transform,
            gene_subset=gene_subset,
        )
    else:
        _log("recompute_graph=False: using existing neighbor/velocity graph from h5ad.")

    # Swap expression layer into adata.layers["Ms"] for confidence calculations
    if xkey == "X":
        expression_layer = adata.X.copy() if hasattr(adata.X, "copy") else adata.X
    else:
        expression_layer = adata.layers[xkey]

    if "Ms" in adata.layers:
        Ms_copy = adata.layers["Ms"].copy()
    else:
        Ms_copy = None

    adata.layers["Ms"] = expression_layer

    num_genes = int(effective_genes.sum())
    _log(f"Computing participation ratio, effective genes, and eff_genes_report using velocity layer '{vkey}' for {num_genes} genes in gene_subset (root_power={root_power})...")
    try:
        V_subset = adata.layers[vkey][:, effective_genes]
        pr_val = compute_participation_ratio(V_subset, root_power=root_power)
        eff_genes_val = compute_eff_genes(V_subset, root_power=root_power)
        power_val = 1.0 / float(root_power)
        eff_rep_out, _ = eff_genes_report(V_subset, power=power_val)
        _log(f"Participation ratio: {pr_val:.4f}, Effective genes: {eff_genes_val:.2f}, PR_pop: {eff_rep_out.get('pr_population', np.nan):.2f}")
    except Exception as e:
        _log(f"WARNING: Failed to compute PR or eff_genes_report for vkey='{vkey}': {e}")
        pr_val = np.nan
        eff_genes_val = np.nan
        eff_rep_out = {}

    result_series_dict = {}
    try:
        if use_confidence_report:
            _log(f"Running confidence_report for vkey='{vkey}'...")
            report_df = vcs.confidence_report(adata, vkey=vkey, use_genes=effective_genes)
            _log(f"confidence_report generated columns: {list(report_df.columns)}")
            for col in report_df.columns:
                result_series_dict[col] = pd.Series(report_df[col].values, index=adata.obs_names)
        else:
            _log(f"Running velocity_confidence for vkey='{vkey}'...")
            vcs.velocity_confidence(adata, vkey=vkey, use_genes=effective_genes)
            conf_key = f"{vkey}_confidence"
            if conf_key not in adata.obs:
                raise KeyError(f"Expected column '{conf_key}' not found in adata.obs after calculation.")
            result_series_dict["default"] = pd.Series(adata.obs[conf_key].values, index=adata.obs_names)
    finally:
        if Ms_copy is not None:
            adata.layers["Ms"] = Ms_copy
            del Ms_copy
        else:
            if "Ms" in adata.layers:
                del adata.layers["Ms"]

    del adata
    gc.collect()

    return result_series_dict, clusters, num_genes, pr_val, eff_genes_val, eff_rep_out


def save_aggregated_results(confidence_collector, dataset_dir_paths, method_order, save_parquet=False, str_suffix=None):
    """Aggregate confidence columns for each dataset across methods and save CSV (and optionally Parquet)."""
    _log("")
    _log("=== Saving Aggregated Results ===")

    suffix = ""
    if str_suffix is not None and str(str_suffix).strip() != "":
        s = str(str_suffix).strip()
        suffix = s if (s.startswith("_") or s.startswith("-")) else f"_{s}"

    for dataset_name, method_dict in confidence_collector.items():
        if not method_dict:
            _log(f"No results collected for dataset '{dataset_name}'; skipping.")
            continue

        dir_path = dataset_dir_paths.get(dataset_name, f"./results/{dataset_name}")
        if not os.path.exists(dir_path):
            _log(f"Creating output directory: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)

        all_barcodes = []
        seen_barcodes = set()
        for method_entry in method_dict.values():
            if isinstance(method_entry, dict) and "versions" in method_entry:
                version_map = method_entry["versions"]
            else:
                version_map = method_entry
            for series in version_map.values():
                for bc in series.index:
                    if bc not in seen_barcodes:
                        seen_barcodes.add(bc)
                        all_barcodes.append(bc)

        df = pd.DataFrame(index=all_barcodes)
        df.index.name = "cell_barcode"


        methods_in_data = [m for m in method_order if m in method_dict]
        for m in method_dict:
            if m not in methods_in_data:
                methods_in_data.append(m)

        # One cluster column per dataset, merged across methods. Methods read the
        # same annotation from their own AnnData, so they should agree wherever
        # they share a cell; disagreement means the objects were annotated
        # differently and is worth a loud warning rather than a silent pick.
        cluster_col = pd.Series(index=df.index, dtype=object)
        n_conflict, conflict_examples = 0, []
        for m_name in methods_in_data:
            entry = method_dict.get(m_name)
            cl = entry.get("clusters") if isinstance(entry, dict) else None
            if cl is None:
                continue
            cl = cl.reindex(df.index)
            both = cluster_col.notna() & cl.notna()
            diff = both & (cluster_col.astype(str) != cl.astype(str))
            if diff.any():
                n_conflict += int(diff.sum())
                if not conflict_examples:
                    bc = df.index[diff][:3]
                    conflict_examples = [
                        f"{b}: {cluster_col[b]!r} vs {cl[b]!r} ({m_name})" for b in bc]
            cluster_col = cluster_col.where(cluster_col.notna(), cl)
        if n_conflict:
            _log(f"  WARNING: {n_conflict} cells have conflicting cluster labels across "
                 f"methods; keeping the first method's label. Examples: {conflict_examples}")
        if cluster_col.notna().any():
            df["cluster"] = cluster_col
            _log(f"  cluster column: {cluster_col.nunique(dropna=True)} levels, "
                 f"{int(cluster_col.notna().sum())}/{len(cluster_col)} cells labelled")
        else:
            _log("  WARNING: no cluster labels collected for this dataset - downstream "
                 "analyses will have no within-dataset unit of replication")

        for method_name in methods_in_data:
            method_entry = method_dict[method_name]
            if isinstance(method_entry, dict) and "versions" in method_entry:
                version_map = method_entry["versions"]
                num_genes = method_entry.get("num_genes", np.nan)
                pr_val = method_entry.get("participation_ratio", np.nan)
                eff_genes_val = method_entry.get("eff_genes", np.nan)
                eff_report = method_entry.get("eff_report", {})
            else:
                version_map = method_entry
                num_genes = np.nan
                pr_val = np.nan
                eff_genes_val = np.nan
                eff_report = {}

            for version_name, series in version_map.items():
                if version_name == "default":
                    col_name = f"{method_name}_velocity_confidence"
                else:
                    col_name = f"{method_name}_{version_name}_velocity_confidence"
                df[col_name] = series.reindex(df.index)

            # Create alias {method_name}_velocity_confidence if using report
            primary_col = f"{method_name}_mad_velocity_confidence"
            alias_col = f"{method_name}_velocity_confidence"
            if primary_col in df.columns and alias_col not in df.columns:
                df[alias_col] = df[primary_col]
            elif f"{method_name}_scvelo_stock_velocity_confidence" in df.columns and alias_col not in df.columns:
                df[alias_col] = df[f"{method_name}_scvelo_stock_velocity_confidence"]

            df[f"{method_name}_participation_ratio"] = pr_val
            df[f"{method_name}_num_genes"] = num_genes
            df[f"{method_name}_eff_genes"] = eff_genes_val

            if isinstance(eff_report, dict):
                for k, val in eff_report.items():
                    if isinstance(val, (int, float, np.integer, np.floating)):
                        df[f"{method_name}_{k}"] = val

        df = df.reset_index()

        out_base = os.path.join(dir_path, f"{dataset_name}_velocity_confidence{suffix}")
        csv_path = f"{out_base}.csv"

        _log(f"Dataset '{dataset_name}': saving aggregated table with shape {df.shape} and columns: {list(df.columns)}")
        df.to_csv(csv_path, index=False)
        _log(f"  Saved CSV    : {csv_path}")

        if save_parquet:
            parquet_path = f"{out_base}.parquet"
            df.to_parquet(parquet_path, index=False)
            _log(f"  Saved Parquet: {parquet_path}")


def _resolve_params(defaults, entry):
    merged = dict(defaults or {})
    for k, v in entry.items():
        if k in ("name", "adata_path", "dir_path"):
            continue
        if k in _PIPELINE_PARAMS:
            merged[k] = v
        else:
            _log(f"  WARNING: ignoring unknown parameter '{k}'")

    resolved = {}
    for k in _PIPELINE_PARAMS:
        if k in merged:
            resolved[k] = merged[k]
        else:
            resolved[k] = _DEFAULT_FALLBACKS[k]

    return resolved


def run_from_config(config_path, str_suffix=None):
    """Run velocity confidence pipeline across methods and datasets defined in YAML config files."""
    try:
        import yaml
    except ImportError as e:
        raise SystemExit(
            "PyYAML is required to read config files. Install it with `pip install pyyaml`."
        ) from e

    with open(config_path, "r") as fh:
        global_cfg = yaml.safe_load(fh)

    if not isinstance(global_cfg, dict) or "methods" not in global_cfg:
        raise ValueError("Global config YAML must be a mapping containing a 'methods:' key.")

    if str_suffix is None:
        str_suffix = global_cfg.get("str_suffix", None)

    methods_dict = global_cfg["methods"]
    if not isinstance(methods_dict, dict) or not methods_dict:
        raise ValueError("'methods' in global config must be a non-empty mapping of method_name -> config_path.")

    default_log = f"velocity_confidence_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_file = global_cfg.get("log_file", default_log)
    log_dir = os.path.dirname(os.path.abspath(log_file))
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    log_fh = _FilteredFile(open(log_file, "a"))
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    sys.stderr = _Tee(sys.__stderr__, log_fh)

    try:
        _log("=== Velocity Confidence Batch Run Started ===")
        _log(f"Global config file : {os.path.abspath(config_path)}")
        _log(f"Log file          : {os.path.abspath(log_file)}")
        _log(f"Methods           : {list(methods_dict.keys())}")
        if str_suffix:
            _log(f"str_suffix        : {str_suffix}")

        # Structure: method_datasets_by_name[d_name] = [(method_name, entry_params), ...]
        dataset_methods_map = defaultdict(list)
        dataset_dir_paths = {}

        method_cfgs = {}
        for method_name, method_cfg_path in methods_dict.items():
            if not os.path.exists(method_cfg_path):
                _log(f"WARNING: Config file for method '{method_name}' not found: {method_cfg_path}")
                continue
            with open(method_cfg_path, "r") as fh:
                m_cfg = yaml.safe_load(fh)
            if not isinstance(m_cfg, dict) or "datasets" not in m_cfg:
                _log(f"WARNING: Config file for method '{method_name}' is not a dict or missing 'datasets' key: {method_cfg_path}")
                continue
            method_cfgs[method_name] = m_cfg
            defaults = m_cfg.get("defaults", {})
            for entry in m_cfg.get("datasets", []):
                d_name = entry.get("name")
                if d_name and "adata_path" in entry and "dir_path" in entry:
                    params = _resolve_params(defaults, entry)
                    params["adata_path"] = entry["adata_path"]
                    params["dir_path"] = entry["dir_path"]
                    dataset_methods_map[d_name].append((method_name, params))
                    if d_name not in dataset_dir_paths:
                        dataset_dir_paths[d_name] = entry["dir_path"]

        # Fail fast on a missing cluster_key, before any heavy compute. Every
        # (method, dataset) must name the .obs column holding cell-type labels;
        # without it the aggregated table has no within-dataset unit and every
        # downstream test is stuck with the dataset (few) or the cell
        # (pseudoreplicated). `cluster_key: none` opts out explicitly.
        missing = [(m, d) for d, lst in dataset_methods_map.items()
                   for m, p in lst if p.get("cluster_key") in (None, "")]
        if missing:
            shown = ", ".join(f"{m}/{d}" for m, d in missing[:8])
            raise ValueError(
                f"cluster_key is required but missing for {len(missing)} "
                f"(method, dataset) pair(s): {shown}"
                f"{' ...' if len(missing) > 8 else ''}.\n"
                "Add it to each method config, either under `defaults:` or per dataset:\n"
                "    defaults:\n"
                "      cluster_key: clusters      # the .obs column with cell-type labels\n"
                "      cluster_map: null          # optional {level_or_code: new_label}\n"
                "To run without cell-type labels anyway, set `cluster_key: none`."
            )

        # Pre-compute gene intersections per dataset if enabled
        dataset_gene_intersections = {}
        for d_name, m_list in dataset_methods_map.items():
            intersection_enabled = any(p.get("intersection_mode", True) for _, p in m_list)
            if intersection_enabled:
                intersected = compute_dataset_gene_intersection(d_name, [(m, p) for m, p in m_list])
                dataset_gene_intersections[d_name] = intersected

        confidence_collector = defaultdict(dict)
        summary = []
        method_order = list(methods_dict.keys())

        for method_name, method_cfg_path in methods_dict.items():
            _log("")
            _log(f"==================================================")
            _log(f" Processing Method: {method_name}")
            _log(f" Config path: {os.path.abspath(method_cfg_path)}")
            _log(f"==================================================")

            if method_name not in method_cfgs:
                _log(f"  ERROR: Method '{method_name}' config not loaded; skipping.")
                summary.append((method_name, "ALL", "SKIPPED", 0.0))
                continue

            m_cfg = method_cfgs[method_name]
            defaults = m_cfg.get("defaults", {})
            datasets = m_cfg.get("datasets", [])

            for i, entry in enumerate(datasets, start=1):
                d_name = entry.get("name", f"dataset_{i}")
                _log("")
                _log(f"----- [{method_name}] dataset {i}/{len(datasets)}: {d_name} -----")

                if "adata_path" not in entry or "dir_path" not in entry:
                    _log(f"  ERROR: entry '{d_name}' missing paths; skipping")
                    summary.append((method_name, d_name, "SKIPPED (missing paths)", 0.0))
                    continue

                params = _resolve_params(defaults, entry)
                _log(f"  adata_path : {entry['adata_path']}")
                _log(f"  dir_path   : {entry['dir_path']}")
                _log(f"  params     : {params}")

                intersected_genes = dataset_gene_intersections.get(d_name)

                t0 = time.time()
                try:
                    res_dict, clusters, num_genes, pr_val, eff_genes_val, eff_rep_out = compute_confidence_for_dataset(
                        name=d_name,
                        adata_path=entry["adata_path"],
                        intersected_genes=intersected_genes,
                        **params,
                    )
                    dt = time.time() - t0
                    confidence_collector[d_name][method_name] = {
                        "versions": res_dict,
                        "clusters": clusters,
                        "num_genes": num_genes,
                        "participation_ratio": pr_val,
                        "eff_genes": eff_genes_val,
                        "eff_report": eff_rep_out,
                    }
                    _log(f"  DONE in {dt/60:.2f} min ({len(res_dict)} metrics; {num_genes} genes; PR={pr_val:.4f}; eff_genes={eff_genes_val:.2f})")
                    summary.append((method_name, d_name, "OK", dt))
                except Exception as e:
                    dt = time.time() - t0
                    _log(f"  FAILED after {dt/60:.2f} min: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    summary.append((method_name, d_name, f"FAILED ({type(e).__name__})", dt))

        # Save aggregated tables per dataset
        save_parquet = global_cfg.get("save_parquet", False)
        save_aggregated_results(
            confidence_collector,
            dataset_dir_paths,
            method_order,
            save_parquet=save_parquet,
            str_suffix=str_suffix,
        )

        _log("")
        _log("=== Batch Run Summary ===")
        for method_name, d_name, status, dt in summary:
            _log(f"  {method_name:<15s} {d_name:<25s} {status:<30s} {dt/60:6.2f} min")
        _log("=== Velocity Confidence Batch Run Finished ===")

    finally:
        log_fh.flush()
        log_fh.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__


# Output capture and filtering helpers
class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def close(self):
        self.flush()


class _FilteredFile:
    def __init__(self, fh):
        self._fh = fh
        self._buf = ""

    def write(self, data):
        self._buf += data
        import re
        _PROGRESS_RE = re.compile(r"^\s*(loss|epoch|\d+%|\[|\.|\*|-)", re.IGNORECASE)
        while "\n" in self._buf:
            nl = self._buf.find("\n")
            cr = self._buf.rfind("\r", 0, nl)
            if cr != -1:
                line = self._buf[cr + 1:nl]
                self._buf = self._buf[nl + 1:]
                eff = line.rsplit("\r", 1)[-1]
                if eff.strip() and _PROGRESS_RE.search(eff):
                    continue
                self._fh.write(eff + "\n")
                self._fh.flush()
                continue
            line = self._buf[:nl]
            self._buf = self._buf[nl + 1:]
            eff = line.rsplit("\r", 1)[-1]
            if eff.strip() and _PROGRESS_RE.search(eff):
                continue
            self._fh.write(eff + "\n")
        self._fh.flush()

    def flush(self):
        try:
            self._fh.flush()
        except Exception:
            pass

    def close(self):
        rem = self._buf.rsplit("\r", 1)[-1]
        import re
        _PROGRESS_RE = re.compile(r"^\s*(loss|epoch|\d+%|\[|\.|\*|-)", re.IGNORECASE)
        if rem.strip() and not _PROGRESS_RE.search(rem):
            self._fh.write(rem)
        self._buf = ""
        self._fh.flush()
        self._fh.close()


def _timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg):
    print(f"[{_timestamp()}] {msg}", flush=True)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute velocity confidence across multiple methods and save aggregated per-dataset tables."
    )
    parser.add_argument(
        "config",
        help="Path to the global YAML config file (see velo_confidence_global_config_template.yaml).",
    )
    parser.add_argument(
        "--str-suffix",
        "--suffix",
        dest="str_suffix",
        default=None,
        help="Optional string suffix to append to all output filenames (e.g. '_v2').",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_from_config(args.config, str_suffix=args.str_suffix)
    sys.exit(0)
