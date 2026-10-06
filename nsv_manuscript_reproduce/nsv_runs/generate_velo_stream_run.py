"""Config-driven wrapper around scvelo and scanpy to generate velocity embedding stream plots.

For each dataset in a YAML config, this:
  1. Loads the AnnData object from `adata_path`.
  2. Optionally filters genes using a pandas query on `adata.var` (e.g. metadata cols).
  3. Computes neighbor graph and velocity graph.
  4. Generates and saves a velocity embedding stream plot.
  5. Optionally computes and saves a velocity pseudotime plot.
  6. Optionally saves the updated AnnData to `dir_path`/`output_filename`.

Usage:
    python generate_velo_stream_run.py velo_stream_config_template_nsv.yaml
"""

import os
import sys
import gc
import time
import traceback
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scvelo as scv
import matplotlib.pyplot as plt

# Make sibling modules (scv_velocity_graph_new, velocity_scale_diagnostics)
# importable when the script is launched from another working directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from scv_velocity_graph_new import velocity_graph

# Pipeline parameters allowed in the global `defaults:` block or per-dataset.
_PIPELINE_PARAMS = (
    "xkey",
    "vkey",
    "basis",
    "n_neighbors",
    "rep",
    "label_col",
    "density",
    "approx",
    "sqrt_transform",
    "sqrt_transform_func",
    "compute_pseudotime",
    "save_adata",
    "output_filename",
    "gene_query",
    "new_layers",
    "quant_filter_col",
    "q_",
    "filter_by_latent_time",
    "latent_time_col",
    "boolean_gene_col",
    "leiden_res",
    "scale_velo",
    "alpha_criterion",
    "alpha_max",
    "scale_mu",
    "scale_type",
)


def run_velo_stream(
    name,
    adata_path,
    dir_path,
    xkey="Ms",
    vkey="velocity",
    basis="umap",
    n_neighbors=30,
    rep="X_pca",
    label_col="clusters",
    density=3,
    approx=False,
    sqrt_transform=True,
    sqrt_transform_func=np.sqrt,
    compute_pseudotime=True,
    save_adata=False,
    output_filename="adata_stream.h5ad",
    gene_query=None,
    new_layers=None,
    quant_filter_col=None,
    q_=0.2,
    filter_by_latent_time=False,
    latent_time_col="fit_t",
    boolean_gene_col="MURK_genes",
    leiden_res=0.8,
    scale_velo=False,
    alpha_criterion="expr_rank_wtd",
    alpha_max=1.0,
    scale_mu=False,
    scale_type="mean_abs",
):
    """Process a single dataset to compute velocity stream and pseudotime plots."""
    _log(f"Loading AnnData from {adata_path}...")
    if not os.path.exists(adata_path):
        raise FileNotFoundError(f"adata_path not found: {adata_path}")
    
    adata = ad.read_h5ad(adata_path)
    _log(f"AnnData loaded successfully. Shape: {adata.shape}")

    # 0. Custom Layer Computation
    if new_layers:
        _log(f"Computing custom new layers...")
        eval_env = {
            'adata': adata,
            'np': np,
            'pd': pd,
        }
        for lay_def in new_layers:
            if not isinstance(lay_def, dict) or 'layer_name' not in lay_def or 'expression' not in lay_def:
                _log(f"WARNING: Invalid layer definition {lay_def}. Must be dict with 'layer_name' and 'expression'. Skipping.")
                continue
            lay_name = lay_def['layer_name']
            expression = lay_def['expression']
            _log(f"Computing layer '{lay_name}' using expression: {expression}")
            try:
                # Evaluate expression
                result = eval(expression, eval_env)
                adata.layers[lay_name] = result
                _log(f"Successfully added layer '{lay_name}' with shape {result.shape}")
            except Exception as e:
                _log(f"ERROR: Failed to compute layer '{lay_name}' with expression '{expression}': {type(e).__name__}: {e}")
                raise e

    # 1. Quantile Computation & Gene Query Filtering
    adata.var['reliable_velo_gene'] = True
    gene_subset = None

    quant_ = None
    local_dict = {}

    if quant_filter_col:
        if quant_filter_col in adata.var.columns:
            try:
                col_vals = adata.var[quant_filter_col].dropna()
                quant_ = float(np.quantile(col_vals, q_))
                local_dict['quant_'] = quant_
                _log(f"Computed quantile q_={q_} for column '{quant_filter_col}': quant_ = {quant_:.6f}")
                if gene_query is None:
                    gene_query = f"{quant_filter_col} >= @quant_"
                    _log(f"Auto-generated gene_query: {gene_query}")
            except Exception as e:
                _log(f"ERROR: Failed to compute quantile for column '{quant_filter_col}' with q_={q_}: {type(e).__name__}: {e}")
                raise e
        else:
            _log(f"WARNING: quant_filter_col '{quant_filter_col}' not found in adata.var columns.")

    if gene_query:
        _log(f"Applying gene query filter: {gene_query}")
        try:
            # Use pandas DataFrame query to select genes, providing local_dict if quant_ was computed
            if local_dict:
                filtered_var = adata.var.query(gene_query, local_dict=local_dict)
            else:
                filtered_var = adata.var.query(gene_query)

            # Store the boolean flag in adata.var['reliable_velo_gene']
            adata.var['reliable_velo_gene'] = adata.var.index.isin(filtered_var.index)
            selected_genes = filtered_var.index.tolist()
            if not selected_genes:
                _log(f"WARNING: Gene query '{gene_query}' returned no genes! Skipping query filtering, using all genes.")
                adata.var['reliable_velo_gene'] = True
            else:
                _log(f"Flagged {len(selected_genes)} genes matching the query as reliable_velo_gene=True.")
                gene_subset = selected_genes
        except Exception as e:
            _log(f"ERROR: Failed to apply gene query '{gene_query}': {type(e).__name__}: {e}")
            raise e

    # 1b. Latent Time Gene Filtering (filter_latent_time_genes.py logic)
    if filter_by_latent_time:
        _log(f"Applying latent_time gene filtering (latent_time_col='{latent_time_col}', boolean_gene_col='{boolean_gene_col}', leiden_res={leiden_res})...")
        try:
            adata_tmp = adata[:, adata.var['reliable_velo_gene']].copy()
            adata_gene = adata_tmp.T

            if latent_time_col in adata_tmp.layers:
                lt_data = adata_tmp.layers[latent_time_col].T
                if hasattr(lt_data, "toarray"):
                    lt_data = lt_data.toarray()
                adata_gene.X = np.asarray(lt_data, dtype=np.float64)
            else:
                raise KeyError(f"latent_time_col '{latent_time_col}' not found in adata.layers")

            n_obs_g = adata_gene.n_obs
            if n_obs_g < 3:
                _log(f"WARNING: Too few genes ({n_obs_g}) to perform latent time gene clustering; skipping.")
            else:
                n_pcs_gene = min(50, max(1, n_obs_g - 1), max(1, adata_gene.n_vars - 1))
                sc.pp.pca(adata_gene, n_comps=n_pcs_gene)
                n_neigh = min(30, max(1, n_obs_g - 1))
                sc.pp.neighbors(adata_gene, n_pcs=n_pcs_gene, n_neighbors=n_neigh)
                sc.tl.leiden(adata_gene, resolution=leiden_res, key_added="leiden", n_iterations=2)

                if boolean_gene_col in adata_gene.obs.columns:
                    # bool_mask = np.asarray(adata_gene.obs[boolean_gene_col], dtype=bool)
                    # unique_clusts_to_rm = np.unique(adata_gene.obs.loc[bool_mask, 'leiden'].values)
                    df_ = adata_gene.obs[['leiden', boolean_gene_col]].value_counts(dropna=False, sort=False).reset_index()
                    df_ = df_[df_[boolean_gene_col]]
                    df_['perc'] = (df_['count'] / df_['count'].sum()) * 100
                    df_ = df_[df_['perc'] > 10.0]
                    unique_clusts_to_rm = df_['leiden'].values.copy().astype(object)
                    _log(f"Found {len(unique_clusts_to_rm)} leiden cluster(s) containing {boolean_gene_col} genes: {unique_clusts_to_rm}")
                    keep_mask = ~adata_gene.obs['leiden'].isin(unique_clusts_to_rm)
                    final_selected_genes = adata_gene.obs_names[keep_mask].tolist()
                else:
                    _log(f"WARNING: boolean_gene_col '{boolean_gene_col}' not found in adata.var; keeping all genes.")
                    final_selected_genes = adata_gene.obs_names.tolist()

                adata.var['reliable_velo_gene'] = adata.var.index.isin(final_selected_genes)
                gene_subset = final_selected_genes
                _log(f"Latent time gene filtering complete: retained {len(gene_subset)} genes (filtered out {n_obs_g - len(gene_subset)} genes).")

            del adata_tmp, adata_gene
            gc.collect()
        except Exception as e:
            _log(f"ERROR: Failed to apply latent time gene filtering: {type(e).__name__}: {e}")
            raise e

    # 1c. Velocity & Expression Scaling via Alpha Selection (velocity_scale_diagnostics.py)
    effective_vkey = vkey
    effective_xkey = xkey
    if scale_velo or scale_mu:
        _log(f"Running alpha selection and scaling (scale_velo={scale_velo}, scale_mu={scale_mu}, scale_type='{scale_type}')...")
        from velocity_scale_diagnostics import select_alpha, _dense, _gene_scale
        try:
            if xkey == "X" and "X" not in adata.layers:
                adata.layers["X"] = adata.X.copy() if hasattr(adata.X, "copy") else adata.X

            if 'reliable_velo_gene' in adata.var.columns:
                adata_filtered = adata[:, adata.var['reliable_velo_gene']].copy()
            else:
                adata_filtered = adata.copy()

            alpha_rec = select_alpha(
                adata_filtered,
                vkey=vkey,
                xkey=xkey,
                criterion=alpha_criterion,
                alpha_max=alpha_max,
                scale=scale_type,
                dataset=name,
                verbose=True,
            )
            alpha_val = float(alpha_rec["alpha"])
            _log(f"Selected alpha for dataset '{name}': alpha = {alpha_val} (scale_type='{scale_type}')")

            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)

            alpha_csv_path = os.path.join(dir_path, f"{name}_alpha_diagnostics.csv")
            pd.DataFrame([alpha_rec]).to_csv(alpha_csv_path, index=False)
            _log(f"Saved alpha diagnostics CSV to: {alpha_csv_path}")

            # Compute velocity-derived denominator using _gene_scale(V, scale_type)**alpha
            V_dense = _dense(adata.layers[vkey]).astype(float)
            sd_v = _gene_scale(V_dense, how=scale_type)
            velocity_denom = sd_v ** alpha_val

            if scale_velo:
                scaled_v = V_dense / velocity_denom
                scaled_vkey = f"{vkey}_a{alpha_val:g}"
                adata.layers[scaled_vkey] = scaled_v
                effective_vkey = scaled_vkey
                _log(f"Scaled velocity saved to layer '{effective_vkey}'. Downstream tasks will use vkey='{effective_vkey}'.")

            if scale_mu:
                if xkey == "X":
                    X_dense = _dense(adata.X).astype(float)
                else:
                    X_dense = _dense(adata.layers[xkey]).astype(float)

                scaled_x = X_dense / velocity_denom
                scaled_xkey = f"{xkey}_a{alpha_val:g}"
                adata.layers[scaled_xkey] = scaled_x
                effective_xkey = scaled_xkey
                _log(f"Scaled expression xkey saved to layer '{effective_xkey}' using velocity-derived denominator _gene_scale(v, '{scale_type}')**alpha. Downstream tasks will use xkey='{effective_xkey}'.")

            del adata_filtered, V_dense
            gc.collect()
        except Exception as e:
            _log(f"ERROR: Failed to perform scaling: {type(e).__name__}: {e}")
            raise e

    # 2. Neighbors computation
    # Check if we need to compute PCA if rep is 'X_pca' but missing
    if rep not in adata.obsm:
        if rep == "X_pca":
            _log("X_pca not found in obsm. Running PCA first...")
            sc.pp.pca(adata)
        else:
            raise KeyError(f"Specified representation rep='{rep}' not found in adata.obsm")

    n_pcs = adata.obsm[rep].shape[1]
    _log(f"Computing neighbor graph with n_neighbors={n_neighbors}, use_rep={rep}, n_pcs={n_pcs}...")
    sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors, use_rep=rep)

    # Resolve sqrt_transform_func if passed as string or callable
    if isinstance(sqrt_transform_func, str):
        s_raw = sqrt_transform_func.strip()
        s_func = s_raw.lower()
        if s_func in ("sqrt", "np.sqrt"):
            effective_sqrt_func = np.sqrt
        elif s_func in ("cbrt", "np.cbrt"):
            effective_sqrt_func = np.cbrt
        elif s_func in ("log", "np.log"):
            effective_sqrt_func = np.log
        elif s_func in ("log1p", "np.log1p"):
            effective_sqrt_func = np.log1p
        elif s_func in ("none", "identity", "null"):
            effective_sqrt_func = None
        elif hasattr(np, s_func):
            effective_sqrt_func = getattr(np, s_func)
        else:
            # Custom expression evaluation, e.g. "x**(1/4)" or "lambda x: np.sign(x)*np.abs(x)**(1/4)"
            try:
                eval_env = {"np": np, "numpy": np}
                if s_raw.startswith("lambda"):
                    effective_sqrt_func = eval(s_raw, eval_env)
                else:
                    effective_sqrt_func = eval(f"lambda x: {s_raw}", eval_env)
                _log(f"Parsed custom sqrt_transform_func expression: '{s_raw}'")
            except Exception as e:
                raise ValueError(f"Failed to parse custom sqrt_transform_func expression '{sqrt_transform_func}': {e}")
    else:
        effective_sqrt_func = sqrt_transform_func

    # 3. Velocity Graph computation
    _log(f"Computing velocity graph (xkey={effective_xkey}, vkey={effective_vkey}, approx={approx}, sqrt_transform={sqrt_transform}, sqrt_transform_func={effective_sqrt_func}, gene_subset={'None' if gene_subset is None else len(gene_subset)})...")
    velocity_graph(
        adata,
        xkey=effective_xkey,
        vkey=effective_vkey,
        approx=approx,
        sqrt_transform=sqrt_transform,
        sqrt_transform_func=effective_sqrt_func,
        gene_subset=gene_subset
    )

    # Ensure output directory exists
    if not os.path.exists(dir_path):
        _log(f"Creating output directory: {dir_path}")
        os.makedirs(dir_path)

    # 4. Stream plot generation
    _log(f"Generating velocity embedding stream plot (basis={basis}, color={label_col}, vkey={effective_vkey})...")
    
    # Check if the coloring column exists in adata.obs. If not, color by default or warn
    color_col = label_col
    if color_col not in adata.obs:
        _log(f"WARNING: label_col '{color_col}' not found in adata.obs. Looking for any obs column to color...")
        if len(adata.obs.columns) > 0:
            color_col = adata.obs.columns[0]
            _log(f"Falling back to coloring by: '{color_col}'")
        else:
            color_col = None
            _log("No obs column available for coloring.")

    # velocity_embedding_stream plot
    scv.pl.velocity_embedding_stream(
        adata,
        basis=basis,
        vkey=effective_vkey,
        density=density,
        color=color_col,
        show=False,
        figsize=(6, 6)
    )
    
    stream_plot_path = os.path.join(dir_path, f"{name}_velocity_stream.png")
    plt.savefig(stream_plot_path, bbox_inches='tight', dpi=150)
    plt.close()
    _log(f"Saved stream plot to {stream_plot_path}")

    # 5. Pseudotime plot generation
    if compute_pseudotime:
        for col_ in ['root_cells', 'end_points']:
            if col_ in adata.obs.columns:
                _log(f"Deleting {col_} from adata.obs")
                del adata.obs[col_]
        _log(f"Computing velocity pseudotime (vkey={effective_vkey}, xkey={effective_xkey})...")
        # scVelo reads expression from layers['Ms']; swap the xkey layer in for
        # the call and restore the original Ms (or remove it) afterwards. This
        # replaces TFvelo's velocity_pseudotime(modality=xkey) used previously.
        _ms_backup = adata.layers["Ms"].copy() if "Ms" in adata.layers else None
        _swap = effective_xkey != "Ms" and (effective_xkey in adata.layers or effective_xkey == "X")
        if _swap:
            adata.layers["Ms"] = adata.X.copy() if effective_xkey == "X" else adata.layers[effective_xkey]
        try:
            scv.tl.velocity_pseudotime(adata, vkey=effective_vkey)
        finally:
            if _swap:
                if _ms_backup is not None:
                    adata.layers["Ms"] = _ms_backup
                else:
                    del adata.layers["Ms"]
            del _ms_backup
        
        pt_key = f"{effective_vkey}_pseudotime"
        if pt_key not in adata.obs:
            if "velocity_pseudotime" in adata.obs:
                pt_key = "velocity_pseudotime"
            else:
                raise KeyError("Could not find velocity pseudotime in adata.obs after calculation.")
        
        _log(f"Generating velocity pseudotime scatter plot (color={pt_key})...")
        scv.pl.scatter(
            adata,
            color=pt_key,
            basis=basis,
            cmap='gnuplot',
            show=False
        )
        
        pt_plot_path = os.path.join(dir_path, f"{name}_velocity_pseudotime.png")
        plt.savefig(pt_plot_path, bbox_inches='tight', dpi=150)
        plt.close()
        _log(f"Saved pseudotime plot to {pt_plot_path}")

    # 6. Save modified AnnData
    if save_adata:
        out_path = os.path.join(dir_path, output_filename)
        _log(f"Saving updated AnnData object to {out_path}...")
        adata.write_h5ad(out_path)
        _log("AnnData saved successfully.")

    plt.close('all')
    return stream_plot_path


# =============================================================================
# Logging / Output capture wrappers
# =============================================================================

class _Tee:
    """Write everything to several streams at once (console + log file)."""

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


import re as _re
_PROGRESS_RE = _re.compile(
    r"\d{1,3}%\s*\|"
    r"|it/s[,\]]"
    r"|s/it[,\]]"
    r"|^Epoch\s+\d+/\d+"
)


class _FilteredFile:
    """File wrapper that drops in-place progress-bar updates from the log."""

    def __init__(self, fh):
        self._fh = fh
        self._buf = ""

    def write(self, data):
        self._buf += data
        while True:
            nl = self._buf.find("\n")
            if nl == -1:
                cr = self._buf.rfind("\r")
                if cr != -1:
                    self._buf = self._buf[cr + 1:]
                break
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
        if rem.strip() and not _PROGRESS_RE.search(rem):
            self._fh.write(rem)
        self._buf = ""
        self._fh.flush()
        self._fh.close()


def _timestamp():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg):
    print(f"[{_timestamp()}] {msg}", flush=True)


def _normalize_config(cfg):
    if isinstance(cfg, list):
        return {}, cfg
    if isinstance(cfg, dict):
        datasets = cfg.get("datasets")
        if datasets is None:
            raise ValueError("Config dict must contain a 'datasets:' list")
        if not isinstance(datasets, list):
            raise ValueError("'datasets' must be a list")
        return cfg, datasets
    raise ValueError("Top-level YAML must be a list or a mapping")


def _resolve_params(defaults, entry):
    merged = dict(defaults or {})
    for k, v in entry.items():
        if k in ("name", "adata_path", "dir_path"):
            continue
        if k in _PIPELINE_PARAMS:
            merged[k] = v
        else:
            _log(f"  WARNING: ignoring unknown parameter '{k}'")
            
    # Resolve all pipeline params with defaults
    default_fallbacks = {
        "xkey": "Ms",
        "vkey": "velocity",
        "basis": "umap",
        "n_neighbors": 30,
        "rep": "X_pca",
        "label_col": "clusters",
        "density": 3,
        "approx": False,
        "sqrt_transform": True,
        "sqrt_transform_func": np.sqrt,
        "compute_pseudotime": True,
        "save_adata": False,
        "output_filename": "adata_stream.h5ad",
        "gene_query": None,
        "new_layers": None,
        "quant_filter_col": None,
        "q_": 0.2,
        "filter_by_latent_time": False,
        "latent_time_col": "fit_t",
        "boolean_gene_col": "MURK_genes",
        "leiden_res": 0.8,
        "scale_velo": False,
        "alpha_criterion": "expr_rank_wtd",
        "alpha_max": 1.0,
        "scale_mu": False,
        "scale_type": "mean_abs",
    }
    
    resolved = {}
    for k in _PIPELINE_PARAMS:
        if k in merged:
            resolved[k] = merged[k]
        else:
            resolved[k] = default_fallbacks[k]
            
    return resolved


def run_from_config(config_path):
    """Run velocity stream pipeline for every dataset in a YAML config, in sequence."""
    try:
        import yaml
    except ImportError as e:
        raise SystemExit(
            "PyYAML is required to read the config file. "
            "Install it with `pip install pyyaml`."
        ) from e

    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)

    global_cfg, datasets = _normalize_config(cfg)
    defaults = global_cfg.get("defaults", {}) if isinstance(global_cfg, dict) else {}

    from datetime import datetime
    default_log = f"velo_stream_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_file = global_cfg.get("log_file", default_log) if isinstance(global_cfg, dict) else default_log
    log_dir = os.path.dirname(os.path.abspath(log_file))
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_fh = _FilteredFile(open(log_file, "a"))
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    sys.stderr = _Tee(sys.__stderr__, log_fh)

    try:
        _log("=== velocity stream batch run started ===")
        _log(f"config file : {os.path.abspath(config_path)}")
        _log(f"log file    : {os.path.abspath(log_file)}")
        _log(f"datasets    : {len(datasets)}")
        if defaults:
            _log(f"defaults    : {defaults}")

        summary = []
        for i, entry in enumerate(datasets, start=1):
            name = entry.get("name", f"dataset_{i}")
            _log("")
            _log(f"----- [{i}/{len(datasets)}] {name} -----")

            if "adata_path" not in entry or "dir_path" not in entry:
                _log(f"  ERROR: entry '{name}' must define both 'adata_path' and 'dir_path'; skipping")
                summary.append((name, "SKIPPED (missing paths)", 0.0))
                continue

            params = _resolve_params(defaults, entry)
            _log(f"  adata_path : {entry['adata_path']}")
            _log(f"  dir_path   : {entry['dir_path']}")
            _log(f"  params     : {params}")

            t0 = time.time()
            try:
                out_path = run_velo_stream(
                    name=name,
                    adata_path=entry["adata_path"],
                    dir_path=entry["dir_path"],
                    **params
                )
                dt = time.time() - t0
                _log(f"  DONE in {dt/60:.2f} min -> {out_path}")
                summary.append((name, "OK", dt))
            except Exception as e:
                dt = time.time() - t0
                _log(f"  FAILED after {dt/60:.2f} min: {type(e).__name__}: {e}")
                traceback.print_exc()
                summary.append((name, f"FAILED ({type(e).__name__})", dt))

            # Clear memory caches and matplotlib figures
            plt.close('all')
            gc.collect()

        _log("")
        _log("=== batch run summary ===")
        for name, status, dt in summary:
            _log(f"  {name:<30s} {status:<28s} {dt/60:6.2f} min")
        _log("=== velocity stream batch run finished ===")
    finally:
        plt.close('all')
        log_fh.flush()
        log_fh.close()
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__


def _parse_args(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate velocity embedding stream and pseudotime plots "
                    "for datasets described in a YAML config file."
    )
    parser.add_argument(
        "config",
        help="Path to the YAML config file (see velo_stream_config_template_nsv.yaml).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_from_config(args.config)
    sys.exit(0)
