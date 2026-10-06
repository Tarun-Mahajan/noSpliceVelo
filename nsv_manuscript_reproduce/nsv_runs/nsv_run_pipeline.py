import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scvelo as scv
import scanpy as sc
import seaborn as sns
import torch
import gc
import importlib
from scipy.special import digamma, polygamma
from scipy.sparse import issparse
import torch.nn.functional as F
import scvi


def compute_mu_var_naive(adata, n_neighbors=30, use_rep='X_latent'):
    """Compute neighbor-smoothed naive mean/variance of raw counts.

    Returns (mu_naive_smooth, var_naive_smooth), each of shape (n_cells, n_genes).
    Used to populate the 'mu_naive_smooth' / 'var_naive_smooth' layers required
    by SCVIModified.setup_anndata when they are missing from the input adata.
    """
    if 'neighbors' not in adata.uns:
        sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors)
    elif adata.uns['neighbors']['params']['n_neighbors'] < n_neighbors or \
         adata.uns['neighbors']['params']['use_rep'] != use_rep:
        print(f"Warning: Existing neighbors were computed with n_neighbors="
              f"{adata.uns['neighbors']['params']['n_neighbors']} and use_rep="
              f"'{adata.uns['neighbors']['params']['use_rep']}'. Recomputing neighbors "
              f"with n_neighbors={n_neighbors} and use_rep='{use_rep}'.")
        sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors)

    counts = adata.layers['counts']
    if issparse(counts):
        counts = counts.A  # Convert to dense if sparse

    n_cells = adata.n_obs

    mu_naive = np.zeros_like(counts, dtype=float)
    var_naive = np.zeros_like(counts, dtype=float)
    for i in range(n_cells):
        neighbors_indices = np.where(
            adata.uns['neighbors']['connectivities'][i, :].toarray().flatten() != 0)[0]
        neighbors_indices = np.union1d(neighbors_indices, [i])
        neighbor_counts = counts[neighbors_indices, :]
        mu_naive[i, :] = np.mean(neighbor_counts, axis=0)
        var_naive[i, :] = np.var(neighbor_counts, axis=0)

    mu_naive_smooth = np.zeros_like(counts, dtype=float)
    var_naive_smooth = np.zeros_like(counts, dtype=float)
    for i in range(n_cells):
        neighbors_indices = np.where(
            adata.uns['neighbors']['connectivities'][i, :].toarray().flatten() != 0)[0]
        neighbors_indices = np.union1d(neighbors_indices, [i])
        neighbor_mu = mu_naive[neighbors_indices, :]
        neighbor_var = var_naive[neighbors_indices, :]
        mu_naive_smooth[i, :] = np.mean(neighbor_mu, axis=0)
        var_naive_smooth[i, :] = np.mean(neighbor_var, axis=0)

    return mu_naive_smooth, var_naive_smooth

# ---------------------------------------------------------------------------
# Make the model code in ../nsv importable, independent of the working
# directory. Set NSV_SRC to point at a different copy of the model code.
# ---------------------------------------------------------------------------
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
NSV_SRC = os.environ.get("NSV_SRC", os.path.abspath(os.path.join(_HERE, "..", "nsv")))
for _p in (NSV_SRC, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# First VAE: joint capture efficiency + expression mean/variance.
from scvi_modified_capture_efficiency_model import SCVIModified
# Second VAE: noSpliceVelo (polar features, Student-t likelihood, shared
# per-cell time-dependence factor for burst frequency and burst size of the
# final up-branch state; see nosplicevelo_module_v5_polar.py).
from nosplicevelo_model_v5_polar import noSpliceVelo as noSpliceVelo_v4_polar

def run_nsv_pipeline(
    adata,
    dir_path,
    continue_from_prev=False,
    n_hidden_scvi=64,
    n_latent_scvi=10,
    n_hidden_nsv=64,
    n_latent_nsv=10,
    n_layers=1,
    n_layers_scvi=None,
    n_layers_nsv=None,
    n_epochs_kl_warmup=1000,
    batch_size=512,
    use_emp_total_count=False,
    match_bottom_left=False,
    w_sep=0.01,
    no_smoothing=False,
    use_time_dependence=True,
    time_loss=0.1,
    max_epochs_scvi=10000,
    max_epochs_nsv=300000,
):
    """Run the noSpliceVelo pipeline (SCVIModified -> smoothing -> noSpliceVelo).

    Parameters
    ----------
    adata :
        Input AnnData (must contain a 'counts' layer, etc.).
    dir_path :
        Directory where models are saved / loaded.
    continue_from_prev :
        If True, reuse previously saved checkpoints instead of retraining:
        a saved SCVIModified model ('model_scvi_modified.pt') is loaded if
        present (the smoothing + ellipse-fit stages still run, since those
        layers are not saved with it), and a saved noSpliceVelo checkpoint
        ('model_nosplicevelo.pt') is resumed if present. When the noSpliceVelo
        checkpoint is resumed, `n_epochs_kl_warmup` is forced to 0 (KL warmup
        only makes sense when training from scratch).
    n_hidden_scvi, n_latent_scvi :
        Hidden width / latent dim of the SCVIModified model.
    n_hidden_nsv, n_latent_nsv :
        Hidden width / latent dim of the noSpliceVelo model.
    n_layers :
        Number of hidden layers used by BOTH the SCVIModified and noSpliceVelo
        encoders/decoders (default 1).
    n_layers_scvi, n_layers_nsv :
        Optional per-model overrides of `n_layers`. If None (default), the
        value of `n_layers` is used for that model.
    n_epochs_kl_warmup :
        KL warmup epochs for the noSpliceVelo training plan. Ignored (set to 0)
        when resuming from a checkpoint.
    batch_size :
        Minibatch size used for training both models.
    use_emp_total_count :
        If True, set the SCVIModified `total_count` to the empirical value
        `int(median(counts.sum(axis=1)) / 0.1)`. If False (default), use a
        fixed `total_count = 1e4`.
    no_smoothing :
        If True, skip the kNN smoothing of the SCVIModified mean/variance and
        use them directly: layers['mu_scvi_smooth'] = mu_scvi and
        layers['var_scvi_smooth'] = var_scvi. Default False.
    use_time_dependence :
        If True (default; used for all manuscript runs), the burst frequency
        and burst size of the final up-branch state are multiplied by a shared
        per-cell factor omega in (0, 1] (`b_t` in nosplicevelo_module_v5_polar.py),
        so the up-branch steady state can drift along the trajectory. If False,
        the factor is fixed to 1 and the time-dependence penalty is inactive.
    time_loss :
        Weight of the penalty that keeps omega close to 1 (lambda_omega in the
        Methods; default 0.1). Only used when `use_time_dependence` is True.
    max_epochs_scvi, max_epochs_nsv :
        Upper bounds on training epochs for SCVIModified (default 10000) and
        noSpliceVelo (default 300000). Both models use early stopping on the
        validation ELBO (patience 100 and 1000 epochs), so the defaults are
        caps, not targets. Lower them only for smoke tests.
    """
    scvi.settings.seed = 0

    # YAML may deliver booleans as strings ("false"); bool("false") is True.
    if isinstance(use_time_dependence, str):
        use_time_dependence = use_time_dependence.strip().lower() in ("1", "true", "yes", "on")
    use_time_dependence = bool(use_time_dependence)

    # Resolve per-model layer counts (fall back to the shared `n_layers`).
    if n_layers_scvi is None:
        n_layers_scvi = n_layers
    if n_layers_nsv is None:
        n_layers_nsv = n_layers
    print(f"n_layers: SCVIModified={n_layers_scvi}, noSpliceVelo={n_layers_nsv}; "
          f"no_smoothing={no_smoothing}; use_time_dependence={use_time_dependence}")
    
    import pandas as pd
    import gzip
    
    device_ = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    gc.collect()
    
    adata_pan = adata.copy()

    # Ensure the naive smoothed layers required by SCVIModified exist; compute
    # them from the raw counts (PCA -> kNN -> neighbor smoothing) if missing.
    if 'mu_naive_smooth' not in adata_pan.layers or \
       'var_naive_smooth' not in adata_pan.layers:
        print("mu_naive_smooth/var_naive_smooth not found in adata.layers; "
              "computing them from the 'counts' layer...")
        sc.pp.pca(adata_pan, n_comps=50, svd_solver='arpack')
        n_neighbors = 30
        sc.pp.neighbors(adata_pan, use_rep='X_pca', n_neighbors=n_neighbors, n_pcs=None)
        mu_naive_smooth, var_naive_smooth = compute_mu_var_naive(
            adata_pan, n_neighbors=n_neighbors, use_rep='X_pca'
        )
        adata_pan.layers['mu_naive_smooth'] = mu_naive_smooth.copy()
        adata_pan.layers['var_naive_smooth'] = var_naive_smooth.copy()

    # setup adata_pan for SCVIModified
    SCVIModified.setup_anndata(
        adata_pan,
        layer="counts",
        mu_naive_key="mu_naive_smooth",
        var_naive_key="var_naive_smooth",
        # batch_key='sequencing.batch'
        # size_factor_key='capture_efficiency'
    )

    # create model object for SCVIModified
    # fac_loss_noise = 1e-4 # to regularize variability in burst size across cells
    fac_loss_noise = 1e-4 # to regularize variability in burst size across cells
    fac_loss_burstB = 0.0 # regularize burst_B gene
    fac_total_count = 0.1
    fac_loss_correlation = 0.01
    kl_weight_fac = 1.0
    # mean_eta_hat = m_cap
    # sigma_eta_prior = np.sqrt(v_cap)
    # print(mean_eta_hat, sigma_eta_prior)
    sigma_eta_prior = 0.3
    mean_eta_hat = 0.1
    total_count = np.array(adata_pan.layers['counts'].sum(axis=-1)).flatten()
    log_total_count_mu = np.mean(np.log(total_count + 1))
    log_total_count_std = np.std(np.log(total_count + 1))
    if use_emp_total_count:
        # empirical total count from the data
        total_count = np.median(np.array(adata_pan.layers['counts'].sum(axis=1))) / 0.1
        total_count = int(total_count)
    else:
        total_count = 1e4
    print("total count:", total_count)


    # Ensure the output directory exists (needed for both saving and resuming).
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    scvi_checkpoint_path = os.path.join(dir_path, 'model_scvi_modified.pt')

    if continue_from_prev and os.path.exists(scvi_checkpoint_path):
        # Reuse a previously trained SCVIModified model instead of retraining.
        # NOTE: the smoothing + ellipse-fit stages below are STILL run, because
        # those layers (mu_scvi_smooth, var_scvi_smooth, best_fit, ...) are
        # computed AFTER the SCVIModified model is saved and are not part of its
        # saved AnnData, so adata_pan_velo must be rebuilt every run.
        print(f"[continue_from_prev] Loading existing SCVIModified checkpoint at "
              f"{scvi_checkpoint_path}; skipping SCVIModified training")
        model_scvi = SCVIModified.load(scvi_checkpoint_path, adata=adata_pan)
    else:
        model_scvi = SCVIModified(adata_pan, burst_B_gene=False, \
                              var_activation=F.softplus, device_=device_, \
                              n_hidden=n_hidden_scvi, n_latent=n_latent_scvi, log_variational=True, \
                              fac_loss_noise=fac_loss_noise, n_layers=n_layers_scvi,
                              fac_total_count=fac_total_count, mean_eta_hat=mean_eta_hat,
                              sigma_eta_prior=sigma_eta_prior,
                              log_total_count_mu=log_total_count_mu, log_total_count_std=log_total_count_std,
                              use_observed_lib_size=False, total_count=total_count,
                              fac_loss_correlation=fac_loss_correlation,
                              kl_weight_fac=kl_weight_fac
                             )

        # train the model
        model_scvi.train(max_epochs=max_epochs_scvi, batch_size=batch_size, \
                     train_size=0.9, validation_size=0.1, early_stopping=True, \
                     early_stopping_patience=100, \
                     early_stopping_monitor='elbo_validation',
                     accelerator="gpu" if device_ == "cuda" else "cpu")

        model_scvi.save(scvi_checkpoint_path, overwrite=True, save_anndata=True)
    
    
    # set numpy random seed
    np.random.seed(0)
    # set torch random seed
    torch.manual_seed(0)
    # set torch cuda random seed
    torch.cuda.manual_seed(0)
    # set torch cuda random seed all
    torch.cuda.manual_seed_all(0)


    # get gene-cell specific estimates of mean and variance of expression 
    # estimated by SCVIModified
    nrepeats = 10 # nume samples for posterior predictive estimate
    for n_ in range(nrepeats):
        print(f'repeat = {n_}')
        params_scvi = model_scvi.get_likelihood_parameters_new()
        mu_tmp = params_scvi['mu'].copy()
        var_tmp = params_scvi['var'].copy()
        library_tmp = params_scvi['library'].copy()
        # total_count_tmp = params_scvi['total_counts'].copy()

        if n_ == 0:
            mu_scvi = mu_tmp.copy()
            var_scvi = var_tmp.copy()
            library_ = library_tmp.copy()
            # total_count = total_count_tmp.copy()
        else:
            mu_scvi += mu_tmp.copy()
            var_scvi += var_tmp.copy()
            library_ += library_tmp.copy()
            # total_count += total_count_tmp.copy()
        torch.cuda.empty_cache()
        gc.collect()
    mu_scvi /= nrepeats
    var_scvi /= nrepeats
    library_ /= nrepeats
    # total_count /= nrepeats
    
    
    adata_pan.layers['mu_scvi'] = mu_scvi.copy()
    adata_pan.layers['var_scvi'] = var_scvi.copy()

    z_latent = model_scvi.get_latent_representation()
    adata_pan.obsm['X_latent'] = z_latent.copy()
    n_neighbors = 30
    sc.pp.neighbors(adata_pan, use_rep='X_latent', n_neighbors=n_neighbors)
    
    from scipy.sparse import issparse

    def pairwise_product_sum_excluding_diagonal(mu):
        """
        Computes the sum of pairwise products for each gene in a matrix,
        excluding the product of an element with itself.

        Args:
        mu: A numpy array of shape (n_cells, n_genes).

        Returns:
        A numpy array of shape (n_genes,) where each element g contains
        the sum of mu[j, g] * mu[k, g] for all j != k.
        """
        n_cells, n_genes = mu.shape

        # Calculate the sum of each column (for each gene)
        sum_g = np.sum(mu, axis=0)

        # Calculate the sum of squares of each column
        sum_sq_g = np.sum(mu**2, axis=0)

        # The sum of all pairwise products (including diagonal) for gene g
        # is (sum of column g) * (sum of column g) = sum_g[g] * sum_g[g]

        # The sum of the diagonal products (mu[i, g] * mu[i, g]) for gene g
        # is the sum of squares of column g = sum_sq_g[g]

        # The sum of pairwise products excluding the diagonal is the
        # total sum of pairwise products minus the sum of the diagonal products.
        result = sum_g**2 - sum_sq_g

        return result

    def sum_i_mul_sum_j_neq_i_per_gene(mu):
        """
        Computes, for each gene g, the product of the sum of all cells'
        expression and the sum of all *other* cells' expression.

        Args:
        mu: A numpy array of shape (n_cells, n_genes).

        Returns:
        A numpy array of shape (n_genes,) where each element g contains
        (sum of mu[:, g]) * (sum of mu[:, g] excluding the i-th element).
        """
        n_cells, n_genes = mu.shape

        # Calculate the sum of each column (for each gene)
        sum_g = np.sum(mu, axis=0)  # Shape: (n_genes,)

        # For each gene g, the sum of all other cells' expression
        # (sum_{j != i} mu[j, g]) can be computed by subtracting each
        # individual cell's expression (mu[:, g]) from the total sum for that gene (sum_g[g]).

        # We can achieve this efficiently using broadcasting.
        sum_j_neq_i_per_gene = sum_g[np.newaxis, :] - mu  # Shape: (n_cells, n_genes)

        # Now, we want to compute sum_i (mu[i, g]) * sum_{j != i} (mu[j, g]) for each gene g.
        # We need to multiply each element of mu with the corresponding
        # sum_j_neq_i_per_gene and then sum along the cell axis (axis=0) for each gene.

        result = np.sum(mu * sum_j_neq_i_per_gene, axis=0)  # Shape: (n_genes,)

        return result

    def compute_mu_var_smooth(adata, mu_layer='mu_scvi', var_layer="var_scvi", n_neighbors=30, use_rep='X_latent'):
        """
        Computes a smoothed cell-gene matrix (mu_smooth) by averaging the 'mu'
        values of each cell with its neighbors in the PCA/latent space.

        Args:
            adata: An AnnData object.
            mu_layer: The key in `adata.layers` containing the cell-gene matrix 'mu'.
            n_neighbors: The number of neighbors to use for smoothing (must match
                         or be less than the n_neighbors used in sc.pp.neighbors).
            use_rep: The representation in `adata.obsm` used for neighbor computation
                     (must match the use_rep used in sc.pp.neighbors).

        Returns:
            A pandas DataFrame of shape (n_cells, n_genes) containing the smoothed 'mu' values.
        """
        if 'neighbors' not in adata.uns:
            sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors)
        elif adata.uns['neighbors']['params']['n_neighbors'] < n_neighbors or \
             adata.uns['neighbors']['params']['use_rep'] != use_rep:
            print(f"Warning: Existing neighbors were computed with n_neighbors="
                  f"{adata.uns['neighbors']['params']['n_neighbors']} and use_rep="
                  f"'{adata.uns['neighbors']['params']['use_rep']}'. Recomputing neighbors "
                  f"with n_neighbors={n_neighbors} and use_rep='{use_rep}'.")
            sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors)

        if mu_layer not in adata.layers:
            raise KeyError(f"Layer '{mu_layer}' not found in adata.layers.")

        mu = adata.layers[mu_layer]
        if issparse(mu):
            mu = mu.A  # Convert to dense if sparse
        var = adata.layers[var_layer]
        if issparse(var):
            var = var.A  # Convert to dense if sparse
        counts = adata.layers['counts']
        if issparse(counts):
            counts = counts.A  # Convert to dense if sparse

        n_cells = adata.n_obs
        n_genes = mu.shape[1]
        mu_smooth = np.zeros_like(mu, dtype=float)
        var_smooth = np.zeros_like(mu, dtype=float)
        mu_naive = np.zeros_like(mu, dtype=float)
        var_naive = np.zeros_like(mu, dtype=float)

        for i in range(n_cells):
            neighbors_indices = np.where(adata.uns['neighbors']['connectivities'][i, :].toarray().flatten() != 0)[0]
            neighbors_indices = np.union1d(neighbors_indices, [i])
            n_nbers = len(neighbors_indices)
            neighbor_mu = mu[neighbors_indices, :]
            neighbor_var = var[neighbors_indices, :]
            neighbor_counts = counts[neighbors_indices, :]
            mu_smooth[i, :] = np.mean(neighbor_mu, axis=0)
            term1 = (((n_nbers - 2) / n_nbers) * (neighbor_var + neighbor_mu**2.0)).sum(axis=0) / n_nbers
            term2 = - 2 * sum_i_mul_sum_j_neq_i_per_gene(neighbor_mu) / n_nbers**2.0
            term3 = np.sum(neighbor_var + neighbor_mu**2, axis=0) / n_nbers**2.0
            term4 = pairwise_product_sum_excluding_diagonal(neighbor_mu) / n_nbers**2.0
            # var_smooth[i, :] = term1 + term2 + term3 + term4

            term1_new = np.mean(neighbor_var, axis=0)
            mu_mean = np.mean(neighbor_mu, axis=0)
            term2_new = np.sum((neighbor_mu - mu_mean)**2, axis=0) / (n_nbers - 1)
            var_smooth[i, :] = term1_new + term2_new

            mu_naive[i, :] = np.mean(neighbor_counts, axis=0)
            var_naive[i, :] = np.var(neighbor_counts, axis=0)

        mu_naive_smooth = np.zeros_like(mu, dtype=float)
        var_naive_smooth = np.zeros_like(mu, dtype=float)
        for i in range(n_cells):
            neighbors_indices = np.where(adata.uns['neighbors']['connectivities'][i, :].toarray().flatten() != 0)[0]
            neighbors_indices = np.union1d(neighbors_indices, [i])
            n_nbers = len(neighbors_indices)
            neighbor_mu = mu_naive[neighbors_indices, :]
            neighbor_var = var_naive[neighbors_indices, :]
            mu_naive_smooth[i, :] = np.mean(neighbor_mu, axis=0)
            var_naive_smooth[i, :] = np.mean(neighbor_var, axis=0)

        return mu_smooth, var_smooth, mu_naive_smooth, var_naive_smooth
    
    if no_smoothing:
        # Skip kNN smoothing: use the (repeat-averaged) SCVIModified mean/var
        # directly as the "smoothed" layers consumed downstream.
        print("[no_smoothing] Skipping mu/var smoothing; using mu_scvi/var_scvi "
              "as mu_scvi_smooth/var_scvi_smooth")
        adata_pan.layers['mu_scvi_smooth'] = mu_scvi.copy()
        adata_pan.layers['var_scvi_smooth'] = var_scvi.copy()
    else:
        mu_scvi_smooth, var_scvi_smooth, mu_naive_smooth, var_naive_smooth = \
            compute_mu_var_smooth(
                adata_pan,
                n_neighbors=n_neighbors,\
                use_rep='X_latent'
            )

        adata_pan.layers['mu_scvi_smooth'] = mu_scvi_smooth.copy()
        adata_pan.layers['var_scvi_smooth'] = var_scvi_smooth.copy()
    
    
    import ellipse_fit
    importlib.reload(ellipse_fit)
    from ellipse_fit import VelocityModelSelector
    selector = VelocityModelSelector(
        adata_pan.copy(),
        mu_layer="mu_scvi_smooth",
        var_layer="var_scvi_smooth"
    )

    # Use strict thresholds for high confidence in non-linear models
    selector.run_selection(n_jobs=-1, thresh_p=2, thresh_e=5)

    # Or use loose thresholds to explore potential candidates
    # selector.run_selection(n_jobs=-1, threshold_p=2, threshold_e=5)

    # Or use loose thresholds to explore potential candidates
    # selector.run_selection(n_jobs=-1, threshold_p=2, threshold_e=5)
    
    adata_pan = selector.adata.copy()
    id_ = adata_pan.layers['branch_assignment'] == -1
    adata_pan.layers['branch_assignment'][id_] = 0
    
    ## remove line genes
    # adata_pan = adata_pan[:, (adata_pan.var['best_fit'] != "line") & 
    #                          (adata_pan.var['best_fit'] != "noise")].copy()
    adata_pan_complete = adata_pan.copy()
    adata_pan = adata_pan_complete[:, (adata_pan_complete.var['best_fit'] != "noise")].copy()
    
    
    mu_scvi_smooth = adata_pan.layers['mu_scvi_smooth'].copy()
    var_scvi_smooth = adata_pan.layers['var_scvi_smooth'].copy()
    
    
    burstB = np.zeros(adata_pan.shape[1])
    for gene_ in np.arange(adata_pan.shape[1]):
        id_corner = np.where((var_scvi_smooth[:, gene_] >= np.quantile(var_scvi_smooth[:, gene_], 0.95)))[0]
        burstB[gene_] = np.mean(var_scvi_smooth[id_corner, gene_]) / np.mean(mu_scvi_smooth[id_corner, gene_])
        
        
    var_thresh = 1e4
    var_quant = np.quantile(var_scvi_smooth, 0.95, axis=0)
    

    adata_pan_velo = adata_pan.copy()
    # id_cells = adata_pan.obs['clusters'] != '5'
    # adata_pan_velo = adata_pan[id_cells].copy()
    # remove genes with a linear fit. that means remove genes in id_genes_lin
    # id_genes_nonLin_keep = np.setdiff1d(np.arange(adata_pan.shape[1]), id_genes_lin)
    id_genes_nonLin_keep = np.arange(adata_pan_velo.shape[1])
    r2_thresh = 0.2
    # id_genes_nonLin_keep = np.where((burstB >= 2) &
    #                                 (r2_mu >= r2_thresh) &
    #                                 (r2_var >= r2_thresh))[0]
    # id_genes_nonLin_keep = np.where((r2_mu >= r2_thresh) &
    #                                 (r2_var >= r2_thresh))[0]
    # id_genes_nonLin_keep = np.where((burstB >= 1.2))[0]
    # id_genes_nonLin_keep = genes_pass.copy()
    # genes_selected = adata_pan.var_names.values[id_genes_nonLin_keep]
    # gene_cluster = 34
    # genes_corr_clusters = clusters[gene_cluster]
    # genes_union = np.intersect1d(genes_corr_clusters, genes_selected)
    # id_genes_nonLin_keep = []
    # for gene_ in genes_union:
    #     id_genes_nonLin_keep.append(list(genes_union).index(gene_))
    adata_pan_velo = adata_pan_velo[:, id_genes_nonLin_keep].copy()
    # print shape of adata_pan_velo
    print(adata_pan_velo.shape)
    adata_pan_velo.layers['mu'] = mu_scvi_smooth[:, id_genes_nonLin_keep].copy()
    adata_pan_velo.layers['std'] = \
        np.sqrt(var_scvi_smooth[:, id_genes_nonLin_keep].copy())
    adata_pan_velo.layers['prior_clusters'] = np.ones(adata_pan_velo.shape)
    
    
    
    quantile_ = 0.95
    mu_scale = np.quantile(mu_scvi_smooth[:, :], q=quantile_, axis=0, keepdims=True)
    var_scale = np.quantile(var_scvi_smooth[:, :], q=quantile_, axis=0, keepdims=True)
    mu_scale = np.ones((mu_scvi_smooth.shape))
    var_scale = np.ones((var_scvi_smooth.shape))
    mu_scvi_smooth_scaled = mu_scvi_smooth.copy() / var_scale
    var_scvi_smooth_scaled = var_scvi_smooth.copy() / var_scale
    
    
    
    adata_pan_velo.layers['mu_scaled'] = mu_scvi_smooth_scaled[:, id_genes_nonLin_keep].copy()
    adata_pan_velo.layers['std_scaled'] = np.sqrt(var_scvi_smooth_scaled[:, id_genes_nonLin_keep].copy()+ 1e-8)
    
    
    adata_pan_velo.var['gene_scale'] = var_scale[0, id_genes_nonLin_keep].copy()
    gene_scale_torch = torch.tensor(var_scale[0, id_genes_nonLin_keep].copy(),
                                    dtype=torch.float, device=device_)
    
    
    fac_multiply = 5.0
    # fac_multiply = 3.0
    fac_var = fac_multiply
    mu_max = np.max(mu_scvi_smooth_scaled[:, id_genes_nonLin_keep], axis=0) * fac_multiply
    mu_max += 1 / adata_pan_velo.var['gene_scale'].values.copy()
    var_max = np.max(var_scvi_smooth_scaled[:, id_genes_nonLin_keep], axis=0) * fac_multiply
    var_max += 1 / adata_pan_velo.var['gene_scale'].values.copy()
    mu_max_torch = torch.tensor(mu_max, dtype=torch.float, device=device_)
    var_max_torch = torch.tensor(var_max, dtype=torch.float, device=device_)
    
    fac_multiply = 5.0
    # fac_multiply = 3.0
    fac_var = fac_multiply
    mu_max = np.quantile(mu_scvi_smooth_scaled[:, id_genes_nonLin_keep], 0.9999, axis=0) * fac_multiply
    mu_max += 1 / adata_pan_velo.var['gene_scale'].values.copy()
    var_max = np.quantile(var_scvi_smooth_scaled[:, id_genes_nonLin_keep], 0.9999, axis=0) * fac_multiply
    var_max += 1 / adata_pan_velo.var['gene_scale'].values.copy()
    mu_max_torch = torch.tensor(mu_max, dtype=torch.float, device=device_)
    var_max_torch = torch.tensor(var_max, dtype=torch.float, device=device_)
    
    
    fac_multiply = 1.0
    # mu_scale = np.max(mu_scvi_smooth_scaled[:, id_genes_nonLin_keep] * fac_multiply, axis=0)
    # std_scale = np.max(np.sqrt(var_scvi_smooth_scaled[:, id_genes_nonLin_keep] * fac_multiply), axis=0)
    # std_ref_scale = np.max(np.sqrt(var_max - var_scvi_smooth_scaled[:, id_genes_nonLin_keep] * fac_multiply), axis=0)
    mu_scale = np.quantile(mu_scvi_smooth_scaled[:, id_genes_nonLin_keep] * fac_multiply, 
                           0.95, axis=0)
    # std_scale = np.quantile(np.sqrt(var_scvi_smooth_scaled[:, id_genes_nonLin_keep] * fac_multiply), 
    #                    0.95, axis=0)
    std_scale = np.quantile((var_scvi_smooth_scaled[:, id_genes_nonLin_keep] * fac_multiply), 
                       0.95, axis=0) # for polar model
    std_ref_scale = np.quantile(np.sqrt(var_max - var_scvi_smooth_scaled[:, id_genes_nonLin_keep] * fac_multiply),
                           0.95, axis=0)

    # mu_scale = np.std(np.log(mu_scvi_smooth_scaled[:, id_genes_nonLin_keep]) * fac_multiply, axis=0)
    # std_scale = np.std(np.log(var_scvi_smooth_scaled[:, id_genes_nonLin_keep] * fac_multiply), axis=0)
    # std_ref_scale = np.std(np.log(var_max - var_scvi_smooth_scaled[:, id_genes_nonLin_keep] * fac_multiply), axis=0)

    mu_center = np.mean(mu_scvi_smooth_scaled[:, id_genes_nonLin_keep] * fac_multiply, axis=0)
    std_center = np.mean(np.sqrt(var_scvi_smooth_scaled[:, id_genes_nonLin_keep] * fac_multiply), axis=0)
    std_ref_center = np.mean(np.sqrt(var_max - var_scvi_smooth_scaled[:, id_genes_nonLin_keep] * fac_multiply), axis=0)

    mu_scale = torch.tensor(mu_scale, dtype=torch.float, device=device_)
    std_scale = torch.tensor(std_scale, dtype=torch.float, device=device_)
    std_ref_scale = torch.tensor(std_ref_scale, dtype=torch.float, device=device_)

    mu_center = torch.tensor(mu_center, dtype=torch.float, device=device_)
    std_center = torch.tensor(std_center, dtype=torch.float, device=device_)
    std_ref_center = torch.tensor(std_ref_center, dtype=torch.float, device=device_)

    mu_center[:] = 0.0
    std_center[:] = 0.0
    std_ref_center[:] = 0.0

    # mu_scale[:] = 1.0
    # std_scale[:] = 1.0
    # std_ref_scale[:] = 1.0
    
    
    std_sum = ((2 * np.sqrt(var_scvi_smooth_scaled[:, id_genes_nonLin_keep] * fac_multiply + 1e-10))**2.0).mean(0)
    std_sum = np.sqrt(std_sum + 1e-10)
    std_sum_torch = torch.tensor(std_sum, dtype=torch.float, device=device_)
    
    
    
    mu_scvi_smooth_scaled = mu_scvi_smooth.copy()
    var_scvi_smooth_scaled = var_scvi_smooth.copy()


    # Extract just the kept genes
    mu_ = mu_scvi_smooth_scaled[:, id_genes_nonLin_keep]   # shape: (cells, genes)
    var_ = var_scvi_smooth_scaled[:, id_genes_nonLin_keep] # shape: (cells, genes)

    quantile_1, quantile_2 = 0.95, 0.99999

    # ---- Case 1: quantiles based on mu ----
    q1_mu = np.quantile(mu_, quantile_1, axis=0)  # shape: (genes,)
    q2_mu = np.quantile(mu_, quantile_2, axis=0)

    mask_mu = (mu_ >= q1_mu) & (mu_ <= q2_mu)     # shape: (cells, genes)

    # avoid empty slices by masking and summing
    mu_cornerMu_obs = (mu_ * mask_mu).sum(0) / mask_mu.sum(0)
    var_cornerMu_obs = (var_ * mask_mu).sum(0) / mask_mu.sum(0)

    mu_cornerMu_obs_np = mu_cornerMu_obs.copy()
    var_cornerMu_obs_np = var_cornerMu_obs.copy()

    # ---- Case 2: quantiles based on var ----
    q1_var = np.quantile(var_, quantile_1, axis=0)
    q2_var = np.quantile(var_, quantile_2, axis=0)

    mask_var = (var_ >= q1_var) & (var_ <= q2_var)

    mu_cornerVar_obs = (mu_ * mask_var).sum(0) / mask_var.sum(0)
    var_cornerVar_obs = (var_ * mask_var).sum(0) / mask_var.sum(0)
    mu_cornerVar_obs_np = mu_cornerVar_obs.copy()
    var_cornerVar_obs_np = var_cornerVar_obs.copy()

    # ---- convert to torch ----
    mu_cornerMu_obs  = torch.tensor(mu_cornerMu_obs,  dtype=torch.float, device=device_)
    var_cornerMu_obs = torch.tensor(var_cornerMu_obs, dtype=torch.float, device=device_)
    mu_cornerVar_obs = torch.tensor(mu_cornerVar_obs, dtype=torch.float, device=device_)
    var_cornerVar_obs = torch.tensor(var_cornerVar_obs, dtype=torch.float, device=device_)

    
    
    
    mu_scvi_smooth_scaled = mu_scvi_smooth.copy()
    var_scvi_smooth_scaled = var_scvi_smooth.copy()


    # Extract just the kept genes
    mu_ = mu_scvi_smooth_scaled[:, id_genes_nonLin_keep]   # shape: (cells, genes)
    var_ = var_scvi_smooth_scaled[:, id_genes_nonLin_keep] # shape: (cells, genes)

    quantile_1 = 0.05
    quantile_2 = 0.01

    # ---- Case 1: quantiles based on mu ----
    q_mu = np.quantile(mu_, quantile_1, axis=0)  # shape: (genes,)
    q_var = np.quantile(var_, quantile_2, axis=0)  # shape: (genes,)

    mask_mu = (mu_ <= q_mu) & (var_ <= q_var)   # shape: (cells, genes)

    # avoid empty slices by masking and summing
    mu_low_obs = np.nansum(mu_ * mask_mu, axis=0) / (mask_mu.sum(0) + 1e-8) + 1e-5
    var_low_obs = np.nansum(var_ * mask_mu, axis=0) / (mask_mu.sum(0) + 1e-8) + 1e-5

    # ---- convert to torch ----
    mu_low_obs  = torch.tensor(mu_low_obs,  dtype=torch.float, device=device_)
    var_low_obs = torch.tensor(var_low_obs, dtype=torch.float, device=device_)
    
    
    gene_recont_weights = torch.log10(var_cornerVar_obs + 1)
    gene_recont_weights = 2 * ((gene_recont_weights - gene_recont_weights.min()) / \
        (gene_recont_weights.max() - gene_recont_weights.min())) + 1
    gene_recont_weights[:] = 1.0
    
    
    torch.cuda.empty_cache()
    gc.collect()
    
    
    
    def classify_cells(A1, A2, corners_bl, corners_tr):
        # 1. Extract coordinates for all genes
        # Shape of these will be (G,)
        x0, y0 = corners_bl[:, 0], corners_bl[:, 1]
        x1, y1 = corners_tr[:, 0], corners_tr[:, 1]

        # 2. Compute the components of the line equation
        dx = x1 - x0  # Shape (G,)
        dy = y1 - y0  # Shape (G,)

        # 3. Calculate relative positions
        # We use broadcasting: (C, G) - (G,) works automatically
        # Term 1: (y - y0) * dx
        term1 = (A2 - y0) * dx

        # Term 2: (x - x0) * dy
        term2 = (A1 - x0) * dy

        # 4. Classify: 1 if above diagonal, 0 otherwise
        # Results in a boolean array (C, G), converted to int
        classification = (term1 > term2).astype(int)

        return classification

    mu_scvi_smooth = adata_pan_velo.layers['mu_scvi_smooth']
    var_scvi_smooth = adata_pan_velo.layers['var_scvi_smooth']
    corners_bl = np.zeros((adata_pan_velo.shape[1], 2))
    # cells_ = adata_pan_velo.obs['range'].values.copy()
    corners_tr = np.vstack((mu_cornerMu_obs_np, var_cornerMu_obs_np)).T
    classification_2 = classify_cells(mu_scvi_smooth, 
                                    var_scvi_smooth, corners_bl, corners_tr)
    dist_2 = (mu_scvi_smooth - mu_cornerMu_obs_np[None, :])**2 + \
        (var_scvi_smooth - var_cornerMu_obs_np[None, :])**2

    corners_tr = np.vstack((mu_cornerVar_obs_np, var_cornerVar_obs_np)).T
    classification_1 = classify_cells(mu_scvi_smooth, 
                                    var_scvi_smooth, corners_bl, corners_tr)
    dist_1 = (mu_scvi_smooth - mu_cornerVar_obs_np[None, :])**2 + \
        (var_scvi_smooth - var_cornerVar_obs_np[None, :])**2

    classification = -1 * np.ones_like(classification_1)
    classification[classification_1 == 1] = 1
    classification[classification_2 == 0] = 0
    classification[(classification_1 == 0) & (classification_2 == 1) & 
                   (dist_1 < dist_2)] = 1
    classification[(classification_1 == 0) & (classification_2 == 1) & 
                   (dist_1 >= dist_2)] = 0
    adata_pan_velo.layers['prior_pi_up'] = classification.copy()
    
    
    mask_ = np.where(adata_pan_velo.var['best_fit'] == "ellipse")[0]
    adata_pan_velo.layers['branch_assignment'][:, mask_] = \
        adata_pan_velo.layers['prior_pi_up'][:, mask_].copy()
    
    
    adata_pan_velo.obs['capture_efficiency_tmp'] = 1.0
    
    torch.cuda.empty_cache()
    gc.collect()
    
    
    noSpliceVelo_v4_polar.setup_anndata(    
        adata_pan_velo, 
        layer='counts', 
        continuous_covariate_keys=['capture_efficiency_tmp'],
        mean_layer='mu_scaled', 
        std_layer='std_scaled',
        prior_pi_up_layer='prior_pi_up', # when use_prior_parabola = False
        # prior_pi_up_layer='branch_assignment', # when use_prior_parabola = True
        prior_cluster='prior_clusters',
    )
    
    
    prior_branch_assignment = None # when use_prior_parabola = False
    use_prior_parabola = False # when use_prior_parabola = False
    
    
    # first gene-specific, and then gene-cell specific
    loss_fac_geneCell = 0.0 ## only gene-level params
    loss_fac_gene = 1.0
    loss_ = 0.1
    loss_0 = 0.1
    loss_1 = 0.1
    # loss_2 = 0.1
    # loss_2_ = 0.1
    loss_2 = 0.1
    loss_2_ = 0.1
    # loss_3 = 0.1
    loss_3 = 0.1
    # loss_2 = 0.0
    loss_prior_cluster = 0.0
    match_upf_down_up = False
    # match_upf_down_up = True
    match_burst_params = True
    edge_loss_fac = 1.0
    w_theta = 1.0
    # print(f'use_prior_parabola = {use_prior_parabola}, prior_branch_assignment = {prior_branch_assignment}')
    # model_nosplicevelo = noSpliceVelo(adata_pan_velo, dispersion='gene-cell', \
    #                                 gene_likelihood='nb', \
    #                                 tmax=24, var_activation=F.softplus, 
    #                                 log_variational=False, \
    #                                 n_states=2, n_hidden=128, \
    #                                 n_latent=10, cluster_states=False, \
    #                                 burst_B_gene=True, burst_f_gene=True, \
    #                                 use_loss_burst=False, use_controlBurst_gene=True, \
    #                                 use_time_cell=False, state_times_unique=None, \
    #                                 burst_f_updown=None, \
    #                                 burst_B_updown=None, \
    #                                 burst_f_next=None, \
    #                                 burst_B_next=None, \
    #                                 burst_f_updown_next=None, \
    #                                 burst_B_updown_next=None, mu_max=mu_max_torch, \
    #                                 var_max=var_max_torch, std_sum=std_sum_torch, \
    #                                 match_burst_params=match_burst_params, match_upf_down_up=match_upf_down_up, \
    #                                 match_burst_params_not_muVar=False,
    #                                 extra_loss_fac_0=loss_0, \
    #                                 extra_loss_fac=loss_, extra_loss_fac_1=loss_1, \
    #                                 extra_loss_fac_2=loss_2,
    #                                 mu_center=mu_center, mu_scale=mu_scale,
    #                                 std_center=std_center, std_scale=std_scale, \
    #                                 std_ref_center=std_ref_center, std_ref_scale=std_ref_scale, \
    #                                 loss_fac_geneCell=loss_fac_geneCell, loss_fac_gene=loss_fac_gene, \
    #                                 mu_ss_obs=mu_cornerVar_obs, var_ss_obs=var_cornerVar_obs, \
    #                                 mu_mean_obs=mu_cornerMu_obs, var_mean_obs=var_cornerMu_obs, \
    #                                 loss_fac_prior_clust=loss_prior_cluster,
    #                                   gene_recont_weights=gene_recont_weights)
    # ---- build a fresh model, or resume from an existing checkpoint ----
    checkpoint_path = os.path.join(dir_path, 'model_nosplicevelo.pt')
    resume = continue_from_prev and os.path.exists(checkpoint_path)

    if resume:
        # Guard: the loaded checkpoint is validated by scvi against the freshly
        # rebuilt adata_pan_velo (var names / layers / covariates). If the
        # ellipse-fit gene selection drifted and the registries no longer match,
        # load() raises -> fall back to training a fresh model instead of dying.
        try:
            print(f"[continue_from_prev] Found checkpoint at {checkpoint_path}; "
                  f"resuming training and forcing n_epochs_kl_warmup = 0")
            model_nosplicevelo = noSpliceVelo_v4_polar.load(
                checkpoint_path
            )
            adata_pan_velo = model_nosplicevelo.adata.copy()
            n_epochs_kl_warmup = 0
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[continue_from_prev] Could not resume from {checkpoint_path} "
                  f"({type(e).__name__}: {e}). This usually means the rebuilt "
                  f"adata no longer matches the saved model registry. Falling "
                  f"back to training a new model from scratch.")
            resume = False

    if not resume:
        if continue_from_prev:
            print(f"[continue_from_prev] No usable checkpoint at {checkpoint_path}; "
                  f"training a new model from scratch")
        model_nosplicevelo = noSpliceVelo_v4_polar(adata_pan_velo, dispersion='gene-cell', \
                                    gene_likelihood='nb', \
                                    tmax=24, var_activation=F.softplus,
                                    log_variational=False, \
                                    n_states=4, n_hidden=n_hidden_nsv, \
                                    n_layers=n_layers_nsv,
                                    n_latent=n_latent_nsv, cluster_states=False, \
                                    burst_B_gene=True, burst_f_gene=True, \
                                    use_loss_burst=False, use_controlBurst_gene=True, \
                                    use_time_cell=False, state_times_unique=None, \
                                    burst_f_updown=None, \
                                    burst_B_updown=None, \
                                    burst_f_next=None, \
                                    burst_B_next=None, \
                                    burst_f_updown_next=None, \
                                    burst_B_updown_next=None, mu_max=mu_max_torch, \
                                    var_max=var_max_torch, std_sum=std_sum_torch, \
                                    match_burst_params=match_burst_params, match_upf_down_up=match_upf_down_up, \
                                    match_burst_params_not_muVar=False,
                                    extra_loss_fac_0=loss_0, \
                                    extra_loss_fac=loss_, extra_loss_fac_1=loss_1, \
                                    extra_loss_fac_2=loss_2, extra_loss_fac_3=loss_3,
                                    extra_loss_fac_2_=loss_2_,
                                    mu_center=mu_center, mu_scale=mu_scale,
                                    std_center=std_center, std_scale=std_scale, \
                                    std_ref_center=std_ref_center, std_ref_scale=std_ref_scale, \
                                    loss_fac_geneCell=loss_fac_geneCell, loss_fac_gene=loss_fac_gene, \
                                    mu_ss_obs=mu_cornerVar_obs, var_ss_obs=var_cornerVar_obs, \
                                    mu_mean_obs=mu_cornerMu_obs, var_mean_obs=var_cornerMu_obs, \
                                    loss_fac_prior_clust=loss_prior_cluster, 
                                    mu_low_obs=mu_low_obs, var_low_obs=var_low_obs, 
                                    edge_loss_fac=edge_loss_fac, 
                                    prior_branch_assignment=prior_branch_assignment,
                                    use_prior_parabola=use_prior_parabola, w_theta=w_theta,
                                    use_time_dependence=use_time_dependence,
                                    time_loss=time_loss if use_time_dependence else 0.0,
                                    match_bottom_left=match_bottom_left,
                                    w_sep=w_sep)
    
    
    # n_epochs_kl_warmup is a function argument (defaults to 1000 for training
    # from scratch); the resume branch above forces it to 0.
    plan_kwargs={
            # "lr": 1e-4,                   # Drop from default 1e-3 to 1e-4
            # "weight_decay": 1e-6,         # Adds slight L2 regularization to prevent extreme weights
            # "reduce_lr_on_plateau": True, # Automatically drops LR when loss gets stuck
            # "lr_patience": 30,            # If loss doesn't improve for 30 epochs, drop LR
            # "lr_factor": 0.5,             # Cut LR in half when plateauing
            "n_epochs_kl_warmup": n_epochs_kl_warmup
    }
    trainer_kwargs={
            "gradient_clip_val": 10.0,           # TRUNCATES EXPLOSIVE GRADIENTS (Crucial for ODEs)
            "gradient_clip_algorithm": "norm",
    }
    # NOTE: trainer_kwargs above (gradient clipping) is NOT passed to train(),
    # exactly as in the manuscript runs.
    model_nosplicevelo.train(
        max_epochs=max_epochs_nsv, batch_size=batch_size, train_size=0.9, \
        validation_size=0.1, 
        early_stopping=True, \
        early_stopping_patience=1000, \
        early_stopping_monitor='elbo_validation', 
        plan_kwargs=plan_kwargs
    ) 
        #                     gradient_clip_val=100.0, # Start here, try 0.5 if it still bounces
        # gradient_clip_algorithm="norm")
        
    model_nosplicevelo.save(checkpoint_path, overwrite=True, save_anndata=True)
    
    
    
    return model_nosplicevelo


# =============================================================================
# Command-line runner: drive one or more datasets from a YAML config file
# =============================================================================

# Names of the optional pipeline parameters accepted per dataset (and in the
# global `defaults:` block). Anything else in a dataset entry is ignored with a
# warning, except the required `adata_path` / `dir_path` and the cosmetic `name`.
_PIPELINE_PARAMS = (
    "continue_from_prev",
    "n_hidden_scvi",
    "n_latent_scvi",
    "n_hidden_nsv",
    "n_latent_nsv",
    "n_layers",
    "n_layers_scvi",
    "n_layers_nsv",
    "n_epochs_kl_warmup",
    "batch_size",
    "use_emp_total_count",
    "w_sep",
    "match_bottom_left",
    "no_smoothing",
    "use_time_dependence",
    "time_loss",
    "max_epochs_scvi",
    "max_epochs_nsv",
)


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


# Matches tqdm / PyTorch-Lightning training progress lines, e.g.
#   "Epoch 1028/300000:   0%| | 1027/300000 [08:01<38:49:13, 2.14it/s, ...]"
import re as _re
_PROGRESS_RE = _re.compile(
    r"\d{1,3}%\s*\|"     # "  0%|"
    r"|it/s[,\]]"        # "2.14it/s]" / "it/s,"
    r"|s/it[,\]]"        # "s/it]" / "s/it,"
    r"|^Epoch\s+\d+/\d+" # "Epoch 1028/300000"
)


class _FilteredFile:
    """File wrapper that drops in-place progress-bar updates from the log.

    tqdm/Lightning redraw progress with carriage returns ('\\r') and never emit
    a newline until the bar finishes, which bloats the log file. This wrapper
    keeps memory bounded (collapsing '\\r' overwrites) and skips any completed
    line that looks like a progress bar. Normal output passes through unchanged.
    """

    def __init__(self, fh):
        self._fh = fh
        self._buf = ""

    def write(self, data):
        self._buf += data
        while True:
            nl = self._buf.find("\n")
            if nl == -1:
                # No complete line yet; collapse '\r' overwrites so an actively
                # redrawing progress bar can't grow the buffer without bound.
                cr = self._buf.rfind("\r")
                if cr != -1:
                    self._buf = self._buf[cr + 1:]
                break
            line = self._buf[:nl]
            self._buf = self._buf[nl + 1:]
            # effective content = text after the last carriage return
            eff = line.rsplit("\r", 1)[-1]
            if eff.strip() and _PROGRESS_RE.search(eff):
                continue  # drop progress-bar line
            self._fh.write(eff + "\n")
        self._fh.flush()

    def flush(self):
        try:
            self._fh.flush()
        except Exception:
            pass

    def close(self):
        # flush any trailing non-progress remainder
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
    """Timestamped line; goes to console and (via the Tee) the log file."""
    print(f"[{_timestamp()}] {msg}", flush=True)


def _load_adata(adata_path):
    """Read an AnnData from disk (.h5ad preferred, falls back to scanpy)."""
    if not os.path.exists(adata_path):
        raise FileNotFoundError(f"adata_path does not exist: {adata_path}")
    if adata_path.endswith(".h5ad"):
        return ad.read_h5ad(adata_path)
    # generic fallback (loom, zarr-as-dir handled by scanpy where possible)
    return sc.read(adata_path)


def _normalize_config(cfg):
    """Return (global_dict, [dataset_dict, ...]) from a parsed YAML config.

    Accepts either a top-level list of datasets, or a dict with a `datasets:`
    key plus optional `log_file`, `log_level`, and `defaults:` blocks.
    """
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
    """Merge global defaults with a dataset entry -> kwargs for run_nsv_pipeline."""
    merged = dict(defaults or {})
    for k, v in entry.items():
        if k in ("name", "adata_path", "dir_path"):
            continue
        if k in _PIPELINE_PARAMS:
            merged[k] = v
        else:
            _log(f"  WARNING: ignoring unknown parameter '{k}'")
    # keep only recognized params
    return {k: merged[k] for k in merged if k in _PIPELINE_PARAMS}


def run_from_config(config_path, no_smoothing=False):
    """Run the pipeline for every dataset listed in a YAML config, in sequence.

    If `no_smoothing` is True (the `--no-smoothing` CLI flag), it overrides any
    `no_smoothing` value in the config and is applied to every dataset.
    """
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

    # ---- set up the detailed log (console + file) ----
    from datetime import datetime
    default_log = f"nsv_pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"
    log_file = global_cfg.get("log_file", default_log) if isinstance(global_cfg, dict) else default_log
    log_dir = os.path.dirname(os.path.abspath(log_file))
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_fh = _FilteredFile(open(log_file, "a"))
    sys.stdout = _Tee(sys.__stdout__, log_fh)
    sys.stderr = _Tee(sys.__stderr__, log_fh)

    import time as _time
    import traceback as _traceback

    _log(f"=== noSpliceVelo batch run started ===")
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
            _log(f"  ERROR: entry '{name}' must define 'adata_path' and 'dir_path'; skipping")
            summary.append((name, "SKIPPED (missing adata_path/dir_path)", 0.0))
            continue

        params = _resolve_params(defaults, entry)
        if no_smoothing:
            params["no_smoothing"] = True
        _log(f"  adata_path : {entry['adata_path']}")
        _log(f"  dir_path   : {entry['dir_path']}")
        _log(f"  params     : {params}")

        t0 = _time.time()
        try:
            adata = _load_adata(entry["adata_path"])
            _log(f"  loaded adata: {adata.shape[0]} cells x {adata.shape[1]} genes")
            run_nsv_pipeline(adata, entry["dir_path"], **params)
            dt = _time.time() - t0
            _log(f"  DONE in {dt/60:.1f} min")
            summary.append((name, "OK", dt))
        except Exception as e:
            dt = _time.time() - t0
            _log(f"  FAILED after {dt/60:.1f} min: {type(e).__name__}: {e}")
            _traceback.print_exc()
            summary.append((name, f"FAILED ({type(e).__name__})", dt))
            # continue with the next dataset rather than aborting the whole run

        # free GPU/host memory between datasets
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

    _log("")
    _log("=== batch run summary ===")
    for name, status, dt in summary:
        _log(f"  {name:<30s} {status:<28s} {dt/60:6.1f} min")
    _log("=== noSpliceVelo batch run finished ===")

    log_fh.flush()
    log_fh.close()


def _parse_args(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        description="Run the noSpliceVelo pipeline over one or more datasets "
                    "described in a YAML config file."
    )
    parser.add_argument(
        "config",
        help="Path to the YAML config file (see nsv_config_runs_studentT_time_template.yaml).",
    )
    parser.add_argument(
        "--no-smoothing",
        dest="no_smoothing",
        action="store_true",
        help="Skip the kNN smoothing of the SCVIModified mean/variance; use "
             "mu_scvi / var_scvi directly as mu_scvi_smooth / var_scvi_smooth "
             "for all datasets.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_from_config(args.config, no_smoothing=args.no_smoothing)