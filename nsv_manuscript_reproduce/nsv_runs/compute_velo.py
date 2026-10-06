"""Compute noSpliceVelo velocity layers/fields from a trained model.

`compute_velo(model)` takes a loaded noSpliceVelo model (with its AnnData) and
returns (adata, prob_state_avg) where `adata` has the velocity layers / fitted
moments / per-gene kinetic parameters populated. The heavy inference is done
through the batched `get_nosplicevelo_ll_params`, so it is safe for large
datasets.
"""

import os
import sys
import gc

import numpy as np
import scipy.stats
import torch
import scanpy as sc
import scvelo as scv

# Make ../nsv (model code) and this folder importable, independent of the
# working directory. Set NSV_SRC to use a different copy of the model code.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.environ.get("NSV_SRC", os.path.abspath(os.path.join(_HERE, "..", "nsv"))), _HERE):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from nosplicevelo_ll_params import (
    get_nosplicevelo_ll_params,
    get_nosplicevelo_ll_params_muVarUp_gene,
)


def compute_velo(
    model_nosplicevelo,
    model_scvi=None,
    nboot=10,
    prob_thresh=0.5,
    rep="X_pca",
    n_neighbors=15,
    cell_batch_size=4096,
    gpu_batch_size=512,
    state_reduction="argmax_stable",
    extra_reductions=None,
    knn_rep="X_latent",
    knn_k=30,
    knn_n_iter=1,
    temper_taus=(4.0,),
):
    """Populate velocity layers / fitted moments / kinetic params on the model's AnnData.

    Parameters
    ----------
    model_nosplicevelo :
        A loaded noSpliceVelo model (its `.adata` supplies the data).
    model_scvi :
        Optional SCVIModified model; only used as the (currently unused)
        `model_0` argument of `get_nosplicevelo_ll_params`. Safe to leave None.
    nboot :
        Bootstrap repeats for the posterior-predictive averaging.
    prob_thresh :
        State probability threshold passed through to the ll-params routines.
    rep :
        obsm key used to build the neighbor graph (for velocity consistency).
        Falls back to 'X_latent', then computes PCA if neither is present.
    n_neighbors :
        Neighbors for the graph.
    cell_batch_size, gpu_batch_size :
        Batching controls forwarded to the (memory-safe) ll-params routines.
    extra_reductions :
        Optional list of additional state reductions, any of "argmax_knn" and
        "tempered" (see nSV/state_reductions.py). They add layers and never
        change the existing ones:
          argmax_knn   -> velocity_mu_argmax_knn, velocity_var_argmax_knn,
                          mu_fit_argmax_knn, var_fit_argmax_knn
          tempered     -> velocity_mu_tempered<tau>, ... for each tau
          vote         -> velocity_mu_vote, velocity_var_vote: per-bootstrap argmax
                          velocity averaged over bootstraps (velocity only)
          vote_knn     -> velocity_mu_vote_knn, velocity_var_vote_knn: as vote, but each
                          bootstrap's posterior is kNN-averaged before its argmax
    knn_rep, knn_k, knn_n_iter :
        Graph for argmax_knn: kNN connectivities on obsm[knn_rep] (falls back to
        X_pca) with knn_k neighbours, plus self loops, row-normalised; the
        posterior is averaged knn_n_iter times. Defaults match the X_latent /
        k=30 graph used to smooth mu_scvi / var_scvi.
    temper_taus :
        Exponent(s) tau for the tempered mixture p_s^tau / sum p^tau.

    Returns
    -------
    (adata, prob_state_avg) :
        adata : the model's AnnData with velocity fields added.
        prob_state_avg : np.ndarray of shape (n_cells, n_genes, n_states),
            the bootstrap-averaged per-state probabilities.
    """
    adata_pan_velo = model_nosplicevelo.adata.copy()
    adata_pan_tmp = adata_pan_velo.copy()

    # ---- neighbor graph (used for gene-velocity consistency) ----
    if rep not in adata_pan_tmp.obsm:
        if "X_latent" in adata_pan_tmp.obsm:
            rep = "X_latent"
        else:
            print(f"[compute_velo] '{rep}' not in obsm; computing PCA")
            sc.pp.pca(adata_pan_tmp, n_comps=30)
            rep = "X_pca"
    n_pcs = adata_pan_tmp.obsm[rep].shape[1]
    scv.pp.neighbors(adata_pan_tmp, n_pcs=n_pcs, n_neighbors=n_neighbors, use_rep=rep)

    mu_scvi_smooth = adata_pan_velo.layers['mu_scvi_smooth'].copy()
    var_scvi_smooth = adata_pan_velo.layers['var_scvi_smooth'].copy()

    # ---- optional: kNN weights for the argmax_knn reduction ----
    extra_reductions = list(extra_reductions or [])
    if isinstance(temper_taus, (int, float)):
        temper_taus = [temper_taus]
    knn_weights = None
    if "argmax_knn" in extra_reductions or "vote_knn" in extra_reductions:
        from state_reductions import row_normalised_knn
        _g = adata_pan_velo.copy()
        _rep = knn_rep if knn_rep in _g.obsm else "X_pca"
        if _rep not in _g.obsm:
            sc.pp.pca(_g, n_comps=30)
        sc.pp.neighbors(_g, use_rep=_rep, n_neighbors=int(knn_k))
        knn_weights = row_normalised_knn(_g.obsp["connectivities"], add_self=True)
        print(f"[compute_velo] argmax_knn graph: obsm['{_rep}'], k={knn_k}, n_iter={knn_n_iter}")
        del _g

    # reproducible sampling for the bootstrap
    np.random.seed(123)
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)

    # ---- main gene / gene-cell velocity parameters (batched, memory-safe) ----
    params_all = get_nosplicevelo_ll_params(
        model_nosplicevelo, model_scvi, adata_pan_tmp,
        nboot=nboot, prob_thresh=prob_thresh,
        cell_batch_size=cell_batch_size, gpu_batch_size=gpu_batch_size,
        state_reduction=state_reduction,
        extra_reductions=extra_reductions, knn_weights=knn_weights,
        knn_n_iter=int(knn_n_iter), temper_taus=tuple(float(t) for t in temper_taus),
    )
    torch.cuda.empty_cache()
    gc.collect()

    # per-gene mu/var correlation (kept for reference)
    corr_mu_var = np.zeros(adata_pan_velo.shape[1])
    for gene_ in range(adata_pan_velo.shape[1]):
        corr_mu_var[gene_] = scipy.stats.pearsonr(
            mu_scvi_smooth[:, gene_], var_scvi_smooth[:, gene_]
        )[0]
    adata_pan_velo.var['corr_mu_var'] = corr_mu_var

    # ---- velocity layers + fitted moments (gene-specific) ----
    adata_pan_velo.layers['time_latent'] = params_all['time_pred_all'].copy()
    adata_pan_velo.layers['velocity_mu'] = params_all['velo_mu_all_gene'].copy()
    adata_pan_velo.layers['mu_fit'] = params_all['mu_all_gene'].copy()
    adata_pan_velo.layers['velocity_var'] = params_all['velo_var_all_gene'].copy()
    adata_pan_velo.layers['var_fit'] = params_all['var_all_gene'].copy()

    # ---- soft (posterior-mixture) mu/var/velocities, when state_reduction='both' ----
    # Use mu_fit/var_fit (argmax_stable, on-manifold) for R2 / phase portraits,
    # and these soft fields (which retain the VAE's cross-state variability) for
    # the downstream velocity graph / streams.
    if params_all.get('velo_mu_soft_all_gene') is not None:
        adata_pan_velo.layers['mu_fit_soft'] = params_all['mu_soft_all_gene'].copy()
        adata_pan_velo.layers['var_fit_soft'] = params_all['var_soft_all_gene'].copy()
        adata_pan_velo.layers['velocity_mu_soft'] = params_all['velo_mu_soft_all_gene'].copy()
        adata_pan_velo.layers['velocity_var_soft'] = params_all['velo_var_soft_all_gene'].copy()

    # ---- optional extra reductions (argmax_knn / tempered<tau>) ----
    _layer_prefix = (("velo_mu_", "velocity_mu_"), ("velo_var_", "velocity_var_"),
                     ("mu_", "mu_fit_"), ("var_", "var_fit_"))
    _extra_layers = []
    for _k, _v in params_all.items():
        if not _k.endswith("_all_gene"):
            continue
        _base = _k[:-len("_all_gene")]
        if _base.startswith("state_change_frac_"):
            adata_pan_velo.uns[_base] = float(np.asarray(_v))
            continue
        _name = next((n for n in ("argmax_knn", "vote_knn", "vote") if _base.endswith("_" + n)), None)
        if _name is None and "_tempered" in _base:
            _name = _base[_base.index("_tempered") + 1:]
        if _name is None:
            continue
        for _src, _dst in _layer_prefix:
            if _base == _src + _name:
                adata_pan_velo.layers[_dst + _name] = np.asarray(_v, dtype=np.float32)
                _extra_layers.append(_dst + _name)
                break
    if _extra_layers:
        adata_pan_velo.uns["extra_state_reductions"] = dict(
            reductions=list(extra_reductions), knn_rep=str(knn_rep), knn_k=int(knn_k),
            knn_n_iter=int(knn_n_iter), temper_taus=[float(t) for t in temper_taus],
            layers=_extra_layers)
        print(f"[compute_velo] extra reduction layers: {_extra_layers}")

    # ---- bootstrap-averaged per-state probabilities ----
    # Reuse the average already produced by get_nosplicevelo_ll_params instead of
    # re-running an un-batched inference loop (which would OOM on large data).
    prob_state_avg = params_all['probs_all'].copy()   # (n_cells, n_genes, n_states)

    # ---- gene-level kinetic parameters (bootstrap-averaged, batched) ----
    mu_0_gene, var_0_gene, mu_up_f_gene, var_up_f_gene, \
        mu_down_up_gene, var_down_up_gene, mu_down_f_gene, var_down_f_gene = \
        get_nosplicevelo_ll_params_muVarUp_gene(
            model_nosplicevelo, adata_=adata_pan_tmp, nboot=nboot,
            prob_thresh=prob_thresh,
            cell_batch_size=cell_batch_size, gpu_batch_size=gpu_batch_size,
        )

    adata_pan_velo.var['mu_0_gene'] = mu_0_gene[0, :].flatten()
    adata_pan_velo.var['var_0_gene'] = var_0_gene[0, :].flatten()
    adata_pan_velo.var['mu_up_f_gene'] = mu_up_f_gene[0, :].flatten()
    adata_pan_velo.var['var_up_f_gene'] = var_up_f_gene[0, :].flatten()
    adata_pan_velo.var['mu_down_up_gene'] = mu_down_up_gene[0, :].flatten()
    adata_pan_velo.var['var_down_up_gene'] = var_down_up_gene[0, :].flatten()
    adata_pan_velo.var['mu_down_f_gene'] = mu_down_f_gene[0, :].flatten()
    adata_pan_velo.var['var_down_f_gene'] = var_down_f_gene[0, :].flatten()

    # time_switch + gamma_mRNA from a single (batched) gene-specific call
    params_gene = model_nosplicevelo.get_likelihood_parameters_gene_specific(
        batch_size=gpu_batch_size
    )
    adata_pan_velo.var['time_switch'] = np.asarray(params_gene['time_ss'])[0, :].flatten()
    adata_pan_velo.var['gamma_mRNA'] = np.asarray(params_gene['gamma_mRNA_all'])[0, :].flatten()
    del params_gene

    torch.cuda.empty_cache()
    gc.collect()

    return adata_pan_velo, prob_state_avg
