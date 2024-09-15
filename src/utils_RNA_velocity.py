import scipy
import scipy.cluster.hierarchy as sch
import numpy as np
import pandas as pd
import torch
import gc

# def cluster_corr(corr_array, inplace=False):
#     """
#     Rearranges the correlation matrix, corr_array, so that groups of highly 
#     correlated variables are next to eachother 
    
#     Parameters
#     ----------
#     corr_array : pandas.DataFrame or numpy.ndarray
#         a NxN correlation matrix 
        
#     Returns
#     -------
#     pandas.DataFrame or numpy.ndarray
#         a NxN correlation matrix with the columns and rows rearranged
#     """
#     pairwise_distances = sch.distance.pdist(corr_array)
#     linkage = sch.linkage(pairwise_distances, method='complete')
#     cluster_distance_threshold = pairwise_distances.max()/2
#     idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
#                                         criterion='distance')
#     idx = np.argsort(idx_to_cluster_array)
    
#     if not inplace:
#         corr_array = corr_array.copy()
    
#     if isinstance(corr_array, pd.DataFrame):
#         return corr_array.iloc[idx, :].T.iloc[idx, :], linkage
#     return corr_array[idx, :][:, idx], linkage

def cluster_corr(corr_array, inplace=False, nclusts=2, \
                 cluster_distance_threshold=None):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly 
    correlated variables are next to eachother 
    
    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix 
        
    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    if cluster_distance_threshold is None:
        cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, 
                                        criterion='distance')
    id_clusters = sch.fcluster(linkage, nclusts, 
                                        criterion='maxclust')
    idx = np.argsort(idx_to_cluster_array)
    
    if not inplace:
        corr_array = corr_array.copy()
    
    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :], id_clusters, linkage
    return corr_array[idx, :][:, idx], id_clusters, linkage


def get_max_ll_at_mode(burst_B, burst_F):
    r = burst_F.copy()
    p = 1 / (1 + burst_B.copy())
    mode_ = np.floor(((r - 1) * (1 - p)) / p)
    mode_[r <= 1] = 0
    out_ = scipy.stats.nbinom.logpmf(mode_, r, p)
    return out_

def get_max_ll_at_obs(burst_B, burst_F, obs_):
    r = burst_F.copy()
    p = 1 / (1 + burst_B.copy())
    out_ = scipy.stats.nbinom.logpmf(obs_, r, p)
    return out_


def initialize_module_noSpliceVelo_params_gene_specific(module_, param_name, param_prior):
    getattr(module_.module, param_name).data.copy_(\
        param_prior
    )

def burst_fb_from_quantiles(mu_, var_, quantile_=0.95, type_="upper", ax=0):
    if type_ == "upper":
        id_mu = mu_ >= np.quantile(mu_, quantile_, axis=ax)
        id_var = var_ >= np.quantile(var_, quantile_, axis=ax)
    elif type_ == "lower":
        id_mu = mu_ <= np.quantile(mu_, quantile_, axis=ax)
        id_var = var_ <= np.quantile(var_, quantile_, axis=ax)

    id_ = id_mu & id_var
    var_quant = \
        np.array([np.mean(var_[id_[:, col], col]) \
                  if np.any(id_[:, col]) \
                    else np.nan \
                    for col in range(var_.shape[1])])
    mu_quant = \
        np.array([np.mean(mu_[id_[:, col], col]) \
                  if np.any(id_[:, col]) \
                    else np.nan \
                    for col in range(mu_.shape[1])])
    b_ = var_quant / mu_quant - 1
    b_[b_ <= 0] = 1e-6
    f_ = mu_quant / b_
    return f_, b_


def initialize_module_noSpliceVelo_priorClust(module_clust, module_priorClust, \
                                              use_encoders=True):
    params_list = ["gamma_mRNA", "burst_B1", "burst_f1", "burst_B2", "burst_f2", \
                "burst_B3", "burst_f3", "burst_B_low", "burst_F_low", "time_Bswitch", \
                "time_ss", "scale_unconstr", "burst_B_states", "burst_f_states"]
    for param_ in params_list:
        getattr(module_priorClust.module, param_).data.copy_(\
            getattr(module_clust.module, param_)
        )
    # model_scvi_pos_prior.module.gamma_mRNA.data.copy_(model_scvi_pos.module.gamma_mRNA.data)

    # copy params for encoders
    if use_encoders:
        module_priorClust.module.z_encoder.load_state_dict(module_clust.module.z_encoder.state_dict())
        module_priorClust.module.l_encoder.load_state_dict(module_clust.module.l_encoder.state_dict())

    # copy params for the decoder
    module_list = ['px_decoder', 'px_burstB1_decoder', 'px_burstF1_decoder', \
                'px_burstB2_decoder', 'px_burstF2_decoder', 'px_burstB3_decoder', \
                'px_burstF3_decoder', 'px_burstB4_decoder', 'px_burstF4_decoder', \
                'px_gamma_decoder', 'time_scale_decoder', 'time_scale_switch_decoder', \
                'time_scale_next_decoder']
    for module_ in module_list:
        # Copy the parameters from model1 to model2
        getattr(module_priorClust.module.decoder, module_).load_state_dict(
            getattr(module_clust.module.decoder, module_).state_dict()
        )


def get_mu_var_max(mu_smooth, var_smooth, device_="cuda"):
    mu_max = np.max(mu_smooth * 1.0, axis=0) + 1
    var_max = np.max(var_smooth * 1.0, axis=0) + 1
    if device_ == "cuda":
        mu_max_torch = torch.tensor(mu_max, dtype=torch.float, device=device_)
        var_max_torch = torch.tensor(var_max, dtype=torch.float, device=device_)
    else:
        mu_max_torch = mu_max.copy()
        var_max_torch = var_max.copy()

    return mu_max_torch, var_max_torch

def get_mu_var_center_scale(mu_smooth, var_smooth, device_="cuda", no_scaling=False):
    mu_max, var_max = get_mu_var_max(mu_smooth, var_smooth, device_="cpu")
    fac_multiply = 1.0
    mu_scale = np.std(mu_smooth * fac_multiply, axis=0)
    std_scale = np.std(np.sqrt(var_smooth * fac_multiply), axis=0)
    std_ref_scale = np.std(np.sqrt(var_max - var_smooth * fac_multiply), axis=0)

    mu_center = np.mean(mu_smooth * fac_multiply, axis=0)
    std_center = np.mean(np.sqrt(var_smooth * fac_multiply), axis=0)
    std_ref_center = np.mean(np.sqrt(var_max - var_smooth * fac_multiply), axis=0)

    mu_scale = torch.tensor(mu_scale, dtype=torch.float, device=device_)
    std_scale = torch.tensor(std_scale, dtype=torch.float, device=device_)
    std_ref_scale = torch.tensor(std_ref_scale, dtype=torch.float, device=device_)

    mu_center = torch.tensor(mu_center, dtype=torch.float, device=device_)
    std_center = torch.tensor(std_center, dtype=torch.float, device=device_)
    std_ref_center = torch.tensor(std_ref_center, dtype=torch.float, device=device_)

    if no_scaling:
        mu_center[:] = 0.0
        std_center[:] = 0.0
        std_ref_center[:] = 0.0

        mu_scale[:] = 1.0
        std_scale[:] = 1.0
        std_ref_scale[:] = 1.0

    return mu_center, std_center, std_ref_center, \
        mu_scale, std_scale, std_ref_scale


def get_std_sum(var_smooth, device_="cuda"):
    fac_multiply = 1.0
    std_sum = ((2 * np.sqrt(var_smooth * fac_multiply + 1e-10))**2.0).mean(0)
    std_sum = np.sqrt(std_sum + 1e-10)
    std_sum_torch = torch.tensor(std_sum, dtype=torch.float, device=device_)
    return std_sum_torch


def get_mu_var_time(mu_0, var_0, mu_f, var_f, gamma_, time):
    p_t = np.exp(-gamma_ * time)
    mu_t = mu_0 * p_t + mu_f * (1 - p_t)
    var_t = (var_0 - mu_0) * p_t**2.0 + (var_f - mu_f) * (1 - p_t**2.0) + mu_t
    return mu_t, var_t


def get_nosplicevelo_ll_params_geneCell(model_, nboot=10, prob_thresh=0.5):
    for b_ in range(nboot):
        print(f'gene-cell boot = {b_}')
        params_ = model_.get_likelihood_parameters_new()
        velo_mu_up = params_['velo_mu_up'].copy()
        velo_var_up = params_['velo_var_up'].copy()
        velo_mu_down = params_['velo_mu_down'].copy()
        velo_var_down = params_['velo_var_down'].copy()
        f1 = params_['burst_f1'].copy()
        b1 = params_['burst_B1'].copy()
        gamma_ = params_['gamma_mRNA_all'].copy()
        mu_0 = f1 * b1 / gamma_
        var_0 = mu_0 * (b1 + 1)
        mu_up_f = params_['mu_up_f'].copy()
        var_up_f = params_['var_up_f'].copy()
        mu_down_up = params_['mu_down_up'].copy()
        var_down_up = params_['var_down_up'].copy()
        mu_down_f = params_['mu_down_f'].copy()
        var_down_f = params_['var_down_f'].copy()
        time_up = params_['tau_up'].copy()
        time_down = params_['tau_down'].copy()
        time_ss = params_['time_ss'].copy()
        mu_up, var_up = get_mu_var_time(mu_0, var_0, mu_up_f, var_up_f, gamma_, time_up)
        mu_down, var_down = get_mu_var_time(mu_down_up, var_down_up, \
                                            mu_down_f, var_down_f, gamma_, time_down)
        gamma_ = params_['gamma_mRNA_all'].copy()
        probs_ = params_['prob_state_list'][:, :, 0].copy()
        # probs_1 = params_scvi_pos['prob_state_list'][:, :, 1].copy()
        id_up = np.where(probs_ >= prob_thresh)
        id_down = np.where(probs_ <= (1 - prob_thresh))
        probs_tmp = probs_.copy()
        probs_tmp[id_up] = 1.0
        probs_tmp[id_down] = 0.0

        # velocities
        velo_mu_ = np.zeros_like(velo_mu_up)
        velo_var_ = np.zeros_like(velo_var_up)
        velo_mu_[id_up] = velo_mu_up[id_up].copy()
        velo_mu_[id_down] = velo_mu_down[id_down].copy()
        velo_var_[id_up] = velo_var_up[id_up].copy()
        velo_var_[id_down] = velo_var_down[id_down].copy()

        # get latent time
        time_up_tmp = time_up.copy()
        time_up_tmp[id_down] = 0.0
        time_down_tmp = time_down.copy()
        time_down_tmp[id_up] = 1e10
        # time_pred_tmp = (time_down_tmp) + np.max(time_up_tmp, axis=0)
        time_pred_tmp = time_down + time_ss
        time_pred = np.zeros_like(time_up)
        time_pred[id_up] = time_up[id_up]
        time_pred[id_down] = time_pred_tmp[id_down]

        # time_pred = time_up * probs_ + time_pred_tmp * (1 - probs_)

        mu_ = np.zeros_like(velo_mu_up)
        mu_[id_up] = mu_up[id_up].copy()
        mu_[id_down] = mu_down[id_down].copy()
        # mu_scvi_pos[id_mid] = mu_up[id_mid] * probs_[id_mid] + mu_down[id_mid] * (1 - probs_[id_mid])
        var_ = np.zeros_like(velo_mu_up)
        var_[id_up] = var_up[id_up].copy()
        var_[id_down] = var_down[id_down].copy()

        del params_
        torch.cuda.empty_cache()
        gc.collect()

        if b_ == 0:
            mu_all = mu_.copy()
            var_all = var_.copy()
            velo_mu_all = velo_mu_.copy()
            velo_var_all = velo_var_.copy()
            time_pred_all = time_pred.copy()
            probs_all = probs_tmp.copy()
        else:
            mu_all += mu_.copy()
            var_all += var_.copy()
            velo_mu_all += velo_mu_.copy()
            velo_var_all += velo_var_.copy()
            time_pred_all += time_pred.copy()
            probs_all += probs_tmp.copy()
    
    mu_all /= nboot
    var_all /= nboot
    velo_mu_all /= nboot
    velo_var_all /= nboot
    time_pred_all /= nboot
    probs_all /= nboot
    return mu_all, var_all, velo_mu_all, velo_var_all, time_pred_all, probs_all


def get_nosplicevelo_ll_params_gene(model_, nboot=10, prob_thresh=0.5):

    for b_ in range(nboot):
        print(f'gene boot = {b_}')
        params_gene = model_.get_likelihood_parameters_gene_specific()
        f1_gene = params_gene['burst_f1'].copy()
        b1_gene = params_gene['burst_B1'].copy()
        gamma_gene = params_gene['gamma_mRNA_all'].copy()
        mu_0_gene = f1_gene * b1_gene / gamma_gene
        var_0_gene = mu_0_gene * (b1_gene + 1)
        mu_up_f_gene = params_gene['mu_up_f'].copy()
        var_up_f_gene = params_gene['var_up_f'].copy()
        mu_down_up_gene = params_gene['mu_down_up'].copy()
        var_down_up_gene = params_gene['var_down_up'].copy()
        mu_down_f_gene = params_gene['mu_down_f'].copy()
        var_down_f_gene = params_gene['var_down_f'].copy()
        time_ss_gene = params_gene['time_ss'].copy()
        time_up_gene = params_gene['tau_up'].copy()
        time_down_gene = params_gene['tau_down'].copy()
        mu_up_gene, var_up_gene = \
            get_mu_var_time(mu_0_gene , var_0_gene , mu_up_f_gene , \
                            var_up_f_gene , gamma_gene , time_up_gene)
        mu_up_ss_gene, var_up_ss_gene = \
            get_mu_var_time(mu_0_gene , var_0_gene , mu_up_f_gene , \
                            var_up_f_gene , gamma_gene , time_ss_gene)
        mu_down_gene, var_down_gene = \
            get_mu_var_time(mu_down_up_gene, var_down_up_gene, \
                            mu_down_f_gene, var_down_f_gene, gamma_gene, time_down_gene)
        
        probs_ = params_gene['prob_state'][:, :, 0].copy()
        # probs_1 = params_scvi_pos['prob_state_list'][:, :, 1].copy()
        id_up = np.where(probs_ >= prob_thresh)
        id_down = np.where(probs_ <= (1 - prob_thresh))
        probs_tmp = probs_.copy()
        probs_tmp[id_up] = 1.0
        probs_tmp[id_down] = 0.0
        
        # mu, var for gene-specific
        mu_gene = np.zeros_like(mu_up_gene)
        mu_gene[id_up] = mu_up_gene[id_up].copy()
        mu_gene[id_down] = mu_down_gene[id_down].copy()
        var_gene = np.zeros_like(mu_up_gene)
        var_gene[id_up] = var_up_gene[id_up].copy()
        var_gene[id_down] = var_down_gene[id_down].copy()

        velo_mu_up_gene = (mu_up_f_gene - mu_up_gene) * gamma_gene
        B_up_f_gene = var_up_f_gene / mu_up_f_gene - 1
        velo_var_up_gene = (mu_up_f_gene * (2 * B_up_f_gene + 1) + mu_up_gene - \
                            2 * var_up_gene) * gamma_gene
        velo_mu_down_gene = (mu_down_f_gene - mu_down_gene) * gamma_gene
        B_down_f_gene = var_down_f_gene / mu_down_f_gene - 1
        velo_var_down_gene = (mu_down_f_gene * (2 * B_down_f_gene + 1) + mu_down_gene - \
                            2 * var_down_gene) * gamma_gene


        velo_mu_gene = np.zeros_like(velo_mu_up_gene)
        velo_var_gene = np.zeros_like(velo_var_up_gene)
        velo_mu_gene[id_up] = velo_mu_up_gene[id_up].copy()
        velo_mu_gene[id_down] = velo_mu_down_gene[id_down].copy()
        velo_var_gene[id_up] = velo_var_up_gene[id_up].copy()
        velo_var_gene[id_down] = velo_var_down_gene[id_down].copy()

        del params_gene
        torch.cuda.empty_cache()
        gc.collect()

        if b_ == 0:
            mu_all_gene = mu_gene.copy()
            var_all_gene = var_gene.copy()
            velo_mu_all_gene = velo_mu_gene.copy()
            velo_var_all_gene = velo_var_gene.copy()
        else:
            mu_all_gene += mu_gene.copy()
            var_all_gene += var_gene.copy()
            velo_mu_all_gene += velo_mu_gene.copy()
            velo_var_all_gene += velo_var_gene.copy()
    
    mu_all_gene /= nboot
    var_all_gene /= nboot
    velo_mu_all_gene /= nboot
    velo_var_all_gene /= nboot
    return mu_all_gene, var_all_gene, velo_mu_all_gene, velo_var_all_gene

def get_nosplicevelo_ll_params(model_, nboot=10, prob_thresh=0.5):
    # get gene specific velo from gene specific params
    mu_all_gene, var_all_gene, velo_mu_all_gene, velo_var_all_gene = \
        get_nosplicevelo_ll_params_gene(model_, prob_thresh=prob_thresh, nboot=nboot)
    
    # get geneCell specific velo from geneCell specific params
    mu_all, var_all, velo_mu_all, velo_var_all, time_pred_all, probs_all = \
        get_nosplicevelo_ll_params_geneCell(model_, prob_thresh=prob_thresh, nboot=nboot)
    
    params_all = {}
    params_all['mu_all_gene'] = mu_all_gene
    params_all['var_all_gene'] = var_all_gene
    params_all['velo_mu_all_gene'] = velo_mu_all_gene
    params_all['velo_var_all_gene'] = velo_var_all_gene

    params_all['mu_all'] = mu_all
    params_all['var_all'] = var_all
    params_all['velo_mu_all'] = velo_mu_all
    params_all['velo_var_all'] = velo_var_all
    params_all['time_pred_all'] = time_pred_all
    params_all['probs_all'] = probs_all

    return params_all


def get_nosplicevelo_ll_params_muVarUp_gene(model_, nboot=10, prob_thresh=0.5):

    for b_ in range(nboot):
        print(f'gene boot = {b_}')
        params_gene = model_.get_likelihood_parameters_gene_specific()
        mu_up_f_gene = params_gene['mu_up_f'].copy()
        var_up_f_gene = params_gene['var_up_f'].copy()
        mu_down_up_gene = params_gene['mu_down_up'].copy()
        var_down_up_gene = params_gene['var_down_up'].copy()

        del params_gene
        torch.cuda.empty_cache()
        gc.collect()

        if b_ == 0:
            mu_up_f = mu_up_f_gene.copy()
            var_up_f = var_up_f_gene.copy()
            mu_down_up = mu_down_up_gene.copy()
            var_down_up = var_down_up_gene.copy()
        else:
            mu_up_f += mu_up_f_gene.copy()
            var_up_f += var_up_f_gene.copy()
            mu_down_up += mu_down_up_gene.copy()
            var_down_up += var_down_up_gene.copy()
    
    mu_up_f /= nboot
    var_up_f /= nboot
    mu_down_up /= nboot
    var_down_up /= nboot
    return mu_up_f, var_up_f, mu_down_up, var_down_up

