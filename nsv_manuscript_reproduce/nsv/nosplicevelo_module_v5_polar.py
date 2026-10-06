# Copyright (c) 2024, Tarun Mahajan, Sergei Maslov
# All rights reserved.
#
# This code is licensed under the BSD 3-Clause License.
# See the LICENSE file for more details.

# ---- Begin Third-Party Copyright Information ----
#
# This file incorporates code from scvi-tools (https://github.com/scverse/scvi-tools), 
# which is licensed under the BSD 3-Clause License.

# Copyright (c) 2024, Adam Gayoso, Romain Lopez, Martin Kim, Pierre Boyeau, Nir Yosef
# All rights reserved.

# See the `external_licenses/scvi_tools_LICENSE` file for more details.
#
# ---- End Third-Party Copyright Information ----

from typing import Callable, Iterable, Literal, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import logsumexp
from torch.distributions import Normal, StudentT, MixtureSameFamily, MultivariateNormal
from torch.distributions import kl_divergence as kl
from torch.distributions import Categorical, Dirichlet

# from scvi import REGISTRY_KEYS
from constants_tmp import REGISTRY_KEYS
from scvi.autotune._types import Tunable
from scvi.data._constants import ADATA_MINIFY_TYPE
from scvi.distributions import NegativeBinomial, Poisson, ZeroInflatedNegativeBinomial
from scvi.module.base import BaseMinifiedModeModuleClass, LossOutput, auto_move_data
from scvi.nn import Encoder, DecoderSCVI, LinearDecoderSCVI, one_hot
from scvi_clonealign_distributions import NegativeBinomialNew

torch.backends.cudnn.benchmark = True

import logging
# from typing import List, Literal, Optional
from typing import Dict, Iterable, Literal, Optional, Sequence, Union, Optional, List

import numpy as np

# from scvi import REGISTRY_KEYS
from scvi.data._constants import ADATA_MINIFY_TYPE
from scvi.nn import FCLayers
from scvi_clonealign_layers import GumbelSoftmax, GumbelSoftmax_3D
from scvi_clonealign_losses import entropy, entropy_no_mean
DEVICE_ = 'cuda'

def _identity(x):
    return x


class EncoderNew(nn.Module):
    """Encode data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.
    Uses a fully-connected neural network of ``n_hidden`` layers.
    Parameters
    ----------
    n_input
        The dimensionality of the input (data space)
    n_output
        The dimensionality of the output (latent space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    distribution
        Distribution of z
    var_eps
        Minimum value for the variance;
        used for numerical stability
    var_activation
        Callable used to ensure positivitgamma_mRNA_tmpy of the variance.
        Defaults to :meth:`torch.exp`.
    return_dist
        Return directly the distribution of z instead of its parameters.
    **kwargs
        Keyword args for :class:`~scvi.nn.FCLayers`
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        n_states: int = 2,
        dropout_rate: float = 0.1,
        distribution: str = "normal",
        var_eps: float = 1e-4,
        var_activation: Optional[Callable] = None,
        return_dist: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.n_states = n_states
        # self.ystate_encoder_first = FCLayers(
        #     n_in=int(n_input),
        #     n_out=n_hidden,
        #     n_cat_list=n_cat_list,
        #     n_layers=n_layers,
        #     n_hidden=n_hidden,
        #     dropout_rate=dropout_rate,
        #     **kwargs,
        # )
        # # self.ystate_encoder_second = GumbelSoftmax(n_hidden, n_states)
        # self.ystate_encoder_second = GumbelSoftmax_3D(n_hidden, int(n_input / 2), n_states)

        self.distribution = distribution
        self.var_eps = var_eps
        self.encoder = FCLayers(
            n_in=(n_input),
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            **kwargs,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
        self.return_dist = return_dist

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = _identity
        self.var_activation = torch.exp if var_activation is None else var_activation

    def forward(self, x: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.
         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim Ne(q_m, \\mathbf{I}q_v) \\)
        Parameters
        ----------
        x
            tensor with shape (n_input,)
        cat_list
            list of category membership(s) for this sample
        Returns
        -------
        3-tuple of :py:class:`torch.Tensor`
            tensors of shape ``(n_latent,)`` for mean and var, and sample
        """
        q = self.encoder(x, *cat_list)
        q_m = self.mean_encoder(q)
        q_v = self.var_activation(self.var_encoder(q)) + self.var_eps
        dist = Normal(q_m, q_v.sqrt())
        latent = self.z_transformation(dist.rsample())
        if self.return_dist:
            return dist, latent
        return q_m, q_v, latent
    

    
class DecoderNoiseVelo(nn.Module):
    """Decodes data from latent space of ``n_input`` dimensions into ``n_output`` dimensions.
    Uses a fully-connected neural network of ``n_hidden`` layers.
    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    scale_activation
        Activation layer to use for px_scale_decoder
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        n_states: int = 5,
        prior_branch_assignment: torch.Tensor = None,
        use_prior_parabola: bool = False,
        cluster_states: bool = False,
        state_loss_type: str = 'cross-entropy',
        inject_covariates: bool = True,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        scale_activation_init: Literal["softmax", "softplus", "sigmoid"] = "sigmoid",
        use_two_rep: bool = False,
        use_splicing: bool = False,
        use_time_cell: bool = False,
        use_alpha_gene:bool = False,
        use_controlBurst_gene: bool = False,
        burst_B_gene: bool = False,
        burst_f_gene:bool = False,
        use_library_time_correction: bool = False,
        timing_relative: bool = False,
        device_: str = "cuda",
        use_noise_ext: bool = False,
        fac_loss_geneCell: float = 1.0,
        use_time_dependence: bool = False,
        match_bottom_left: bool = False,
    ):
        r"""TODO: add docstring"""
        super().__init__()
        self.n_output = n_output
        self.state_loss_type = state_loss_type
        self.fac_loss_geneCell = fac_loss_geneCell
        self.prior_branch_assignment = prior_branch_assignment
        self.use_prior_parabola = use_prior_parabola
        self.match_bottom_left = match_bottom_left

        if cluster_states:
            if state_loss_type == 'cross-entropy':
                self.state_predictor = nn.Linear(n_input, n_states)
            else:
                self.state_predictor = nn.Linear(n_input, n_states - 1)

        n_input_new = n_input
        
        self.px_decoder = FCLayers(
            n_in=n_input_new,
            n_out=n_hidden,
            n_cat_list=None,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )

        self.pi_first_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=None,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0.0,
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        self.px_pi_decoder = nn.Linear(n_hidden, n_states * n_output)
        # self.px_pi_decoder = GumbelSoftmax_3D(n_hidden, n_output, n_states)
        
        self.use_two_rep = use_two_rep
        self.use_splicing = use_splicing
        self.use_time_cell = use_time_cell
        self.burst_B_gene = burst_B_gene
        self.burst_f_gene = burst_f_gene
        self.use_alpha_gene = use_alpha_gene
        self.use_library_time_correction = use_library_time_correction
        self.n_states = n_states
        self.use_noise_ext = use_noise_ext
        self.cluster_states = cluster_states
        self.timing_relative = timing_relative
        self.use_controlBurst_gene = use_controlBurst_gene
        self.use_time_dependence = use_time_dependence
        
        if use_time_dependence:
            self.b_t = nn.Linear(n_hidden, int(n_output))
            self.f_t = nn.Linear(n_hidden, int(n_output))
        
        if self.fac_loss_geneCell > 0:
            # if prior_branch_assignment is None:
            self.px_burstB1_decoder = nn.Linear(n_hidden, int(n_output))
            self.px_burstF1_decoder = nn.Linear(n_hidden, int(n_output))
            # else:
            #     mask_ = prior_branch_assignment == "parabola"
            #     self.px_burstB1_decoder = nn.Linear(n_hidden, int(torch.sum(mask_)))
            #     self.px_burstF1_decoder = nn.Linear(n_hidden, int(torch.sum(mask_)))
            self.px_burstB2_decoder = nn.Linear(n_hidden, int(n_output))
            self.px_burstF2_decoder = nn.Linear(n_hidden, int(n_output))
            self.px_burstB3_decoder = nn.Linear(n_hidden, int(n_output))
            self.px_burstF3_decoder = nn.Linear(n_hidden, int(n_output))
            self.px_burstB4_decoder = nn.Linear(n_hidden, int(n_output))
            self.px_burstF4_decoder = nn.Linear(n_hidden, int(n_output))
            self.px_gamma_decoder = nn.Linear(n_hidden, int(n_output))
        
        self.time_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            # nn.Sigmoid(),
        )
        
        # self.time_scale_switch_decoder = nn.Sequential(
        #     nn.Linear(n_hidden, n_output),
        #     # nn.Sigmoid(),
        # )

        self.time_scale_next_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            # nn.Sigmoid(),
        )

    def dist_from_connecting_line(self, mu_0, var_0, mu_f, var_f, mu_, var_):
        slope_ = (var_f - var_0) / (mu_f - mu_0 + 1e-9)
        intercept_ = (var_0 * mu_f - var_f * mu_0) / (mu_f - mu_0 + 1e-9)
        dist_ = var_ - slope_ * mu_ - intercept_
        return dist_ / torch.sqrt(1 + slope_**2.0)
    
    def get_slope_intercept(self, mu_0, var_0, mu_f, var_f):
        slope_ = (var_f - var_0) / (mu_f - mu_0)
        intercept_ = (var_0 * mu_f - var_f * mu_0) / (mu_f - mu_0)
        return slope_, intercept_
    
    def mu_from_connecting_line(self, mu_0, var_0, mu_f, var_f):
        slope_ = (var_f - var_0) / (mu_f - mu_0)
        intercept_ = (var_0 * mu_f - var_f * mu_0) / (mu_f - mu_0)
        mu_min, _ = \
            torch.max(torch.stack([- intercept_ / (slope_ - 1), \
                                   - intercept_ / (slope_)]), dim=0)
        mu_min = torch.clamp(mu_min, min=1e-9) + 0.1
        # val_ = mu_min * (slope_) + intercept_
        val_ = (slope_ - 1)
        id_ = val_ < 0
        if len(mu_min[id_]) > 0:
            print(f'mu_min neg')
            tmp = val_[id_]
            print(torch.min(tmp.detach()))
        return mu_min
    
    def var_from_connecting_line(self, mu_0, var_0, mu_f, var_f, mu_):
        slope_ = (var_f - var_0) / (mu_f - mu_0)
        intercept_ = (var_0 * mu_f - var_f * mu_0) / (mu_f - mu_0)
        var_max = slope_ * mu_ + intercept_
        return var_max
    
    def deriv2(self,gamma_mRNA_tmp, delta_mu, delta_var, p_t, mu_, var_):
        dmu_dt = gamma_mRNA_tmp * delta_mu * p_t
        dvar_dt = gamma_mRNA_tmp * (2 * delta_var * p_t**2.0 + \
                                       delta_mu * p_t * (1 - 2  * p_t))
        d2mu_dt2 = - gamma_mRNA_tmp**2.0 * delta_mu  * p_t
        d2var_dt2 = - gamma_mRNA_tmp**2.0 * (4 * delta_var * p_t**2.0 + \
                                             delta_mu * p_t * (1 - 4 * p_t))
        ones_ = torch.ones_like(var_, requires_grad=False)
        deriv2_fac1 = - (0.5 * torch.log(var_ + 1e-4) + torch.log(2 * ones_)) + \
            torch.log(torch.abs(d2var_dt2) + 1e-4) - \
            2 * torch.log(torch.abs(dmu_dt) + 1e-4)
        d2var_dt2_sign = 2 * (F.relu(d2var_dt2 * 1e5) / (F.relu(d2var_dt2 * 1e5) + 1e-9)) - \
            1
        deriv2_fac1 = d2var_dt2_sign * torch.exp(deriv2_fac1)

        deriv2_fac2 = - ((3 / 2) * torch.log(var_ + 1e-4) + torch.log(4 * ones_)) + \
             2  * torch.log(torch.abs(dvar_dt) + 1e-4) - \
            2 * torch.log(torch.abs(dmu_dt) + 1e-4)
        deriv2_fac2 = torch.exp(deriv2_fac2)

        deriv2_fac3 = - (0.5 * torch.log(var_ + 1e-4) + torch.log(2 * ones_)) + \
            torch.log(torch.abs(dvar_dt) + 1e-4) + \
            torch.log(torch.abs(d2mu_dt2) + 1e-4) - 3 * torch.log(torch.abs(dmu_dt) + 1e-4)
        dvar_dt_sign = 2 * (F.relu(dvar_dt * 1e5) / (F.relu(dvar_dt * 1e5) + 1e-9)) - \
            1
        dmu_dt_sign = 2 * (F.relu(dmu_dt * 1e5) / (F.relu(dmu_dt * 1e5) + 1e-9)) - \
            1
        d2mu_dt2_sign = 2 * (F.relu(d2mu_dt2 * 1e5) / (F.relu(d2mu_dt2 * 1e5) + 1e-9)) - \
            1
        deriv2_fac3 = dvar_dt_sign * dmu_dt_sign * d2mu_dt2_sign * \
            torch.exp(deriv2_fac3)
        
        d2sigma_dmu2 = deriv2_fac1 - deriv2_fac2 - deriv2_fac3

        return d2sigma_dmu2

    def deriv2_from_max(self,gamma_mRNA_tmp, delta_mu, delta_var, p_t, mu_, var_, \
                        mu_max, var_max):
        ones_ = torch.ones_like(var_, requires_grad=False)
        dmu_dt = gamma_mRNA_tmp * delta_mu * p_t
        dmuM_dt = - dmu_dt
        dvar_dt = gamma_mRNA_tmp * (2 * delta_var * p_t**2.0 + \
                                       delta_mu * p_t * (1 - 2  * p_t))
        dvar_dt_sign = 2 * (F.relu(dvar_dt * 1e5) / (F.relu(dvar_dt * 1e5) + 1e-9)) - \
            1
        dsigmaM_dt = - (0.5 * torch.log(var_max - var_ + 1e-4) + torch.log(2 * ones_)) + \
            torch.log(torch.abs(dvar_dt) + 1e-4)
        dsigmaM_dt = dvar_dt_sign * torch.exp(dsigmaM_dt)
        dsig_dt = \
            2 * torch.exp(torch.log(dvar_dt + 1e-4) - \
                          torch.log(torch.sqrt(var_ + 1e-4) + 1e-4))
        d2mu_dt2 = - gamma_mRNA_tmp**2.0 * delta_mu  * p_t
        d2muM_dt2 = - d2mu_dt2
        d2var_dt2 = - gamma_mRNA_tmp**2.0 * (4 * delta_var * p_t**2.0 + \
                                             delta_mu * p_t * (1 - 4 * p_t))
        deriv2_fac1 = - (0.5 * torch.log(var_max - var_ + 1e-4) + torch.log(2 * ones_)) + \
            torch.log(torch.abs(d2var_dt2) + 1e-4) - \
            2 * torch.log(torch.abs(dmuM_dt) + 1e-4)
        d2var_dt2_sign = 2 * (F.relu(d2var_dt2 * 1e5) / (F.relu(d2var_dt2 * 1e5) + 1e-9)) - \
            1
        # dmuM_dt_sign = 2 * (F.relu(dmuM_dt * 1e5) / (F.relu(dmuM_dt * 1e5) + 1e-9)) - \
        #     1
        deriv2_fac1 = d2var_dt2_sign * torch.exp(deriv2_fac1)

        deriv2_fac2 = - ((3 / 2) * torch.log(var_max - var_ + 1e-4) + torch.log(4 * ones_)) + \
             2  * torch.log(torch.abs(dvar_dt) + 1e-4) - \
            2 * torch.log(torch.abs(dmuM_dt) + 1e-4)
        deriv2_fac2 = torch.exp(deriv2_fac2)

        deriv2_fac3 = - (0.5 * torch.log(var_max - var_ + 1e-4) + torch.log(2 * ones_)) + \
            torch.log(torch.abs(dvar_dt) + 1e-4) + \
            torch.log(torch.abs(d2muM_dt2) + 1e-4) - 3 * torch.log(torch.abs(dmuM_dt) + 1e-4)
        dvar_dt_sign = 2 * (F.relu(dvar_dt * 1e5) / (F.relu(dvar_dt * 1e5) + 1e-9)) - \
            1
        dmuM_dt_sign = 2 * (F.relu(dmuM_dt * 1e5) / (F.relu(dmuM_dt * 1e5) + 1e-9)) - \
            1
        d2muM_dt2_sign = 2 * (F.relu(d2muM_dt2 * 1e5) / (F.relu(d2muM_dt2 * 1e5) + 1e-9)) - \
            1
        deriv2_fac3 = dvar_dt_sign * dmuM_dt_sign * d2muM_dt2_sign * \
            torch.exp(deriv2_fac3)
        
        d2sigma_dmu2 = - deriv2_fac1 - deriv2_fac2 + deriv2_fac3

        return d2sigma_dmu2


    def forward(
        self,
        z: torch.Tensor,
        scaling_factor: torch.Tensor = None,
        gamma_mRNA: torch.Tensor = None,
        use_gamma_mRNA: bool = False,
        capture_eff: torch.Tensor = None,
        tmax: float = 12,
        burst_B1: torch.Tensor = None,
        burst_F1: torch.Tensor = None,
        burst_B2: torch.Tensor = None,
        burst_F2: torch.Tensor = None,
        burst_B3: torch.Tensor = None,
        burst_F3: torch.Tensor = None,
        burst_B_low: torch.Tensor = None,
        burst_F_low: torch.Tensor = None,
        time_Bswitch: torch.Tensor = None,
        time_ss: torch.Tensor = None,
        mu_obs: torch.Tensor = None,
        var_obs: torch.Tensor = None,
        mu_ss_obs: torch.Tensor = None,
        var_ss_obs: torch.Tensor = None,
        mu_ss1_obs: torch.Tensor = None,
        var_ss1_obs: torch.Tensor = None,
        scale_dist_sigmoid: torch.Tensor = None,
        scale_dist_sigmoid2: torch.Tensor = None,
        scale_dist_sigmoid3: torch.Tensor = None,
        scale_dist_sigmoid4: torch.Tensor = None,
        scale_dist_sigmoid5: torch.Tensor = None,
        mu_max_tmp: torch.Tensor = None,
        var_max_tmp: torch.Tensor = None,
        match_upf_down_up: bool = False,
        max_fac: int = 2,
        device_: str = "cuda",
        mu_center: torch.Tensor = None,
        std_center: torch.Tensor = None,
        mu_scale: torch.Tensor = None,
        std_scale: torch.Tensor = None,
        match_burst_params_not_muVar: bool = False,
        scale_mu: torch.Tensor = None,
        scale_std: torch.Tensor = None,
        fac_var: float = 1.0,
        *cat_list: int,
    ):
        r"""TODO: add docstring"""
        
        # eps_ratio = mu_max_tmp / (var_max_tmp + 1e-8)
        eps_ratio = 1.0
        
        if self.cluster_states:
            state_pred = self.state_predictor(z)
        else:
            state_pred = None

        pow_fac = 0.5

        z_new = z
        px = self.px_decoder(z_new, *cat_list)

        # cells by genes by n_states
        pi_first = self.pi_first_decoder(z)
        temperature=0.1
        hard=1
        # logits_state, px_pi, y_state = \
        #     self.px_pi_decoder(pi_first, temperature, hard)
        

        px_pi = nn.Softplus()(
                    torch.reshape(
                        self.px_pi_decoder(pi_first), 
                        (z.shape[0], self.n_output, self.n_states)
                    )
                ) + 1e-6

        # get gene-specific burst frequency and burst size for the starting 
        # steady state for the upregulated branch. Also get mu and var.
        # var_max = (var_max_tmp)
        px_B1_gene = F.softplus(burst_B_low.repeat(px.shape[0], 1)) + 1e-9
        px_f1_by_gam_gene = F.softplus(burst_F_low.repeat(px.shape[0], 1)) + 1e-9
        # f1_by_gam_max_gene = torch.min((mu_max_tmp * 0.5 / fac_var) / px_B1_gene, \
        #                           ((var_max_tmp * 0.5 / fac_var) / (px_B1_gene * (px_B1_gene + 1))))
        # px_f1_by_gam_gene = torch.clamp(px_f1_by_gam_gene, max=f1_by_gam_max_gene)
        mu_0_gene = px_B1_gene * px_f1_by_gam_gene
        var_0_gene = eps_ratio * mu_0_gene * (px_B1_gene + 1)

        # get gene-cell-specific burst frequency and burst size for the starting 
        # steady state for the upregulated branch. Also get mu and var.
        if self.fac_loss_geneCell > 0:
            px_B1 = px_B1_gene
            px_f1_by_gam = px_f1_by_gam_gene
            mu_0 = px_B1 * px_f1_by_gam
            var_0 = eps_ratio * mu_0 * (px_B1 + 1)
        else:
            px_B1 = None
            px_f1_by_gam = None
            mu_0 = None
            var_0 = None

        # assign gene-cell-specific time for the upregulated branch
        time_ss_full = F.softplus(time_ss.repeat(px.shape[0], 1))
        time_ss_full = torch.clamp(time_ss_full, max=tmax)
        
        if self.use_prior_parabola:
            mask_ = self.prior_branch_assignment == 1 # down regulation branch
            mask_reshaped = mask_.view(1, -1)
            time_ss_full = torch.where(mask_reshaped, torch.zeros_like(time_ss_full), 
                                       time_ss_full)
            
        time_ss_full_gene = time_ss_full
        
        px_time_scale = F.softplus(self.time_scale_decoder(px))
        px_time_rate_tmp = torch.clamp(px_time_scale, max=time_ss_full)
        px_time_rate_tmp_gene = px_time_rate_tmp

        # assign gene-cell-specific time for the downregulated branch
        px_time_scale_next = F.softplus(self.time_scale_next_decoder(px))
        px_time_rate_rep_tmp = torch.clamp(px_time_scale_next, \
                                           max=(tmax - time_ss_full))
        px_time_rate_rep_tmp_gene = px_time_rate_rep_tmp

        p_up_branch = px_pi[..., 0]
        p_up_branch = F.relu(p_up_branch - 0.5)
        p_down_branch = 1 - p_up_branch
        time_total = (px_time_rate_tmp * p_up_branch) + \
            ((px_time_rate_rep_tmp + time_ss_full) * p_down_branch)
        loss_time = torch.mean(torch.var(time_total, dim=-1))
        # Compute mean time per cell
        mean_time_per_cell = torch.mean(time_total, dim=-1, keepdim=True)

        # Compute absolute deviation per gene in each cell
        time_deviation = torch.abs(time_total - mean_time_per_cell)

        # Compute weights using learnable scaling factor
        # weights_time = torch.exp(-scaling_factor * time_deviation)
        weights_time = torch.ones_like(time_total)

        def get_mu_std_scaled_loss(mu_0, std_0, mu_1, std_1, mu_center, mu_scale,
                                   std_center, std_scale,
                                   scale_mu, scale_std,
                                   weights_genes=None):
            if weights_genes is None:
                weights_genes = torch.ones_like(mu_0)
            mu_0_scaled = (mu_0 - mu_center) / mu_scale
            mu_1_scaled = (mu_1 - mu_center) / mu_scale
            std_0_scaled = (std_0 - std_center) / std_scale
            std_1_scaled = (std_1 - std_center) / std_scale

            loss_mu = (((mu_0_scaled - mu_1_scaled)**2.0) / (2 * scale_mu**2.0))
            loss_mu = (loss_mu * weights_genes).sum(-1)
            loss_std = (((std_0_scaled - std_1_scaled)**2.0) / (2 * scale_std**2.0))
            loss_std = (loss_std * weights_genes).sum(-1)
            loss_total = (loss_mu + loss_std).mean()
            return loss_total

        # match gene-specific and gene-cell-specific mu and var for the starting
        # steady state for the upregulated branch
        if not match_burst_params_not_muVar:
            loss_match_mu_var_0 = 0.0
            if self.fac_loss_geneCell > 0:
                loss_match_mu_var_0 = \
                    get_mu_std_scaled_loss(mu_0, torch.sqrt(var_0 + 1e-4), \
                                           mu_0_gene, torch.sqrt(var_0_gene + 1e-4), \
                                           mu_center, mu_scale, \
                                           std_center, std_scale,
                                           scale_mu, scale_std, weights_time)
        else:
            loss_match_mu_var_0 = 0.0
            if self.fac_loss_geneCell > 0:
                loss_match_mu_var_0 = \
                    torch.sqrt((torch.log(px_f1_by_gam + 1e-4) - \
                                torch.log(px_f1_by_gam_gene + 1e-4))**2.0 + \
                               (torch.log(px_B1 + 1e-4) - \
                                torch.log(px_B1_gene + 1e-4))**2.0 + 1e-4).sum(-1).mean()

        # get gene-specific burst frequency and burst size for the final 
        # steady state for the upregulated branch. Also get mu and var.
        px_f2_by_gam_gene = F.softplus(burst_F2.repeat(px.shape[0], 1)) + \
            px_f1_by_gam_gene * 1.2
        # f2_max_gene = (var_max - px_B1_gene**2.0 * px_f1_by_gam_gene)**2.0 / \
        #     (px_B1_gene**2.0 * px_f1_by_gam_gene + 1e-4)
        # px_f2_by_gam_gene = torch.clamp(px_f2_by_gam_gene, max=f2_max_gene)
        px_B2_gene = F.softplus(burst_B2.repeat(px.shape[0], 1)) + px_B1_gene * \
            torch.sqrt(px_f1_by_gam_gene / px_f2_by_gam_gene + 1e-9) + 1e-9
        # B2_max_gene = (-1 + torch.sqrt(1 + 4 * var_max / px_f2_by_gam_gene)) / 2
        # px_B2_gene = torch.clamp(px_B2_gene, max=B2_max_gene)
        if self.use_time_dependence:
            b_t = F.sigmoid(self.b_t(px)) + 1e-4
            f_t = F.sigmoid(self.b_t(px)) + 1e-4
            # f_t = F.sigmoid(self.f_t(px)) + 1e-4
        else:
            b_t = torch.ones_like(px_B2_gene)
            f_t = torch.ones_like(px_B2_gene)
            
        px_B2_gene = px_B2_gene * b_t
        px_f2_by_gam_gene = px_f2_by_gam_gene * f_t
        mu_up_f_gene = (px_B2_gene) * (px_f2_by_gam_gene)
        var_up_f_gene = eps_ratio * mu_up_f_gene * ((px_B2_gene) + 1)

        if self.fac_loss_geneCell > 0:
            # get gene-cell-specific burst frequency and burst size for the final
            # steady state for the upregulated branch. Also get mu and var.
            px_f2_by_gam = F.softplus(self.px_burstF2_decoder(px)) + px_f1_by_gam * 1.2
            # f2_max = (var_max - px_B1**2.0 * px_f1_by_gam)**2.0 / \
            #     (px_B1**2.0 * px_f1_by_gam + 1e-4)
            # px_f2_by_gam = torch.clamp(px_f2_by_gam, max=f2_max)
            px_B2 = F.softplus(self.px_burstB2_decoder(px)) + 1e-9 + \
                px_B1 * torch.sqrt(px_f1_by_gam / px_f2_by_gam + 1e-9)
            # B2_max = (-1 + torch.sqrt(1 + 4 * var_max / px_f2_by_gam)) / 2
            # px_B2 = torch.clamp(px_B2, max=B2_max)
            mu_up_f = px_B2 * px_f2_by_gam
            var_up_f = eps_ratio * mu_up_f * (px_B2 + 1)
        else:
            mu_up_f = None
            var_up_f = None
            px_f2_by_gam = None
            px_B2 = None

        # match gene-specific and gene-cell-specific mu and var for the final
        # steady state for the upregulated branch
        if not match_burst_params_not_muVar:
            if self.fac_loss_geneCell > 0.0:
                loss_match_mu_var_up_f = \
                    get_mu_std_scaled_loss(mu_up_f, torch.sqrt(var_up_f + 1e-4), \
                                           mu_up_f_gene, torch.sqrt(var_up_f_gene + 1e-4), \
                                           mu_center, mu_scale, \
                                           std_center, std_scale,
                                           scale_mu, scale_std,
                                           weights_time)
            else:
                loss_match_mu_var_up_f = 0.0
        else:
            if self.fac_loss_geneCell > 0.0:
                loss_match_mu_var_up_f = \
                    torch.sqrt((torch.log(px_f2_by_gam + 1e-4) - \
                                torch.log(px_f2_by_gam_gene + 1e-4))**2.0 + \
                            (torch.log(px_B2 + 1e-4) - \
                                torch.log(px_B2_gene + 1e-4))**2.0 + 1e-4).sum(-1).mean()
            else:
                loss_match_mu_var_up_f = 0.0
        if self.fac_loss_geneCell == 0.0:
            loss_match_mu_var_up_f = 0.0
        
        # get gene-specific degradation rates
        gamma_mRNA_tmp = F.softplus(gamma_mRNA.repeat(px.shape[0], 1)) + 1e-9
        # gamma_mRNA_tmp = torch.clamp(
        #     gamma_mRNA_tmp, 
        #     min=0.01, max=10
        # )
        gamma_mRNA_tmp_gene = gamma_mRNA_tmp

        # get gene-specific switch time; the time when genes switch from the 
        # upregulated branch to the downregulated branch

        loss_match_gamma = 0.0
        
        # convert burst frequencies from relative scale (wrt gamma) to absolute scale
        if self.fac_loss_geneCell > 0.0:
            px_f1 = px_f1_by_gam * gamma_mRNA_tmp
            px_f2 = px_f2_by_gam *  gamma_mRNA_tmp
        else:
            px_f1 = None
            px_f2 = None

        px_f1_gene = px_f1_by_gam_gene * gamma_mRNA_tmp_gene
        px_f2_gene = px_f2_by_gam_gene *  gamma_mRNA_tmp_gene



        # get mu and var, and convert to burst freq. and burst size for the 
        # upregulated branch achievable at switch time. Use gene-cell-specific 
        # initial and final steady state mu and var
        if self.fac_loss_geneCell > 0.0:
            p_up_ss = torch.exp(-gamma_mRNA_tmp * time_ss_full)
            mu_up_ss = mu_0 * p_up_ss + \
                mu_up_f * (1 - p_up_ss)
            var_up_ss = (var_0 - eps_ratio * mu_0) * \
                p_up_ss**2.0 + (var_up_f - eps_ratio * mu_up_f) * (1 - p_up_ss**2.0) + \
                eps_ratio * mu_up_ss
            velo_mu_up_ss = torch.zeros_like(mu_up_f)
            velo_var_up_ss = torch.zeros_like(mu_up_f)
            px_B_ss = var_up_ss / mu_up_ss - 1 + 1e-4
            px_f_ss = (mu_up_ss / px_B_ss) * gamma_mRNA_tmp
        else:
            px_B_ss = None
            px_f_Ss = None
            mu_up_ss = None
            var_up_ss = None
            velo_mu_up_ss = None
            velo_var_up_ss = None

        # get mu and var, and convert to burst freq. and burst size for the
        # downregulated branch achievable at switch time. Use gene-specific
        # initial and final steady state mu and var
        p_up_ss_gene = torch.exp(-gamma_mRNA_tmp_gene * time_ss_full_gene)
        mu_up_ss_gene = mu_0_gene * p_up_ss_gene + \
            mu_up_f_gene * (1 - p_up_ss_gene)
        var_up_ss_gene = (var_0_gene - eps_ratio * mu_0_gene) * \
            p_up_ss_gene**2.0 + (var_up_f_gene - eps_ratio * mu_up_f_gene) * (1 - p_up_ss_gene**2.0) + \
            eps_ratio * mu_up_ss_gene
        velo_mu_up_ss_gene = torch.zeros_like(mu_up_f_gene)
        velo_var_up_ss_gene = torch.zeros_like(mu_up_f_gene)
        px_B_ss_gene = var_up_ss_gene / mu_up_ss_gene - 1 + 1e-4
        px_f_ss_gene = (mu_up_ss_gene / px_B_ss_gene) * gamma_mRNA_tmp_gene

        # get gene-specific burst frequency and burst size for the starting 
        # steady state for the downregulated branch. Also get mu and var.
        px_B3_up_gene = F.softplus(burst_B1.repeat(px.shape[0], 1)) + 2e-9
        px_f3_up_gene = F.softplus(burst_F1.repeat(px.shape[0], 1)) + px_f1_gene + 1e-9
        # f3_max_gene = ((var_max_tmp) * gamma_mRNA_tmp_gene) / (px_B3_up_gene * (px_B3_up_gene + 1))
        f3_min_gene_new = (1.1 * mu_0_gene * gamma_mRNA_tmp_gene) / px_B3_up_gene
        # px_f3_up_gene = torch.clamp(px_f3_up_gene, max=f3_max_gene, min=f3_min_gene_new)
        px_f3_up_gene = torch.clamp(px_f3_up_gene, max=None, min=f3_min_gene_new)
        
        # px_B3_up_gene = px_B_ss_gene
        # px_f3_up_gene = px_f_ss_gene
        
        ## for single parabola genes, align the corners
        if self.prior_branch_assignment is not None:
            # 1. Create the mask
            # Ensure it has the same dimensionality as your data if needed, 
            # though torch.where usually handles broadcasting well.
            mask_ = self.prior_branch_assignment == 0
            mask_reshaped = mask_.view(1, -1)

            # 2. Use torch.where to create NEW tensors instead of modifying px_B3_up_gene
            px_B3_up_gene = torch.where(mask_reshaped, px_B_ss_gene, px_B3_up_gene)
            px_f3_up_gene = torch.where(mask_reshaped, px_f_ss_gene, px_f3_up_gene)
        
        mu_down_up_gene = px_B3_up_gene * px_f3_up_gene / gamma_mRNA_tmp_gene
        var_down_up_gene = eps_ratio * mu_down_up_gene * (px_B3_up_gene + 1)

        # get gene-specific burst frequency and burst size for the final 
        # steady state for the downregulated branch. Also get mu and var.
        if not self.match_bottom_left:
            px_f3_gene = F.softplus(burst_F3.repeat(px.shape[0], 1)) + 1e-9
            id_f3_check_1_gene = var_0_gene * gamma_mRNA_tmp_gene > px_B3_up_gene**2.0 * px_f3_up_gene
            id_f3_check_0_gene = var_0_gene * gamma_mRNA_tmp_gene <= px_B3_up_gene**2.0 * px_f3_up_gene
            f3_check_fac_1_gene = \
                (var_0_gene * gamma_mRNA_tmp_gene - px_B3_up_gene**2.0 * px_f3_up_gene)**2.0 / \
                (px_B3_up_gene**2.0 * px_f3_up_gene)
            f3_check_fac_0_gene = torch.zeros_like(px_f3_gene, requires_grad=False)
            f3_min_0_gene = f3_check_fac_1_gene * id_f3_check_1_gene + \
                f3_check_fac_0_gene * id_f3_check_0_gene
            f3_min_1_gene  = px_f1_gene 
            f3_min_2_gene  = (mu_0_gene * gamma_mRNA_tmp_gene / px_B3_up_gene )**2.0 / px_f3_up_gene
            f3_min_gene = torch.max(f3_min_1_gene , f3_min_2_gene)
            f3_min_gene = torch.max(f3_min_gene , f3_min_0_gene)
            px_f3_gene = torch.clamp(px_f3_gene, max=px_f3_up_gene * 1.0, min=f3_min_gene)
            px_B3_gene = F.softplus(burst_B3.repeat(px.shape[0], 1)) + 1e-9
            B3_max_tmp_gene = px_B3_up_gene * torch.sqrt(px_f3_up_gene / px_f3_gene + 1e-9)
            B3_min_tmp_gene = torch.max((px_f1_gene * px_B1_gene / px_f3_gene), \
                                        (-1 + \
                                         torch.sqrt(1 + \
                                                    (4 * var_0_gene * \
                                                     gamma_mRNA_tmp_gene) / px_f3_gene)) / 2)
            px_B3_gene = torch.clamp(px_B3_gene, max=B3_max_tmp_gene, \
                                     min=B3_min_tmp_gene)
        else:
            px_B3_gene = px_B1_gene
            px_f3_gene = px_f1_gene
        mu_down_f_gene = px_B3_gene * px_f3_gene / gamma_mRNA_tmp_gene
        var_down_f_gene = eps_ratio * mu_down_f_gene * (px_B3_gene + 1)

        # get gene-cell-specific burst frequency and burst size for the starting
        # steady state for the downregulated branch. Also get mu and var.
        if self.fac_loss_geneCell > 0:
            px_B3_up = F.softplus(self.px_burstB3_decoder(px)) + 2e-9
            px_f3_up = F.softplus(self.px_burstF3_decoder(px)) + px_f1 + 1e-9
            # f3_max = ((var_max_tmp) * gamma_mRNA_tmp) / (px_B3_up * (px_B3_up + 1))
            f3_min = (1.1 * mu_0 * gamma_mRNA_tmp) / px_B3_up
            px_f3_up = torch.clamp(px_f3_up, max=None, min=f3_min)

            # px_B3_up = px_B_ss
            # px_f3_up = px_f_ss
            mu_down_up = px_B3_up * px_f3_up / gamma_mRNA_tmp
            var_down_up = eps_ratio * mu_down_up * (px_B3_up + 1)
        else:
            px_B3_up = None
            px_f3_up = None
            px_f3_up = None
            mu_down_up = None
            var_down_up = None

        # match gene-specific and gene-cell-specific mu and var or 
        # burst freq. and burst size for the starting steady state 
        # for the downregulated branch
        loss_match_mu_var_down_up = 0.0
        if not match_burst_params_not_muVar:
            if self.fac_loss_geneCell > 0:
                loss_match_mu_var_down_up = \
                    get_mu_std_scaled_loss(mu_down_up, torch.sqrt(var_down_up + 1e-4), \
                                           mu_down_up_gene, torch.sqrt(var_down_up_gene + 1e-4), \
                                           mu_center, mu_scale, \
                                           std_center, std_scale,
                                           scale_mu, scale_std,
                                           weights_time)
        else:
            if self.fac_loss_geneCell > 0:
                loss_match_mu_var_down_up = \
                    torch.sqrt((torch.log(px_f3_up + 1e-4) - \
                                torch.log(px_f3_up_gene + 1e-4))**2.0 + \
                            (torch.log(px_B3_up + 1e-4) - \
                                torch.log(px_B3_up_gene + 1e-4))**2.0 + 1e-4).sum(-1).mean()

        # get gene-cell-specific burst frequency and burst size for the final
        # steady state for the downregulated branch. Also get mu and var.
        if self.fac_loss_geneCell > 0:
            if not self.match_bottom_left:
                px_f3 = F.softplus(self.px_burstF4_decoder(px)) + 1e-9
                id_f3_check_1 = var_0 * gamma_mRNA_tmp > px_B3_up**2.0 * px_f3_up
                id_f3_check_0 = var_0 * gamma_mRNA_tmp <= px_B3_up**2.0 * px_f3_up
                f3_check_fac_1 = \
                    (var_0 * gamma_mRNA_tmp - px_B3_up**2.0 * px_f3_up)**2.0 / \
                    (px_B3_up**2.0 * px_f3_up)
                f3_check_fac_0 = torch.zeros_like(px_f3, requires_grad=False)
                f3_min_0 = f3_check_fac_1 * id_f3_check_1 + f3_check_fac_0 * id_f3_check_0
                f3_min_1 = px_f1
                f3_min_2 = (mu_0 * gamma_mRNA_tmp / px_B3_up)**2.0 / px_f3_up
                f3_min_ = torch.max(f3_min_1, f3_min_2)
                f3_min_ = torch.max(f3_min_ , f3_min_0)
                px_f3 = torch.clamp(px_f3, max=px_f3_up * 1.0, min=f3_min_)
                px_B3 = F.softplus(self.px_burstB4_decoder(px)) + 1e-9
                B3_max_tmp = px_B3_up * torch.sqrt(px_f3_up / px_f3 + 1e-9)
                B3_min_tmp = torch.max((px_f1 * px_B1 / px_f3), \
                                            (-1 + \
                                             torch.sqrt(1 + (4 * var_0 * gamma_mRNA_tmp) / px_f3)) / 2)
                px_B3 = torch.clamp(px_B3, max=B3_max_tmp, min=B3_min_tmp)
                mu_down_f = px_B3 * px_f3 / gamma_mRNA_tmp
                var_down_f = eps_ratio * mu_down_f * (px_B3 + 1)
            else:
                px_B3 = px_B1
                px_f3 = px_f1
        else:
            px_f3 = None
            px_B3 = None
            mu_down_f = None
            var_down_f = None

        # match gene-specific and gene-cell-specific mu and var or
        # burst freq. and burst size for the final steady state
        # for the downregulated branch
        loss_match_mu_var_down_f = 0.0
        if not match_burst_params_not_muVar:
            if self.fac_loss_geneCell > 0:
                loss_match_mu_var_down_f = \
                    get_mu_std_scaled_loss(mu_down_f, torch.sqrt(var_down_f + 1e-4), \
                                        mu_down_f_gene, torch.sqrt(var_down_f_gene + 1e-4), \
                                        mu_center, mu_scale, \
                                        std_center, std_scale,
                                        scale_mu, scale_std,
                                        weights_time)
            # loss_match_mu_var_down_f = \
            #     ((mu_down_f - mu_down_f_gene)**2.0 / loss_fac_div + \
            #             (torch.sqrt(var_down_f + 1e-4) - \
            #                 torch.sqrt(var_down_f_gene + 1e-4))**2.0 / loss_fac_div_2 + 1e-4).mean(0).sum()
        else:
            if self.fac_loss_geneCell > 0:
                loss_match_mu_var_down_f = \
                    torch.sqrt((torch.log(px_f3 + 1e-4) - \
                                torch.log(px_f3_gene + 1e-4))**2.0 + \
                            (torch.log(px_B3 + 1e-4) - \
                                torch.log(px_B3_gene + 1e-4))**2.0 + 1e-4).sum(-1).mean()

        # uncomment the following lines to match the mu and var for the final
        # initial state for the downregulated branch to the mu and var achievable at 
        # switch time for the upregulated branch
        # if not match_upf_down_up:
        #     if not match_burst_params_not_muVar:
        #         loss_match_mu_var_down_up_ss = \
        #             get_mu_std_scaled_loss(mu_down_up, torch.sqrt(var_down_up + 1e-4), \
        #                                 mu_up_ss, torch.sqrt(var_up_ss + 1e-4), \
        #                                 mu_center, mu_scale, \
        #                                 std_center, std_scale, scale_mu, scale_std)
        #     else:
        #         loss_match_mu_var_down_up_ss = \
        #         torch.sqrt((torch.log(px_f3_up + 1e-4) - \
        #                     torch.log(px_f_ss + 1e-4))**2.0 + \
        #                 (torch.log(px_B3_up + 1e-4) - \
        #                     torch.log(px_B_ss + 1e-4))**2.0 + 1e-4).sum(-1).mean()

        # else:
        #     loss_match_mu_var_down_up_ss = 0.0
        loss_match_mu_var_down_up_ss = 0.0

        # TODO: Unused variables; remove
        if self.fac_loss_geneCell > 0:
            loss_up0_downf_match = torch.sqrt(((mu_0[0, :].flatten() - \
                            mu_down_f[0, :].flatten()).pow(2)).sum() + 1e-4) + \
                        torch.sqrt(((torch.sqrt(var_0[0, :].flatten() + 1e-4) - \
                         torch.sqrt(var_down_f[0, :].flatten() + 1e-4)).pow(2)).sum() + 1e-4)
            dist_down_up_to_upper = \
                self.dist_from_connecting_line(mu_0[:, :], \
                                               (var_0[:, :] + 1e-4).pow(pow_fac), \
                                               mu_up_ss[:, :], \
                                               (var_up_ss[:, :] + 1e-4).pow(pow_fac), \
                                               mu_down_up[:, :], \
                                               (var_down_up[:, :] + 1e-4).pow(pow_fac))
            loss_down_up_to_upper = torch.sum(F.relu(dist_down_up_to_upper), dim=-1).mean()
        else:
            loss_up0_downf_match = None
            dist_down_up_to_upper = None
            loss_down_up_to_upper = None

        # TODO: Unused variables; remove
        if self.fac_loss_geneCell > 0:
            p_down_min = torch.exp(-gamma_mRNA_tmp * (tmax - time_ss_full))
            mu_down_min = mu_down_up * p_down_min + \
                mu_down_f * (1 - p_down_min)
            var_down_min = (var_down_up - eps_ratio * mu_down_up) * p_down_min**2.0 + \
                (var_down_f - eps_ratio * mu_down_f) * (1 - p_down_min**2.0) + \
                eps_ratio * mu_down_min
            dist_down_f_to_upper = \
                self.dist_from_connecting_line(mu_0[0, :], \
                                               (var_0[0, :] + 1e-4).pow(pow_fac), \
                                               mu_up_ss[0, :], \
                                               (var_up_ss[0, :] + 1e-4).pow(pow_fac), \
                                               mu_down_f[0, :], \
                                               (var_down_f[0, :] + 1e-4).pow(pow_fac))
            loss_down_f_to_upper = torch.sum(F.relu(dist_down_f_to_upper))
            loss_match_top_corner = torch.sqrt(((mu_up_ss[0, :].flatten() + 1e-4 - \
                            (mu_down_up[0, :].flatten())).pow(2)).sum() + 1e-4) + \
                        torch.sqrt(((torch.sqrt(var_up_ss[0, :].flatten() - 1e-8 + 1e-4) - \
                         torch.sqrt(var_down_up[0, :].flatten() + 1e-4)).pow(2)).sum() + 1e-4)
        else:
            p_down_min = None
            mu_down_min = None
            var_down_min = None
            dist_down_f_to_upper = None
            loss_down_f_to_upper = None
            loss_match_top_corner = None
        
        # TODO: Unused variables; remove
        if self.fac_loss_geneCell > 0:
            distance_cond = (mu_up_f - mu_obs) * (mu_up_f + mu_obs - 2 * mu_0) + \
                (var_up_f - var_obs) * (var_up_f + var_obs - 2 * var_0)
            loss_distance_cell = F.relu(-distance_cond).sum(-1) / \
                (F.relu(-distance_cond).sum(-1) + 1e-5)

            dist_to_up_f = (mu_up_f - mu_obs)**2.0 + \
                (var_up_f - var_obs)**2.0
            scale_dist_sigmoid_full3 = F.softplus(scale_dist_sigmoid3) + 1.0
            scale_dist_sigmoid_full3 = scale_dist_sigmoid_full3.repeat(px.shape[0], 1) 
            p_to_up_f = 2 * F.sigmoid(-scale_dist_sigmoid_full3 * dist_to_up_f)
        else:
            distance_cond = None
            loss_distance_cell = None

            dist_to_up_f = None
            scale_dist_sigmoid_full3 = None
            scale_dist_sigmoid_full3 = None
            p_to_up_f = None
        
        # get time-dependent mu and var for the upregulated branch using the
        # gene-cell-specific params
        if self.fac_loss_geneCell > 0:
            p_up = torch.exp(-gamma_mRNA_tmp * px_time_rate_tmp)
            mu_up = mu_0 * p_up + \
                mu_up_f * (1 - p_up)
            var_up = (var_0 - eps_ratio * mu_0) * p_up**2.0 + \
                (var_up_f - eps_ratio * mu_up_f) * (1 - p_up**2.0) + \
                eps_ratio * mu_up
            # velo_mu_up = (mu_up_f - mu_up) * gamma_mRNA_tmp
            # velo_var_up = (eps_ratio * mu_up_f * (2 * px_B2 + 1) + eps_ratio * mu_up - 2  * var_up) * gamma_mRNA_tmp
            
            velo_mu_up = (mu_up_f - mu_obs) * gamma_mRNA_tmp
            velo_var_up = (eps_ratio * mu_up_f * (2 * px_B2 + 1) + eps_ratio * mu_obs - 2  * var_obs) * gamma_mRNA_tmp
        else:
            p_up = None
            mu_up = None
            var_up = None
            velo_mu_up = None
            velo_var_up = None

        # get time-dependent mu and var for the downregulated branch using the
        # gene-specific params
        p_up_gene = torch.exp(-gamma_mRNA_tmp * px_time_rate_tmp_gene)
        mu_up_gene = mu_0_gene * p_up_gene + \
            mu_up_f_gene * (1 - p_up_gene)
        var_up_gene = (var_0_gene - eps_ratio * mu_0_gene) * p_up_gene**2.0 + \
            (var_up_f_gene - eps_ratio * mu_up_f_gene) * (1 - p_up_gene**2.0) + \
            eps_ratio * mu_up_gene
        # velo_mu_up_gene = (mu_up_f_gene - mu_up_gene) * gamma_mRNA_tmp_gene
        # velo_var_up_gene = (eps_ratio * mu_up_f_gene * (2 * px_B2_gene + 1) + eps_ratio * mu_up_gene - \
        #                     2  * var_up_gene) * gamma_mRNA_tmp_gene
        velo_mu_up_gene = (mu_up_f_gene - mu_obs) * gamma_mRNA_tmp_gene
        velo_var_up_gene = (eps_ratio * mu_up_f_gene * (2 * px_B2_gene + 1) + eps_ratio * mu_obs - \
                            2  * var_obs) * gamma_mRNA_tmp_gene
        
        # get time-dependent mu and var for the downregulated branch using the
        # gene-cell-specific params
        if self.fac_loss_geneCell > 0:
            p_down = torch.exp(-gamma_mRNA_tmp * px_time_rate_rep_tmp)
            mu_down = mu_down_up * p_down + \
                mu_down_f * (1 - p_down)
            var_down = (var_down_up) * p_down**2.0 + \
                (var_down_f) * (1 - p_down**2.0) - \
                (eps_ratio * mu_down_f - eps_ratio * mu_down_up) * (p_down - p_down**2.0)
            # velo_mu_down = (mu_down_f - mu_down) * gamma_mRNA_tmp
            # velo_var_down = \
            #     (eps_ratio * mu_down_f * (2 * px_B3 + 1) + eps_ratio * mu_down - 2  * var_down) * gamma_mRNA_tmp
            velo_mu_down = (mu_down_f - mu_obs) * gamma_mRNA_tmp
            velo_var_down = \
                (eps_ratio * mu_down_f * (2 * px_B3 + 1) + eps_ratio * mu_obs - 2  * var_obs) * gamma_mRNA_tmp
        else:
            p_down = None
            mu_down = None
            var_down = None
            velo_mu_down = None
            velo_var_down = None
        
        # get time-dependent mu and var for the downregulated branch using the
        # gene-specific params
        p_down_gene = torch.exp(-gamma_mRNA_tmp_gene * px_time_rate_rep_tmp_gene)
        mu_down_gene = mu_down_up_gene * p_down_gene + \
            mu_down_f_gene * (1 - p_down_gene)
        var_down_gene = (var_down_up_gene) * p_down_gene**2.0 + \
            (var_down_f_gene) * (1 - p_down_gene**2.0) - \
            (eps_ratio * mu_down_f_gene - eps_ratio * mu_down_up_gene) * (p_down_gene - p_down_gene**2.0)
        # velo_mu_down_gene = (mu_down_f_gene - mu_down_gene) * gamma_mRNA_tmp_gene
        # velo_var_down_gene = \
        #     (eps_ratio * mu_down_f_gene * (2 * px_B3_gene + 1) + eps_ratio * mu_down_gene - \
        #      2  * var_down_gene) * gamma_mRNA_tmp_gene
        
        velo_mu_down_gene = (mu_down_f_gene - mu_obs) * gamma_mRNA_tmp_gene
        velo_var_down_gene = \
            (eps_ratio * mu_down_f_gene * (2 * px_B3_gene + 1) + eps_ratio * mu_obs - \
             2  * var_obs) * gamma_mRNA_tmp_gene

        # TODO: Unused variables; remove
        if self.fac_loss_geneCell > 0:
            dist_up = self.dist_from_connecting_line(mu_0, \
                                                     (var_0 + 1e-4).pow(pow_fac), \
                                                     mu_up_ss, \
                                                     (var_up_ss + 1e-4).pow(pow_fac), \
                                                     mu_obs, \
                                                     (var_obs + 1e-4).pow(pow_fac))
            dist_up_lin = self.dist_from_connecting_line(mu_0, (var_0), \
                                                     mu_up_ss, (var_up_ss), \
                                                     mu_obs, (var_obs))
            dist_up_pred = self.dist_from_connecting_line(mu_0, \
                                                          (var_0 + 1e-4).pow(pow_fac), \
                                                          mu_up_ss, \
                                                          (var_up_ss + 1e-4).pow(pow_fac), \
                                                          mu_up, \
                                                          (var_up + 1e-4).pow(pow_fac))
            scale_dist_sigmoid_full = F.softplus(scale_dist_sigmoid) + 1
            scale_dist_sigmoid_full = scale_dist_sigmoid_full.repeat(px.shape[0], 1) 
            # id_up_branch = F.sigmoid(scale_dist_sigmoid_full * dist_up)
            up_pos = (F.relu(scale_dist_sigmoid_full * dist_up))
            up_neg = (F.relu(-scale_dist_sigmoid_full * dist_up))
            dist_up_corner = torch.sqrt((mu_obs - mu_up_f)**2.0 + \
                                        (var_obs - var_up_f)**2.0)
            dist_down_corner = (torch.sqrt((mu_obs - mu_down_f)**2.0 + \
                                        (var_obs - var_down_f)**2.0))
            id_ss_up = \
                F.sigmoid(1 * (dist_down_corner - \
                                                      dist_up_corner))
            id_ss_down = 1 - id_ss_up
        else:
            dist_up = None
            dist_up_lin = None
            dist_up_pred = None
            scale_dist_sigmoid_full = None
            scale_dist_sigmoid_full = None
            # id_up_branch = F.sigmoid(scale_dist_sigmoid_full * dist_up)
            up_pos = None
            up_neg = None
            dist_up_corner = None
            dist_down_corner = None
            id_ss_up = None
            id_ss_down = None

        # TODO: Unused variables; remove
        mu_max = (mu_max_tmp + 1)
        var_max = (var_max_tmp + 1)
        if self.fac_loss_geneCell > 0:
            dist_down = self.dist_from_connecting_line(mu_down_up, \
                                                       var_down_up, \
                                                       mu_down_f, var_down_f, mu_obs, var_obs)
            dist_down_sqrt = self.dist_from_connecting_line(mu_max - mu_down_up, \
                                                       (var_max - var_down_up + 1e-4).pow(pow_fac), \
                                                       mu_max - mu_down_f, \
                                                        (var_max - var_down_f + 1e-4).pow(pow_fac), \
                                                        mu_max - mu_obs, \
                                                        (var_max - var_obs + 1e-4).pow(pow_fac))
            dist_down_pred = self.dist_from_connecting_line(mu_max - mu_down_up, \
                                                       (var_max - var_down_up + 1e-4).pow(pow_fac), \
                                                       mu_max - mu_down_f, \
                                                        (var_max - var_down_f + 1e-4).pow(pow_fac), \
                                                        mu_max - mu_down, \
                                                        (var_max - var_down + 1e-4).pow(pow_fac))
            scale_dist_sigmoid2_full = F.softplus(scale_dist_sigmoid2) + 1
            scale_dist_sigmoid2_full = scale_dist_sigmoid2_full.repeat(px.shape[0], 1) 
            loss_match_dist_down = None
            loss_match_dist_up = None
            down_pos = (F.relu(scale_dist_sigmoid2_full * dist_down_sqrt))
            down_neg = (F.relu(-scale_dist_sigmoid2_full * dist_down_sqrt))
            loss_down_branch = None
            loss_up_branch = None
        else:
            dist_down = None
            dist_down_sqrt = None
            dist_down_pred = None
            scale_dist_sigmoid2_full = None
            scale_dist_sigmoid2_full = None
            loss_match_dist_down = None
            loss_match_dist_up = None
            down_pos = None
            down_neg = None
            loss_down_branch = None
            loss_up_branch = None
        
        # TODO: Unused variables; remove
        d2sigma_dmu2_up = None
        d2sigma_dmu2_down = None
        d2sigmaM_dmu2_up = None
        d2sigmaM_dmu2_down = None


        library_ = capture_eff.repeat(1, px_f1_gene.shape[1])

        return px_time_rate_tmp, \
            gamma_mRNA_tmp, library_, px_time_rate_rep_tmp, \
            px_B1, px_f1, mu_down_up, var_down_up, state_pred, \
            mu_up, var_up, \
            mu_up_f, var_up_f, mu_down, var_down, \
            mu_down_f, var_down_f, \
            velo_mu_up, velo_var_up, \
            velo_mu_up_ss, velo_var_up_ss, velo_mu_down, velo_var_down, \
            px_pi, mu_up_ss, var_up_ss, \
            loss_distance_cell, id_ss_up, id_ss_down, \
            p_to_up_f, \
            None, loss_down_up_to_upper, \
            loss_down_f_to_upper, loss_up_branch, loss_down_branch, time_ss_full, \
            loss_match_top_corner, loss_up0_downf_match, \
            loss_match_dist_up, loss_match_dist_down, \
            loss_match_mu_var_up_f, loss_match_mu_var_down_up, loss_match_mu_var_down_f, \
            loss_match_mu_var_0, \
            d2sigma_dmu2_up, d2sigma_dmu2_down, d2sigmaM_dmu2_up, d2sigmaM_dmu2_down, \
            mu_up_gene, var_up_gene, mu_down_gene, var_down_gene, \
            loss_match_gamma, loss_match_mu_var_down_up_ss, \
            px_time_rate_tmp_gene, \
            gamma_mRNA_tmp_gene, px_time_rate_rep_tmp_gene, \
            px_B1_gene, px_f1_gene, mu_down_up_gene, var_down_up_gene, \
            mu_up_f_gene, var_up_f_gene, \
            mu_down_f_gene, var_down_f_gene, \
            velo_mu_up_gene, velo_var_up_gene, time_ss_full_gene, \
            velo_mu_down_gene, velo_var_down_gene, \
            mu_up_ss_gene, var_up_ss_gene, weights_time, loss_time, b_t, f_t
    
class VAENoiseVelo(BaseMinifiedModeModuleClass):
    """Variational auto-encoder model.
    TODO: Add details.
    """

    def __init__(
        self,
        n_input: int,
        gamma_mRNA: torch.Tensor = None,
        use_gamma_mRNA: bool = False,
        use_two_rep: bool = False,
        use_splicing: bool = False,
        use_time_cell: bool = False,
        use_prior_parabola: bool = False,
        w_theta: float = 1.0,
        burst_B_gene: bool = False,
        burst_f_gene: bool = False,
        burst_f_updown: torch.Tensor = None,
        burst_B_updown: torch.Tensor = None, 
        burst_f_updown_next: torch.Tensor = None,
        burst_B_updown_next: torch.Tensor = None,
        burst_f_previous: torch.Tensor = None,
        burst_B_previous: torch.Tensor = None,
        burst_f_next: torch.Tensor = None,
        burst_B_next: torch.Tensor = None,
        mu_neighbors: torch.Tensor = None,
        var_neighbors: torch.Tensor = None,
        mu_ss_obs: torch.Tensor = None,
        var_ss_obs: torch.Tensor = None,
        mu_std: torch.Tensor = None,
        var_std: torch.Tensor = None,
        mu_mean_obs: torch.Tensor = None,
        var_mean_obs: torch.Tensor = None,
        mu_center: torch.Tensor = None,
        std_center: torch.Tensor = None,
        std_ref_center: torch.Tensor = None,
        mu_scale: torch.Tensor = None,
        std_scale: torch.Tensor = None,
        std_ref_scale: torch.Tensor = None,
        match_burst_params: bool = True,
        match_burst_params_not_muVar: bool = False,
        mu_low_obs: torch.Tensor = None,
        var_low_obs: torch.Tensor = None,
        mu_max: torch.Tensor = None,
        var_max: torch.Tensor = None,
        std_sum: torch.Tensor = None,
        y_state0_super: torch.Tensor = None,
        y_state1_super: torch.Tensor = None,
        mat_where_prev: torch.Tensor = None,
        mat_where_next: torch.Tensor = None,
        state_times_unique: torch.Tensor = None,
        use_library_time_correction: bool = False,
        use_tr_gene:bool = False,
        use_alpha_gene: bool = False,
        use_noise_ext: bool = False,
        extra_loss_fac: Tunable[float] = 0.5,
        extra_loss_fac_1: Tunable[float] = 0.5,
        extra_loss_fac_2: float = 0.1,
        extra_loss_fac_2_: float = 0.1,
        extra_loss_fac_0: float = 0.1,
        extra_loss_fac_3: float = 0.1,
        edge_loss_fac: float = 1.0,
        prior_branch_assignment: torch.Tensor = None,
        fac_var: float = 1.0,
        loss_fac_geneCell: float = 1.0,
        loss_fac_gene: float = 1.0,
        loss_fac_prior_clust: float = 0.1,
        capture_eff: torch.Tensor = None,
        use_time_dependence: bool = False,
        time_loss: float = 0.0,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: Tunable[int] = 128,
        n_latent: Tunable[int] = 10,
        n_layers: Tunable[int] = 1,
        n_states: int = 4,
        match_upf_down_up: bool = False,
        match_bottom_left: bool = False,
        use_loss_burst: bool = False,
        use_controlBurst_gene: bool = False,
        cluster_states: bool = False,
        state_loss_type: str = "cross-entropy",
        states_vec: torch.Tensor = None,
        state_time_max_vec: torch.Tensor = None,
        timing_relative: bool = False,
        timing_relative_mat_bin: torch.Tensor = None,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate: Tunable[float] = 0.1,
        dispersion: Tunable[
            Literal["gene", "gene-batch", "gene-label", "gene-cell"]
        ] = "gene",
        log_variational: bool = True,
        gene_likelihood: Tunable[Literal["zinb", "nb", "poisson"]] = "zinb",
        latent_distribution: Tunable[Literal["normal", "ln"]] = "normal",
        encode_covariates: Tunable[bool] = False,
        deeply_inject_covariates: Tunable[bool] = True,
        use_batch_norm: Tunable[Literal["encoder", "decoder", "none", "both"]] = "both",
        use_layer_norm: Tunable[Literal["encoder", "decoder", "none", "both"]] = "none",
        use_size_factor_key: bool = False,
        use_observed_lib_size: bool = True,
        library_log_means: Optional[np.ndarray] = None,
        library_log_vars: Optional[np.ndarray] = None,
        var_activation: Optional[Callable] = None,
        tmax: int = 12, 
        t_d: float = 13, 
        t_r: float = 6.5,
        sample_prob: float = 0.4,
        use_student_t: bool = True,
        student_t_df: float = 1000.0,
        student_t_df_final: float = 4.0,
        min_scale: float = 1e-2,
        w_sep: float = 0.01,
        sep_margin: float = 0.0,
        sep_n_grid: int = 32,
        sep_bandwidth: float = 0.05,
    ):
        super().__init__()
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self.tmax = tmax
        self.t_d = t_d
        self.t_r = t_r
        self.use_gamma_mRNA = use_gamma_mRNA
        self.use_tr_gene = use_tr_gene
        self.use_two_rep = use_two_rep
        self.use_splicing = use_splicing
        self.use_time_cell = use_time_cell
        self.use_prior_parabola = use_prior_parabola
        self.use_time_dependence = use_time_dependence
        print(f'using time depence of b, f = {use_time_dependence}')
        self.time_loss = time_loss
        self.w_theta = w_theta
        # ---- v5 additions -------------------------------------------------
        # Robust (heavy-tailed) observation model for the magnitude features
        # (mu, sqrt_r, r). A finite degrees-of-freedom StudentT down-weights the
        # high-variance points at the top of the mean-variance cloud so they no
        # longer dominate the (quadratic) Gaussian NLL. Set to a large value or
        # switch `self.use_student_t = False` to recover the Normal likelihood.
        self.use_student_t = use_student_t
        self.student_t_df = student_t_df          # kept for backward compatibility
        self.student_t_df_warmup = student_t_df    # df during warmup (e.g. 1000.0)
        self.student_t_df_final = student_t_df_final  # df after warmup (e.g. 4.0)
        self.min_scale = min_scale          # floor on scale tensors to prevent divide-by-zero / overflow
        # Branch-separation (repulsion) loss: keeps the up/down arcs from
        # collapsing onto each other. See `_branch_repulsion_loss`. Disabled by
        # default (weight 0) so it is strictly opt-in.
        self.w_sep = w_sep                  # loss weight; try ~1e-2..1e-1
        self.sep_margin = sep_margin        # minimum theta gap (radians) at matched sqrt_r
        self.sep_n_grid = sep_n_grid        # radius grid resolution
        self.sep_bandwidth = sep_bandwidth  # kernel bandwidth (in normalized-radius units)
        # -------------------------------------------------------------------
        self.burst_B_gene = burst_B_gene
        self.burst_f_gene = burst_f_gene
        self.capture_eff = capture_eff
        self.use_alpha_gene = use_alpha_gene
        self.sample_prob = sample_prob
        self.use_library_time_correction = use_library_time_correction
        self.n_states = n_states
        self.states_vec = states_vec
        self.state_time_max_vec = state_time_max_vec
        self.n_genes = n_input
        self.cluster_states = cluster_states
        self.timing_relative_mat_bin = timing_relative_mat_bin
        self.timing_relative = timing_relative
        self.use_loss_burst = use_loss_burst
        self.use_controlBurst_gene = use_controlBurst_gene
        self.state_loss_type = state_loss_type
        self.burst_f_updown = burst_f_updown
        self.burst_B_updown = burst_B_updown
        self.state_times_unique = state_times_unique
        self.burst_f_previous = burst_f_previous
        self.burst_B_previous = burst_B_previous
        self.burst_f_next = burst_f_next
        self.burst_B_next = burst_B_next
        self.burst_f_updown_next = burst_f_updown_next
        self.burst_B_updown_next = burst_B_updown_next
        self.n_input = n_input
        self.mat_where_prev = mat_where_prev
        self.mat_where_next = mat_where_next
        self.mu_neighbors = mu_neighbors
        self.var_neighbors = var_neighbors
        self.mu_ss_obs = mu_ss_obs
        self.var_ss_obs = var_ss_obs
        self.mu_std = mu_std
        self.var_std = var_std
        self.mu_mean_obs = mu_mean_obs
        self.var_mean_obs = var_mean_obs
        self.mu_low_obs = mu_low_obs
        self.var_low_obs = var_low_obs
        self.mu_max = mu_max
        self.var_max = var_max
        # self.burst_B_low = torch.nn.Parameter(-1 * torch.randn(n_input))
        # self.burst_F_low = torch.nn.Parameter(-1 * torch.randn(n_input))
        self.y_state0_super = y_state0_super
        self.y_state1_super = y_state1_super
        self.std_sum = std_sum
        self.max_fac = 1
        self.match_burst_params = match_burst_params
        self.extra_loss_fac = extra_loss_fac
        self.extra_loss_fac_1 = extra_loss_fac_1
        self.edge_loss_fac = edge_loss_fac
        self.match_upf_down_up = match_upf_down_up
        self.prior_branch_assignment = prior_branch_assignment
        self.mu_center = mu_center
        self.std_center = std_center
        self.std_ref_center = std_ref_center
        self.mu_scale = mu_scale
        self.std_scale = std_scale
        self.std_ref_scale = std_ref_scale
        self.match_burst_params_not_muVar = match_burst_params_not_muVar
        self.loss_fac_geneCell = loss_fac_geneCell
        self.loss_fac_gene = loss_fac_gene
        self.extra_loss_fac_2 = extra_loss_fac_2
        self.extra_loss_fac_2_ = extra_loss_fac_2_
        self.extra_loss_fac_0 = extra_loss_fac_0
        self.extra_loss_fac_3 = extra_loss_fac_3
        self.fac_var = fac_var
        self.loss_fac_prior_clust = loss_fac_prior_clust
        self.dirichlet_concentration = 1 / self.n_states
        self.use_prior_only = False
        self.match_bottom_left = match_bottom_left

        self.burst_B_low = torch.nn.Parameter(var_low_obs / mu_low_obs - 1)
        self.burst_F_low = torch.nn.Parameter(mu_low_obs / (var_low_obs / mu_low_obs - 1 + 1e-8))
        self.burst_B3 = torch.nn.Parameter(var_low_obs / mu_low_obs - 1)
        self.burst_f3 = torch.nn.Parameter(mu_low_obs / (var_low_obs / mu_low_obs - 1 + 1e-8))

        # scale param for the variance of the normal distributions used for the 
        # likelihood of mu and var
        
        # 1. Create a normal tensor (no grad tracked yet)
        init_tensor = 0.54 * torch.ones(n_input, 6)

        # 2. Safely apply your specific initial values
        init_tensor[..., 0] = 7.14
        init_tensor[..., 1] = 2.7
        init_tensor[..., 2] = 0.36
        init_tensor[..., 3] = 0.13

        # 3. Wrap the fully initialized tensor into the Parameter
        self.scale_unconstr = torch.nn.Parameter(init_tensor)
        
        # self.scale_unconstr = torch.nn.Parameter(0.54 * torch.ones(n_input, 6))
        # self.scale_unconstr[..., 0] = 7.14
        # self.scale_unconstr[..., 2] = 1.0
        # self.scale_unconstr[..., 3] = 0.13
        self.scaling_factor = torch.nn.Parameter(torch.tensor(5.0))

        # gene-specific switch time
        self.time_ss = \
            torch.nn.Parameter(-1 * torch.randn(n_input))
        
        # TODO: Unused variable; remove
        self.time_Bswitch = \
            torch.nn.Parameter(-1 * torch.randn(n_input))
        
        # TODO: add description for these variables
        self.burst_B_states = \
            torch.nn.Parameter(-1 * torch.randn((n_states, n_input)))
        self.burst_f_states = \
            torch.nn.Parameter(-1 * torch.randn((n_states, n_input)))
        if use_controlBurst_gene:
            # self.burst_B1 = torch.nn.Parameter(-1 * torch.randn(n_input))
            # self.burst_f1 = torch.nn.Parameter(-1 * torch.randn(n_input))
            self.burst_B1 = torch.nn.Parameter(var_mean_obs / mu_mean_obs - 1)
            self.burst_f1 = torch.nn.Parameter(mu_mean_obs / (var_mean_obs / mu_mean_obs - 1 + 1e-8))
        else:
            self.burst_B1 = None
            self.burst_f1 = None
        # self.burst_B3 = torch.nn.Parameter(-1 * torch.randn(n_input))
        # self.burst_f3 = torch.nn.Parameter(-1 * torch.randn(n_input))
        # self.burst_B2 = torch.nn.Parameter(-1 * torch.randn(n_input))
        # self.burst_f2 = torch.nn.Parameter(-1 * torch.randn(int(n_input)))
        
        self.burst_B2 = torch.nn.Parameter(var_ss_obs / mu_ss_obs - 1)
        self.burst_f2 = torch.nn.Parameter(mu_ss_obs / (var_ss_obs / mu_ss_obs - 1 + 1e-8))
        # b_tmp = var_ss_obs / mu_ss_obs - 1

        # TODO: Unused variables; remove
        self.scale_dist_sigmoid = torch.nn.Parameter(-1 * torch.randn(n_input))
        self.scale_dist_sigmoid2 = torch.nn.Parameter(-1 * torch.randn(n_input))
        self.scale_dist_sigmoid3 = torch.nn.Parameter(-1 * torch.randn(n_input))
        self.scale_dist_sigmoid4 = torch.nn.Parameter(-1 * torch.randn(n_input))
        self.scale_dist_sigmoid5 = torch.nn.Parameter(-1 * torch.randn(n_input))

        self.use_size_factor_key = use_size_factor_key
        self.use_observed_lib_size = use_size_factor_key or use_observed_lib_size
        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_vars is None:
                raise ValueError(
                    "If not using observed_lib_size, "
                    "must provide library_log_means and library_log_vars."
                )

            self.register_buffer(
                "library_log_means", torch.from_numpy(library_log_means).float()
            )
            self.register_buffer(
                "library_log_vars", torch.from_numpy(library_log_vars).float()
            )
        
        # TODO: Unused variable; remove
        if self.use_alpha_gene:
            if self.use_time_cell or self.use_splicing:
                self.alpha_gene = torch.nn.Parameter(-1 * torch.randn(int(n_input / 2)))
            else:
                self.alpha_gene = torch.nn.Parameter(-1 * torch.randn(int(n_input)))
        else:
            self.alpha_gene = None
        if self.use_tr_gene:
            if self.use_splicing:
                self.tr_gene = torch.nn.Parameter(-1 * torch.randn(int(n_input / 2)))
            else:
                self.tr_gene = torch.nn.Parameter(-1 * torch.randn(int(n_input)))
        else:
            self.tr_gene = None
        if self.use_gamma_mRNA:
            self.gamma_mRNA = gamma_mRNA
        else:
            if self.use_splicing:
                self.gamma_mRNA = torch.nn.Parameter(-1 * torch.randn(int(n_input / 2)))
            else:
                self.gamma_mRNA = torch.nn.Parameter(-1 * torch.randn(int(n_input)))

        # from SCVI
        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError(
                "dispersion must be one of ['gene', 'gene-batch',"
                " 'gene-label', 'gene-cell'], but input was "
                "{}.format(self.dispersion)"
            )
        
        # TODO: Unused variable; remove
        self.gene_max_time = \
            torch.nn.Parameter(-1 * torch.randn(int(n_input)))

        # from SCVI
        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # Define encoder
        n_input_encoder = n_input
        cat_list = [n_batch] + list([])
        encoder_cat_list = None
        self.z_encoder = EncoderNew(
            n_input_encoder * 4,
            n_latent,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            n_states=n_states,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
        )

        # l encoder goes from n_input-dimensional data to 1-d library size
        self.l_encoder = Encoder(
            n_input_encoder,
            1,
            n_layers=1,
            n_cat_list=encoder_cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
        )

        # Define decoder
        n_input_decoder = n_latent      
        self.decoder = DecoderNoiseVelo(
            n_input_decoder,
            n_input,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            n_states=n_states,
            prior_branch_assignment=prior_branch_assignment,
            use_prior_parabola=use_prior_parabola,
            cluster_states=cluster_states,
            state_loss_type=state_loss_type,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            scale_activation_init="sigmoid",
            use_two_rep=use_two_rep,
            use_splicing=use_splicing,
            use_time_cell=use_time_cell,
            use_alpha_gene=use_alpha_gene,
            use_controlBurst_gene=use_controlBurst_gene,
            burst_B_gene=burst_B_gene,
            burst_f_gene=burst_f_gene,
            timing_relative=timing_relative,
            use_library_time_correction=use_library_time_correction,
            use_noise_ext=use_noise_ext,
            fac_loss_geneCell=self.loss_fac_geneCell,
            use_time_dependence=use_time_dependence,
            match_bottom_left=match_bottom_left,
        )
        
    def _to_polar_features(self, x, y):
        eps = 1e-8
        # 1. Normalize the aspect ratio to a 1x1 space to open up the angle theta
        x_norm = (x) / self.mu_scale
        y_norm = (y) / self.var_scale
        
        # 2. Compute radius in the normalized space
        r_norm = torch.sqrt(torch.clamp(x_norm**2 + y_norm**2, min=eps))
        
        # 3. Variance-stabilize the normalized radius
        sqrt_r = torch.sqrt(r_norm + eps)
        
        # 4. Compute theta on normalized vectors (now cleanly spreads between 0 and pi/2)
        theta = torch.atan2(y_norm, x_norm)
        
        # We return the original raw x for longitudinal tracking,
        # but the normalized space's sqrt_r and theta for geometry tracking.
        return sqrt_r, theta, r_norm

    def _magnitude_dist_for_df(self, loc, scale, df_val):
        """Observation distribution for the magnitude features (mu, sqrt_r, r) with specific df.
        """
        min_scale = getattr(self, "min_scale", 1e-2)
        scale = torch.clamp(scale, min=min_scale)
        if getattr(self, "use_student_t", False):
            df = torch.as_tensor(df_val, dtype=loc.dtype, device=loc.device)
            return StudentT(df, loc, scale)
        return Normal(loc, scale)

    def _magnitude_dist(self, loc, scale):
        """Observation distribution for the magnitude features (mu, sqrt_r, r).

        Uses a heavy-tailed StudentT when ``self.use_student_t`` is set so that
        high-variance outliers (the points at the top of the mean-variance
        cloud) contribute ~log|residual| instead of residual**2 and therefore
        stop dragging the fitted branches upward. Falls back to Normal
        otherwise. StudentT.scale matches Normal.scale in the small-residual
        limit, so existing `scale_*` parameters keep their meaning.
        """
        df_val = getattr(self, "student_t_df_final", getattr(self, "student_t_df", 4.0))
        return self._magnitude_dist_for_df(loc, scale, df_val)

    def _branch_polar_grid(self, mu_s, var_s, mu_e, var_e, eps_ratio, p):
        """Sample a branch analytically on a mixing-fraction grid and return its
        (sqrt_r, theta) polar coordinates.

        The branch runs from the start steady state (mu_s, var_s) at p=1 to the
        end steady state (mu_e, var_e) at p=0. mean is linear in p and variance
        is the exact quadratic used by both the up and down branches in
        `forward`, so this reproduces the fitted curve rather than approximating
        it.  mu_s/var_s/... are per-gene, shape (G,); p is (K,).  Returns two
        (G, K) tensors.
        """
        mu_s = mu_s.reshape(-1, 1)
        var_s = var_s.reshape(-1, 1)
        mu_e = mu_e.reshape(-1, 1)
        var_e = var_e.reshape(-1, 1)
        p = p.reshape(1, -1)
        mu = mu_s * p + mu_e * (1.0 - p)
        var = (var_s - eps_ratio * mu_s) * p ** 2 \
            + (var_e - eps_ratio * mu_e) * (1.0 - p ** 2) \
            + eps_ratio * mu
        var = torch.clamp(var, min=1e-8)
        x = mu / self.mu_scale.reshape(-1, 1)
        y = var / self.var_scale.reshape(-1, 1)
        r_norm = torch.sqrt(torch.clamp(x ** 2 + y ** 2, min=1e-8))
        sqrt_r = torch.sqrt(r_norm + 1e-8)
        theta = torch.atan2(y, x)
        return sqrt_r, theta

    def _branch_repulsion_loss(
        self,
        px_pi,
        mu_up_s, var_up_s, mu_up_e, var_up_e,
        mu_dn_s, var_dn_s, mu_dn_e, var_dn_e,
        eps_ratio,
    ):
        """Separation loss between the up and down branches, measured as a theta
        gap at *matched* sqrt_r.

        Geometry: both branches are traced out as curves in the (mu, var) plane
        and mapped to polar (sqrt_r, theta). Two arcs that overlap have the same
        theta at a given radius; two distinct arcs (a real hysteresis loop)
        differ in theta at matched radius. We therefore compare theta at a
        shared set of radii and penalise insufficient separation.

        Guards against manufacturing hysteresis (three independent brakes):
          1. Occupancy gate: the per-gene loss is scaled by
             sqrt(mass_up * mass_down) from the (detached) branch posterior. If
             a gene is a single arc, one branch is essentially empty, the gate
             -> 0, and no separation is imposed.
          2. Data-driven sign: the required ordering follows the sign the fit
             already shows (detached), so we never impose a loop orientation the
             data does not support - we only stop occupied branches from
             collapsing/crossing.
          3. Zero default margin: with sep_margin = 0 the hinge fires only on
             actual crossings, never inflating a gap between coincident arcs.
        """
        eps = 1e-8
        ref = self.mu_scale.reshape(-1)
        device, dtype = ref.device, ref.dtype

        # per-gene endpoints (gene-level tensors are (N, G) with identical rows)
        def _g(t):
            return t.mean(0) if t.dim() > 1 else t
        mu_up_s, var_up_s = _g(mu_up_s), _g(var_up_s)
        mu_up_e, var_up_e = _g(mu_up_e), _g(var_up_e)
        mu_dn_s, var_dn_s = _g(mu_dn_s), _g(var_dn_s)
        mu_dn_e, var_dn_e = _g(mu_dn_e), _g(var_dn_e)

        K = int(self.sep_n_grid)
        M = int(self.sep_n_grid)
        p = torch.linspace(1.0, 0.0, K, device=device, dtype=dtype)
        
        # eps_ratio = 1.0
        sr_up, th_up = self._branch_polar_grid(
            mu_up_s, var_up_s, mu_up_e, var_up_e, eps_ratio, p)   # (G, K)
        sr_dn, th_dn = self._branch_polar_grid(
            mu_dn_s, var_dn_s, mu_dn_e, var_dn_e, eps_ratio, p)   # (G, K)

        # shared radius grid over the overlapping radius range of the two arcs
        lo = torch.maximum(sr_up.min(-1).values, sr_dn.min(-1).values)   # (G,)
        hi = torch.minimum(sr_up.max(-1).values, sr_dn.max(-1).values)   # (G,)
        overlap = (hi > lo + eps).to(dtype)                             # (G,)
        w = torch.linspace(0.0, 1.0, M, device=device, dtype=dtype).view(1, -1)
        rho = lo.unsqueeze(-1) + w * (hi - lo).clamp(min=0.0).unsqueeze(-1)  # (G, M)

        # differentiable "theta at matched sqrt_r" via a Gaussian kernel along
        # the radius axis (Nadaraya-Watson smoothing of each arc).
        h = float(self.sep_bandwidth)
        def _theta_at(sr, th):
            d = rho.unsqueeze(-1) - sr.unsqueeze(1)          # (G, M, K)
            wk = torch.softmax(-(d ** 2) / (2.0 * h ** 2 + eps), dim=-1)
            return (wk * th.unsqueeze(1)).sum(-1)            # (G, M)
        th_up_at = _theta_at(sr_up, th_up)
        th_dn_at = _theta_at(sr_dn, th_dn)

        diff = th_up_at - th_dn_at                           # (G, M)
        # sign follows the current fit (brake #2): never impose a loop direction
        sign = torch.sign(diff.mean(-1, keepdim=True)).detach()
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        hinge = F.relu(self.sep_margin - sign * diff).mean(-1)   # (G,)

        # occupancy gate (brake #1), detached so the model can't dodge the loss
        # by artificially emptying a branch.
        pn = px_pi / (px_pi.sum(-1, keepdim=True) + eps)
        mass_up = pn[..., 0:2].sum(-1).mean(0)               # (G,)
        mass_dn = pn[..., 2:4].sum(-1).mean(0)               # (G,)
        gate = torch.sqrt(mass_up * mass_dn + eps).detach()  # (G,)

        return (gate * overlap * hinge).mean()

    def _get_inference_input(
        self,
        tensors,
    ):
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        if self.minified_data_type is None:
            x = tensors[REGISTRY_KEYS.X_KEY]
            mu = tensors[REGISTRY_KEYS.M_KEY]
            std_ = tensors[REGISTRY_KEYS.V_KEY]
            input_dict = {
                "x": x,
                "mu" : mu,
                "std_" : std_,
                "batch_index": batch_index,
                "cont_covs": cont_covs,
                "cat_covs": cat_covs,
            }
        else:
            if self.minified_data_type == ADATA_MINIFY_TYPE.LATENT_POSTERIOR:
                qzm = tensors[REGISTRY_KEYS.LATENT_QZM_KEY]
                qzv = tensors[REGISTRY_KEYS.LATENT_QZV_KEY]
                observed_lib_size = tensors[REGISTRY_KEYS.OBSERVED_LIB_SIZE]
                input_dict = {
                    "qzm": qzm,
                    "qzv": qzv,
                    "observed_lib_size": observed_lib_size,
                }
            else:
                raise NotImplementedError(
                    f"Unknown minified-data type: {self.minified_data_type}"
                )

        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        z = inference_outputs["z"]
        library = inference_outputs["library"]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        y = tensors[REGISTRY_KEYS.LABELS_KEY]
        mu_ = tensors[REGISTRY_KEYS.M_KEY]
        var_ = tensors[REGISTRY_KEYS.V_KEY]**2.0
        prior_pi_up = tensors['prior_pi_up']

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        size_factor_key = REGISTRY_KEYS.SIZE_FACTOR_KEY
        size_factor = (
            torch.log(tensors[size_factor_key])
            if size_factor_key in tensors.keys()
            else None
        )


        input_dict = {
            "z": z,
            "library": library,
            "mu_obs" : mu_,
            "var_obs": var_,
            "prior_pi_up": prior_pi_up,
            "batch_index": batch_index,
            "y": y,
            "cont_covs": cont_covs,
            "cat_covs": cat_covs,
            "size_factor": size_factor
        }
        return input_dict

    def _compute_local_library_params(self, batch_index):
        """Computes local library parameters.
        Compute two tensors of shape (batch_index.shape[0], 1) where each
        element corresponds to the mean and variances, respectively, of the
        log library sizes in the batch the cell corresponds to.
        """
        n_batch = self.library_log_means.shape[1]
        local_library_log_means = F.linear(
            one_hot(batch_index, n_batch), self.library_log_means
        )
        local_library_log_vars = F.linear(
            one_hot(batch_index, n_batch), self.library_log_vars
        )
        return local_library_log_means, local_library_log_vars

    @auto_move_data
    def _regular_inference(
        self, x, mu, std_, \
            batch_index, cont_covs=None, cat_covs=None, n_samples=1
    ):
        """High level inference method.
        Runs the inference (encoder) model.
        """
        x_ = x
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)
            
            
        self.mu_scale = torch.quantile(
            mu, 
            0.95, 
            dim=0
        )
        
        self.std_scale = torch.quantile(
            std_, 
            0.95, 
            dim=0
        )
        
        self.var_scale = torch.quantile(
            std_**2.0, 
            0.95, 
            dim=0
        )

        # input variables for the encoder
        # var_max = (self.var_max + 1) * 1
        # std_reflected = torch.sqrt(F.softplus(var_max - std_**2.0) + 1e-4)
        mu_scaled = (mu - self.mu_center)
        # std_scaled = (std_ - self.std_center) / self.std_scale
        # std_ref_scaled = (std_reflected - self.std_ref_center) / self.std_ref_scale
        # x_ = (torch.cat(((mu_scaled), \
        #                 (std_scaled), \
        #                 (std_ref_scaled)), dim=-1))
        
        
        sqrt_r, theta, r = self._to_polar_features(mu, std_**2.0)
        x_ = (torch.cat(((mu_scaled), \
                        (sqrt_r), \
                        (theta), r), dim=-1))
        if self.log_variational:
            x_ = torch.log(1 + x_)

        encoder_input = x_
        categorical_input = ()

        qz, z \
            = self.z_encoder(encoder_input, \
                            batch_index, *categorical_input)
        ql = None
        if not self.use_observed_lib_size:
            ql, library_encoded = self.l_encoder(
                encoder_input, batch_index, *categorical_input
            )
            library = library_encoded

        if n_samples > 1:
            untran_z = qz.sample((n_samples,))
            z = self.z_encoder.z_transformation(untran_z)
            if self.use_observed_lib_size:
                library = library.unsqueeze(0).expand(
                    (n_samples, library.size(0), library.size(1))
                )
            else:
                library = ql.sample((n_samples,))
        outputs = {"z": z, "qz": qz, "ql": ql, "library": library}
        return outputs

    @auto_move_data
    def _cached_inference(self, qzm, qzv, observed_lib_size, n_samples=1):
        if self.minified_data_type == ADATA_MINIFY_TYPE.LATENT_POSTERIOR:
            dist = Normal(qzm, qzv.sqrt())
            # use dist.sample() rather than rsample because we aren't optimizing the z here
            untran_z = dist.sample() if n_samples == 1 else dist.sample((n_samples,))
            z = self.z_encoder.z_transformation(untran_z)
            library = torch.log(observed_lib_size)
            if n_samples > 1:
                library = library.unsqueeze(0).expand(
                    (n_samples, library.size(0), library.size(1))
                )
        else:
            raise NotImplementedError(
                f"Unknown minified-data type: {self.minified_data_type}"
            )
        outputs = {"z": z, "qz_m": qzm, "qz_v": qzv, "ql": None, "library": library}
        return outputs
    
    @auto_move_data
    def entropy_loss(self, prob_):
        """
        Computes entropy loss to encourage probabilities to be close to 0 or 1.
        prob_: (batch_size, num_genes, 2) - Probability matrix
        """
        # Compute entropy per probability entry
        entropy = -prob_ * torch.log(prob_ + 1e-8)  # Avoid log(0)
        entropy = torch.sum(entropy, dim=-1)  # Sum over the two probabilities per gene
        return torch.mean(torch.sum(entropy, dim=-1))  # Average across all cells and genes
    
    @auto_move_data
    def get_mu_std_scaled_loss(self, mu_0, var_0, mu_1, var_1, 
                          scale_mu, scale_r, scale_sqrt_r, scale_theta,
                          weights=None):
        if weights is None:
            weights = torch.ones_like(mu_0)

        sqrt_r0, theta0, r0 = self._to_polar_features(mu_0, var_0)
        sqrt_r1, theta1, r1 = self._to_polar_features(mu_1, var_1)
        # mu_0_scaled = (mu_0 - mu_center) / mu_scale
        # mu_1_scaled = (mu_1 - mu_center) / mu_scale
        # std_0_scaled = (std_0 - std_center) / std_scale
        # std_1_scaled = (std_1 - std_center) / std_scale
        
        def loss_type(x_0, x_1, scale_x, weights):
            loss_x = (((x_0 - x_1)**2.0) / (2 * scale_x**2.0))
            loss_x = (loss_x * weights).sum(-1)
            return loss_x

        loss_mu = loss_type(mu_0, mu_1, scale_mu, weights)
        loss_r = loss_type(r0, r1, scale_r, weights)
        loss_sqrt_r = loss_type(sqrt_r0, sqrt_r1, scale_sqrt_r, weights)
        loss_theta = loss_type(theta0, theta1, scale_theta, weights)
        loss_total = (loss_mu + loss_sqrt_r + loss_theta + loss_r).mean()
        return loss_total

    @auto_move_data
    def generative(
        self,
        z,
        library,
        mu_obs, 
        var_obs,
        prior_pi_up,
        batch_index,
        cont_covs=None,
        cat_covs=None,
        size_factor=None,
        y=None,
        transform_batch=None,
        device_="cuda",
    ):
        """Runs the generative model."""
        decoder_input = z

        categorical_input = ()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        if not self.use_size_factor_key:
            size_factor = library
            
            
        self.mu_scale = torch.quantile(
            mu_obs, 
            0.95, 
            dim=0
        )
        
        self.std_scale = torch.quantile(
            torch.sqrt(var_obs), 
            0.95, 
            dim=0
        )
        
        self.var_scale = torch.quantile(
            var_obs, 
            0.95, 
            dim=0
        )
            
        # get the scale parameters for the normal distributions used for the likelihood
        scale_ = (self.scale_unconstr)
        scale_ = scale_[: self.n_input, :].expand(z.shape[0], self.n_input, 6)
        scale_1 = (F.softplus(scale_[..., 0]) + 1e-6).sqrt() # mu
        scale_2 = (F.softplus(scale_[..., 1]) + 1e-6).sqrt() # r
        scale_3 = (F.softplus(scale_[..., 2]) + 1e-6).sqrt() # sqrt_r
        scale_4 = (F.softplus(scale_[..., 3]) + 1e-6).sqrt()
        scale_4 = torch.clamp(scale_4, min=1e-6, max=0.5) # theta
        scale_5 = (F.softplus(scale_[..., 4]) + 1e-6).sqrt()
        scale_6 = (F.softplus(scale_[..., 5]) + 1e-6).sqrt()

        scaling_factor_weights_time = F.softplus(self.scaling_factor) + 1

        # forward pass through the decoder
        px_time_rate_tmp, \
            gamma_mRNA_tmp, library_, px_time_rate_rep_tmp, \
            px_B1, px_f1, mu_down_up, var_down_up, state_pred, mu_up, var_up, \
            mu_up_f, var_up_f, mu_down, var_down, mu_down_f, var_down_f, \
            velo_mu_up, velo_var_up, \
            velo_mu_up_ss, velo_var_up_ss, velo_mu_down, velo_var_down, \
            px_pi_alpha, \
            mu_up_ss, var_up_ss, loss_distance_cell, id_ss_up, id_ss_down, \
            p_to_up_f, _, \
            loss_down_up_to_upper, \
            loss_down_f_to_upper, loss_up_branch, loss_down_branch, \
            time_ss_full, loss_match_top_corner, \
            loss_up0_downf_match, \
            loss_match_dist_up, loss_match_dist_down, \
            loss_match_mu_var_up_f, loss_match_mu_var_down_up, loss_match_mu_var_down_f, \
            loss_match_mu_var_0, \
            d2sigma_dmu2_up, d2sigma_dmu2_down, d2sigmaM_dmu2_up, d2sigmaM_dmu2_down, \
            mu_up_gene, var_up_gene, mu_down_gene, var_down_gene, \
            loss_match_gamma, loss_match_mu_var_down_up_ss, \
            px_time_rate_tmp_gene, \
            gamma_mRNA_tmp_gene, px_time_rate_rep_tmp_gene, \
            px_B1_gene, px_f1_gene, mu_down_up_gene, var_down_up_gene, \
            mu_up_f_gene, var_up_f_gene, \
            mu_down_f_gene, var_down_f_gene, \
            velo_mu_up_gene, velo_var_up_gene, time_ss_full_gene, \
            velo_mu_down_gene, velo_var_down_gene, \
                mu_up_ss_gene, var_up_ss_gene, weights_time, loss_time, b_t, f_t = \
            self.decoder(
                decoder_input,
                scaling_factor_weights_time,
                self.gamma_mRNA,
                self.use_gamma_mRNA,
                cont_covs,
                self.tmax,
                self.burst_B1,
                self.burst_f1,
                self.burst_B2,
                self.burst_f2,
                self.burst_B3,
                self.burst_f3,
                self.burst_B_low,
                self.burst_F_low,
                self.time_Bswitch,
                self.time_ss,
                mu_obs, 
                var_obs,
                self.mu_mean_obs,
                self.var_mean_obs,
                self.mu_ss_obs,
                self.var_ss_obs,
                self.scale_dist_sigmoid,
                self.scale_dist_sigmoid2,
                self.scale_dist_sigmoid3,
                self.scale_dist_sigmoid4,
                self.scale_dist_sigmoid5,
                self.mu_max, 
                self.var_max,
                self.match_upf_down_up,
                self.max_fac,
                device_,
                self.mu_center,
                self.std_center,
                self.mu_scale,
                self.std_scale,
                self.match_burst_params_not_muVar,
                scale_1,
                scale_5,
                self.fac_var,
                batch_index,
                *categorical_input,
                y,
            )
        
        px_pi_prior = torch.stack(
            (
                prior_pi_up,
                torch.zeros_like(prior_pi_up),
                1 - prior_pi_up,
                torch.zeros_like(prior_pi_up)
            ), dim=2
        )
        px_pi = Dirichlet(px_pi_alpha).rsample()
        if self.use_prior_parabola:  
            mask_ = self.prior_branch_assignment != 2 # not ellipse
            mask_reshaped = mask_.view(1, -1, 1)
            
            px_pi = torch.where(mask_reshaped, px_pi_prior, px_pi)
        
        
        def get_mu_std_scaled_loss_old(mu_0, std_0, mu_1, std_1, mu_center, mu_scale, \
                              std_center, std_scale,
                              scale_mu, scale_std,
                              weights=None):
            if weights is None:
                weights = torch.ones_like(mu_0)
            mu_0_scaled = (mu_0 - mu_center) / mu_scale
            mu_1_scaled = (mu_1 - mu_center) / mu_scale
            std_0_scaled = (std_0 - std_center) / std_scale
            std_1_scaled = (std_1 - std_center) / std_scale

            loss_mu = (((mu_0_scaled - mu_1_scaled)**2.0) / (2 * scale_mu**2.0))
            loss_mu = (loss_mu * weights).sum(-1)
            loss_std = (((std_0_scaled - std_1_scaled)**2.0) / (2 * scale_std**2.0))
            loss_std = (loss_std * weights).sum(-1)
            loss_total = (loss_mu + loss_std).mean()
            return loss_total
        
        

        
        # match mu and var achievable at switch time for the upregulated branch
        # to the observed mu and var near the corner where var is maximum
        mask_reshaped = None
        if self.prior_branch_assignment is not None:
            mask_ = self.prior_branch_assignment != 1 # not down-regulation
            mask_reshaped = mask_.view(1, -1)
        # else:
        
        if self.mu_ss_obs is not None:
            # loss_match_mu_var_switch_1 = \
            #     get_mu_std_scaled_loss(mu_up_ss_gene, torch.sqrt(var_up_ss_gene + 1e-4), \
            #                             self.mu_ss_obs[None, ...], \
            #                             torch.sqrt(self.var_ss_obs[None, ...] + 1e-4), \
            #                             self.mu_center, self.mu_scale, \
            #                             self.std_center,
            #                             self.std_scale, scale_1, scale_5,
            #                             None)
            loss_match_mu_var_switch_1 = self.get_mu_std_scaled_loss(
                mu_up_ss_gene, var_up_ss_gene, 
                self.mu_ss_obs[None, ...], self.var_ss_obs[None, ...], 
                scale_1, scale_2, scale_3, scale_4, weights=None)
            if self.match_burst_params:
                loss_match_mu_var_switch_2 = 0.0
                if self.loss_fac_geneCell > 0.0:
                    # loss_match_mu_var_switch_2 = \
                    #     get_mu_std_scaled_loss(mu_up_ss, torch.sqrt(var_up_ss + 1e-4), \
                    #                             self.mu_ss_obs[None, ...], \
                    #                             torch.sqrt(self.var_ss_obs[None, ...] + 1e-4), \
                    #                             self.mu_center, self.mu_scale, \
                    #                             self.std_center,
                    #                             self.std_scale, scale_1, scale_5,
                    #                             None)
                    loss_match_mu_var_switch_2 = self.get_mu_std_scaled_loss(
                        mu_up_ss, var_up_ss, 
                        self.mu_ss_obs[None, ...], self.var_ss_obs[None, ...], 
                        scale_1, scale_2, scale_3, scale_4, weights=None)
            else:
                loss_match_mu_var_switch_2 = 0.0
            loss_match_mu_var_switch = \
                (loss_match_mu_var_switch_1 + loss_match_mu_var_switch_2)
        else:
            loss_match_mu_var_switch = 0.0

        # match mu and var for the initial steady-state of the downregulated branch
        # to the observed mu and var near the corner where mu is maximum
        # if self.mu_mean_obs is not None:            
        #     # match mu_down_up
        #     if not self.match_upf_down_up:
        #         loss_match_mu_var_switch_1_ = \
        #             get_mu_std_scaled_loss(mu_down_up_gene, torch.sqrt(var_down_up_gene + 1e-4), \
        #                                     self.mu_mean_obs[None, ...], \
        #                                     torch.sqrt(self.var_mean_obs[None, ...] + 1e-4), \
        #                                     self.mu_center, self.mu_scale, \
        #                                     self.std_center,
        #                                     self.std_scale, scale_1, scale_3,
        #                                     None)
        #         if self.match_burst_params:
        #             loss_match_mu_var_switch_2_ = \
        #                 get_mu_std_scaled_loss(mu_down_up, torch.sqrt(var_down_up + 1e-4), \
        #                                         self.mu_mean_obs[None, ...], \
        #                                         torch.sqrt(self.var_mean_obs[None, ...] + 1e-4), \
        #                                         self.mu_center, self.mu_scale, \
        #                                         self.std_center,
        #                                         self.std_scale, scale_1,
        #                                         scale_3, weights_time)
        #         else:
        #             loss_match_mu_var_switch_2_ = 0.0
        #         loss_match_mu_var_switch_ = \
        #             (loss_match_mu_var_switch_1_ + loss_match_mu_var_switch_2_)
        #     else:
        #         loss_match_mu_var_switch_ = 0.0
        # else:
        #     loss_match_mu_var_switch_ = 0.0

        if self.mu_mean_obs is not None:   

            # match mu_down_up
            if not self.match_upf_down_up:
                # loss_match_mu_var_switch_1_ = \
                #     get_mu_std_scaled_loss(mu_down_up_gene, torch.sqrt(var_down_up_gene + 1e-4), \
                #                             mu_up_ss_gene, \
                #                             torch.sqrt(var_up_ss_gene + 1e-4), \
                #                             self.mu_center, self.mu_scale, \
                #                             self.std_center,
                #                             self.std_scale, scale_1, scale_5,
                #                             mask_reshaped)
                # loss_match_mu_var_switch_1_ = self.get_mu_std_scaled_loss(
                #         mu_down_up_gene, var_down_up_gene, 
                #         self.mu_mean_obs[None, ...], self.var_mean_obs[None, ...], 
                #         scale_1, scale_2, scale_3, scale_4, weights=None)
                loss_match_mu_var_switch_1_ = self.get_mu_std_scaled_loss(
                        mu_down_up_gene, var_down_up_gene, 
                        mu_up_ss_gene, var_up_ss_gene, 
                        scale_1, scale_2, scale_3, scale_4, weights=None)
                if self.match_burst_params:
                    loss_match_mu_var_switch_2_ = 0.0
                    if self.loss_fac_geneCell > 0.0:
                        # loss_match_mu_var_switch_2_ = \
                        #     get_mu_std_scaled_loss(mu_down_up, torch.sqrt(var_down_up + 1e-4), \
                        #                             mu_up_ss, \
                        #                             torch.sqrt(var_up_ss + 1e-4), \
                        #                             self.mu_center, self.mu_scale, \
                        #                             self.std_center,
                        #                             self.std_scale, scale_1,
                        #                             scale_5, mask_reshaped)
                        # loss_match_mu_var_switch_2_ = self.get_mu_std_scaled_loss(
                        #         mu_down_up, var_down_up, 
                        #         self.mu_mean_obs[None, ...], self.var_mean_obs[None, ...], 
                        #         scale_1, scale_2, scale_3, scale_4, weights=None)
                        loss_match_mu_var_switch_2_ = self.get_mu_std_scaled_loss(
                                mu_down_up, var_down_up, 
                                mu_up_ss, var_up_ss, 
                                scale_1, scale_2, scale_3, scale_4, weights=None)
                else:
                    loss_match_mu_var_switch_2_ = 0.0
                loss_match_mu_var_switch_ = \
                    (loss_match_mu_var_switch_1_ + loss_match_mu_var_switch_2_)
            else:
                loss_match_mu_var_switch_ = 0.0
        else:
            loss_match_mu_var_switch_ = 0.0
        
        # TODO: Unused variables; remove
        end_penalty = None

        # gene-cell-specific mu and var for the initial steady-state of the upregulated branch
        if self.loss_fac_geneCell > 0.0:
            mu_0 = px_f1 * px_B1 / gamma_mRNA_tmp
            var_0 = mu_0 * (px_B1 + 1)
        else:
            mu_0 = None
            var_0 = None
            
        mu_0_gene = (px_B1_gene * px_B1_gene) / gamma_mRNA_tmp
        var_0_gene = mu_0_gene * (px_B1_gene + 1)
        
        # loss_match_mu_var_0 = \
        #     get_mu_std_scaled_loss(mu_0_gene, torch.sqrt(var_0_gene + 1e-4), \
        #                             self.mu_low_obs, \
        #                             torch.sqrt(self.var_low_obs + 1e-4), \
        #                             self.mu_center, self.mu_scale, \
        #                             self.std_center,
        #                             self.std_scale, scale_1, scale_5,
        #                             None)
        loss_match_mu_var_0 = self.get_mu_std_scaled_loss(
                mu_0_gene, var_0_gene, 
                self.mu_low_obs, self.var_low_obs, 
                scale_1, scale_2, scale_3, scale_4, weights=None)
        
        # gene-cell-specific probabilites for being in the upregulated (state 0) 
        # or downregulated (state 1) branches
        probs_ = px_pi
        # time_total = (px_time_rate_tmp * p_up_branch) + \
        #     ((px_time_rate_rep_tmp + time_ss_full) * p_down_branch)
        # loss_time = torch.mean(torch.var(time_total, dim=1))
        # # Compute mean time per cell
        # mean_time_per_cell = torch.mean(time_total, dim=-1, keepdim=True)

        # # Compute absolute deviation per gene in each cell
        # time_deviation = torch.abs(time_total - mean_time_per_cell)

        # # Compute weights using learnable scaling factor
        # weights_time = torch.exp(-self.scaling_factor * time_deviation)

        # categorical distribution for the states (upregulated and downregulated branches)
        comp_dist = Categorical(probs=px_pi)
        comp_dist_prior = Categorical(probs=px_pi_prior)

        # average mu and var using the gene-cell-specific probabilities
        # mu_ = mu_up * p_up_branch + mu_down * p_down_branch
        # var_ = var_up * p_up_branch + var_down * p_down_branch
        # mu_gene = mu_up_gene * p_up_branch + mu_down_gene * p_down_branch
        # var_gene = var_up_gene * p_up_branch + var_down_gene * p_down_branch

        # gene-cell-specific: center and scale the time-dependent mu and std for 
        # the up- and down-regulated branches
        if self.loss_fac_geneCell > 0.0:
            mu1 = px_f1 * px_B1 / gamma_mRNA_tmp
            # mu0_scaled = (mu1 - self.mu_center) / self.mu_scale
            # mu_up_scaled = (mu_up - self.mu_center) / self.mu_scale
            # mu_up_f_scaled = (mu_up_f - self.mu_center) / self.mu_scale
            # mu_down_up_scaled = (mu_down_up - self.mu_center) / self.mu_scale
            # mu_down_scaled = (mu_down - self.mu_center) / self.mu_scale
            # mu_down_f_scaled = (mu_down_f - self.mu_center) / self.mu_scale
            mu0_scaled = (mu1 - self.mu_center)
            mu_up_scaled = (mu_up - self.mu_center)
            mu_up_f_scaled = (mu_up_f - self.mu_center)
            mu_down_up_scaled = (mu_down_up - self.mu_center)
            mu_down_scaled = (mu_down - self.mu_center)
            mu_down_f_scaled = (mu_down_f - self.mu_center)
            var1 = mu1 * (1 + px_B1)
            
            sqrt_r0, theta0, r0 = self._to_polar_features(mu1, var1)
            sqrt_r_up, theta_up, r_up = self._to_polar_features(mu_up, var_up)
            sqrt_r_up_f, theta_up_f, r_up_f = self._to_polar_features(mu_up_f, var_up_f)
            sqrt_r_down_up, theta_down_up, r_down_up = self._to_polar_features(
                mu_down_up, var_down_up)
            sqrt_r_down, theta_down, r_down = self._to_polar_features(
                mu_down, var_down)
            sqrt_r_down_f, theta_down_f, r_down_f = self._to_polar_features(
                mu_down_f, var_down_f)
            
            # std0_scaled = (torch.sqrt(var1 + 1e-4) - self.std_center) / \
            #     self.std_scale
            # std_up_scaled = (torch.sqrt(var_up + 1e-4) - self.std_center) / \
            #     self.std_scale
            # std_up_f_scaled = (torch.sqrt(var_up_f + 1e-4) - self.std_center) / \
            #     self.std_scale
            # std_down_up_scaled = (torch.sqrt(var_down_up + 1e-4) - self.std_center) / \
            #     self.std_scale
            # std_down_scaled = (torch.sqrt(var_down + 1e-4) - self.std_center) / \
            #     self.std_scale
            # std_down_f_scaled = (torch.sqrt(var_down_f + 1e-4) - self.std_center) / \
            #     self.std_scale
        else:
            mu1 = None
            mu0_scaled = None
            mu_up_scaled = None
            mu_up_f_scaled = None
            mu_down_up_scaled = None
            mu_down_scaled = None
            mu_down_f_scaled = None
            var1 = None
            
            sqrt_r0, theta0 = None, None
            sqrt_r_up, theta_up = None, None
            sqrt_r_up_f, theta_up_f = None, None
            sqrt_r_down_up, theta_down_up = None, None
            sqrt_r_down, theta_down = None, None
            sqrt_r_down_f, theta_down_f = None, None
            
            
            # var1 = None
            # std0_scaled = None
            # std_up_scaled = None
            # std_up_f_scaled = None
            # std_down_up_scaled = None
            # std_down_scaled = None
            # std_down_f_scaled = None
        
        # gene-specific: center and scale the time-dependent mu and std for
        # the up- and down-regulated branches
        mu1_gene = px_f1_gene * px_B1_gene / gamma_mRNA_tmp
        # mu0_gene_scaled = (mu1_gene - self.mu_center) / self.mu_scale
        # mu_up_gene_scaled = (mu_up_gene - self.mu_center) / self.mu_scale
        # mu_up_f_gene_scaled = (mu_up_f_gene - self.mu_center) / self.mu_scale
        # mu_down_up_gene_scaled = (mu_down_up_gene - self.mu_center) / self.mu_scale
        # mu_down_gene_scaled = (mu_down_gene - self.mu_center) / self.mu_scale
        # mu_down_f_gene_scaled = (mu_down_f_gene - self.mu_center) / self.mu_scale
        mu0_gene_scaled = (mu1_gene - self.mu_center)
        mu_up_gene_scaled = (mu_up_gene - self.mu_center)
        mu_up_f_gene_scaled = (mu_up_f_gene - self.mu_center)
        mu_down_up_gene_scaled = (mu_down_up_gene - self.mu_center)
        mu_down_gene_scaled = (mu_down_gene - self.mu_center)
        mu_down_f_gene_scaled = (mu_down_f_gene - self.mu_center)
        var1_gene = mu1_gene * (1 + px_B1_gene)
        
        
        sqrt_r0_gene, theta0_gene, r0_gene = self._to_polar_features(mu1_gene, var1_gene)
        sqrt_r_up_gene, theta_up_gene, r_up_gene = self._to_polar_features(mu_up_gene, var_up_gene)
        sqrt_r_up_f_gene, theta_up_f_gene, r_up_f_gene = self._to_polar_features(
            mu_up_f_gene, var_up_f_gene)
        sqrt_r_down_up_gene, theta_down_up_gene, r_down_up_gene = self._to_polar_features(
            mu_down_up_gene, var_down_up_gene)
        sqrt_r_down_gene, theta_down_gene, r_down_gene = self._to_polar_features(
            mu_down_gene, var_down_gene)
        sqrt_r_down_f_gene, theta_down_f_gene, r_down_f_gene = self._to_polar_features(
            mu_down_f_gene, var_down_f_gene)
        
        
        # std0_gene_scaled = (torch.sqrt(var1_gene + 1e-4) - self.std_center) / \
        #     self.std_scale
        # std_up_gene_scaled = (torch.sqrt(var_up_gene + 1e-4) - self.std_center) / \
        #     self.std_scale
        # std_up_f_gene_scaled = (torch.sqrt(var_up_f_gene + 1e-4) - self.std_center) / \
        #     self.std_scale
        # std_down_up_gene_scaled = (torch.sqrt(var_down_up_gene + 1e-4) - self.std_center) / \
        #     self.std_scale
        # std_down_gene_scaled = (torch.sqrt(var_down_gene + 1e-4) - self.std_center) / \
        #     self.std_scale
        # std_down_f_gene_scaled = (torch.sqrt(var_down_f_gene + 1e-4) - self.std_center) / \
        #     self.std_scale
        
        # stack mu and std for the up- and down-regulated branches
        if self.loss_fac_geneCell > 0.0:
            mu_stacked = torch.stack(
                (
                    # mu0_scaled,
                    mu_up_scaled,
                    mu_up_f_scaled,
                    # mu_down_up_scaled,
                    mu_down_scaled,
                    mu_down_f_scaled
                ),
                dim=2
            )
            scale_mu_stacked = torch.stack(
                (
                    # scale_1,
                    scale_1,
                    scale_1,
                    # scale_1,
                    scale_1,
                    scale_1,
                ),
                dim=2
            )
            # std_stacked = torch.stack(
            #     (
            #         # std0_scaled,
            #         std_up_scaled,
            #         std_up_f_scaled,
            #         # std_down_up_scaled,
            #         std_down_scaled,
            #         std_down_f_scaled
            #     ),
            #     dim=2
            # )
            std_stacked = torch.stack(
                (
                    # std0_scaled,
                    sqrt_r_up,
                    sqrt_r_up_f,
                    # std_down_up_scaled,
                    sqrt_r_down,
                    sqrt_r_down_f
                ),
                dim=2
            )
            scale_std_stacked = torch.stack(
                (
                    # scale_3,
                    scale_3,
                    scale_3,
                    # scale_3,
                    scale_3,
                    scale_3
                ),
                dim=2
            )
            
            r_stacked = torch.stack(
                (
                    # std0_scaled,
                    r_up,
                    r_up_f,
                    # std_down_up_scaled,
                    r_down,
                    r_down_f
                ),
                dim=2
            )
            scale_r_stacked = torch.stack(
                (
                    # scale_3,
                    scale_2,
                    scale_2,
                    # scale_3,
                    scale_2,
                    scale_2
                ),
                dim=2
            )

            # stack reflected std for the up- and down-regulated branches
            # var_max = (self.var_max + 1)
            # std_ref0_stacked = \
            #     (torch.sqrt(F.softplus(var_max - var1) + 1e-4) - self.std_ref_center) / \
            #     self.std_ref_scale
            # std_ref_up_stacked = \
            #     (torch.sqrt(F.softplus(var_max - var_up) + 1e-4) - self.std_ref_center) / \
            #     self.std_ref_scale
            # std_ref_up_f_stacked = \
            #     (torch.sqrt(F.softplus(var_max - var_up_f) + 1e-4) - self.std_ref_center) / \
            #     self.std_ref_scale
            # std_ref_down_up_stacked = \
            #     (torch.sqrt(F.softplus(var_max - var_down_up) + 1e-4) - self.std_ref_center) / \
            #     self.std_ref_scale
            # std_ref_down_stacked = \
            #     (torch.sqrt(F.softplus(var_max - var_down) + 1e-4) - self.std_ref_center) / \
            #     self.std_ref_scale
            # std_ref_down_f_stacked = \
            #     (torch.sqrt(F.softplus(var_max - var_down_f) + 1e-4) - self.std_ref_center) / \
            #     self.std_ref_scale
            # std_ref_stacked = torch.stack(
            #     (
            #         # std_ref0_stacked,
            #         std_ref_up_stacked,
            #         std_ref_up_f_stacked,
            #         # std_ref_down_up_stacked,
            #         std_ref_down_stacked,
            #         std_ref_down_f_stacked
            #     ),
            #     dim=2
            # )
            std_ref_stacked = torch.stack(
                (
                    # std_ref0_stacked,
                    theta_up,
                    theta_up_f,
                    # std_ref_down_up_stacked,
                    theta_down,
                    theta_down_f
                ),
                dim=2
            )
            scale_std_ref_stacked = torch.stack(
                (
                    # scale_4,
                    scale_4,
                    scale_4,
                    # scale_4,
                    scale_4,
                    scale_4
                ),
                dim=2
            )

        # stack gene-specific mu, std, and reflected std for the up- and down-regulated branches
        mu_gene_stacked = torch.stack(
            (
                # mu0_gene_scaled,
                mu_up_gene_scaled,
                mu_up_f_gene_scaled,
                # mu_down_up_gene_scaled,
                mu_down_gene_scaled,
                mu_down_f_gene_scaled
            ),
            dim=2
        )
        scale_mu_gene_stacked = torch.stack(
            (
                # scale_1,
                scale_1,
                scale_1,
                # scale_1,
                scale_1,
                scale_1
            ),
            dim=2
        )
        # std_gene_stacked = torch.stack(
        #     (
        #         # std0_gene_scaled,
        #         std_up_gene_scaled,
        #         std_up_f_gene_scaled,
        #         # std_down_up_gene_scaled,
        #         std_down_gene_scaled,
        #         std_down_f_gene_scaled
        #     ),
        #     dim=2
        # )
        std_gene_stacked = torch.stack(
            (
                # std0_gene_scaled,
                sqrt_r_up_gene,
                sqrt_r_up_f_gene,
                # std_down_up_gene_scaled,
                sqrt_r_down_gene,
                sqrt_r_down_f_gene
            ),
            dim=2
        )
        scale_std_gene_stacked = torch.stack(
            (
                # scale_3,
                scale_3,
                scale_3,
                # scale_3,
                scale_3,
                scale_3
            ),
            dim=2
        )
        
        r_gene_stacked = torch.stack(
            (
                # std0_gene_scaled,
                r_up_gene,
                r_up_f_gene,
                # std_down_up_gene_scaled,
                r_down_gene,
                r_down_f_gene
            ),
            dim=2
        )
        scale_r_gene_stacked = torch.stack(
            (
                # scale_3,
                scale_2,
                scale_2,
                # scale_3,
                scale_2,
                scale_2
            ),
            dim=2
        )
        # var_max = (self.var_max + 1)
        # std_ref0_gene_stacked = \
        #     (torch.sqrt(F.softplus(var_max - var1_gene) + 1e-4) - self.std_ref_center) / \
        #     self.std_ref_scale
        # std_ref_up_gene_stacked = \
        #     (torch.sqrt(F.softplus(var_max - var_up_gene) + 1e-4) - self.std_ref_center) / \
        #     self.std_ref_scale
        # std_ref_up_f_gene_stacked = \
        #     (torch.sqrt(F.softplus(var_max - var_up_f_gene) + 1e-4) - self.std_ref_center) / \
        #     self.std_ref_scale
        # std_ref_down_up_gene_stacked = \
        #     (torch.sqrt(F.softplus(var_max - var_down_up_gene) + 1e-4) - self.std_ref_center) / \
        #     self.std_ref_scale
        # std_ref_down_gene_stacked = \
        #     (torch.sqrt(F.softplus(var_max - var_down_gene) + 1e-4) - self.std_ref_center) / \
        #     self.std_ref_scale
        # std_ref_down_f_gene_stacked = \
        #     (torch.sqrt(F.softplus(var_max - var_down_f_gene) + 1e-4) - self.std_ref_center) / \
        #     self.std_ref_scale
        # std_ref_gene_stacked = torch.stack(
        #     (
        #         # std_ref0_gene_stacked,
        #         std_ref_up_gene_stacked,
        #         std_ref_up_f_gene_stacked,
        #         # std_ref_down_up_gene_stacked,
        #         std_ref_down_gene_stacked,
        #         std_ref_down_f_gene_stacked
        #     ),
        #     dim=2
        # )
        std_ref_gene_stacked = torch.stack(
            (
                # std_ref0_gene_stacked,
                theta_up_gene,
                theta_up_f_gene,
                # std_ref_down_up_gene_stacked,
                theta_down_gene,
                theta_down_f_gene
            ),
            dim=2
        )
        scale_std_ref_gene_stacked = torch.stack(
            (
                # scale_4,
                scale_4,
                scale_4,
                # scale_4,
                scale_4,
                scale_4
            ),
            dim=2
        )

        loss_fac_div = 1
        sqrt_r_obs, theta_obs, r_obs = self._to_polar_features(mu_obs, var_obs)
        # likelihood and loss for mu using the gene-cell-specific params
        # mu_obs_scaled = (mu_obs - self.mu_center) / self.mu_scale
        mu_obs_scaled = (mu_obs - self.mu_center)

        df_warmup = getattr(self, "student_t_df_warmup", getattr(self, "student_t_df", 1000.0))
        df_final = getattr(self, "student_t_df_final", 4.0)

        if self.loss_fac_geneCell > 0.0:
            dist_mu_warmup = self._magnitude_dist_for_df((mu_stacked / loss_fac_div), scale_mu_stacked, df_warmup)
            px_mu_geneCell_warmup = MixtureSameFamily(comp_dist, dist_mu_warmup)
            loss_mu_geneCell_warmup = (-px_mu_geneCell_warmup.log_prob((mu_obs_scaled / loss_fac_div)))
            px_mu_geneCell_prior_warmup = MixtureSameFamily(comp_dist_prior, dist_mu_warmup)
            loss_mu_geneCell_prior_warmup = (-px_mu_geneCell_prior_warmup.log_prob((mu_obs_scaled / loss_fac_div)))

            dist_mu_final = self._magnitude_dist_for_df((mu_stacked / loss_fac_div), scale_mu_stacked, df_final)
            px_mu_geneCell_final = MixtureSameFamily(comp_dist, dist_mu_final)
            loss_mu_geneCell_final = (-px_mu_geneCell_final.log_prob((mu_obs_scaled / loss_fac_div)))
            px_mu_geneCell_prior_final = MixtureSameFamily(comp_dist_prior, dist_mu_final)
            loss_mu_geneCell_prior_final = (-px_mu_geneCell_prior_final.log_prob((mu_obs_scaled / loss_fac_div)))
        else:
            loss_mu_geneCell_warmup = 0.0
            loss_mu_geneCell_prior_warmup = 0.0
            loss_mu_geneCell_final = 0.0
            loss_mu_geneCell_prior_final = 0.0

        # likelihood and loss for mu using the gene-specific params
        dist_mu_gene_warmup = self._magnitude_dist_for_df((mu_gene_stacked / loss_fac_div), scale_mu_gene_stacked, df_warmup)
        px_mu_gene_warmup = MixtureSameFamily(comp_dist, dist_mu_gene_warmup)
        loss_mu_gene_warmup = (-px_mu_gene_warmup.log_prob((mu_obs_scaled / loss_fac_div)))
        px_mu_gene_prior_warmup = MixtureSameFamily(comp_dist_prior, dist_mu_gene_warmup)
        loss_mu_gene_prior_warmup = (-px_mu_gene_prior_warmup.log_prob((mu_obs_scaled / loss_fac_div)))

        dist_mu_gene_final = self._magnitude_dist_for_df((mu_gene_stacked / loss_fac_div), scale_mu_gene_stacked, df_final)
        px_mu_gene_final = MixtureSameFamily(comp_dist, dist_mu_gene_final)
        loss_mu_gene_final = (-px_mu_gene_final.log_prob((mu_obs_scaled / loss_fac_div)))
        px_mu_gene_prior_final = MixtureSameFamily(comp_dist_prior, dist_mu_gene_final)
        loss_mu_gene_prior_final = (-px_mu_gene_prior_final.log_prob((mu_obs_scaled / loss_fac_div)))

        # total loss for mu
        if self.loss_fac_geneCell > 0.0:
            loss_mu_warmup = (((self.loss_fac_geneCell * loss_mu_geneCell_warmup + \
                        self.loss_fac_gene * loss_mu_gene_warmup) / \
                        (self.loss_fac_geneCell + self.loss_fac_gene)))
            loss_mu_prior_warmup = ((self.loss_fac_geneCell * loss_mu_geneCell_prior_warmup + \
                        self.loss_fac_gene * loss_mu_gene_prior_warmup) / \
                        (self.loss_fac_geneCell + self.loss_fac_gene))

            loss_mu_final = (((self.loss_fac_geneCell * loss_mu_geneCell_final + \
                        self.loss_fac_gene * loss_mu_gene_final) / \
                        (self.loss_fac_geneCell + self.loss_fac_gene)))
            loss_mu_prior_final = ((self.loss_fac_geneCell * loss_mu_geneCell_prior_final + \
                        self.loss_fac_gene * loss_mu_gene_prior_final) / \
                        (self.loss_fac_geneCell + self.loss_fac_gene))
            loss_mu_full = (self.loss_fac_geneCell * loss_mu_geneCell_final + \
                        self.loss_fac_gene * loss_mu_gene_final) / \
                        (self.loss_fac_geneCell + self.loss_fac_gene)
        else:
            loss_mu_warmup = (((self.loss_fac_gene * loss_mu_gene_warmup) / \
                        (self.loss_fac_gene)))
            loss_mu_prior_warmup = ((self.loss_fac_gene * loss_mu_gene_prior_warmup) / \
                        (self.loss_fac_gene))

            loss_mu_final = (((self.loss_fac_gene * loss_mu_gene_final) / \
                        (self.loss_fac_gene)))
            loss_mu_prior_final = ((self.loss_fac_gene * loss_mu_gene_prior_final) / \
                        (self.loss_fac_gene))
            loss_mu_full = (self.loss_fac_gene * loss_mu_gene_final) / \
                        (self.loss_fac_gene)

        if self.use_prior_parabola:
            mask_ = self.prior_branch_assignment != 2 # not ellipse
            mask_reshaped = mask_.view(1, -1)
            loss_mu_warmup_ = loss_mu_warmup[:, ~mask_].sum(-1) + loss_mu_prior_warmup[:, mask_].sum(-1)
            loss_mu_final_ = loss_mu_final[:, ~mask_].sum(-1) + loss_mu_prior_final[:, mask_].sum(-1)
        else:
            loss_mu_warmup_ = loss_mu_warmup.sum(-1)
            loss_mu_final_ = loss_mu_final.sum(-1)
        loss_mu_prior_warmup = loss_mu_prior_warmup.sum(-1)
        loss_mu_prior_final = loss_mu_prior_final.sum(-1)

        # likelihood and loss for std using the gene-cell-specific params
        if self.loss_fac_geneCell > 0.0:
            dist_std_warmup = self._magnitude_dist_for_df((std_stacked / loss_fac_div), scale_std_stacked, df_warmup)
            px_std_geneCell_warmup = MixtureSameFamily(comp_dist, dist_std_warmup)
            loss_std_geneCell_warmup = (-px_std_geneCell_warmup.log_prob(sqrt_r_obs))

            dist_std_final = self._magnitude_dist_for_df((std_stacked / loss_fac_div), scale_std_stacked, df_final)
            px_std_geneCell_final = MixtureSameFamily(comp_dist, dist_std_final)
            loss_std_geneCell_final = (-px_std_geneCell_final.log_prob(sqrt_r_obs))

            dist_r_warmup = self._magnitude_dist_for_df((r_stacked / loss_fac_div), scale_r_stacked, df_warmup)
            px_r_geneCell_warmup = MixtureSameFamily(comp_dist, dist_r_warmup)
            loss_r_geneCell_warmup = (-px_r_geneCell_warmup.log_prob(r_obs))

            dist_r_final = self._magnitude_dist_for_df((r_stacked / loss_fac_div), scale_r_stacked, df_final)
            px_r_geneCell_final = MixtureSameFamily(comp_dist, dist_r_final)
            loss_r_geneCell_final = (-px_r_geneCell_final.log_prob(r_obs))

            px_std_geneCell_prior_warmup = MixtureSameFamily(comp_dist_prior, dist_std_warmup)
            loss_std_geneCell_prior_warmup = (-px_std_geneCell_prior_warmup.log_prob(sqrt_r_obs))

            px_std_geneCell_prior_final = MixtureSameFamily(comp_dist_prior, dist_std_final)
            loss_std_geneCell_prior_final = (-px_std_geneCell_prior_final.log_prob(sqrt_r_obs))

            px_r_geneCell_prior_warmup = MixtureSameFamily(comp_dist_prior, dist_r_warmup)
            loss_r_geneCell_prior_warmup = (-px_r_geneCell_prior_warmup.log_prob(r_obs))

            px_r_geneCell_prior_final = MixtureSameFamily(comp_dist_prior, dist_r_final)
            loss_r_geneCell_prior_final = (-px_r_geneCell_prior_final.log_prob(r_obs))
        else:
            loss_std_geneCell_warmup = 0.0
            loss_std_geneCell_prior_warmup = 0.0
            loss_std_geneCell_final = 0.0
            loss_std_geneCell_prior_final = 0.0
            loss_r_geneCell_warmup = 0.0
            loss_r_geneCell_prior_warmup = 0.0
            loss_r_geneCell_final = 0.0
            loss_r_geneCell_prior_final = 0.0

        # likelihood and loss for std using the gene-specific params
        dist_std_gene_warmup = self._magnitude_dist_for_df((std_gene_stacked / loss_fac_div), scale_std_gene_stacked, df_warmup)
        px_std_gene_warmup = MixtureSameFamily(comp_dist, dist_std_gene_warmup)
        loss_std_gene_warmup = (-px_std_gene_warmup.log_prob(sqrt_r_obs))

        dist_std_gene_final = self._magnitude_dist_for_df((std_gene_stacked / loss_fac_div), scale_std_gene_stacked, df_final)
        px_std_gene_final = MixtureSameFamily(comp_dist, dist_std_gene_final)
        loss_std_gene_final = (-px_std_gene_final.log_prob(sqrt_r_obs))

        px_std_gene_prior_warmup = MixtureSameFamily(comp_dist_prior, dist_std_gene_warmup)
        loss_std_gene_prior_warmup = (-px_std_gene_prior_warmup.log_prob(sqrt_r_obs))

        px_std_gene_prior_final = MixtureSameFamily(comp_dist_prior, dist_std_gene_final)
        loss_std_gene_prior_final = (-px_std_gene_prior_final.log_prob(sqrt_r_obs))

        dist_r_gene_warmup = self._magnitude_dist_for_df((r_gene_stacked / loss_fac_div), scale_r_gene_stacked, df_warmup)
        px_r_gene_warmup = MixtureSameFamily(comp_dist, dist_r_gene_warmup)
        loss_r_gene_warmup = (-px_r_gene_warmup.log_prob(r_obs))

        dist_r_gene_final = self._magnitude_dist_for_df((r_gene_stacked / loss_fac_div), scale_r_gene_stacked, df_final)
        px_r_gene_final = MixtureSameFamily(comp_dist, dist_r_gene_final)
        loss_r_gene_final = (-px_r_gene_final.log_prob(r_obs))

        px_r_gene_prior_warmup = MixtureSameFamily(comp_dist_prior, dist_r_gene_warmup)
        loss_r_gene_prior_warmup = (-px_r_gene_prior_warmup.log_prob(r_obs))

        px_r_gene_prior_final = MixtureSameFamily(comp_dist_prior, dist_r_gene_final)
        loss_r_gene_prior_final = (-px_r_gene_prior_final.log_prob(r_obs))

        # likelihood and loss for reflected std using the gene-cell-specific params (theta uses Normal)
        if self.loss_fac_geneCell > 0.0:
            dist_std_ref = Normal((std_ref_stacked / loss_fac_div), scale_std_ref_stacked)
            px_std_ref_geneCell = MixtureSameFamily(comp_dist, dist_std_ref)
            loss_std_ref_geneCell = (-px_std_ref_geneCell.log_prob(theta_obs))

            px_std_ref_geneCell_prior = MixtureSameFamily(comp_dist_prior, dist_std_ref)
            loss_std_ref_geneCell_prior = (-px_std_ref_geneCell_prior.log_prob(theta_obs))
        else:
            loss_std_ref_geneCell = None
            loss_std_ref_geneCell_prior = None

        # likelihood and loss for reflected std using the gene-specific params
        dist_std_ref_gene = Normal((std_ref_gene_stacked / loss_fac_div), scale_std_ref_gene_stacked)
        px_std_ref_gene = MixtureSameFamily(comp_dist, dist_std_ref_gene)
        loss_std_ref_gene = (-px_std_ref_gene.log_prob(theta_obs))

        px_std_ref_gene_prior = MixtureSameFamily(comp_dist_prior, dist_std_ref_gene)
        loss_std_ref_gene_prior = (-px_std_ref_gene_prior.log_prob(theta_obs))

        # total loss for std
        if self.loss_fac_geneCell > 0.0:
            loss_std_warmup = ((self.loss_fac_geneCell * loss_std_geneCell_warmup + \
                        self.loss_fac_gene * loss_std_gene_warmup) / \
                        (self.loss_fac_geneCell + self.loss_fac_gene) + \
                        (self.loss_fac_geneCell * loss_std_ref_geneCell * self.w_theta + \
                        self.loss_fac_gene * loss_std_ref_gene * self.w_theta) / \
                        (self.loss_fac_geneCell + self.loss_fac_gene))
            loss_std_final = ((self.loss_fac_geneCell * loss_std_geneCell_final + \
                        self.loss_fac_gene * loss_std_gene_final) / \
                        (self.loss_fac_geneCell + self.loss_fac_gene) + \
                        (self.loss_fac_geneCell * loss_std_ref_geneCell * self.w_theta + \
                        self.loss_fac_gene * loss_std_ref_gene * self.w_theta) / \
                        (self.loss_fac_geneCell + self.loss_fac_gene))

            loss_r_warmup = (self.loss_fac_geneCell * loss_r_geneCell_warmup + \
                        self.loss_fac_gene * loss_r_gene_warmup) / \
                        (self.loss_fac_geneCell + self.loss_fac_gene)
            loss_r_final = (self.loss_fac_geneCell * loss_r_geneCell_final + \
                        self.loss_fac_gene * loss_r_gene_final) / \
                        (self.loss_fac_geneCell + self.loss_fac_gene)

            loss_std_full = (((self.loss_fac_geneCell * loss_std_geneCell_final + \
                        self.loss_fac_gene * loss_std_gene_final) / \
                        (self.loss_fac_geneCell + self.loss_fac_gene)) + \
                        ((self.loss_fac_geneCell * loss_std_ref_geneCell * self.w_theta + \
                        self.loss_fac_gene * loss_std_ref_gene * self.w_theta) / \
                        (self.loss_fac_geneCell + self.loss_fac_gene)))

            loss_std_prior_warmup = ((self.loss_fac_geneCell * loss_std_geneCell_prior_warmup + \
                        self.loss_fac_gene * loss_std_gene_prior_warmup) / \
                        (self.loss_fac_geneCell + self.loss_fac_gene) + \
                        (self.loss_fac_geneCell * loss_std_ref_geneCell_prior * self.w_theta + \
                        self.loss_fac_gene * loss_std_ref_gene_prior * self.w_theta) / \
                        (self.loss_fac_geneCell + self.loss_fac_gene))
            loss_std_prior_final = ((self.loss_fac_geneCell * loss_std_geneCell_prior_final + \
                        self.loss_fac_gene * loss_std_gene_prior_final) / \
                        (self.loss_fac_geneCell + self.loss_fac_gene) + \
                        (self.loss_fac_geneCell * loss_std_ref_geneCell_prior * self.w_theta + \
                        self.loss_fac_gene * loss_std_ref_gene_prior * self.w_theta) / \
                        (self.loss_fac_geneCell + self.loss_fac_gene))

            loss_r_prior_warmup = (self.loss_fac_geneCell * loss_r_geneCell_prior_warmup + \
                        self.loss_fac_gene * loss_r_gene_prior_warmup) / \
                        (self.loss_fac_geneCell + self.loss_fac_gene)
            loss_r_prior_final = (self.loss_fac_geneCell * loss_r_geneCell_prior_final + \
                        self.loss_fac_gene * loss_r_gene_prior_final) / \
                        (self.loss_fac_geneCell + self.loss_fac_gene)
        else:
            loss_std_warmup = ((self.loss_fac_gene * loss_std_gene_warmup) / \
                        (self.loss_fac_gene) + \
                        (self.loss_fac_gene * loss_std_ref_gene * self.w_theta) / \
                        (self.loss_fac_gene))
            loss_std_final = ((self.loss_fac_gene * loss_std_gene_final) / \
                        (self.loss_fac_gene) + \
                        (self.loss_fac_gene * loss_std_ref_gene * self.w_theta) / \
                        (self.loss_fac_gene))

            loss_r_warmup = loss_r_gene_warmup
            loss_r_final = loss_r_gene_final

            loss_std_full = (((self.loss_fac_gene * loss_std_gene_final) / \
                        (self.loss_fac_gene)) + \
                        ((self.loss_fac_gene * loss_std_ref_gene * self.w_theta) / \
                        (self.loss_fac_gene)))

            loss_std_prior_warmup = ((self.loss_fac_gene * loss_std_gene_prior_warmup) / \
                        (self.loss_fac_gene) + \
                        (self.loss_fac_gene * loss_std_ref_gene_prior * self.w_theta) / \
                        (self.loss_fac_gene))
            loss_std_prior_final = ((self.loss_fac_gene * loss_std_gene_prior_final) / \
                        (self.loss_fac_gene) + \
                        (self.loss_fac_gene * loss_std_ref_gene_prior * self.w_theta) / \
                        (self.loss_fac_gene))

            loss_r_prior_warmup = loss_r_gene_prior_warmup
            loss_r_prior_final = loss_r_gene_prior_final

        if self.use_prior_parabola:
            mask_ = self.prior_branch_assignment != 2 # not ellipse
            mask_reshaped = mask_.view(1, -1)
            loss_std_warmup_ = loss_std_warmup[:, ~mask_].sum(-1) + loss_std_prior_warmup[:, mask_].sum(-1)
            loss_std_final_ = loss_std_final[:, ~mask_].sum(-1) + loss_std_prior_final[:, mask_].sum(-1)
        else:
            loss_std_warmup_ = loss_std_warmup.sum(-1)
            loss_std_final_ = loss_std_final.sum(-1)

        loss_std_prior_warmup = loss_std_prior_warmup.sum(-1)
        loss_std_prior_final = loss_std_prior_final.sum(-1)

        loss_r_warmup = loss_r_warmup.sum(-1)
        loss_r_final = loss_r_final.sum(-1)
        loss_r_prior_warmup = loss_r_prior_warmup.sum(-1)
        loss_r_prior_final = loss_r_prior_final.sum(-1)

        # total reconstruction loss (warmup and final)
        reconst_loss_warmup = loss_mu_warmup_ + loss_std_warmup_ + loss_r_warmup
        reconst_loss_prior_warmup = loss_mu_prior_warmup + loss_std_prior_warmup + loss_r_prior_warmup

        reconst_loss_final = loss_mu_final_ + loss_std_final_ + loss_r_final
        reconst_loss_prior_final = loss_mu_prior_final + loss_std_prior_final + loss_r_prior_final

        reconst_loss = reconst_loss_final
        reconst_loss_prior = reconst_loss_prior_final

        # branch-separation (repulsion) loss on theta at matched sqrt_r.
        # Up branch:   (mu_0_gene, var_0_gene)        -> (mu_up_f_gene, var_up_f_gene)
        # Down branch: (mu_down_up_gene, var_down_up_gene) -> (mu_down_f_gene, var_down_f_gene)
        if getattr(self, "w_sep", 0.0) and self.w_sep > 0.0:
            loss_sep = self._branch_repulsion_loss(
                px_pi,
                mu_0_gene, var_0_gene, mu_up_f_gene, var_up_f_gene,
                mu_down_up_gene, var_down_up_gene, mu_down_f_gene, var_down_f_gene,
                eps_ratio=1.0,
            )
        else:
            loss_sep = torch.zeros((), device=reconst_loss.device, dtype=reconst_loss.dtype)

        # Priors
        if self.use_observed_lib_size:
            pl = None
        else:
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)
            pl = Normal(local_library_log_means, local_library_log_vars.sqrt())
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))
        
        
        
        return {
            "reconst_loss" : reconst_loss,
            "reconst_loss_prior": reconst_loss_prior,
            "reconst_loss_warmup": reconst_loss_warmup,
            "reconst_loss_final": reconst_loss_final,
            "reconst_loss_prior_warmup": reconst_loss_prior_warmup,
            "reconst_loss_prior_final": reconst_loss_prior_final,
            "pl": pl,
            "pz": pz,
            "burst_b1" : px_B1,
            "burst_f1" : px_f1, 
            "burst_b2" : var_down_up,
            "burst_f2" : mu_down_up,
            "mu_up_f" : mu_up_f,
            "var_up_f" : var_up_f,
            "mu_down_f" : mu_down_f,
            "var_down_f" : var_down_f,
            "time_ss" : time_ss_full, 
            "velo_mu_up" : velo_mu_up,
            "velo_var_up" : velo_var_up,
            "velo_mu_down" : velo_mu_down,
            "velo_var_down" : velo_var_down,
            "gamma_mRNA" : gamma_mRNA_tmp,
            "time_cell" : px_time_rate_tmp,
            "library": library_,
            "prob_state" : px_pi, 
            "time_down" : px_time_rate_rep_tmp,
            "scale" : scale_1,
            "scale_var" : scale_3,
            "scale_var_ref" : scale_4, 
            "burst_b1_gene" : px_B1_gene,
            "burst_f1_gene" : px_f1_gene, 
            "burst_b2_gene" : var_down_up_gene,
            "burst_f2_gene" : mu_down_up_gene,
            "mu_up_f_gene" : mu_up_f_gene,
            "var_up_f_gene" : var_up_f_gene,
            "mu_down_f_gene" : mu_down_f_gene,
            "var_down_f_gene" : var_down_f_gene,
            "time_ss_gene" : time_ss_full_gene, 
            "velo_mu_up_gene" : velo_mu_up_gene,
            "velo_var_up_gene" : velo_var_up_gene,
            "velo_mu_down_gene" : velo_mu_down_gene,
            "velo_var_down_gene" : velo_var_down_gene,
            "gamma_mRNA_gene" : gamma_mRNA_tmp,
            "time_up_gene" : px_time_rate_tmp_gene,
            "time_down_gene" : px_time_rate_rep_tmp_gene,
            "scale_gene" : torch.stack((scale_1, scale_3, scale_4), dim=2),
            "end_penalty" : end_penalty,
            "px_pi": px_pi,
            'px_pi_alpha': px_pi_alpha,
            'px_pi_prior': px_pi_prior,
            "loss_up_branch" : loss_up_branch,
            "loss_down_branch" : loss_down_branch,
            "loss_down_up_to_upper" : loss_down_up_to_upper,
            "loss_down_f_to_upper" : loss_down_f_to_upper,
            "loss_match_dist_up" : loss_match_dist_up,
            "loss_match_dist_down" : loss_match_dist_down,
            "loss_match_mu_var_up_f" : loss_match_mu_var_up_f,
            "loss_match_mu_var_down_up" : loss_match_mu_var_down_up,
            "loss_match_mu_var_down_f" : loss_match_mu_var_down_f,
            "loss_match_mu_var_0" : loss_match_mu_var_0,
            "loss_match_gamma" : loss_match_gamma,
            "loss_match_mu_var_down_up_ss" : loss_match_mu_var_down_up_ss,
            "loss_match_mu_var_switch" : loss_match_mu_var_switch,
            "loss_match_mu_var_switch_" : loss_match_mu_var_switch_,
            "loss_match_mu_var_0": loss_match_mu_var_0,
            "weights_time" : weights_time,
            "loss_time" : loss_time,
            "loss_mu_full" : loss_mu_full,
            "loss_std_full" : loss_std_full,
            "loss_sep" : loss_sep,
            "b_t": b_t,
            "f_t": f_t
        }
    
    def expected_value_guide_loss(self, px_pi_alpha, prior_probs):
        # 1. Calculate the expected probability from the predicted alphas
        sum_alpha = px_pi_alpha.sum(dim=-1, keepdim=True)
        expected_pi = px_pi_alpha / (sum_alpha + 1e-8)

        # 2. Convert to log probabilities (add small eps to prevent log(0))
        log_expected_pi = torch.log(expected_pi + 1e-8)

        # 3. Compute KL Divergence against your heuristic prior
        loss = F.kl_div(log_expected_pi, prior_probs, reduction='none').sum(dim=-1)

        return loss

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ):
        """Computes the loss function for the model."""
        x = tensors[REGISTRY_KEYS.X_KEY]
        px_pi = generative_outputs['px_pi']

        reconst_loss_decoded_warmup = generative_outputs.get('reconst_loss_warmup', generative_outputs['reconst_loss'])
        reconst_loss_decoded_final = generative_outputs.get('reconst_loss_final', generative_outputs['reconst_loss'])
        reconst_loss_prior_warmup = generative_outputs.get('reconst_loss_prior_warmup', generative_outputs['reconst_loss_prior'])
        reconst_loss_prior_final = generative_outputs.get('reconst_loss_prior_final', generative_outputs['reconst_loss_prior'])

        reconst_loss_decoded = (1.0 - kl_weight**3.0) * reconst_loss_decoded_warmup + kl_weight**3.0 * reconst_loss_decoded_final
        reconst_loss_prior = (1.0 - kl_weight**3.0) * reconst_loss_prior_warmup + kl_weight**3.0 * reconst_loss_prior_final
        
        if self.use_time_dependence:
            b_t = generative_outputs['b_t']
            f_t = generative_outputs['f_t']
            loss_b_t_time = torch.sum(
                (-torch.log(b_t).mean(0)) + (-torch.log(f_t).mean(0))
            )
        else:
            loss_b_t_time = 0.0
        
        
        if kl_weight < 0.1:
            prior_inv_weight = 0.0
        else:
            prior_inv_weight = kl_weight
        reconst_loss = prior_inv_weight * reconst_loss_decoded + \
            (1 - prior_inv_weight) * reconst_loss_prior
            
        
        loss_match_mu_var_up_f = generative_outputs['loss_match_mu_var_up_f']
        loss_match_mu_var_down_up = generative_outputs['loss_match_mu_var_down_up']
        loss_match_mu_var_down_f = generative_outputs['loss_match_mu_var_down_f']
        loss_match_mu_var_0 = generative_outputs['loss_match_mu_var_0']
        loss_match_mu_var_switch = generative_outputs['loss_match_mu_var_switch']
        loss_match_mu_var_switch_ = generative_outputs['loss_match_mu_var_switch_']
        loss_match_mu_var_0 = generative_outputs['loss_match_mu_var_0']
        loss_time = generative_outputs['loss_time']
        px_pi_alpha = generative_outputs['px_pi_alpha']
        prior_probs = generative_outputs['px_pi_prior']
        loss_sep = generative_outputs.get('loss_sep', 0.0)
    

        # KL divergence loss for the latent variable z
        pz_prior = generative_outputs["pz"]
        kl_divergence_z = kl(inference_outputs["qz"], \
                             pz_prior).sum(
            dim=1
        )

        kl_divergence_l = torch.tensor(0.0, device=x.device)

        # KL divergence loss for state assignment
        # kl_pi = \
        #     (-entropy_no_mean(logits_state, px_pi) - \
        #     np.log(1 / self.n_states)).sum(-1)
        # prob_entropy_loss = self.entropy_loss(px_pi)
        # kl_pi_prior = torch.tensor(0.0, device=self.device)
        kl_pi_decoded = kl(
                Dirichlet(px_pi_alpha),
                Dirichlet(self.dirichlet_concentration * torch.ones_like(px_pi)),
            )
        
        kl_pi_prior_emp = self.expected_value_guide_loss(px_pi_alpha, prior_probs)
        
        if self.use_prior_parabola:  
            mask_ = self.prior_branch_assignment != 2 # not ellipse
            mask_reshaped = mask_.view(1, -1)
            
            # kl_pi_zeros = torch.zeros_like(kl_pi_decoded)
            # kl_pi_safe = torch.clamp(
            #     kl_pi_prior_emp,
            #     min=1e-5,
            #     max=1-1e-5
            # )
            # kl_pi_decoded = torch.where(mask_reshaped, kl_pi_safe, kl_pi_decoded)
            kl_pi_decoded_safe = kl_pi_decoded[:, ~mask_].sum(-1)
            kl_pi_prior_emp_safe = kl_pi_prior_emp[:, ~mask_].sum(-1)
        else:
            kl_pi_decoded_safe = kl_pi_decoded.sum(-1)
            kl_pi_prior_emp_safe = kl_pi_prior_emp.sum(-1)
            
        kl_pi = prior_inv_weight * kl_pi_decoded_safe + \
            (1 - prior_inv_weight) * kl_pi_prior_emp_safe

        kl_local_for_warmup = kl_divergence_z
        kl_local_no_warmup = 0 * kl_divergence_l
        # kl_local = kl_divergence_z + kl_pi
        weighted_kl_local = kl_weight * kl_local_for_warmup + \
            kl_local_no_warmup + kl_pi
        
        if self.match_burst_params:
            # loss = torch.mean(reconst_loss + weighted_kl_local) + \
            #     self.extra_loss_fac * (loss_match_mu_var_up_f) + \
            #     self.extra_loss_fac_0 * (loss_match_mu_var_down_up) + \
            #     self.extra_loss_fac_1 * (loss_match_mu_var_down_f + loss_match_mu_var_0) + \
            #     (self.extra_loss_fac_2 * (loss_match_mu_var_switch) + \
            #     self.extra_loss_fac_2_ * loss_match_mu_var_switch_) * (1 - kl_weight) + \
            #     (self.extra_loss_fac_2 * (loss_match_mu_var_switch) + \
            #     self.extra_loss_fac_2_ * loss_match_mu_var_switch_) * (kl_weight) * self.edge_loss_fac
            #     # (self.extra_loss_fac_2_ * loss_match_mu_var_switch) * (kl_weight)
                
            loss = torch.mean(reconst_loss + weighted_kl_local) + \
                self.extra_loss_fac * (loss_match_mu_var_up_f) + \
                self.extra_loss_fac_0 * (loss_match_mu_var_down_up) + \
                self.extra_loss_fac_1 * (loss_match_mu_var_down_f + loss_match_mu_var_0) + \
                (((1 - kl_weight) * self.extra_loss_fac_2_ + kl_weight * (self.extra_loss_fac_2_ / 1000)) * \
                     (loss_match_mu_var_switch_)) + \
                (self.extra_loss_fac_2 * loss_match_mu_var_switch) * (1 - kl_weight) * self.edge_loss_fac + \
                self.w_sep * loss_sep + self.time_loss * loss_b_t_time

                #
        else:
            loss = torch.mean(reconst_loss + weighted_kl_local) + \
                self.extra_loss_fac_2 * (loss_match_mu_var_switch + \
                                         loss_match_mu_var_switch_) + \
                self.w_sep * loss_sep

        kl_local = {
            "kl_divergence_l": kl_divergence_l,
            "kl_divergence_z": kl_divergence_z + kl_pi,
        }
        return LossOutput(
            loss=loss, reconstruction_loss=reconst_loss, kl_local=kl_local
        )

    @torch.inference_mode()
    def sample(
        self,
        tensors,
        n_samples=1,
        library_size=1,
    ) -> np.ndarray:
        r"""Generate observation samples from the posterior predictive distribution.
        The posterior predictive distribution is written as :math:`p(\hat{x} \mid x)`.
        Parameters
        ----------
        tensors
            Tensors dict
        n_samples
            Number of required samples for each cell
        library_size
            Library size to scale samples to
        Returns
        -------
        x_new : :py:class:`torch.Tensor`
            tensor with shape (n_cells, n_genes, n_samples)
        """
        inference_kwargs = {"n_samples": n_samples}
        (
            _,
            generative_outputs,
        ) = self.forward(
            tensors,
            inference_kwargs=inference_kwargs,
            compute_loss=False,
        )

        dist = generative_outputs["px"]
        if self.gene_likelihood == "poisson":
            l_train = generative_outputs["px"].rate
            l_train = torch.clamp(l_train, max=1e8)
            dist = torch.distributions.Poisson(
                l_train
            )  # Shape : (n_samples, n_cells_batch, n_genes)
        if n_samples > 1:
            exprs = dist.sample().permute(
                [1, 2, 0]
            )  # Shape : (n_cells_batch, n_genes, n_samples)
        else:
            exprs = dist.sample()

        return exprs.cpu()

    @torch.inference_mode()
    @auto_move_data
    def marginal_ll(self, tensors, n_mc_samples):
        """Computes the marginal log likelihood of the model."""
        sample_batch = tensors[REGISTRY_KEYS.X_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        to_sum = torch.zeros(sample_batch.size()[0], n_mc_samples)

        for i in range(n_mc_samples):
            # Distribution parameters and sampled variables
            inference_outputs, _, losses = self.forward(tensors)
            qz = inference_outputs["qz"]
            ql = inference_outputs["ql"]
            z = inference_outputs["z"]
            library = inference_outputs["library"]

            # Reconstruction Loss
            reconst_loss = losses.dict_sum(losses.reconstruction_loss)

            # Log-probabilities
            p_z = (
                Normal(torch.zeros_like(qz.loc), torch.ones_like(qz.scale))
                .log_prob(z)
                .sum(dim=-1)
            )
            p_x_zl = -reconst_loss
            q_z_x = qz.log_prob(z).sum(dim=-1)
            log_prob_sum = p_z + p_x_zl - q_z_x

            if not self.use_observed_lib_size:
                (
                    local_library_log_means,
                    local_library_log_vars,
                ) = self._compute_local_library_params(batch_index)

                p_l = (
                    Normal(local_library_log_means, local_library_log_vars.sqrt())
                    .log_prob(library)
                    .sum(dim=-1)
                )
                q_l_x = ql.log_prob(library).sum(dim=-1)

                log_prob_sum += p_l - q_l_x

            to_sum[:, i] = log_prob_sum

        batch_log_lkl = logsumexp(to_sum, dim=-1) - np.log(n_mc_samples)
        log_lkl = torch.sum(batch_log_lkl).item()
        return log_lkl
    
