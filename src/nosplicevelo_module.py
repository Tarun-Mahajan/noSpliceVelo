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
from torch.distributions import Normal, MixtureSameFamily, MultivariateNormal
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
from coral_pytorch.losses import corn_loss
from coral_pytorch.dataset import corn_label_from_logits

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
        n_states: int = 2,
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
    ):
        r"""TODO: add docstring"""
        super().__init__()
        self.n_output = n_output
        self.state_loss_type = state_loss_type

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
        # self.px_pi_decoder = nn.Linear(n_hidden, n_states * n_output)
        self.px_pi_decoder = GumbelSoftmax_3D(n_hidden, n_output, n_states)
        
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

        self.px_burstB1_decoder = nn.Linear(n_hidden, int(n_output))
        self.px_burstF1_decoder = nn.Linear(n_hidden, int(n_output))
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
        
        self.time_scale_switch_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            # nn.Sigmoid(),
        )

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
        deriv2_fac1 = - (0.5 * torch.log(var_ + 1e-10) + torch.log(2 * ones_)) + \
            torch.log(torch.abs(d2var_dt2) + 1e-10) - \
            2 * torch.log(torch.abs(dmu_dt) + 1e-10)
        d2var_dt2_sign = 2 * (F.relu(d2var_dt2 * 1e5) / (F.relu(d2var_dt2 * 1e5) + 1e-9)) - \
            1
        deriv2_fac1 = d2var_dt2_sign * torch.exp(deriv2_fac1)

        deriv2_fac2 = - ((3 / 2) * torch.log(var_ + 1e-10) + torch.log(4 * ones_)) + \
             2  * torch.log(torch.abs(dvar_dt) + 1e-10) - \
            2 * torch.log(torch.abs(dmu_dt) + 1e-10)
        deriv2_fac2 = torch.exp(deriv2_fac2)

        deriv2_fac3 = - (0.5 * torch.log(var_ + 1e-10) + torch.log(2 * ones_)) + \
            torch.log(torch.abs(dvar_dt) + 1e-10) + \
            torch.log(torch.abs(d2mu_dt2) + 1e-10) - 3 * torch.log(torch.abs(dmu_dt) + 1e-10)
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
        dsigmaM_dt = - (0.5 * torch.log(var_max - var_ + 1e-10) + torch.log(2 * ones_)) + \
            torch.log(torch.abs(dvar_dt) + 1e-10)
        dsigmaM_dt = dvar_dt_sign * torch.exp(dsigmaM_dt)
        dsig_dt = \
            2 * torch.exp(torch.log(dvar_dt + 1e-10) - \
                          torch.log(torch.sqrt(var_ + 1e-10) + 1e-10))
        d2mu_dt2 = - gamma_mRNA_tmp**2.0 * delta_mu  * p_t
        d2muM_dt2 = - d2mu_dt2
        d2var_dt2 = - gamma_mRNA_tmp**2.0 * (4 * delta_var * p_t**2.0 + \
                                             delta_mu * p_t * (1 - 4 * p_t))
        deriv2_fac1 = - (0.5 * torch.log(var_max - var_ + 1e-10) + torch.log(2 * ones_)) + \
            torch.log(torch.abs(d2var_dt2) + 1e-10) - \
            2 * torch.log(torch.abs(dmuM_dt) + 1e-10)
        d2var_dt2_sign = 2 * (F.relu(d2var_dt2 * 1e5) / (F.relu(d2var_dt2 * 1e5) + 1e-9)) - \
            1
        # dmuM_dt_sign = 2 * (F.relu(dmuM_dt * 1e5) / (F.relu(dmuM_dt * 1e5) + 1e-9)) - \
        #     1
        deriv2_fac1 = d2var_dt2_sign * torch.exp(deriv2_fac1)

        deriv2_fac2 = - ((3 / 2) * torch.log(var_max - var_ + 1e-10) + torch.log(4 * ones_)) + \
             2  * torch.log(torch.abs(dvar_dt) + 1e-10) - \
            2 * torch.log(torch.abs(dmuM_dt) + 1e-10)
        deriv2_fac2 = torch.exp(deriv2_fac2)

        deriv2_fac3 = - (0.5 * torch.log(var_max - var_ + 1e-10) + torch.log(2 * ones_)) + \
            torch.log(torch.abs(dvar_dt) + 1e-10) + \
            torch.log(torch.abs(d2muM_dt2) + 1e-10) - 3 * torch.log(torch.abs(dmuM_dt) + 1e-10)
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
        if self.cluster_states:
            state_pred = self.state_predictor(z)
        else:
            state_pred = None

        pow_fac = 0.5

        z_new = z
        px = self.px_decoder(z_new, *cat_list)

        # cells by genes by n_states
        pi_first = self.pi_first_decoder(z)
        temperature=1.0
        hard=0
        logits_state, px_pi, y_state = \
            self.px_pi_decoder(pi_first, temperature, hard)

        # get gene-specific burst frequency and burst size for the starting 
        # steady state for the upregulated branch. Also get mu and var.
        var_max = (var_max_tmp)
        px_B1_gene = F.softplus(burst_B_low.repeat(px.shape[0], 1)) + 1e-9
        px_f1_by_gam_gene = F.softplus(burst_F_low.repeat(px.shape[0], 1)) + 1e-9
        f1_by_gam_max_gene = torch.min((mu_max_tmp * 0.5 / fac_var) / px_B1_gene, \
                                  ((var_max_tmp * 0.5 / fac_var) / (px_B1_gene * (px_B1_gene + 1))))
        px_f1_by_gam_gene = torch.clamp(px_f1_by_gam_gene, max=f1_by_gam_max_gene)
        mu_0_gene = px_B1_gene * px_f1_by_gam_gene
        var_0_gene = mu_0_gene * (px_B1_gene + 1)

        # get gene-cell-specific burst frequency and burst size for the starting 
        # steady state for the upregulated branch. Also get mu and var.
        px_B1 = px_B1_gene
        px_f1_by_gam = px_f1_by_gam_gene
        mu_0 = px_B1 * px_f1_by_gam
        var_0 = mu_0 * (px_B1 + 1)

        def get_mu_std_scaled_loss(mu_0, std_0, mu_1, std_1, mu_center, mu_scale, \
                              std_center, std_scale, scale_mu, scale_std):
            mu_0_scaled = (mu_0 - mu_center) / mu_scale
            mu_1_scaled = (mu_1 - mu_center) / mu_scale
            std_0_scaled = (std_0 - std_center) / std_scale
            std_1_scaled = (std_1 - std_center) / std_scale

            loss_mu = (((mu_0_scaled - mu_1_scaled)**2.0) / (2 * scale_mu**2.0)).sum(-1)
            loss_std = (((std_0_scaled - std_1_scaled)**2.0) / (2 * scale_std**2.0)).sum(-1)
            loss_total = (loss_mu + loss_std).mean()
            return loss_total

        # match gene-specific and gene-cell-specific mu and var for the starting
        # steady state for the upregulated branch
        if not match_burst_params_not_muVar:
            loss_match_mu_var_0 = \
                get_mu_std_scaled_loss(mu_0, torch.sqrt(var_0 + 1e-10), \
                                       mu_0_gene, torch.sqrt(var_0_gene + 1e-10), \
                                       mu_center, mu_scale, \
                                       std_center, std_scale, scale_mu, scale_std)
        else:
            loss_match_mu_var_0 = \
                torch.sqrt((torch.log(px_f1_by_gam + 1e-10) - \
                            torch.log(px_f1_by_gam_gene + 1e-10))**2.0 + \
                           (torch.log(px_B1 + 1e-10) - \
                            torch.log(px_B1_gene + 1e-10))**2.0 + 1e-10).sum(-1).mean()

        # get gene-specific burst frequency and burst size for the final 
        # steady state for the upregulated branch. Also get mu and var.
        px_f2_by_gam_gene = F.softplus(burst_F2.repeat(px.shape[0], 1)) + \
            px_f1_by_gam_gene * 1.2
        f2_max_gene = (var_max - px_B1_gene**2.0 * px_f1_by_gam_gene)**2.0 / \
            (px_B1_gene**2.0 * px_f1_by_gam_gene + 1e-10)
        px_f2_by_gam_gene = torch.clamp(px_f2_by_gam_gene, max=f2_max_gene)
        px_B2_gene = F.softplus(burst_B2.repeat(px.shape[0], 1)) + px_B1_gene * \
            torch.sqrt(px_f1_by_gam_gene / px_f2_by_gam_gene + 1e-9) + 1e-9
        B2_max_gene = (-1 + torch.sqrt(1 + 4 * var_max / px_f2_by_gam_gene)) / 2
        px_B2_gene = torch.clamp(px_B2_gene, max=B2_max_gene)
        mu_up_f_gene = px_B2_gene * px_f2_by_gam_gene
        var_up_f_gene = mu_up_f_gene * (px_B2_gene + 1)

        # get gene-cell-specific burst frequency and burst size for the final
        # steady state for the upregulated branch. Also get mu and var.
        px_f2_by_gam = F.softplus(self.px_burstF2_decoder(px)) + px_f1_by_gam * 1.2
        f2_max = (var_max - px_B1**2.0 * px_f1_by_gam)**2.0 / \
            (px_B1**2.0 * px_f1_by_gam + 1e-10)
        px_f2_by_gam = torch.clamp(px_f2_by_gam, max=f2_max)
        px_B2 = F.softplus(self.px_burstB2_decoder(px)) + 1e-9 + \
            px_B1 * torch.sqrt(px_f1_by_gam / px_f2_by_gam + 1e-9)
        B2_max = (-1 + torch.sqrt(1 + 4 * var_max / px_f2_by_gam)) / 2
        px_B2 = torch.clamp(px_B2, max=B2_max)
        mu_up_f = px_B2 * px_f2_by_gam
        var_up_f = mu_up_f * (px_B2 + 1)

        # match gene-specific and gene-cell-specific mu and var for the final
        # steady state for the upregulated branch
        if not match_burst_params_not_muVar:
            loss_match_mu_var_up_f = \
                get_mu_std_scaled_loss(mu_up_f, torch.sqrt(var_up_f + 1e-10), \
                                       mu_up_f_gene, torch.sqrt(var_up_f_gene + 1e-10), \
                                       mu_center, mu_scale, \
                                       std_center, std_scale, scale_mu, scale_std)
        else:
            loss_match_mu_var_up_f = \
                torch.sqrt((torch.log(px_f2_by_gam + 1e-10) - \
                            torch.log(px_f2_by_gam_gene + 1e-10))**2.0 + \
                        (torch.log(px_B2 + 1e-10) - \
                            torch.log(px_B2_gene + 1e-10))**2.0 + 1e-10).sum(-1).mean()
        
        # get gene-specific degradation rates
        gamma_mRNA_tmp = F.softplus(gamma_mRNA.repeat(px.shape[0], 1)) + 1e-9
        gamma_mRNA_tmp_gene = gamma_mRNA_tmp

        # get gene-specific switch time; the time when genes switch from the 
        # upregulated branch to the downregulated branch
        time_ss_full = F.softplus(time_ss.repeat(px.shape[0], 1))
        time_ss_full = torch.clamp(time_ss_full, max=tmax)
        time_ss_full_gene = time_ss_full
        loss_match_gamma = \
            torch.sqrt((((gamma_mRNA_tmp + 1e-10) - \
                        (gamma_mRNA_tmp_gene + 1e-10))**2.0).mean(0) + 1e-10).sum()
        
        # convert burst frequencies from relative scale (wrt gamma) to absolute scale
        px_f1 = px_f1_by_gam * gamma_mRNA_tmp
        px_f2 = px_f2_by_gam *  gamma_mRNA_tmp

        px_f1_gene = px_f1_by_gam_gene * gamma_mRNA_tmp_gene
        px_f2_gene = px_f2_by_gam_gene *  gamma_mRNA_tmp_gene

        # assign gene-cell-specific time for the upregulated branch
        px_time_scale = F.softplus(self.time_scale_decoder(px))
        px_time_rate_tmp = torch.clamp(px_time_scale, max=time_ss_full)
        px_time_rate_tmp_gene = px_time_rate_tmp

        # assign gene-cell-specific time for the downregulated branch
        px_time_scale_next = F.softplus(self.time_scale_next_decoder(px))
        px_time_rate_rep_tmp = torch.clamp(px_time_scale_next, \
                                           max=(tmax - time_ss_full))
        px_time_rate_rep_tmp_gene = px_time_rate_rep_tmp

        # get mu and var, and convert to burst freq. and burst size for the 
        # upregulated branch achievable at switch time. Use gene-cell-specific 
        # initial and final steady state mu and var
        p_up_ss = torch.exp(-gamma_mRNA_tmp * time_ss_full)
        mu_up_ss = mu_0 * p_up_ss + \
            mu_up_f * (1 - p_up_ss)
        var_up_ss = (var_0 - mu_0) * \
            p_up_ss**2.0 + (var_up_f - mu_up_f) * (1 - p_up_ss**2.0) + \
            mu_up_ss
        velo_mu_up_ss = torch.zeros_like(mu_up_f)
        velo_var_up_ss = torch.zeros_like(mu_up_f)
        px_B_ss = var_up_ss / mu_up_ss - 1 + 1e-10
        px_f_ss = (mu_up_ss / px_B_ss) * gamma_mRNA_tmp

        # get mu and var, and convert to burst freq. and burst size for the
        # downregulated branch achievable at switch time. Use gene-specific
        # initial and final steady state mu and var
        p_up_ss_gene = torch.exp(-gamma_mRNA_tmp_gene * time_ss_full_gene)
        mu_up_ss_gene = mu_0_gene * p_up_ss_gene + \
            mu_up_f_gene * (1 - p_up_ss_gene)
        var_up_ss_gene = (var_0_gene - mu_0_gene) * \
            p_up_ss_gene**2.0 + (var_up_f_gene - mu_up_f_gene) * (1 - p_up_ss_gene**2.0) + \
            mu_up_ss_gene
        velo_mu_up_ss_gene = torch.zeros_like(mu_up_f_gene)
        velo_var_up_ss_gene = torch.zeros_like(mu_up_f_gene)
        px_B_ss_gene = var_up_ss_gene / mu_up_ss_gene - 1 + 1e-10
        px_f_ss_gene = (mu_up_ss_gene / px_B_ss_gene) * gamma_mRNA_tmp_gene

        # get gene-specific burst frequency and burst size for the starting 
        # steady state for the downregulated branch. Also get mu and var.
        px_B3_up_gene = F.softplus(burst_B1.repeat(px.shape[0], 1)) + 2e-9
        px_f3_up_gene = F.softplus(burst_F1.repeat(px.shape[0], 1)) + px_f1_gene + 1e-9
        f3_max_gene = ((var_max_tmp) * gamma_mRNA_tmp_gene) / (px_B3_up_gene * (px_B3_up_gene + 1))
        f3_min_gene_new = (1.1 * mu_0_gene.clone() * gamma_mRNA_tmp_gene) / px_B3_up_gene
        px_f3_up_gene = torch.clamp(px_f3_up_gene, max=f3_max_gene, min=f3_min_gene_new)
        mu_down_up_gene = px_B3_up_gene * px_f3_up_gene / gamma_mRNA_tmp_gene
        var_down_up_gene = mu_down_up_gene * (px_B3_up_gene + 1)

        # get gene-specific burst frequency and burst size for the final 
        # steady state for the downregulated branch. Also get mu and var.
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
        mu_down_f_gene = px_B3_gene * px_f3_gene / gamma_mRNA_tmp_gene
        var_down_f_gene = mu_down_f_gene * (px_B3_gene + 1)

        # get gene-cell-specific burst frequency and burst size for the starting
        # steady state for the downregulated branch. Also get mu and var.
        px_B3_up = F.softplus(self.px_burstB3_decoder(px)) + 2e-9
        px_f3_up = F.softplus(self.px_burstF3_decoder(px)) + px_f1 + 1e-9
        f3_max = ((var_max_tmp) * gamma_mRNA_tmp) / (px_B3_up * (px_B3_up + 1))
        f3_min = (1.1 * mu_0 * gamma_mRNA_tmp) / px_B3_up
        px_f3_up = torch.clamp(px_f3_up, max=f3_max, min=f3_min)
        mu_down_up = px_B3_up * px_f3_up / gamma_mRNA_tmp
        var_down_up = mu_down_up * (px_B3_up + 1)

        # match gene-specific and gene-cell-specific mu and var or 
        # burst freq. and burst size for the starting steady state 
        # for the downregulated branch
        if not match_burst_params_not_muVar:
            loss_match_mu_var_down_up = \
                get_mu_std_scaled_loss(mu_down_up, torch.sqrt(var_down_up + 1e-10), \
                                       mu_down_up_gene, torch.sqrt(var_down_up_gene + 1e-10), \
                                       mu_center, mu_scale, \
                                       std_center, std_scale, scale_mu, scale_std)
        else:
            loss_match_mu_var_down_up = \
                torch.sqrt((torch.log(px_f3_up + 1e-10) - \
                            torch.log(px_f3_up_gene + 1e-10))**2.0 + \
                        (torch.log(px_B3_up + 1e-10) - \
                            torch.log(px_B3_up_gene + 1e-10))**2.0 + 1e-10).sum(-1).mean()

        # get gene-cell-specific burst frequency and burst size for the final
        # steady state for the downregulated branch. Also get mu and var.
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
        var_down_f = mu_down_f * (px_B3 + 1)

        # match gene-specific and gene-cell-specific mu and var or
        # burst freq. and burst size for the final steady state
        # for the downregulated branch
        if not match_burst_params_not_muVar:
            loss_match_mu_var_down_f = \
                get_mu_std_scaled_loss(mu_down_f, torch.sqrt(var_down_f + 1e-10), \
                                    mu_down_f_gene, torch.sqrt(var_down_f_gene + 1e-10), \
                                    mu_center, mu_scale, \
                                    std_center, std_scale, scale_mu, scale_std)
            # loss_match_mu_var_down_f = \
            #     ((mu_down_f - mu_down_f_gene)**2.0 / loss_fac_div + \
            #             (torch.sqrt(var_down_f + 1e-10) - \
            #                 torch.sqrt(var_down_f_gene + 1e-10))**2.0 / loss_fac_div_2 + 1e-10).mean(0).sum()
        else:
            loss_match_mu_var_down_f = \
                torch.sqrt((torch.log(px_f3 + 1e-10) - \
                            torch.log(px_f3_gene + 1e-10))**2.0 + \
                        (torch.log(px_B3 + 1e-10) - \
                            torch.log(px_B3_gene + 1e-10))**2.0 + 1e-10).sum(-1).mean()

        # uncomment the following lines to match the mu and var for the final
        # initial state for the downregulated branch to the mu and var achievable at 
        # switch time for the upregulated branch
        # if not match_upf_down_up:
        #     if not match_burst_params_not_muVar:
        #         loss_match_mu_var_down_up_ss = \
        #             get_mu_std_scaled_loss(mu_down_up, torch.sqrt(var_down_up + 1e-10), \
        #                                 mu_up_ss, torch.sqrt(var_up_ss + 1e-10), \
        #                                 mu_center, mu_scale, \
        #                                 std_center, std_scale, scale_mu, scale_std)
        #     else:
        #         loss_match_mu_var_down_up_ss = \
        #         torch.sqrt((torch.log(px_f3_up + 1e-10) - \
        #                     torch.log(px_f_ss + 1e-10))**2.0 + \
        #                 (torch.log(px_B3_up + 1e-10) - \
        #                     torch.log(px_B_ss + 1e-10))**2.0 + 1e-10).sum(-1).mean()

        # else:
        #     loss_match_mu_var_down_up_ss = 0.0
        loss_match_mu_var_down_up_ss = 0.0

        # TODO: Unused variables; remove
        loss_up0_downf_match = torch.sqrt(((mu_0[0, :].flatten() - \
                        mu_down_f[0, :].flatten()).pow(2)).sum() + 1e-12) + \
                    torch.sqrt(((torch.sqrt(var_0[0, :].flatten() + 1e-12) - \
                     torch.sqrt(var_down_f[0, :].flatten() + 1e-12)).pow(2)).sum() + 1e-12)
        dist_down_up_to_upper = \
            self.dist_from_connecting_line(mu_0[:, :], \
                                           (var_0[:, :] + 1e-10).pow(pow_fac), \
                                           mu_up_ss[:, :], \
                                           (var_up_ss[:, :] + 1e-10).pow(pow_fac), \
                                           mu_down_up[:, :], \
                                           (var_down_up[:, :] + 1e-10).pow(pow_fac))
        loss_down_up_to_upper = torch.sum(F.relu(dist_down_up_to_upper), dim=-1).mean()

        # TODO: Unused variables; remove
        p_down_min = torch.exp(-gamma_mRNA_tmp * (tmax - time_ss_full))
        mu_down_min = mu_down_up * p_down_min + \
            mu_down_f * (1 - p_down_min)
        var_down_min = (var_down_up - mu_down_up) * p_down_min**2.0 + \
            (var_down_f - mu_down_f) * (1 - p_down_min**2.0) + \
            mu_down_min
        dist_down_f_to_upper = \
            self.dist_from_connecting_line(mu_0[0, :], \
                                           (var_0[0, :] + 1e-10).pow(pow_fac), \
                                           mu_up_ss[0, :], \
                                           (var_up_ss[0, :] + 1e-10).pow(pow_fac), \
                                           mu_down_f[0, :], \
                                           (var_down_f[0, :] + 1e-10).pow(pow_fac))
        loss_down_f_to_upper = torch.sum(F.relu(dist_down_f_to_upper))
        loss_match_top_corner = torch.sqrt(((mu_up_ss[0, :].flatten() + 1e-4 - \
                        (mu_down_up[0, :].flatten())).pow(2)).sum() + 1e-10) + \
                    torch.sqrt(((torch.sqrt(var_up_ss[0, :].flatten() - 1e-8 + 1e-12) - \
                     torch.sqrt(var_down_up[0, :].flatten() + 1e-12)).pow(2)).sum() + 1e-10)
        
        # TODO: Unused variables; remove
        distance_cond = (mu_up_f - mu_obs) * (mu_up_f + mu_obs - 2 * mu_0) + \
            (var_up_f - var_obs) * (var_up_f + var_obs - 2 * var_0)
        loss_distance_cell = F.relu(-distance_cond).sum(-1) / \
            (F.relu(-distance_cond).sum(-1) + 1e-5)
        
        dist_to_up_f = (mu_up_f - mu_obs)**2.0 + \
            (var_up_f - var_obs)**2.0
        scale_dist_sigmoid_full3 = F.softplus(scale_dist_sigmoid3) + 1.0
        scale_dist_sigmoid_full3 = scale_dist_sigmoid_full3.repeat(px.shape[0], 1) 
        p_to_up_f = 2 * F.sigmoid(-scale_dist_sigmoid_full3 * dist_to_up_f)
        
        # get time-dependent mu and var for the upregulated branch using the
        # gene-cell-specific params
        p_up = torch.exp(-gamma_mRNA_tmp * px_time_rate_tmp)
        mu_up = mu_0 * p_up + \
            mu_up_f * (1 - p_up)
        var_up = (var_0 - mu_0) * p_up**2.0 + \
            (var_up_f - mu_up_f) * (1 - p_up**2.0) + \
            mu_up
        velo_mu_up = (mu_up_f - mu_up) * gamma_mRNA_tmp
        velo_var_up = (mu_up_f * (2 * px_B2 + 1) + mu_up - 2  * var_up) * gamma_mRNA_tmp

        # get time-dependent mu and var for the downregulated branch using the
        # gene-specific params
        p_up_gene = torch.exp(-gamma_mRNA_tmp * px_time_rate_tmp_gene)
        mu_up_gene = mu_0_gene * p_up_gene + \
            mu_up_f_gene * (1 - p_up_gene)
        var_up_gene = (var_0_gene - mu_0_gene) * p_up_gene**2.0 + \
            (var_up_f_gene - mu_up_f_gene) * (1 - p_up_gene**2.0) + \
            mu_up_gene
        velo_mu_up_gene = (mu_up_f_gene - mu_up_gene) * gamma_mRNA_tmp_gene
        velo_var_up_gene = (mu_up_f_gene * (2 * px_B2_gene + 1) + mu_up_gene - \
                            2  * var_up_gene) * gamma_mRNA_tmp_gene
        
        # get time-dependent mu and var for the downregulated branch using the
        # gene-cell-specific params
        p_down = torch.exp(-gamma_mRNA_tmp * px_time_rate_rep_tmp)
        mu_down = mu_down_up * p_down + \
            mu_down_f * (1 - p_down)
        var_down = (var_down_up) * p_down**2.0 + \
            (var_down_f) * (1 - p_down**2.0) - \
            (mu_down_f - mu_down_up) * (p_down - p_down**2.0)
        velo_mu_down = (mu_down_f - mu_down) * gamma_mRNA_tmp
        velo_var_down = \
            (mu_down_f * (2 * px_B3 + 1) + mu_down - 2  * var_down) * gamma_mRNA_tmp
        
        # get time-dependent mu and var for the downregulated branch using the
        # gene-specific params
        p_down_gene = torch.exp(-gamma_mRNA_tmp_gene * px_time_rate_rep_tmp_gene)
        mu_down_gene = mu_down_up_gene * p_down_gene + \
            mu_down_f_gene * (1 - p_down_gene)
        var_down_gene = (var_down_up_gene) * p_down_gene**2.0 + \
            (var_down_f_gene) * (1 - p_down_gene**2.0) - \
            (mu_down_f_gene - mu_down_up_gene) * (p_down_gene - p_down_gene**2.0)
        velo_mu_down_gene = (mu_down_f_gene - mu_down_gene) * gamma_mRNA_tmp_gene
        velo_var_down_gene = \
            (mu_down_f_gene * (2 * px_B3_gene + 1) + mu_down_gene - \
             2  * var_down_gene) * gamma_mRNA_tmp_gene

        # TODO: Unused variables; remove
        dist_up = self.dist_from_connecting_line(mu_0, \
                                                 (var_0 + 1e-10).pow(pow_fac), \
                                                 mu_up_ss, \
                                                 (var_up_ss + 1e-10).pow(pow_fac), \
                                                 mu_obs, \
                                                 (var_obs + 1e-10).pow(pow_fac))
        dist_up_lin = self.dist_from_connecting_line(mu_0, (var_0), \
                                                 mu_up_ss, (var_up_ss), \
                                                 mu_obs, (var_obs))
        dist_up_pred = self.dist_from_connecting_line(mu_0, \
                                                      (var_0 + 1e-10).pow(pow_fac), \
                                                      mu_up_ss, \
                                                      (var_up_ss + 1e-10).pow(pow_fac), \
                                                      mu_up, \
                                                      (var_up + 1e-10).pow(pow_fac))
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

        # TODO: Unused variables; remove
        mu_max = (mu_max_tmp + 1)
        var_max = (var_max_tmp + 1)
        dist_down = self.dist_from_connecting_line(mu_down_up, \
                                                   var_down_up, \
                                                   mu_down_f, var_down_f, mu_obs, var_obs)
        dist_down_sqrt = self.dist_from_connecting_line(mu_max - mu_down_up, \
                                                   (var_max - var_down_up + 1e-10).pow(pow_fac), \
                                                   mu_max - mu_down_f, \
                                                    (var_max - var_down_f + 1e-10).pow(pow_fac), \
                                                    mu_max - mu_obs, \
                                                    (var_max - var_obs + 1e-10).pow(pow_fac))
        dist_down_pred = self.dist_from_connecting_line(mu_max - mu_down_up, \
                                                   (var_max - var_down_up + 1e-10).pow(pow_fac), \
                                                   mu_max - mu_down_f, \
                                                    (var_max - var_down_f + 1e-10).pow(pow_fac), \
                                                    mu_max - mu_down, \
                                                    (var_max - var_down + 1e-10).pow(pow_fac))
        scale_dist_sigmoid2_full = F.softplus(scale_dist_sigmoid2) + 1
        scale_dist_sigmoid2_full = scale_dist_sigmoid2_full.repeat(px.shape[0], 1) 
        loss_match_dist_down = None
        loss_match_dist_up = None
        down_pos = (F.relu(scale_dist_sigmoid2_full * dist_down_sqrt))
        down_neg = (F.relu(-scale_dist_sigmoid2_full * dist_down_sqrt))
        loss_down_branch = None
        loss_up_branch = None
        
        # TODO: Unused variables; remove
        d2sigma_dmu2_up = None
        d2sigma_dmu2_down = None
        d2sigmaM_dmu2_up = None
        d2sigmaM_dmu2_down = None


        library_ = capture_eff.repeat(1, px_f1.shape[1])

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
            loss_match_top_corner, loss_up0_downf_match, logits_state, y_state, \
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
            mu_up_ss_gene, var_up_ss_gene
    
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
        extra_loss_fac_0: float = 0.1,
        fac_var: float = 1.0,
        loss_fac_geneCell: float = 1.0,
        loss_fac_gene: float = 1.0,
        loss_fac_prior_clust: float = 0.1,
        capture_eff: torch.Tensor = None,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: Tunable[int] = 128,
        n_latent: Tunable[int] = 10,
        n_layers: Tunable[int] = 1,
        n_states: int = 2,
        match_upf_down_up: bool = False,
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
        self.mu_max = mu_max
        self.var_max = var_max
        self.burst_B_low = torch.nn.Parameter(-1 * torch.randn(n_input))
        self.burst_F_low = torch.nn.Parameter(-1 * torch.randn(n_input))
        self.y_state0_super = y_state0_super
        self.y_state1_super = y_state1_super
        self.std_sum = std_sum
        self.max_fac = 1
        self.match_burst_params = match_burst_params
        self.extra_loss_fac = extra_loss_fac
        self.extra_loss_fac_1 = extra_loss_fac_1
        self.match_upf_down_up = match_upf_down_up
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
        self.extra_loss_fac_0 = extra_loss_fac_0
        self.fac_var = fac_var
        self.loss_fac_prior_clust = loss_fac_prior_clust

        # scale param for the variance of the normal distributions used for the 
        # likelihood of mu and var
        self.scale_unconstr = torch.nn.Parameter(-1 * torch.ones(n_input, 6))

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
            self.burst_B1 = torch.nn.Parameter(-1 * torch.randn(n_input))
            self.burst_f1 = torch.nn.Parameter(-1 * torch.randn(n_input))
        else:
            self.burst_B1 = None
            self.burst_f1 = None
        self.burst_B3 = torch.nn.Parameter(-1 * torch.randn(n_input))
        self.burst_f3 = torch.nn.Parameter(-1 * torch.randn(n_input))
        self.burst_B2 = torch.nn.Parameter(-1 * torch.randn(n_input))
        self.burst_f2 = torch.nn.Parameter(-1 * torch.randn(int(n_input)))

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
            n_input_encoder * 3,
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
            use_noise_ext=use_noise_ext
        )

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
            "batch_index": batch_index,
            "y": y,
            "cont_covs": cont_covs,
            "cat_covs": cat_covs,
            "size_factor": size_factor,
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

        # input variables for the encoder
        var_max = (self.var_max + 1) * 1
        std_reflected = torch.sqrt(var_max - std_**2.0 + 1e-10)
        mu_scaled = (mu - self.mu_center) / self.mu_scale
        std_scaled = (std_ - self.std_center) / self.std_scale
        std_ref_scaled = (std_reflected - self.std_ref_center) / self.std_ref_scale
        x_ = (torch.cat(((mu_scaled), \
                        (std_scaled), \
                        (std_ref_scaled)), dim=-1))
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
    def generative(
        self,
        z,
        library,
        mu_obs, 
        var_obs,
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
            
        # get the scale parameters for the normal distributions used for the likelihood
        scale_ = (self.scale_unconstr)
        scale_ = scale_[: self.n_input, :].expand(z.shape[0], self.n_input, 6)
        scale_1 = (F.softplus(scale_[..., 0]) + 1e-8).sqrt()
        scale_2 = (F.softplus(scale_[..., 1]) + 1e-8).sqrt()
        scale_3 = (F.softplus(scale_[..., 2]) + 1e-8).sqrt()
        scale_4 = (F.softplus(scale_[..., 3]) + 1e-8).sqrt()
        scale_5 = (F.softplus(scale_[..., 4]) + 1e-8).sqrt()
        scale_6 = (F.softplus(scale_[..., 5]) + 1e-8).sqrt()

        # forward pass through the decoder
        px_time_rate_tmp, \
            gamma_mRNA_tmp, library_, px_time_rate_rep_tmp, \
            px_B1, px_f1, mu_down_up, var_down_up, state_pred, mu_up, var_up, \
            mu_up_f, var_up_f, mu_down, var_down, mu_down_f, var_down_f, \
            velo_mu_up, velo_var_up, \
            velo_mu_up_ss, velo_var_up_ss, velo_mu_down, velo_var_down, \
            px_pi, \
            mu_up_ss, var_up_ss, loss_distance_cell, id_ss_up, id_ss_down, \
            p_to_up_f, _, \
            loss_down_up_to_upper, \
            loss_down_f_to_upper, loss_up_branch, loss_down_branch, \
            time_ss_full, loss_match_top_corner, \
            loss_up0_downf_match, logits_state, y_state, \
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
            velo_mu_down_gene, velo_var_down_gene, mu_up_ss_gene, var_up_ss_gene = \
            self.decoder(
                decoder_input,
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
                scale_3,
                self.fac_var,
                batch_index,
                *categorical_input,
                y,
            )
        
        # categorical distribution for the states (upregulated and downregulated branches)
        comp_dist = Categorical(probs=px_pi)
        
        def get_mu_std_scaled_loss(mu_0, std_0, mu_1, std_1, mu_center, mu_scale, \
                              std_center, std_scale, scale_mu, scale_std):
            mu_0_scaled = (mu_0 - mu_center) / mu_scale
            mu_1_scaled = (mu_1 - mu_center) / mu_scale
            std_0_scaled = (std_0 - std_center) / std_scale
            std_1_scaled = (std_1 - std_center) / std_scale

            loss_mu = (((mu_0_scaled - mu_1_scaled)**2.0) / (2 * scale_mu**2.0)).sum(-1)
            loss_std = (((std_0_scaled - std_1_scaled)**2.0) / (2 * scale_std**2.0)).sum(-1)
            loss_total = (loss_mu + loss_std).mean()
            return loss_total
        
        # match mu and var achievable at switch time for the upregulated branch
        # to the observed mu and var near the corner where var is maximum
        if self.mu_ss_obs is not None:
            loss_match_mu_var_switch_1 = \
                get_mu_std_scaled_loss(mu_up_ss_gene, torch.sqrt(var_up_ss_gene + 1e-10), \
                                        self.mu_ss_obs[None, ...], \
                                        torch.sqrt(self.var_ss_obs[None, ...] + 1e-10), \
                                        self.mu_center, self.mu_scale, \
                                        self.std_center, self.std_scale, scale_1, scale_3)
            if self.match_burst_params:
                loss_match_mu_var_switch_2 = \
                    get_mu_std_scaled_loss(mu_up_ss, torch.sqrt(var_up_ss + 1e-10), \
                                            self.mu_ss_obs[None, ...], \
                                            torch.sqrt(self.var_ss_obs[None, ...] + 1e-10), \
                                            self.mu_center, self.mu_scale, \
                                            self.std_center, self.std_scale, scale_1, scale_3)
            else:
                loss_match_mu_var_switch_2 = 0.0
            loss_match_mu_var_switch = \
                (loss_match_mu_var_switch_1 + loss_match_mu_var_switch_2)
        else:
            loss_match_mu_var_switch = 0.0

        # match mu and var for the initial steady-state of the downregulated branch
        # to the observed mu and var near the corner where mu is maximum
        if self.mu_mean_obs is not None:            
            # match mu_down_up
            if not self.match_upf_down_up:
                loss_match_mu_var_switch_1_ = \
                    get_mu_std_scaled_loss(mu_down_up_gene, torch.sqrt(var_down_up_gene + 1e-10), \
                                            self.mu_mean_obs[None, ...], \
                                            torch.sqrt(self.var_mean_obs[None, ...] + 1e-10), \
                                            self.mu_center, self.mu_scale, \
                                            self.std_center, self.std_scale, scale_1, scale_3)
                if self.match_burst_params:
                    loss_match_mu_var_switch_2_ = \
                        get_mu_std_scaled_loss(mu_down_up, torch.sqrt(var_down_up + 1e-10), \
                                                self.mu_mean_obs[None, ...], \
                                                torch.sqrt(self.var_mean_obs[None, ...] + 1e-10), \
                                                self.mu_center, self.mu_scale, \
                                                self.std_center, self.std_scale, scale_1, scale_3)
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
        mu_0 = px_f1 * px_B1 / gamma_mRNA_tmp
        var_0 = mu_0 * (px_B1 + 1)
        
        # gene-cell-specific probabilites for being in the upregulated (state 0) 
        # or downregulated (state 1) branches
        p_up_branch = px_pi[..., 0]
        p_down_branch = px_pi[..., 1]

        # average mu and var using the gene-cell-specific probabilities
        mu_ = mu_up * p_up_branch + mu_down * p_down_branch
        var_ = var_up * p_up_branch + var_down * p_down_branch
        mu_gene = mu_up_gene * p_up_branch + mu_down_gene * p_down_branch
        var_gene = var_up_gene * p_up_branch + var_down_gene * p_down_branch

        # gene-cell-specific: center and scale the time-dependent mu and std for 
        # the up- and down-regulated branches
        mu_up_scaled = (mu_up - self.mu_center) / self.mu_scale
        mu_down_scaled = (mu_down - self.mu_center) / self.mu_scale
        std_up_scaled = (torch.sqrt(var_up + 1e-10) - self.std_center) / \
            self.std_scale
        std_down_scaled = (torch.sqrt(var_down + 1e-10) - self.std_center) / \
            self.std_scale
        
        # gene-specific: center and scale the time-dependent mu and std for
        # the up- and down-regulated branches
        mu_up_gene_scaled = (mu_up_gene - self.mu_center) / self.mu_scale
        mu_down_gene_scaled = (mu_down_gene - self.mu_center) / self.mu_scale
        std_up_gene_scaled = (torch.sqrt(var_up_gene + 1e-10) - self.std_center) / \
            self.std_scale
        std_down_gene_scaled = (torch.sqrt(var_down_gene + 1e-10) - self.std_center) / \
            self.std_scale
        
        # stack mu and std for the up- and down-regulated branches
        mu_stacked = torch.stack(
            (
                mu_up_scaled,
                mu_down_scaled
            ),
            dim=2
        )
        scale_mu_stacked = torch.stack(
            (
                scale_1,
                scale_1
            ),
            dim=2
        )
        std_stacked = torch.stack(
            (
                std_up_scaled,
                std_down_scaled
            ),
            dim=2
        )
        scale_std_stacked = torch.stack(
            (
                scale_3,
                scale_3
            ),
            dim=2
        )

        # stack reflected std for the up- and down-regulated branches
        var_max = (self.var_max + 1)
        std_ref_up_stacked = \
            (torch.sqrt(var_max - var_up + 1e-10) - self.std_ref_center) / \
            self.std_ref_scale
        std_ref_down_stacked = \
            (torch.sqrt(var_max - var_down + 1e-10) - self.std_ref_center) / \
            self.std_ref_scale
        std_ref_stacked = torch.stack(
            (
                std_ref_up_stacked,
                std_ref_down_stacked
            ),
            dim=2
        )
        scale_std_ref_stacked = torch.stack(
            (
                scale_4,
                scale_4
            ),
            dim=2
        )

        # stack gene-specific mu, std, and reflected std for the up- and down-regulated branches
        mu_gene_stacked = torch.stack(
            (
                mu_up_gene_scaled,
                mu_down_gene_scaled
            ),
            dim=2
        )
        scale_mu_gene_stacked = torch.stack(
            (
                scale_1,
                scale_1
            ),
            dim=2
        )
        std_gene_stacked = torch.stack(
            (
                std_up_gene_scaled,
                std_down_gene_scaled
            ),
            dim=2
        )
        scale_std_gene_stacked = torch.stack(
            (
                scale_3,
                scale_3
            ),
            dim=2
        )
        var_max = (self.var_max + 1)
        std_ref_up_gene_stacked = \
            (torch.sqrt(var_max - var_up_gene + 1e-10) - self.std_ref_center) / \
            self.std_ref_scale
        std_ref_down_gene_stacked = \
            (torch.sqrt(var_max - var_down_gene + 1e-10) - self.std_ref_center) / \
            self.std_ref_scale
        std_ref_gene_stacked = torch.stack(
            (
                std_ref_up_gene_stacked,
                std_ref_down_gene_stacked
            ),
            dim=2
        )
        scale_std_ref_gene_stacked = torch.stack(
            (
                scale_4,
                scale_4
            ),
            dim=2
        )

        loss_fac_div = 1
        # likelihood and loss for mu using the gene-cell-specific params
        dist_mu = Normal((mu_stacked / loss_fac_div), scale_mu_stacked)
        px_mu_geneCell = MixtureSameFamily(comp_dist, dist_mu)
        mu_obs_scaled = (mu_obs - self.mu_center) / self.mu_scale
        loss_mu_geneCell = (-px_mu_geneCell.log_prob((mu_obs_scaled / loss_fac_div)))

        # likelihood and loss for mu using the gene-specific params
        dist_mu_gene = Normal((mu_gene_stacked / loss_fac_div), scale_mu_gene_stacked)
        px_mu_gene = MixtureSameFamily(comp_dist, dist_mu_gene)
        mu_obs_scaled = (mu_obs - self.mu_center) / self.mu_scale
        loss_mu_gene = (-px_mu_gene.log_prob((mu_obs_scaled / loss_fac_div)))

        # total loss for mu
        loss_mu = ((self.loss_fac_geneCell * loss_mu_geneCell + \
                    self.loss_fac_gene * loss_mu_gene) / \
                    (self.loss_fac_geneCell + self.loss_fac_gene)).sum(-1)

        # likelihood and loss for std using the gene-cell-specific params
        var_max = (self.var_max + 1)
        std_max = torch.sqrt(var_max)
        # px_std_up = Normal(torch.sqrt(var_ + 1e-10), scale_3)
        dist_std = Normal((std_stacked / loss_fac_div), scale_std_stacked)
        px_std_geneCell = MixtureSameFamily(comp_dist, dist_std)
        std_obs_scaled = (torch.sqrt(var_obs + 1e-10) - self.std_center) / self.std_scale
        loss_std_geneCell = \
            (-px_std_geneCell.log_prob(std_obs_scaled / loss_fac_div))
        
        # likelihood and loss for std using the gene-specific params
        dist_std_gene = Normal((std_gene_stacked / loss_fac_div), scale_std_gene_stacked)
        px_std_gene = MixtureSameFamily(comp_dist, dist_std_gene)
        std_obs_scaled = (torch.sqrt(var_obs + 1e-10) - self.std_center) / self.std_scale
        loss_std_gene = \
            (-px_std_gene.log_prob(std_obs_scaled / loss_fac_div))

        # likelihood and loss for reflected std using the gene-cell-specific params
        dist_std_ref = Normal((std_ref_stacked / loss_fac_div), scale_std_ref_stacked)
        px_std_ref_geneCell = MixtureSameFamily(comp_dist, dist_std_ref)
        std_ref_obs_scaled = (torch.sqrt(var_max - var_obs + 1e-10) - self.std_ref_center) / \
            self.std_ref_scale
        loss_std_ref_geneCell = \
            (-px_std_ref_geneCell.log_prob(std_ref_obs_scaled / loss_fac_div))
        
        # likelihood and loss for reflected std using the gene-specific params
        dist_std_ref_gene = Normal((std_ref_gene_stacked / loss_fac_div), scale_std_ref_gene_stacked)
        px_std_ref_gene = MixtureSameFamily(comp_dist, dist_std_ref_gene)
        std_ref_obs_scaled = (torch.sqrt(var_max - var_obs + 1e-10) - self.std_ref_center) / \
            self.std_ref_scale
        loss_std_ref_gene = \
            (-px_std_ref_gene.log_prob(std_ref_obs_scaled / loss_fac_div))

        # total loss for std
        loss_std = ((self.loss_fac_geneCell * loss_std_geneCell + \
                    self.loss_fac_gene * loss_std_gene) / \
                    (self.loss_fac_geneCell + self.loss_fac_gene) + \
                    (self.loss_fac_geneCell * loss_std_ref_geneCell + \
                    self.loss_fac_gene * loss_std_ref_gene) / \
                    (self.loss_fac_geneCell + self.loss_fac_gene)).sum(-1) / 2

        # total reconstruction loss
        reconst_loss = loss_mu + loss_std

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
            "pl": pl,
            "pz": pz,
            "n_t" : mu_,
            "var" : var_, 
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
            "scale" : scale_[..., 0],
            "scale_var" : scale_[..., 1],
            "n_t_gene" : mu_gene,
            "var_gene" : var_gene, 
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
            "logits_state" : logits_state,
            "y_state" : y_state,
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
        }

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ):
        """Computes the loss function for the model."""
        x = tensors[REGISTRY_KEYS.X_KEY]
        logits_state = generative_outputs['logits_state']
        px_pi = generative_outputs['px_pi']
        reconst_loss = generative_outputs['reconst_loss']
        loss_match_mu_var_up_f = generative_outputs['loss_match_mu_var_up_f']
        loss_match_mu_var_down_up = generative_outputs['loss_match_mu_var_down_up']
        loss_match_mu_var_down_f = generative_outputs['loss_match_mu_var_down_f']
        loss_match_mu_var_0 = generative_outputs['loss_match_mu_var_0']
        loss_match_mu_var_switch = generative_outputs['loss_match_mu_var_switch']
        loss_match_mu_var_switch_ = generative_outputs['loss_match_mu_var_switch_']

        # KL divergence loss for the latent variable z
        pz_prior = generative_outputs["pz"]
        kl_divergence_z = kl(inference_outputs["qz"], \
                             pz_prior).sum(
            dim=1
        )

        kl_divergence_l = torch.tensor(0.0, device=x.device)

        # KL divergence loss for state assignment
        kl_pi = \
            (-entropy_no_mean(logits_state, px_pi) - \
            np.log(1 / self.n_states)).sum(-1)

        kl_local_for_warmup = kl_divergence_z
        kl_local_no_warmup = 0 * kl_divergence_l
        # kl_local = kl_divergence_z + kl_pi
        weighted_kl_local = kl_weight * kl_local_for_warmup + \
            kl_local_no_warmup + kl_pi
        
        if self.match_burst_params:
            loss = torch.mean(reconst_loss + weighted_kl_local) + \
                self.extra_loss_fac * (loss_match_mu_var_up_f) + \
                self.extra_loss_fac_0 * (loss_match_mu_var_down_up) + \
                self.extra_loss_fac_1 * (loss_match_mu_var_down_f + loss_match_mu_var_0) + \
                self.extra_loss_fac_2 * (loss_match_mu_var_switch + \
                                         loss_match_mu_var_switch_)
        else:
            loss = torch.mean(reconst_loss + weighted_kl_local) + \
                self.extra_loss_fac_2 * (loss_match_mu_var_switch + \
                                         loss_match_mu_var_switch_)

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
    
