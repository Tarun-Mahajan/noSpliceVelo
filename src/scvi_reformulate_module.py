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
from torch.distributions import Normal, MixtureSameFamily
from torch.distributions import kl_divergence as kl
from torch.distributions import Categorical, Dirichlet

from scvi import REGISTRY_KEYS
from scvi.autotune._types import Tunable
from scvi.data._constants import ADATA_MINIFY_TYPE
from scvi.distributions import NegativeBinomial, Poisson, ZeroInflatedNegativeBinomial
from scvi.module.base import BaseMinifiedModeModuleClass, LossOutput, auto_move_data
from scvi.nn import Encoder, DecoderSCVI, LinearDecoderSCVI, one_hot
from distributions import NegativeBinomialNew

torch.backends.cudnn.benchmark = True

import logging
# from typing import List, Literal, Optional
from typing import Dict, Iterable, Literal, Optional, Sequence, Union, Optional, List

import numpy as np

from scvi import REGISTRY_KEYS
from scvi.data._constants import ADATA_MINIFY_TYPE
from scvi.nn import FCLayers
from layers import GumbelSoftmax
from losses import entropy


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
        # n_gene_clusts: int = 2,
        n_gene_types: int = 2,
        n_clones: int = 4,
        gene_clusts: Iterable[torch.Tensor] = None,
        # n_gene_states: Iterable[str] = ['up-down', 'down-up'],
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        distribution: str = "normal",
        var_eps: float = 1e-4,
        var_activation: Optional[Callable] = None,
        return_dist: bool = False,
        edge_index: Iterable[torch.Tensor] = None,
        edge_attr: Iterable[torch.Tensor] = None,
        clusters_exclusive: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.hid = 8
        self.in_head = 8
        self.hid_1 = 4
        self.in_head_1 = 4
        self.out_head = 1
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.gene_clusts = gene_clusts
        self.clusters_exclusive = clusters_exclusive

        self.distribution = distribution
        self.var_eps = var_eps
        self.encoder_list = nn.ModuleList()
        # for n_ in range(self.num_gene_clusts):
        # self.encoder_list.append(
        #     FCLayers(
        #         n_in=int(n_input),
        #         n_out=n_hidden,
        #         n_cat_list=n_cat_list,
        #         n_layers=n_layers,
        #         n_hidden=n_hidden,
        #         dropout_rate=dropout_rate,
        #         **kwargs,
        #     )
        # )
        self.encoder = FCLayers(
            n_in=int(n_input),
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            **kwargs,
        )

        # self.conv1_list = nn.ModuleList()
        # for n_ in range(self.num_gene_clusts):
        #     self.conv1_list.append(GATv2Conv(n_hidden, \
        #                                     self.hid, \
        #                                 heads=self.in_head, \
        #                                     dropout=0.6, \
        #                                     edge_dim=1))
            
        # self.conv2_list = nn.ModuleList()
        # for n_ in range(self.num_gene_clusts):
        #     self.conv2_list.append(GATv2Conv(self.hid*self.in_head, \
        #                                     self.hid_1, \
        #                                 heads=self.in_head_1, \
        #                                     dropout=0.6, \
        #                                     edge_dim=1))
        # self.conv1 = \
        #     GATv2Conv(n_hidden, self.hid, \
        #             heads=self.in_head, dropout=0.6)
        # self.conv2 = \
        #     GATv2Conv(self.hid*self.in_head, \
        #               self.hid_1, \
        #             heads=self.in_head_1, dropout=0.6)
        # self.mean_encoder_list = nn.ModuleList()
        # self.var_encoder_list = nn.ModuleList()
        # for n_ in range(self.num_gene_clusts):
        #     self.mean_encoder_list.append(
        #         GATv2Conv(self.hid_1*self.in_head_1, \
        #                 n_output, \
        #                 concat=False,
        #                 heads=self.out_head, \
        #                 dropout=0.6, edge_dim=1)
        #     )
        #     self.var_encoder_list.append(
        #         GATv2Conv(self.hid_1*self.in_head_1, \
        #                 n_output, \
        #                 concat=False,
        #                 heads=self.out_head, \
        #                 dropout=0.6, edge_dim=1)
        #     )

        # if self.clusters_exclusive:
        #     self.mean_encoder = \
        #         nn.Linear(n_output * self.num_gene_clusts, \
        #                   n_output)

        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
        
        # self.var_encoder = \
        #     nn.Linear(n_output * self.num_gene_clusts, \
        #               n_output)
        # self.mean_encoder = GATv2Conv(self.hid*self.in_head, \
        #                             n_output, \
        #                             concat=False,
        #                             heads=self.out_head, \
        #                             dropout=0.6)
        # self.var_encoder = GATv2Conv(self.hid*self.in_head, \
        #                             n_output, \
        #                             concat=False,
        #                             heads=self.out_head, \
        #                             dropout=0.6)

        # self.mean_encoder = nn.Linear(n_hidden, n_output)
        # self.var_encoder = nn.Linear(n_hidden, n_output)
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
        nclones: int = 4,
        n_gene_types: int = 2,
        inject_covariates: bool = True,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        scale_activation_init: Literal["softmax", "softplus", "sigmoid"] = "sigmoid",
        burst_B_gene: bool = False,
        gene_clusts: Iterable[torch.Tensor] = None,
        direction_up: torch.Tensor = None,
        edge_index: Iterable[torch.Tensor] = None,
        edge_attr: Iterable[torch.Tensor] = None,
        decoder_type: Literal["mu_var", "burst"] = "mu_var",
    ):
        super().__init__()
        self.gene_clusts = gene_clusts
        self.n_input = n_input
        self.direction_up = direction_up
        self.hid = 16
        self.in_head = 16
        self.out_head = 1
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.decoder_type = decoder_type
        # p(x|z)
        # self.px_decoder_list = nn.ModuleList()
        # for n_ in range(self.num_gene_clusts):
        #     self.px_decoder_list.append(
        #         FCLayers(
        #             n_in=n_input,
        #             n_out=n_hidden,
        #             n_cat_list=n_cat_list,
        #             n_layers=n_layers,
        #             n_hidden=n_hidden,
        #             dropout_rate=0,
        #             inject_covariates=inject_covariates,
        #             use_batch_norm=use_batch_norm,
        #             use_layer_norm=use_layer_norm,
        #         )
        #     )
        self.px_decoder = \
            FCLayers(
                n_in=n_input,
                n_out=n_hidden,
                n_cat_list=n_cat_list,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=0,
                inject_covariates=inject_covariates,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
        )
        

        self.n_output = n_output
        self.nclones = nclones
        self.n_gene_types = n_gene_types
#         n_output = int(n_output_orig / 2)
        self.burst_B_gene = burst_B_gene

        if not self.burst_B_gene:
            # self.px_scale_B_decoder_list = \
            #     nn.ModuleList()
            # for n_ in range(self.num_gene_clusts):
            #     self.px_scale_B_decoder_list.append(
            #         nn.Sequential(
            #             nn.Linear(n_hidden, len(gene_clusts[n_])),
            #         )
            #     )
            self.px_scale_B_decoder = nn.Sequential(
                nn.Linear(n_hidden, n_output),
            )

        self.px_r_final_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output),
#             nn.Sigmoid(),
        )

    
    def forward(
        self,
        dispersion: str,
        z: torch.Tensor,
        library: torch.Tensor,
        cat_covs,
        gamma_mRNA: torch.Tensor=None,
        lib_full: torch.Tensor=None,
        mu_full: torch.Tensor=None,
        var_full: torch.Tensor=None,
        tmax: float = 20,
        burst_B_: torch.Tensor = None,
        capture_eff: torch.Tensor = None,
        *cat_list: int,
    ):
        """The forward computation for a single sample.
         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``
        Parameters
        ----------
        dispersion
            One of the following
            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell
        z :
            tensor with shape ``(n_input,)``
        library_size
            library size
        cat_list
            list of category membership(s) for this sample
        Returns
        -------
        4-tuple of :py:class:`torch.Tensor`
            parameters for the ZINB distribution of expression
        """
        # print(f'yclone = {y_clone.shape}')
        # p(y_clone | y_state)
        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z, *cat_list)
        if self.decoder_type == "burst":
            if not self.burst_B_gene:
                burst_B_mat = \
                    F.softplus(self.px_scale_B_decoder(px)) + 1
            else:
                burst_B_mat = F.softplus(burst_B_) + 1
                # burst_B_mat = F.softplus(burst_B_, threshold=10) + 1
                burst_B_mat = burst_B_mat.repeat(px.shape[0], 1)
        #         px_r_final = self.px_r_final_decoder(px) * px_final_rate

            library_ = capture_eff.repeat(1, burst_B_mat.shape[1])
            px_r_final = \
                F.softplus(self.px_r_final_decoder(px)) + \
                1e-8
            mu_ = px_r_final * burst_B_mat
            var_ = mu_ * (1 + burst_B_mat) / 2
            # mu_ = px_r_final * (burst_B_mat - 1 + 1e-8)
            # var_ = mu_ * burst_B_mat
            # px_r_final = mu_ / burst_B_mat


        if self.decoder_type == "mu_var":
            mu_ = F.softplus(self.px_r_final_decoder(px)) + \
                1e-8
            # var_ = F.softplus(self.px_scale_B_decoder(px)) + \
            #     mu_
            if not self.burst_B_gene:
                phi_ = torch.exp(self.px_scale_B_decoder(px)) + \
                    1e-8
            else:
                phi_ = torch.exp(burst_B_) + 1e-8
                # burst_B_mat = F.softplus(burst_B_, threshold=10) + 1
                phi_ = phi_.repeat(px.shape[0], 1)
            # noise_ = F.softplus(self.px_scale_B_decoder(px)) + \
            #     1e-4 
            # phi_ = torch.clamp(phi_, max=10)
            var_ = mu_ * (1 + mu_ * phi_)
            # var_ = mu_ + noise_
            px_r_final = 1 / phi_
            # px_r_final = mu_ / (var_ / mu_ - 1)
            # px_r_final = mu_ / (var_ / mu_ - 1)
            # burst_B_mat = var_ / mu_ - 1
            burst_B_mat = mu_ * phi_
            # burst_B_mat = (var_ - mu_) / mu_

        if self.decoder_type == "mu_var2":
            mu_ = F.softplus(self.px_r_final_decoder(px)) + \
                1e-8
            # var_ = F.softplus(self.px_scale_B_decoder(px)) + \
            #     mu_
            if not self.burst_B_gene:
                tmp = (self.px_scale_B_decoder(px))
            else:
                tmp = (burst_B_)
                # burst_B_mat = F.softplus(burst_B_, threshold=10) + 1
                tmp = tmp.repeat(px.shape[0], 1)
            # noise_ = F.softplus(self.px_scale_B_decoder(px)) + \
            #     1e-4 
            # phi_ = torch.clamp(phi_, max=10)
            # phi_ = 1 / px_r_final
            # phi_ = torch.exp(tmp)
            # phi_ = F.softplus(tmp) + 1e-8
            px_r_final = F.softplus(tmp) + 1e-8
            phi_ =  1 / px_r_final
            var_ = mu_ * (1 + mu_ * phi_)
            # var_ = mu_ + tmp
            # px_r_final = mu_**2.0 / tmp
            # var_ = mu_ + noise_
            # px_r_final = torch.exp(-tmp)
            # px_r_final = 1 / phi_
            # px_r_final = mu_ / (var_ / mu_ - 1)
            # px_r_final = mu_ / (var_ / mu_ - 1)
            # burst_B_mat = var_ / mu_ - 1
            # burst_B_mat = mu_ * phi_
            burst_B_mat = mu_ * phi_
            # burst_B_mat = (var_ - mu_) / mu_
        library_ = capture_eff.repeat(1, px_r_final.shape[1])
        # mu_ = mu_0 * noise1
        # var_ = mu_ * (1 + burst_B_mat + mu_ * noise2)
        
        
        return px_r_final, mu_, var_, burst_B_mat, library_
    

class VAENoiseVelo(BaseMinifiedModeModuleClass):
    """Variational auto-encoder model.
    This is an implementation of the scVI model described in :cite:p:`Lopez18`.
    Parameters
    ----------
    n_input
        Number of input genes
    n_batch
        Number of batches, if 0, no batch correction is performed.
    n_labels
        Number of labels
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    n_continuous_cov
        Number of continuous covarites
    n_cats_per_cov
        Number of categories for each extra categorical covariate
    dropout_rate
        Dropout rate for neural networks
    dispersion
        One of the following
        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    gene_likelihood
        One of
        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    latent_distribution
        One of
        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)
    encode_covariates
        Whether to concatenate covariates to expression in encoder
    deeply_inject_covariates
        Whether to concatenate covariates into output of hidden layers in encoder/decoder. This option
        only applies when `n_layers` > 1. The covariates are concatenated to the input of subsequent hidden layers.
    use_batch_norm
        Whether to use batch norm in layers.
    use_layer_norm
        Whether to use layer norm in layers.
    use_size_factor_key
        Use size_factor AnnDataField defined by the user as scaling factor in mean of conditional distribution.
        Takes priority over `use_observed_lib_size`.
    use_observed_lib_size
        Use observed library size for RNA as scaling factor in mean of conditional distribution
    library_log_means
        1 x n_batch array of means of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    library_log_vars
        1 x n_batch array of variances of the log library sizes. Parameterizes prior on library size if
        not using observed library size.
    var_activation
        Callable used to ensure positivity of the variational distributions' variance.
        When `None`, defaults to `torch.exp`.
    """

    def __init__(
        self,
        n_input: int,
        capture_eff: torch.Tensor = None,
        burst_B_gene: bool = False,
        mean_up: torch.Tensor = None, 
        var_up: torch.Tensor = None,
        mean_down: torch.Tensor = None, 
        var_down: torch.Tensor = None,
        lib_full: torch.Tensor = None,
        gene_clusters = None,
        gene_clust_param: bool = False,
        pi_up_down: torch.Tensor = None,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: Tunable[int] = 128,
        n_latent: Tunable[int] = 10,
        n_layers: Tunable[int] = 1,
        edge_index: Iterable[torch.Tensor] = None,
        edge_attr: Iterable[torch.Tensor] = None,
        gene_clusts: Iterable[torch.Tensor] = None,
        clusters_exclusive: bool = False,
        decoder_type: Literal["mu_var", "burst"] = "mu_var",
        nclones: int = 4,
        n_gene_types: int = 2,
        n_neighbors: int = 10,
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
        tmax: float = 20,
        direction_up: torch.Tensor = None,
    ):
        super().__init__()
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        # Automatically deactivate if useless
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates
        self.tmax = tmax
        self.capture_eff = capture_eff
        self.burst_B_gene = burst_B_gene
        self.dirichlet_concentration = 0.5
        self.penalty_scale = 0.2
        self.mean_up = mean_up
        self.var_up = var_up
        self.mean_down = mean_down
        self.var_down = var_down
        self.mu_full = mean_down
        self.var_full = var_down
        self.nclones = nclones
        self.n_gene_types = n_gene_types
        self.gene_clust_param = gene_clust_param
        self.gene_clusts = gene_clusts
        self.n_neighbors = n_neighbors
        self.lib_full = lib_full
        self.decoder_type = decoder_type

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
        
        size_ = int(n_input)

        if self.burst_B_gene:
            self.burst_B_ = torch.nn.Parameter(-1 * torch.randn(size_))
        else:
            self.burst_B_ = None
        self.gene_clusters = gene_clusters

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        # n_input_encoder = n_input + n_continuous_cov * encode_covariates
        # cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        n_input_encoder = n_input
        cat_list = [n_batch] + list([])
        encoder_cat_list = cat_list if encode_covariates else None
        self.z_encoder = EncoderNew(
            n_input_encoder,
            n_latent,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
            edge_index=edge_index,
            edge_attr=edge_attr,
            gene_clusts=gene_clusts,
            clusters_exclusive=clusters_exclusive,
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
        # decoder goes from n_latent-dimensional space to n_input-d data
#         n_input_decoder = n_latent + n_continuous_cov
        n_input_decoder = n_latent
#         self.decoder = DecoderNoiseVelo(
#             n_input_decoder,
#             n_input,
#             n_cat_list=cat_list,
#             n_layers=n_layers,
#             n_hidden=n_hidden,
#             inject_covariates=deeply_inject_covariates,
#             use_batch_norm=use_batch_norm_decoder,
#             use_layer_norm=use_layer_norm_decoder,
#             scale_activation_init="sigmoid" if use_size_factor_key else "softmax",
#         )
        
        self.decoder = DecoderNoiseVelo(
            n_input_decoder,
            n_input,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            scale_activation_init="sigmoid",
            burst_B_gene=burst_B_gene,
            gene_clusts=gene_clusts,
            direction_up=direction_up,
            edge_index=edge_index,
            edge_attr=edge_attr,
            decoder_type=decoder_type,
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
            input_dict = {
                "x": x,
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
        self, x, batch_index, cont_covs=None, cat_covs=None, n_samples=1
    ):
        """High level inference method.
        Runs the inference (encoder) model.
        """
        x_ = x
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)

        # if cont_covs is not None and self.encode_covariates:
        #     encoder_input = torch.cat((x_, cont_covs), dim=-1)
        # else:
        #     encoder_input = x_
        encoder_input = x_
        # if cat_covs is not None and self.encode_covariates:
        #     categorical_input = torch.split(cat_covs, 1, dim=1)
        # else:
        #     categorical_input = ()

        categorical_input = ()

        qz, z = \
            self.z_encoder(encoder_input, batch_index, *categorical_input)
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
    def get_nb_params(self, px_r_final, \
              px_init_rate, px_final_rate, px_time_rate, \
              px_var_init, gamma_mRNA_):
#             gamma_mRNA_tmp = torch.exp(gamma_mRNA.clone())
#             gamma_mRNA_tmp = gamma_mRNA_tmp.repeat(px_init_rate.shape[0], 1)
#             print(f'px_init_rate = {px_init_rate}')
#             print(f'px_final_rate = {px_final_rate}')
        mu_ = px_init_rate * torch.exp(-gamma_mRNA_ * px_time_rate) + \
            px_final_rate * (1 - torch.exp(-gamma_mRNA_ * px_time_rate))
        burstB_ = (px_final_rate) / (px_r_final)
#             print(f'burstB_ = {burstB_}')
#             var_ = px_final_rate * (burstB_ + 1) * (1 - torch.exp(-2 * gamma_mRNA_ * px_time_rate)) + \
#                 px_var_init * torch.exp(-2 * gamma_mRNA_ * px_time_rate) + \
#                 (px_init_rate - px_final_rate) * torch.exp(-gamma_mRNA_ * px_time_rate) * \
#                 (1 - torch.exp(-gamma_mRNA_ * px_time_rate))
        var_ = px_final_rate * (burstB_ + 1) * (1 - torch.exp(-2 * gamma_mRNA_ * px_time_rate)) + \
            px_var_init * torch.exp(-2 * gamma_mRNA_ * px_time_rate) + \
            (px_init_rate - px_final_rate) * torch.exp(-gamma_mRNA_ * px_time_rate) * \
            (1 - torch.exp(-gamma_mRNA_ * px_time_rate))
#             var_ = px_final_rate * (burstB_ - 1) * (1 - torch.exp(-2 * gamma_mRNA_ * px_time_rate)) + \
#                 (px_var_init - px_init_rate) * torch.exp(-2 * gamma_mRNA_ * px_time_rate) + \
#                 mu_
#             var_ = px_final_rate * (burstB_ - 1) * (1 - torch.exp(-2 * gamma_mRNA_ * px_time_rate)) + \
#                 (px_var_init) * torch.exp(-2 * gamma_mRNA_ * px_time_rate) + \
#                 px_rate * (1 - torch.exp(-gamma_mRNA_ * px_time_rate))  + \
#                 px_final_rate * (1 - torch.exp(-gamma_mRNA_ * px_time_rate)) * \
#                 torch.exp(-gamma_mRNA_ * px_time_rate)

#             var_1 = px_final_rate * burstB_ * (1 - torch.exp(-2 * gamma_mRNA_tmp * px_time_rate))
#             var_ = torch.clamp(var_, max=1e4)
        theta_ = mu_ / ((var_ / (mu_)) - 1 + 1e-5) + 1e-4
        # theta_[(mu_ == 0)] = 1e-2
        return mu_, theta_, var_

    @auto_move_data
    def generative(
        self,
        z,
        library,
        batch_index,
        cont_covs=None,
        cat_covs=None,
        size_factor=None,
        y=None,
        transform_batch=None,
        device_='cpu'
    ):
        """Runs the generative model."""
        # TODO: refactor forward function to not rely on y
        # Likelihood distribution
#         if cont_covs is None:
#             decoder_input = z
#         elif z.dim() != cont_covs.dim():
#             decoder_input = torch.cat(
#                 [z, cont_covs.unsqueeze(0).expand(z.size(0), -1, -1)], dim=-1
#             )
#         else:
#             decoder_input = torch.cat([z, cont_covs], dim=-1)
        decoder_input = z

#         if cat_covs is not None:
#             categorical_input = torch.split(cat_covs, 1, dim=1)
#         else:
#             categorical_input = ()
        categorical_input = ()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        if not self.use_size_factor_key:
            size_factor = library
        # torch.autograd.set_detect_anomaly(True)
            
    
#         px_init_scale, px_final_scale, px_r_final, px_init_rate, px_final_rate, px_time_rate, \
#             px_var_init, px_dropout, gamma_mRNA_tmp = \
#             self.decoder(
#                 self.dispersion,
#                 decoder_input,
#                 size_factor,
#                 self.gamma_mRNA,
#                 self.tmax,
#                 batch_index,
#                 *categorical_input,
#                 y,
#             )


#         px_scale, px_init_scale_1, px_final_scale, px_r_final, px_init_rate, \
#             px_final_rate, px_time_rate, \
#             px_var_init, px_dropout, gamma_mRNA_tmp, px_rate, px_r = \
#             self.decoder(
#                 self.dispersion,
#                 decoder_input,
#                 size_factor,
#                 self.gamma_mRNA,
#                 self.tmax,
#                 batch_index,
#                 *categorical_input,
#                 y,
#             )

        px_r_final, mu_, var_, burst_B_mat, library_ = \
            self.decoder(
                self.dispersion,
                decoder_input,
                size_factor,
                cat_covs,
                None,
                self.lib_full,
                self.mu_full,
                self.var_full,
                self.tmax,
                self.burst_B_,
                cont_covs,
                batch_index,
                *categorical_input,
                y,
            )
        

        # mu_tmp = mu_.clone()
        # mu_tmp = mu_tmp.detach()
        # print(torch.max(mu_tmp))
        # print(torch.max(theta_))
        # print(f'total cells = {num_cells}')
        # var_ = mu_ * (1 + mu_ / theta_)
        # theta_ = torch.clamp(theta_, min=1e-4, max=1e6)
        # mu_ = torch.clamp(mu_, min=1e-2, max=1e4)
        # px = NegativeBinomial(mu=mu_ * library_, \
        #                       theta=theta_, \
        #                       scale=1 / (theta_ + 1e-4))
        # theta_ = mean_down / (var_down / mean_down - 1 + 1e-5)
        # theta_ = px_r_final.clone()
        theta_ = mu_ / ((var_ / mu_) - 1 + 1e-5) + 1e-4
        # if self.decoder_type == "mu_var":
        #     px = NegativeBinomial(mu=mu_ * library_, \
        #                         theta=theta_, \
        #                         scale=1 / (theta_ + 1e-4))
        # if self.decoder_type == "burst":
        #     px = NegativeBinomial(mu=mu_ * library_, \
        #                         theta=theta_, \
        #                         scale=1 / (theta_ + 1e-4))
            
        px = NegativeBinomial(mu=mu_ * library_, \
                              theta=theta_, \
                              scale=1 / (theta_ + 1e-4))
        # px = Normal(loc=mu_, scale=var_.sqrt())

        # px_B = burst_B_mat
        # px_f = px_r_final * gamma_mRNA_tmp
        # print(len(theta_[theta_ <= 0]))

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
        # pz = Normal(self.z_prior, torch.ones_like(z))
        
        return {
            "px": px,
            "pl": pl,
            "pz": pz,
            "px_r_final" : px_r_final,\
            "burst_B" : burst_B_mat,\
            "mu" : mu_, \
            "theta" : theta_,
            "var" : var_,
            "library" : library_,
            
        }

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1,
    ):
        """Computes the loss function for the model."""
        x = tensors[REGISTRY_KEYS.X_KEY]
        pz_prior = generative_outputs["pz"]

        # pz_prior = Normal(z_prior, torch.ones_like(z_prior))
        # kl_divergence_z = kl(inference_outputs["qz"], generative_outputs["pz"]).sum(
        #     dim=1
        # )
        kl_divergence_z = kl(inference_outputs["qz"], \
                             pz_prior).sum(
            dim=1
        )
        kl_divergence_z[torch.isnan(kl_divergence_z)] = 0.0
        if not self.use_observed_lib_size:
            kl_divergence_l = kl(
                inference_outputs["ql"],
                generative_outputs["pl"],
            ).sum(dim=1)
        else:
            kl_divergence_l = torch.tensor(0.0, device=x.device)
        # tmp = generative_outputs["px"].scale
        # print(len(tmp[tmp <= 0]))

        reconst_loss = -generative_outputs["px"].log_prob(x).sum(-1)

        kl_local_for_warmup = kl_divergence_z
        kl_local_no_warmup = kl_divergence_l

        weighted_kl_local = kl_weight * kl_divergence_z + \
            kl_local_no_warmup
        weighted_kl_local[torch.isnan(weighted_kl_local)] = 0.0
        
        def loss_diff(x, y):
            return(torch.abs(x - y))
        

        # loss = torch.mean(reconst_loss + weighted_kl_local + \
        #     loss_cosine_mu + loss_cosine_var)
        loss = torch.mean(reconst_loss + \
                          weighted_kl_local)
        # loss = torch.mean(weighted_kl_local + \
        #     torch.tensor(1.0) * (loss_cosine_mu + loss_cosine_var) + \
        #     torch.tensor(1.0) * (loss_mu + loss_var))
        # loss = torch.mean(0 * reconst_loss + weighted_kl_local) + \
        #     torch.sum(mean_loss + var_loss)
        # loss = torch.mean(weighted_kl_local + \
        #                   mean_loss + var_loss)

        kl_local = {
            "kl_divergence_l": kl_divergence_l,
            "kl_divergence_z": kl_divergence_z,
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
    
