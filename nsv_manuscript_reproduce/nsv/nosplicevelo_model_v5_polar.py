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

import logging
# from typing import List, Literal, Optional
from typing import Dict, Iterable, Literal, Optional, \
    Sequence, Union, Optional, List, Tuple

import numpy as np
from anndata import AnnData
import torch.nn.functional as F

# from scvi import REGISTRY_KEYS
from constants_tmp import REGISTRY_KEYS
from scvi._types import MinifiedDataType
from scvi.data import AnnDataManager
from scvi.data._constants import _ADATA_MINIFY_TYPE_UNS_KEY, ADATA_MINIFY_TYPE
from scvi.data._utils import _get_adata_minify_type
from scvi.data.fields import (
    BaseAnnDataField,
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
    ObsmField,
    StringUnsField,
)
from scvi.model._utils import _init_library_size
from scvi.model.base import UnsupervisedTrainingMixin
from scvi.model.utils import get_minified_adata_scrna
from scvi.module import VAE
from scvi.utils import setup_anndata_dsp
from scvi.distributions import NegativeBinomial

from scvi.model.base import ArchesMixin, BaseMinifiedModeModelClass, RNASeqMixin, VAEMixin
from scvi.nn import FCLayers
from nosplicevelo_module_v5_polar import VAENoiseVelo
import torch
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import ttest_ind
DEVICE_ = 'cuda'


class noSpliceVelo(
    RNASeqMixin,
    VAEMixin,
    ArchesMixin,
    UnsupervisedTrainingMixin,
    BaseMinifiedModeModelClass,
):
    """add docstring
    """

    _module_cls = VAENoiseVelo

    def __init__(
        self,
        adata: AnnData,
        gamma_mRNA: torch.Tensor = None,
        use_gamma_mRNA: bool = False,
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
        gene_recont_weights: torch.Tensor = None,
        state_times_unique: torch.Tensor = None,
        use_library_time_correction:bool = False,
        use_two_rep: bool = False,
        use_splicing:bool = False,
        use_time_cell:bool = False,
        use_tr_gene: bool = False,
        use_alpha_gene: bool = False,
        match_upf_down_up: bool = False,
        capture_eff: torch.Tensor = None,
        use_noise_ext: bool = False,
        mu_neighbors: torch.Tensor = None,
        var_neighbors: torch.Tensor = None,
        mu_ss_obs: torch.Tensor = None,
        var_ss_obs: torch.Tensor = None,
        std_sum: torch.Tensor = None,
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
        mu_max: torch.Tensor = None,
        var_max: torch.Tensor = None,
        y_state0_super: torch.Tensor = None,
        y_state1_super: torch.Tensor = None,
        mat_where_prev: torch.Tensor = None,
        mat_where_next: torch.Tensor = None,
        prior_branch_assignment: torch.Tensor = None,
        extra_loss_fac: float = 0.5,
        extra_loss_fac_1: float = 0.5,
        extra_loss_fac_2: float = 0.1,
        extra_loss_fac_2_: float = 0.1,
        extra_loss_fac_0: float = 0.1,
        extra_loss_fac_3: float = 0.1,
        edge_loss_fac: float = 1.0,
        loss_fac_geneCell: float = 1.0,
        loss_fac_gene: float = 1.0,
        loss_fac_prior_clust: float = 0.1,
        fac_var: float = 1.0,
        tmax: int = 12, 
        t_d: float = 13,
        t_r: float = 6.5,
        sample_prob: float = 0.4,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        n_states: int = 4,
        use_loss_burst: bool = False,
        match_burst_params: bool = True,
        match_burst_params_not_muVar: bool = False,
        cluster_states: bool = False,
        state_loss_type: str = 'cross-entropy',
        states_vec: torch.Tensor = None,
        state_time_max_vec: torch.Tensor = None,
        use_controlBurst_gene: bool = False,
        timing_relative: bool = False,
        timing_relative_mat_bin: torch.Tensor = None,
        dropout_rate: float = 0.4,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        **model_kwargs,
    ):
        super().__init__(adata)

        self.cluster_states = cluster_states
        self.state_loss_type = state_loss_type
        self.loss_fac_geneCell = loss_fac_geneCell

        n_cats_per_cov = (
            self.adata_manager.get_state_registry(
                REGISTRY_KEYS.CAT_COVS_KEY
            ).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        n_batch = self.summary_stats.n_batch
        use_size_factor_key = (
            REGISTRY_KEYS.SIZE_FACTOR_KEY in self.adata_manager.data_registry
        )
        library_log_means, library_log_vars = None, None
        if not use_size_factor_key and self.minified_data_type is None:
            library_log_means, library_log_vars = _init_library_size(
                self.adata_manager, n_batch
            )

        self.module = self._module_cls(
            n_input=self.summary_stats.n_vars,
            gamma_mRNA=gamma_mRNA,
            use_gamma_mRNA=use_gamma_mRNA,
            use_two_rep=use_two_rep,
            use_splicing=use_splicing,
            use_time_cell=use_time_cell,
            use_tr_gene=use_tr_gene,
            use_alpha_gene=use_alpha_gene,
            use_prior_parabola=use_prior_parabola,
            w_theta=w_theta,
            burst_B_gene=burst_B_gene,
            burst_f_gene=burst_f_gene,
            burst_f_updown=burst_f_updown,
            burst_B_updown=burst_B_updown,
            burst_f_updown_next=burst_f_updown_next,
            burst_B_updown_next=burst_B_updown_next,
            burst_f_next=burst_f_next,
            burst_B_next=burst_B_next,
            burst_f_previous = burst_f_previous,
            burst_B_previous = burst_B_previous,
            match_burst_params=match_burst_params,
            match_burst_params_not_muVar=match_burst_params_not_muVar,
            state_times_unique=state_times_unique,
            use_library_time_correction=use_library_time_correction,
            match_upf_down_up=match_upf_down_up,
            capture_eff=capture_eff,
            mu_neighbors=mu_neighbors,
            var_neighbors=var_neighbors,
            mu_ss_obs=mu_ss_obs,
            var_ss_obs=var_ss_obs,
            mu_std=mu_std,
            var_std=var_std,
            mu_center=mu_center,
            mu_scale=mu_scale,
            std_center=std_center,
            std_scale=std_scale,
            std_ref_center=std_ref_center,
            std_ref_scale=std_ref_scale,
            mu_mean_obs=mu_mean_obs,
            var_mean_obs=var_mean_obs,
            extra_loss_fac=extra_loss_fac,
            extra_loss_fac_1=extra_loss_fac_1,
            extra_loss_fac_2=extra_loss_fac_2,
            extra_loss_fac_2_=extra_loss_fac_2_,
            extra_loss_fac_0=extra_loss_fac_0,
            extra_loss_fac_3=extra_loss_fac_3,
            edge_loss_fac=edge_loss_fac,
            prior_branch_assignment=prior_branch_assignment,
            fac_var=fac_var,
            loss_fac_geneCell=loss_fac_geneCell,
            loss_fac_gene=loss_fac_gene,
            loss_fac_prior_clust=loss_fac_prior_clust,
            mu_max=mu_max,
            var_max=var_max,
            std_sum=std_sum,
            y_state0_super=y_state0_super,
            y_state1_super=y_state1_super,
            mat_where_prev=mat_where_prev,
            mat_where_next=mat_where_next,
            tmax=tmax,
            t_d=t_d,
            t_r=t_r,
            use_noise_ext=use_noise_ext,
            sample_prob=sample_prob,
            n_batch=n_batch,
            n_labels=self.summary_stats.n_labels,
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            n_cats_per_cov=n_cats_per_cov,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            n_states=n_states,
            use_loss_burst=use_loss_burst,
            cluster_states=cluster_states,
            state_loss_type=state_loss_type,
            states_vec=states_vec,
            state_time_max_vec=state_time_max_vec,
            use_controlBurst_gene=use_controlBurst_gene,
            timing_relative=timing_relative,
            timing_relative_mat_bin=timing_relative_mat_bin,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            use_size_factor_key=use_size_factor_key,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            **model_kwargs,
        )
        self.module.minified_data_type = self.minified_data_type
        self._model_summary_string = (
            "noSpliceVelo Model with the following params: \nn_hidden: {}, n_latent: {}, n_layers: {}, dropout_rate: "
            "{}, latent_distribution: {}"
        ).format(
            n_hidden,
            n_latent,
            n_layers,
            dropout_rate,
            latent_distribution,
        )
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: Optional[str] = None,
        mean_layer: Optional[str] = None,
        std_layer: Optional[str] = None,
        prior_pi_up_layer:Optional[str] = None,
        prior_cluster: Optional[str] = None,
        # mean_neighbors_layer: Optional[str] = None,
        # var_neighbors_layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        size_factor_key: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """%(summary)s.
        Parameters
        ----------
        %(param_adata)s
        %(param_layer)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            LayerField(REGISTRY_KEYS.M_KEY, mean_layer, is_count_data=False),
            LayerField(REGISTRY_KEYS.V_KEY, std_layer, is_count_data=False),
            LayerField("prior_pi_up", prior_pi_up_layer, is_count_data=False),
            LayerField("prior_cluster", prior_cluster, is_count_data=False),
            # LayerField("deriv2", deriv2_layer_1, is_count_data=False),
            # LayerField("deriv2_from_max", deriv2_layer_2, is_count_data=False),
            # LayerField(REGISTRY_KEYS.Mn_KEY, mean_neighbors_layer, is_count_data=False),
            # LayerField(REGISTRY_KEYS.Vn_KEY, var_neighbors_layer, is_count_data=False),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False
            ),
            CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
        ]
        # register new fields if the adata is minified
        adata_minify_type = _get_adata_minify_type(adata)
        if adata_minify_type is not None:
            anndata_fields += cls._get_fields_for_adata_minification(adata_minify_type)
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @staticmethod
    def _get_fields_for_adata_minification(
        minified_data_type: MinifiedDataType,
    ) -> List[BaseAnnDataField]:
        """Return the anndata fields required for adata minification of the given minified_data_type."""
        if minified_data_type == ADATA_MINIFY_TYPE.LATENT_POSTERIOR:
            fields = [
                ObsmField(
                    REGISTRY_KEYS.LATENT_QZM_KEY,
                    _SCVI_LATENT_QZM,
                ),
                ObsmField(
                    REGISTRY_KEYS.LATENT_QZV_KEY,
                    _SCVI_LATENT_QZV,
                ),
                NumericalObsField(
                    REGISTRY_KEYS.OBSERVED_LIB_SIZE,
                    _SCVI_OBSERVED_LIB_SIZE,
                ),
            ]
        else:
            raise NotImplementedError(f"Unknown MinifiedDataType: {minified_data_type}")
        fields.append(
            StringUnsField(
                REGISTRY_KEYS.MINIFY_TYPE_KEY,
                _ADATA_MINIFY_TYPE_UNS_KEY,
            ),
        )
        return fields

    def minify_adata(
        self,
        minified_data_type: MinifiedDataType = ADATA_MINIFY_TYPE.LATENT_POSTERIOR,
        use_latent_qzm_key: str = "X_latent_qzm",
        use_latent_qzv_key: str = "X_latent_qzv",
    ) -> None:
        """Minifies the model's adata.
        Minifies the adata, and registers new anndata fields: latent qzm, latent qzv, adata uns
        containing minified-adata type, and library size.
        This also sets the appropriate property on the module to indicate that the adata is minified.
        Parameters
        ----------
        minified_data_type
            How to minify the data. Currently only supports `latent_posterior_parameters`.
            If minified_data_type == `latent_posterior_parameters`:
            * the original count data is removed (`adata.X`, adata.raw, and any layers)
            * the parameters of the latent representation of the original data is stored
            * everything else is left untouched
        use_latent_qzm_key
            Key to use in `adata.obsm` where the latent qzm params are stored
        use_latent_qzv_key
            Key to use in `adata.obsm` where the latent qzv params are stored
        Notes
        -----
        The modification is not done inplace -- instead the model is assigned a new (minified)
        version of the adata.
        """
        # TODO(adamgayoso): Add support for a scenario where we want to cache the latent posterior
        # without removing the original counts.
        if minified_data_type != ADATA_MINIFY_TYPE.LATENT_POSTERIOR:
            raise NotImplementedError(f"Unknown MinifiedDataType: {minified_data_type}")

        if self.module.use_observed_lib_size is False:
            raise ValueError(
                "Cannot minify the data if `use_observed_lib_size` is False"
            )

        minified_adata = get_minified_adata_scrna(self.adata, minified_data_type)
        minified_adata.obsm[_SCVI_LATENT_QZM] = self.adata.obsm[use_latent_qzm_key]
        minified_adata.obsm[_SCVI_LATENT_QZV] = self.adata.obsm[use_latent_qzv_key]
        counts = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
        minified_adata.obs[_SCVI_OBSERVED_LIB_SIZE] = np.squeeze(
            np.asarray(counts.sum(axis=1))
        )
        self._update_adata_and_manager_post_minification(
            minified_adata, minified_data_type
        )
        self.module.minified_data_type = minified_data_type
        
    @torch.inference_mode()
    def get_likelihood_parameters_new(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        n_samples: Optional[int] = 1,
        give_mean: Optional[bool] = False,
        batch_size: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        r"""Estimates for the parameters of the likelihood :math:`p(x \mid z)`.
        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        n_samples
            Number of posterior samples to use for estimation.
        give_mean
            Return expected value of parameters or a samples
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        """
        adata = self._validate_anndata(adata)

        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        dropout_list = []
        mean_list = []
        dispersion_list = []
        B1_list = []
        f1_list = []
        B2_list = []
        f2_list = []
        mu_up_f_list = []
        var_up_f_list = []
        mu_down_f_list = []
        var_down_f_list = []
        time_ss_list = []
        velo_mu_up_list = []
        velo_var_up_list = []
        velo_mu_down_list = []
        velo_var_down_list = []
        tau_list = []
        gamma_mRNA_list = []
        library_list = []
        # prob_list = []
        gene_max_time_list = []
        if self.cluster_states:
            cc_phase_pred_list = []
        scale_list = []
        scale1_list = []
        scale2_list = []
        prob_state_list = []
        loss_mu_list = []
        loss_std_list = []

        for tensors in scdl:
            inference_kwargs = {"n_samples": n_samples}
            _, generative_outputs = self.module.forward(
                tensors=tensors,
                inference_kwargs=inference_kwargs,
                compute_loss=False,
            )
            # px = generative_outputs["px"]
            # px_r = px.theta
            # px_rate = px.mu
            if self.loss_fac_geneCell > 0.0:
                px_B1 = generative_outputs["burst_b1"]
                px_f1 = generative_outputs["burst_f1"]
                px_B2 = generative_outputs["burst_b2"]
                px_f2 = generative_outputs["burst_f2"]
                mu_up_f = generative_outputs['mu_up_f']
                var_up_f = generative_outputs['var_up_f']
                mu_down_f = generative_outputs['mu_down_f']
                var_down_f = generative_outputs['var_down_f']
            else:
                px_B1 = generative_outputs["burst_b1_gene"]
                px_f1 = generative_outputs["burst_f1_gene"]
                px_B2 = generative_outputs["burst_b2_gene"]
                px_f2 = generative_outputs["burst_f2_gene"]
                mu_up_f = generative_outputs['mu_up_f_gene']
                var_up_f = generative_outputs['var_up_f_gene']
                mu_down_f = generative_outputs['mu_down_f_gene']
                var_down_f = generative_outputs['var_down_f_gene']
            time_ss = generative_outputs["time_ss"]
            
            if self.loss_fac_geneCell > 0.0:
                velo_mu_up = generative_outputs["velo_mu_up"]
                velo_var_up = generative_outputs["velo_var_up"]
                velo_mu_down = generative_outputs["velo_mu_down"]
                velo_var_down = generative_outputs["velo_var_down"]
            else:
                velo_mu_up = generative_outputs["velo_mu_up_gene"]
                velo_var_up = generative_outputs["velo_var_up_gene"]
                velo_mu_down = generative_outputs["velo_mu_down_gene"]
                velo_var_down = generative_outputs["velo_var_down_gene"]
            px_time_rate = generative_outputs["time_cell"]
            gamma_mRNA = generative_outputs["gamma_mRNA"]
            library_tmp = generative_outputs['library']
            # prob_tmp = generative_outputs['prob_state']
            gene_max_time_tmp = \
                generative_outputs['time_down']
            if self.cluster_states:
                cc_phase_pred = generative_outputs['prob_state']
            scale_tmp = generative_outputs['scale']
            scale1_tmp = generative_outputs['scale_var']
            scale2_tmp = generative_outputs['scale_var_ref']
            prob_state_tmp = generative_outputs['prob_state']
            loss_mu = generative_outputs['loss_mu_full']
            loss_std = generative_outputs['loss_std_full']

            # if self.module.gene_likelihood == "zinb":
            #     px_dropout = px.zi_probs

            # n_batch = px_rate.size(0) if n_samples == 1 else px_rate.size(1)

            # px_r = px_r.cpu().numpy()
            # if len(px_r.shape) == 1:
            #     dispersion_list += [np.repeat(px_r[np.newaxis, :], n_batch, axis=0)]
            # else:
            #     dispersion_list += [px_r]
#                 px_r_list += [px_r_]
                
            # mean_list += [px_rate.cpu().numpy()]
            
            B1_list += [px_B1.cpu().numpy()]
            f1_list += [px_f1.cpu().numpy()]
            B2_list += [px_B2.cpu().numpy()]
            f2_list += [px_f2.cpu().numpy()]
            mu_up_f_list += [mu_up_f.cpu().numpy()]
            var_up_f_list += [var_up_f.cpu().numpy()]
            mu_down_f_list += [mu_down_f.cpu().numpy()]
            var_down_f_list += [var_down_f.cpu().numpy()]
            time_ss_list += [time_ss.cpu().numpy()]
            velo_mu_up_list += [velo_mu_up.cpu().numpy()]
            velo_var_up_list += [velo_var_up.cpu().numpy()]
            velo_mu_down_list += [velo_mu_down.cpu().numpy()]
            velo_var_down_list += [velo_var_down.cpu().numpy()]
            tau_list += [px_time_rate.cpu().numpy()]
            gamma_mRNA_list += [gamma_mRNA.cpu().numpy()]
            library_list += [library_tmp.cpu().numpy()]
            # prob_list += [prob_tmp.cpu().numpy()]
            gene_max_time_list += [gene_max_time_tmp.cpu().numpy()]
            if self.cluster_states:
                cc_phase_pred_list += [cc_phase_pred.cpu().numpy()]
            scale_list += [scale_tmp.cpu().numpy()]
            scale1_list += [scale1_tmp.cpu().numpy()]
            scale2_list += [scale2_tmp.cpu().numpy()]
            prob_state_list += [prob_state_tmp.cpu().numpy()]
            loss_mu_list += [loss_mu.cpu().numpy()]
            loss_std_list += [loss_std.cpu().numpy()]

            
            # if self.module.gene_likelihood == "zinb":
            #     dropout_list += [px_dropout.cpu().numpy()]
            #     dropout = np.concatenate(dropout_list, axis=-2)
        # means = np.concatenate(mean_list, axis=-2)
        # dispersions = np.concatenate(dispersion_list, axis=-2)
        burst_B1_final = np.concatenate(B1_list, axis=-2)
        burst_f1_final = np.concatenate(f1_list, axis=-2)
        burst_B2_final = np.concatenate(B2_list, axis=-2)
        burst_f2_final = np.concatenate(f2_list, axis=-2)
        mu_up_f_final = np.concatenate(mu_up_f_list, axis=-2)
        var_up_f_final = np.concatenate(var_up_f_list, axis=-2)
        mu_down_f_final = np.concatenate(mu_down_f_list, axis=-2)
        var_down_f_final = np.concatenate(var_down_f_list, axis=-2)
        time_ss_final = np.concatenate(time_ss_list, axis=-2)
        velo_mu_up_final = np.concatenate(velo_mu_up_list, axis=-2)
        velo_var_up_final = np.concatenate(velo_var_up_list, axis=-2)
        velo_mu_down_final = np.concatenate(velo_mu_down_list, axis=-2)
        velo_var_down_final = np.concatenate(velo_var_down_list, axis=-2)
#         means_final = np.concatenate(mean_final_list, axis=-2)
        tau_ = np.concatenate(tau_list, axis=-2)
        gamma_mRNA_all = np.concatenate(gamma_mRNA_list, axis=-2)
        library_list = np.concatenate(library_list, axis=-2)
        # prob_list = np.concatenate(prob_list, axis=-1)
        gene_max_time_list = np.concatenate(gene_max_time_list, axis=-2)
        if self.cluster_states:
            if self.state_loss_type == 'cross-entropy':
                cc_phase_pred_list = np.concatenate(cc_phase_pred_list, axis=-2)
            else:
                cc_phase_pred_list = np.concatenate(cc_phase_pred_list, axis=-1)
        scale_list = np.concatenate(scale_list, axis=-2)
        scale1_list = np.concatenate(scale1_list, axis=-2)
        scale2_list = np.concatenate(scale2_list, axis=-2)
        prob_state_list = np.concatenate(prob_state_list, axis=-3)
        loss_mu_list = np.concatenate(loss_mu_list, axis=-2)
        loss_std_list = np.concatenate(loss_std_list, axis=-2)

        if give_mean and n_samples > 1:
            # if self.module.gene_likelihood == "zinb":
            #     dropout = dropout.mean(0)
            # means = means.mean(0)
            # dispersions = dispersions.mean(0)
            burst_B1_final = burst_B1_final.mean(0)
            burst_f1_final = burst_f1_final.mean(0)
            burst_B2_final = burst_B2_final.mean(0)
            burst_f2_final = burst_f2_final.mean(0)
            mu_up_f_final = mu_up_f_final.mean(0)
            var_up_f_final = var_up_f_final.mean(0)
            mu_down_f_final = mu_down_f_final.mean(0)
            var_down_f_final = var_down_f_final.mean(0)
            time_ss_final = time_ss_final.mean(0)
            velo_mu_up_final = velo_mu_up_final.mean(0)
            velo_var_up_final = velo_var_up_final.mean(0)
            velo_mu_down_final = velo_mu_down_final.mean(0)
            velo_var_down_final = velo_var_down_final.mean(0)
            tau_ = tau_.mean(0)
            gamma_mRNA_all = gamma_mRNA_all.mean(0)
            library_list = library_list.mean(0)
            # prob_list = prob_list.mean(0)
            gene_max_time_list = gene_max_time_list.mean(0)
            if self.cluster_states:
                cc_phase_pred_list = cc_phase_pred_list.mean(0)
            scale_list = scale_list.mean(0)
            scale1_list = scale1_list.mean(0)
            scale2_list = scale2_list.mean(0)
            prob_state_list = prob_state_list.mean(0)
            loss_mu_list = loss_mu_list.mean(0)
            loss_std_list = loss_std_list.mean(0)

        return_dict = {}
        # return_dict["mean"] = means
        return_dict["burst_B1"] = burst_B1_final
        return_dict["burst_f1"] = burst_f1_final
        return_dict["var_down_up"] = burst_B2_final
        return_dict["mu_down_up"] = burst_f2_final
        return_dict['mu_up_f'] = mu_up_f_final
        return_dict['var_up_f'] = var_up_f_final
        return_dict['mu_down_f'] = mu_down_f_final
        return_dict['var_down_f'] = var_down_f_final
        return_dict['time_ss'] = time_ss_final
        return_dict["velo_mu_up"] = velo_mu_up_final
        return_dict["velo_var_up"] = velo_var_up_final
        return_dict["velo_mu_down"] = velo_mu_down_final
        return_dict["velo_var_down"] = velo_var_down_final
        return_dict["tau_up"] = tau_
        return_dict["gamma_mRNA_all"] = gamma_mRNA_all
        return_dict['library'] = library_list
        # return_dict['prob_state'] = prob_list
        return_dict['tau_down'] = gene_max_time_list
        if self.cluster_states:
            return_dict["cc_phase_pred"] = cc_phase_pred_list
        return_dict['scale_mu'] = scale_list
        return_dict['scale_var'] = scale1_list
        return_dict['scale_var_ref'] = scale2_list
        return_dict['prob_state'] = prob_state_list
        return_dict['loss_mu'] = loss_mu_list
        return_dict['loss_std'] = loss_std_list

        # if self.module.gene_likelihood == "zinb":
        #     return_dict["dropout"] = dropout
        #     return_dict["dispersions"] = dispersions
        # if self.module.gene_likelihood == "nb":
        #     return_dict["dispersions"] = dispersions

        return return_dict

    @torch.inference_mode()
    def get_likelihood_parameters_gene_specific(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        n_samples: Optional[int] = 1,
        give_mean: Optional[bool] = False,
        batch_size: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        r"""Estimates for the parameters of the likelihood :math:`p(x \mid z)`.
        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        n_samples
            Number of posterior samples to use for estimation.
        give_mean
            Return expected value of parameters or a samples
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        """
        adata = self._validate_anndata(adata)

        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        B1_list = []
        f1_list = []
        B2_list = []
        f2_list = []
        mu_up_f_list = []
        var_up_f_list = []
        mu_down_f_list = []
        var_down_f_list = []
        time_ss_list = []
        velo_mu_up_list = []
        velo_var_up_list = []
        velo_mu_down_list = []
        velo_var_down_list = []
        tau_list = []
        gamma_mRNA_list = []
        # prob_list = []
        tau_down_list = []
        scale_list = []
        prob_list = []

        for tensors in scdl:
            inference_kwargs = {"n_samples": n_samples}
            _, generative_outputs = self.module.forward(
                tensors=tensors,
                inference_kwargs=inference_kwargs,
                compute_loss=False,
            )
            # px = generative_outputs["px"]
            # px_r = px.theta
            # px_rate = px.mu
            px_B1 = generative_outputs["burst_b1_gene"]
            px_f1 = generative_outputs["burst_f1_gene"]
            px_B2 = generative_outputs["burst_b2_gene"]
            px_f2 = generative_outputs["burst_f2_gene"]
            mu_up_f = generative_outputs['mu_up_f_gene']
            var_up_f = generative_outputs['var_up_f_gene']
            mu_down_f = generative_outputs['mu_down_f_gene']
            var_down_f = generative_outputs['var_down_f_gene']
            time_ss = generative_outputs["time_ss_gene"]
            velo_mu_up = generative_outputs["velo_mu_up_gene"]
            velo_var_up = generative_outputs["velo_var_up_gene"]
            velo_mu_down = generative_outputs["velo_mu_down_gene"]
            velo_var_down = generative_outputs["velo_var_down_gene"]
            px_time_rate = generative_outputs["time_up_gene"]
            gamma_mRNA = generative_outputs["gamma_mRNA_gene"]
            # prob_tmp = generative_outputs['prob_state']
            time_down_tmp = \
                generative_outputs['time_down_gene']
            scale_tmp = generative_outputs['scale_gene']
            prob_tmp = generative_outputs['prob_state']

            # if self.module.gene_likelihood == "zinb":
            #     px_dropout = px.zi_probs

            # n_batch = px_rate.size(0) if n_samples == 1 else px_rate.size(1)

            # px_r = px_r.cpu().numpy()
            # if len(px_r.shape) == 1:
            #     dispersion_list += [np.repeat(px_r[np.newaxis, :], n_batch, axis=0)]
            # else:
            #     dispersion_list += [px_r]
#                 px_r_list += [px_r_]
                
            # mean_list += [px_rate.cpu().numpy()]
            B1_list += [px_B1.cpu().numpy()]
            f1_list += [px_f1.cpu().numpy()]
            B2_list += [px_B2.cpu().numpy()]
            f2_list += [px_f2.cpu().numpy()]
            mu_up_f_list += [mu_up_f.cpu().numpy()]
            var_up_f_list += [var_up_f.cpu().numpy()]
            mu_down_f_list += [mu_down_f.cpu().numpy()]
            var_down_f_list += [var_down_f.cpu().numpy()]
            time_ss_list += [time_ss.cpu().numpy()]
            velo_mu_up_list += [velo_mu_up.cpu().numpy()]
            velo_var_up_list += [velo_var_up.cpu().numpy()]
            velo_mu_down_list += [velo_mu_down.cpu().numpy()]
            velo_var_down_list += [velo_var_down.cpu().numpy()]
            tau_list += [px_time_rate.cpu().numpy()]
            gamma_mRNA_list += [gamma_mRNA.cpu().numpy()]
            tau_down_list += [time_down_tmp.cpu().numpy()]
            scale_list += [scale_tmp.cpu().numpy()]
            prob_list += [prob_tmp.cpu().numpy()]
            
            # if self.module.gene_likelihood == "zinb":
            #     dropout_list += [px_dropout.cpu().numpy()]
            #     dropout = np.concatenate(dropout_list, axis=-2)
        # means = np.concatenate(mean_list, axis=-2)
        # dispersions = np.concatenate(dispersion_list, axis=-2)
        burst_B1_final = np.concatenate(B1_list, axis=-2)
        burst_f1_final = np.concatenate(f1_list, axis=-2)
        burst_B2_final = np.concatenate(B2_list, axis=-2)
        burst_f2_final = np.concatenate(f2_list, axis=-2)
        mu_up_f_final = np.concatenate(mu_up_f_list, axis=-2)
        var_up_f_final = np.concatenate(var_up_f_list, axis=-2)
        mu_down_f_final = np.concatenate(mu_down_f_list, axis=-2)
        var_down_f_final = np.concatenate(var_down_f_list, axis=-2)
        time_ss_final = np.concatenate(time_ss_list, axis=-2)
        velo_mu_up_final = np.concatenate(velo_mu_up_list, axis=-2)
        velo_var_up_final = np.concatenate(velo_var_up_list, axis=-2)
        velo_mu_down_final = np.concatenate(velo_mu_down_list, axis=-2)
        velo_var_down_final = np.concatenate(velo_var_down_list, axis=-2)
#         means_final = np.concatenate(mean_final_list, axis=-2)
        tau_ = np.concatenate(tau_list, axis=-2)
        gamma_mRNA_all = np.concatenate(gamma_mRNA_list, axis=-2)
        # prob_list = np.concatenate(prob_list, axis=-1)
        tau_down_list = np.concatenate(tau_down_list, axis=-2)
        scale_list = np.concatenate(scale_list, axis=-3)
        prob_list = np.concatenate(prob_list, axis=-3)

        if give_mean and n_samples > 1:
            # if self.module.gene_likelihood == "zinb":
            #     dropout = dropout.mean(0)
            # means = means.mean(0)
            # dispersions = dispersions.mean(0)
            burst_B1_final = burst_B1_final.mean(0)
            burst_f1_final = burst_f1_final.mean(0)
            burst_B2_final = burst_B2_final.mean(0)
            burst_f2_final = burst_f2_final.mean(0)
            mu_up_f_final = mu_up_f_final.mean(0)
            var_up_f_final = var_up_f_final.mean(0)
            mu_down_f_final = mu_down_f_final.mean(0)
            var_down_f_final = var_down_f_final.mean(0)
            time_ss_final = time_ss_final.mean(0)
            velo_mu_up_final = velo_mu_up_final.mean(0)
            velo_var_up_final = velo_var_up_final.mean(0)
            velo_mu_down_final = velo_mu_down_final.mean(0)
            velo_var_down_final = velo_var_down_final.mean(0)
            tau_ = tau_.mean(0)
            gamma_mRNA_all = gamma_mRNA_all.mean(0)
            tau_down_list = tau_down_list.mean(0)
            scale_list = scale_list.mean(0)
            prob_list = prob_list.mean(0)

        return_dict = {}
        # return_dict["mean"] = means
        return_dict["burst_B1"] = burst_B1_final
        return_dict["burst_f1"] = burst_f1_final
        return_dict["var_down_up"] = burst_B2_final
        return_dict["mu_down_up"] = burst_f2_final
        return_dict['mu_up_f'] = mu_up_f_final
        return_dict['var_up_f'] = var_up_f_final
        return_dict['mu_down_f'] = mu_down_f_final
        return_dict['var_down_f'] = var_down_f_final
        return_dict['time_ss'] = time_ss_final
        return_dict["velo_mu_up"] = velo_mu_up_final
        return_dict["velo_var_up"] = velo_var_up_final
        return_dict["velo_mu_down"] = velo_mu_down_final
        return_dict["velo_var_down"] = velo_var_down_final
        return_dict["tau_up"] = tau_
        return_dict["gamma_mRNA_all"] = gamma_mRNA_all
        return_dict['tau_down'] = tau_down_list
        return_dict['scale_mu'] = scale_list
        return_dict['prob_state'] = prob_list

        return return_dict
    
    def get_directional_uncertainty(
        self,
        adata: Optional[AnnData] = None,
        n_samples: int = 50,
        gene_list: Iterable[str] = None,
        n_jobs: int = -1,
    ):
        adata = self._validate_anndata(adata)

        # logger.info("Sampling from model...")
        velocities_all = np.zeros(
            (n_samples, adata.n_obs, self.module.n_genes)
        )
        for b_ in range(n_samples):
            print(f'gene-cell boot = {b_}')
            params_ = self.get_likelihood_parameters_new()
            velo_mu_up = params_['velo_mu_up'].copy()
            velo_var_up = params_['velo_var_up'].copy()
            velo_mu_down = params_['velo_mu_down'].copy()
            velo_var_down = params_['velo_var_down'].copy()

            
            print(params_.keys())
            probs_ = params_['prob_state'].copy()
            # probs_1 = params_scvi_pos['prob_state_list'][:, :, 1].copy()
            
            # Step 1: find index of max probability along last axis
            max_idx = np.argmax(probs_, axis=2)  # shape (N, G)
            N, G = adata.shape
            zeros_array = np.zeros((N, G))
            ones_array = np.zeros((N, G))
            velo_mu_gene_stacked = np.stack(
                (
                    # zeros_array,
                    velo_mu_up,
                    zeros_array,
                    # zeros_array,
                    velo_mu_down,
                    zeros_array
                ),
                axis=2
            )
            
            # velo_var_gene_stacked = np.stack(
            #     (
            #         # zeros_array,
            #         velo_var_up,
            #         zeros_array,
            #         # zeros_array,
            #         velo_var_down,
            #         zeros_array
            #     ),
            #     axis=2
            # )
            
            velo_mu_ = velo_mu_gene_stacked[np.arange(N)[:, None], np.arange(G)[None, :], max_idx]
            # velo_var_ = velo_var_gene_stacked[np.arange(N)[:, None], np.arange(G)[None, :], max_idx]
            del params_
            velocities_all[b_, :, :] = velo_mu_


        # velocities_all = self.get_velocity(
        #     n_samples=n_samples, return_mean=False, gene_list=gene_list
        # )  # (n_samples, n_cells, n_genes)

        df, cosine_sims = _compute_directional_statistics_tensor(
            tensor=velocities_all, n_jobs=n_jobs, n_cells=adata.n_obs
        )
        df.index = adata.obs_names

        return df, cosine_sims
    
    def get_permutation_scores(
        self, labels_key: str, adata: Optional[AnnData] = None, N_repeat=10
    ) -> Tuple[pd.DataFrame, AnnData]:
        """Compute permutation scores.

        Parameters
        ----------
        labels_key
            Key in adata.obs encoding cell types
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.

        Returns
        -------
        Tuple of DataFrame and AnnData. DataFrame is genes by cell types with score per cell type.
        AnnData is the permutated version of the original AnnData.
        """
        def get_mu_std(params_nosplicevelo):
            def get_mu_var_time(mu_0, var_0, mu_f, var_f, gamma_, time):
                p_t = np.exp(-gamma_ * time)
                mu_t = mu_0 * p_t + mu_f * (1 - p_t)
                var_t = (var_0 - mu_0) * p_t**2.0 + (var_f - mu_f) * (1 - p_t**2.0) + mu_t
                return mu_t, var_t

            def get_new_time(mu_0, var_0, mu_f, var_f, mu_t, var_t, gamma_):
                mu_ratio = (mu_t - mu_0) / (mu_f - mu_0)
                mu_ratio = np.clip(mu_ratio, a_min=0, a_max=1)
                time_mu = - (np.log(1 - mu_ratio) / gamma_)
                var_ratio = (((var_f - mu_f) - (var_0 - mu_0))) / ((var_f - mu_f) - (var_t - mu_t))
                time_var = - (np.log(var_ratio)) / gamma_
                return time_mu, time_var
            f1 = params_nosplicevelo['burst_f1'].copy()
            b1 = params_nosplicevelo['burst_B1'].copy()
            gamma_ = params_nosplicevelo['gamma_mRNA_all'].copy()
            mu_0 = f1 * b1 / gamma_
            var_0 = mu_0 * (b1 + 1)
            mu_up_f = params_nosplicevelo['mu_up_f'].copy()
            var_up_f = params_nosplicevelo['var_up_f'].copy()
            mu_down_up = params_nosplicevelo['mu_down_up'].copy()
            var_down_up = params_nosplicevelo['var_down_up'].copy()
            mu_down_f = params_nosplicevelo['mu_down_f'].copy()
            var_down_f = params_nosplicevelo['var_down_f'].copy()
            time_up = params_nosplicevelo['tau_up'].copy()
            time_down = params_nosplicevelo['tau_down'].copy()
            mu_up, var_up = get_mu_var_time(mu_0, var_0, mu_up_f, var_up_f, gamma_, time_up)
            mu_down, var_down = get_mu_var_time(mu_down_up, var_down_up, \
                                                mu_down_f, var_down_f, gamma_, time_down)



            #######################
            mu_gene_stacked = np.stack(
                (
                    # mu_0,
                    mu_up,
                    mu_up_f,
                    mu_down,
                    mu_down_f
                ),
                axis=2
            )
            var_gene_stacked = np.stack(
                (
                    # var_0,
                    var_up,
                    var_up_f,
                    var_down,
                    var_down_f
                ),
                axis=2
            )


            probs_ = params_nosplicevelo['prob_state'].copy()
            # probs_1 = params_scvi_pos['prob_state_list'][:, :, 1].copy()

            # Step 1: find index of max probability along last axis
            max_idx = np.argmax(probs_, axis=2)  # shape (N, G)

            # Step 2: use fancy indexing to select from array_
            N, G, _ = mu_gene_stacked.shape
            mu_nosplicevelo = mu_gene_stacked[np.arange(N)[:, None], np.arange(G)[None, :], max_idx]
            var_nosplicevelo = var_gene_stacked[np.arange(N)[:, None], np.arange(G)[None, :], max_idx]
            return mu_nosplicevelo, var_nosplicevelo
        
        
        adata = self._validate_anndata(adata)
        adata_manager = self.get_anndata_manager(adata)
        if labels_key not in adata.obs:
            raise ValueError(f"{labels_key} not found in adata.obs")

        # shuffle spliced then unspliced
        bdata = self._shuffle_layer_celltype(
            adata_manager, labels_key, REGISTRY_KEYS.M_KEY
        )
        bdata_manager = self.get_anndata_manager(bdata)
        bdata = self._shuffle_layer_celltype(
            bdata_manager, labels_key, REGISTRY_KEYS.V_KEY
        )
        bdata_manager = self.get_anndata_manager(bdata)

        mu_ = adata_manager.get_from_registry(REGISTRY_KEYS.M_KEY)
        std_ = adata_manager.get_from_registry(REGISTRY_KEYS.V_KEY)

        mu_p = bdata_manager.get_from_registry(REGISTRY_KEYS.M_KEY)
        std_p = bdata_manager.get_from_registry(REGISTRY_KEYS.V_KEY)
        
        # N_repeat = 10
        
        for n_ in np.arange(N_repeat):
            print(f'gene param loop = {n_}')
            params_ = self.get_likelihood_parameters_gene_specific(adata)
            mu_tmp, var_tmp = get_mu_std(params_)
            
            if n_ == 0:
                mu_vae = mu_tmp
                var_vae = var_tmp
            else:
                mu_vae += mu_tmp
                var_vae += var_tmp
        mu_vae /= N_repeat
        var_vae /= N_repeat
        std_vae = np.sqrt(var_vae)
        
        for n_ in np.arange(N_repeat):
            print(f'gene param loop, perm = {n_}')
            params_ = self.get_likelihood_parameters_gene_specific(bdata)
            mu_tmp, var_tmp = get_mu_std(params_)
            
            if n_ == 0:
                mu_vae_p = mu_tmp
                var_vae_p = var_tmp
            else:
                mu_vae_p += mu_tmp
                var_vae_p += var_tmp
        mu_vae_p /= N_repeat
        var_vae_p /= N_repeat
        std_vae_p = np.sqrt(var_vae_p)
        
        root_squared_error = np.abs(mu_vae - mu_)
        root_squared_error += np.abs(std_vae - std_)

        root_squared_error_p = np.abs(mu_vae_p - mu_p)
        root_squared_error_p += np.abs(std_vae - std_p)

        celltypes = np.unique(adata.obs[labels_key])

        dynamical_df = pd.DataFrame(
            index=adata.var_names,
            columns=celltypes,
            data=np.zeros((adata.shape[1], len(celltypes))),
        )
        N = 200
        for ct in celltypes:
            print(f'celltype = {ct}')
            for count_, g in enumerate(adata.var_names.tolist()):
                # print(f'celltype = {ct}, gene # = {count_}')
                id_ = np.where(adata.obs[labels_key].values == ct)[0]
                x = root_squared_error_p[id_, count_].flatten()
                y = root_squared_error[id_, count_].flatten()
                ratio = ttest_ind(x[:N], y[:N])[0]
                dynamical_df.loc[g, ct] = ratio

        return dynamical_df, bdata
    
    def _shuffle_layer_celltype(
        self, adata_manager: AnnDataManager, labels_key: str, registry_key: str
    ) -> AnnData:
        """Shuffle cells within cell types for each gene."""
        from scvi.data._constants import _SCVI_UUID_KEY

        bdata = adata_manager.adata.copy()
        labels = bdata.obs[labels_key]
        del bdata.uns[_SCVI_UUID_KEY]
        self._validate_anndata(bdata)
        bdata_manager = self.get_anndata_manager(bdata)

        # get registry info to later set data back in bdata
        # in a way that doesn't require actual knowledge of location
        unspliced = bdata_manager.get_from_registry(registry_key)
        u_registry = bdata_manager.data_registry[registry_key]
        attr_name = u_registry.attr_name
        attr_key = u_registry.attr_key

        for lab in np.unique(labels):
            mask = np.asarray(labels == lab)
            unspliced_ct = unspliced[mask].copy()
            unspliced_ct = np.apply_along_axis(
                np.random.permutation, axis=0, arr=unspliced_ct
            )
            unspliced[mask] = unspliced_ct
        # e.g., if using adata.X
        if attr_key is None:
            setattr(bdata, attr_name, unspliced)
        # e.g., if using a layer
        elif attr_key is not None:
            attribute = getattr(bdata, attr_name)
            attribute[attr_key] = unspliced
            setattr(bdata, attr_name, attribute)

        return bdata
    
def _compute_directional_statistics_tensor(
    tensor: np.ndarray, n_jobs: int, n_cells: int
) -> pd.DataFrame:
    df = pd.DataFrame(index=np.arange(n_cells))
    df["directional_variance"] = np.nan
    df["directional_difference"] = np.nan
    df["directional_cosine_sim_variance"] = np.nan
    df["directional_cosine_sim_difference"] = np.nan
    df["directional_cosine_sim_mean"] = np.nan
    # logger.info("Computing the uncertainties...")
    results = Parallel(n_jobs=n_jobs, verbose=3)(
        delayed(_directional_statistics_per_cell)(tensor[:, cell_index, :])
        for cell_index in range(n_cells)
    )
    # cells by samples
    cosine_sims = np.stack([results[i][0] for i in range(n_cells)])
    df.loc[:, "directional_cosine_sim_variance"] = [
        results[i][1] for i in range(n_cells)
    ]
    df.loc[:, "directional_cosine_sim_difference"] = [
        results[i][2] for i in range(n_cells)
    ]
    df.loc[:, "directional_variance"] = [results[i][3] for i in range(n_cells)]
    df.loc[:, "directional_difference"] = [results[i][4] for i in range(n_cells)]
    df.loc[:, "directional_cosine_sim_mean"] = [results[i][5] for i in range(n_cells)]

    return df, cosine_sims


def _directional_statistics_per_cell(
    tensor: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Internal function for parallelization.

    Parameters
    ----------
    tensor
        Shape of samples by genes for a given cell.
    """
    n_samples = tensor.shape[0]
    # over samples axis
    mean_velocity_of_cell = tensor.mean(0)
    cosine_sims = [
        _cosine_sim(tensor[i, :], mean_velocity_of_cell) for i in range(n_samples)
    ]
    angle_samples = [np.arccos(el) for el in cosine_sims]
    return (
        cosine_sims,
        np.var(cosine_sims),
        np.percentile(cosine_sims, 95) - np.percentile(cosine_sims, 5),
        np.var(angle_samples),
        np.percentile(angle_samples, 95) - np.percentile(angle_samples, 5),
        np.mean(cosine_sims),
    )


def _centered_unit_vector(vector: np.ndarray) -> np.ndarray:
    """Returns the centered unit vector of the vector."""
    vector = vector - np.mean(vector)
    return vector / np.linalg.norm(vector)


def _cosine_sim(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Returns cosine similarity of the vectors."""
    v1_u = _centered_unit_vector(v1)
    v2_u = _centered_unit_vector(v2)
    return np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
