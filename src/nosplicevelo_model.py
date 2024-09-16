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
from nosplicevelo_module \
    import VAENoiseVelo
import torch
import pandas as pd
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
        extra_loss_fac: float = 0.5,
        extra_loss_fac_1: float = 0.5,
        extra_loss_fac_2: float = 0.1,
        extra_loss_fac_0: float = 0.1,
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
        n_states: int = 2,
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
            extra_loss_fac_0=extra_loss_fac_0,
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
        nt_list = []
        var_list = []
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
        prob_state_list = []

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
            px_nt = generative_outputs["n_t"]
            px_var = generative_outputs['var']
            px_B1 = generative_outputs["burst_b1"]
            px_f1 = generative_outputs["burst_f1"]
            px_B2 = generative_outputs["burst_b2"]
            px_f2 = generative_outputs["burst_f2"]
            mu_up_f = generative_outputs['mu_up_f']
            var_up_f = generative_outputs['var_up_f']
            mu_down_f = generative_outputs['mu_down_f']
            var_down_f = generative_outputs['var_down_f']
            time_ss = generative_outputs["time_ss"]
            velo_mu_up = generative_outputs["velo_mu_up"]
            velo_var_up = generative_outputs["velo_var_up"]
            velo_mu_down = generative_outputs["velo_mu_down"]
            velo_var_down = generative_outputs["velo_var_down"]
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
            prob_state_tmp = generative_outputs['prob_state']

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
            nt_list += [px_nt.cpu().numpy()]
            var_list += [px_var.cpu().numpy()]
            if self.cluster_states:
                cc_phase_pred_list += [cc_phase_pred.cpu().numpy()]
            scale_list += [scale_tmp.cpu().numpy()]
            scale1_list += [scale1_tmp.cpu().numpy()]
            prob_state_list += [prob_state_tmp.cpu().numpy()]
            
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
        nt_list = np.concatenate(nt_list, axis=-2)
        var_list = np.concatenate(var_list, axis=-2)
        if self.cluster_states:
            if self.state_loss_type == 'cross-entropy':
                cc_phase_pred_list = np.concatenate(cc_phase_pred_list, axis=-2)
            else:
                cc_phase_pred_list = np.concatenate(cc_phase_pred_list, axis=-1)
        scale_list = np.concatenate(scale_list, axis=-2)
        scale1_list = np.concatenate(scale1_list, axis=-2)
        prob_state_list = np.concatenate(prob_state_list, axis=-3)

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
            nt_list = nt_list.mean(0)
            var_list = var_list.mean(0)
            if self.cluster_states:
                cc_phase_pred_list = cc_phase_pred_list.mean(0)
            scale_list = scale_list.mean(0)
            scale1_list = scale1_list.mean(0)
            prob_state_list = prob_state_list.mean(0)

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
        return_dict["n_t"] = nt_list
        return_dict["var"] = var_list
        if self.cluster_states:
            return_dict["cc_phase_pred"] = cc_phase_pred_list
        return_dict['scale_mu'] = scale_list
        return_dict['scale_var'] = scale1_list
        return_dict['prob_state_list'] = prob_state_list

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

        mean_list = []
        nt_list = []
        var_list = []
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
            px_nt = generative_outputs["n_t_gene"]
            px_var = generative_outputs['var_gene']
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
            nt_list += [px_nt.cpu().numpy()]
            var_list += [px_var.cpu().numpy()]
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
        nt_list = np.concatenate(nt_list, axis=-2)
        var_list = np.concatenate(var_list, axis=-2)
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
            nt_list = nt_list.mean(0)
            var_list = var_list.mean(0)
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
        return_dict["n_t"] = nt_list
        return_dict["var"] = var_list
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