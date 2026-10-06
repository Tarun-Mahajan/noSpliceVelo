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
from scvi.model._utils import _get_batch_code_from_category, scrna_raw_counts_properties
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
from scvi.utils import setup_anndata_dsp
from scvi._types import Number

from scvi.model.base import ArchesMixin, BaseMinifiedModeModelClass, RNASeqMixin, VAEMixin
from scvi.nn import FCLayers
from scvi_modified_capture_efficiency_module import VAE
import torch
import pandas as pd

class SCVIModified(
    RNASeqMixin,
    VAEMixin,
    ArchesMixin,
    UnsupervisedTrainingMixin,
    BaseMinifiedModeModelClass,
):
    """single-cell Variational Inference :cite:p:`Lopez18`.
    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~scvi.model.SCVI.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    dropout_rate
        Dropout rate for neural networks.
    dispersion
        One of the following:
        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    gene_likelihood
        One of:
        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    latent_distribution
        One of:
        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic normal distribution (Normal(0, I) transformed by softmax)
    **model_kwargs
        Keyword args for :class:`~scvi.module.VAE`
    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> scvi.model.SCVI.setup_anndata(adata, batch_key="batch")
    >>> vae = scvi.model.SCVI(adata)
    >>> vae.train()
    >>> adata.obsm["X_scVI"] = vae.get_latent_representation()
    >>> adata.obsm["X_normalized_scVI"] = vae.get_normalized_expression()
    Notes
    -----
    See further usage examples in the following tutorials:
    1. :doc:`/tutorials/notebooks/api_overview`
    2. :doc:`/tutorials/notebooks/harmonization`
    3. :doc:`/tutorials/notebooks/scarches_scvi_tools`
    4. :doc:`/tutorials/notebooks/scvi_in_R`
    """

    _module_cls = VAE
    # self.device_ = "cpu"

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        fac_loss_noise: float = 0.1,
        fac_total_count: float = 0.0,
        sigma_eta_prior: float = 0.1,
        log_total_count_mu: float = 1,
        log_total_count_std: float = 10,
        total_count: float = 1e3,
        fac_loss_correlation: float = 0.0,
        kl_weight_fac: float = 1.0,
        use_constant_total_count: bool = True,
        mean_eta_hat: float = 0.01,
        tmax: float = 20,
        burst_B_gene: bool = False,
        gene_clusters = None,
        dropout_rate: float = 0.1,
        latent_distribution: Literal["normal", "ln"] = "normal",
        device_: Literal["cpu", "cuda"] = "cpu",
        **model_kwargs,
    ):
        super().__init__(adata)
        self.device_ = device_
        self.gene_clusters = gene_clusters

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
            n_batch=n_batch,
            n_labels=self.summary_stats.n_labels,
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            n_cats_per_cov=n_cats_per_cov,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            fac_loss_noise=fac_loss_noise,
            fac_total_count=fac_total_count,
            kl_weight_fac=kl_weight_fac,
            sigma_eta_prior=sigma_eta_prior,
            log_total_count_mu=log_total_count_mu,
            log_total_count_std=log_total_count_std,
            total_count=total_count,
            fac_loss_correlation=fac_loss_correlation,
            use_constant_total_count=use_constant_total_count,
            mean_eta_hat=mean_eta_hat,
            tmax=tmax,
            burst_B_gene=burst_B_gene,
            dropout_rate=dropout_rate,
            latent_distribution=latent_distribution,
            use_size_factor_key=use_size_factor_key,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            **model_kwargs,
        )
        self.module.minified_data_type = self.minified_data_type
        self._model_summary_string = (
            "SCVI Model with the following params: \nn_hidden: {}, n_latent: {}, n_layers: {}, dropout_rate: "
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
        batch_key: Optional[str] = None,
        labels_key: Optional[str] = None,
        size_factor_key: Optional[str] = None,
        mu_naive_key: Optional[str] = None,
        var_naive_key: Optional[str] = None,
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
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False
            ),
            LayerField(
                REGISTRY_KEYS.Mn_KEY, mu_naive_key, is_count_data=False
            ),
            LayerField(
                REGISTRY_KEYS.Vn_KEY, var_naive_key, is_count_data=False
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

        mu_list = []
        var_list = []
        library_list = []
        # total_counts_list = []

        for tensors in scdl:
            inference_kwargs = {"n_samples": n_samples}
            inference_outputs, generative_outputs = self.module.forward(
                tensors=tensors,
                inference_kwargs=inference_kwargs,
                compute_loss=False,
            )
            mu_tmp = generative_outputs["mu"]
            var_tmp = generative_outputs["var"]
            library_tmp = F.sigmoid(inference_outputs['library'])
            # total_counts_tmp = F.softplus(generative_outputs["total_count_pred"])
                
            if 'cuda' in self.device_:
                mu_list += [mu_tmp.cpu().numpy()]
                var_list += [var_tmp.cpu().numpy()]
                library_list += [library_tmp.cpu().numpy()]
                # total_counts_list += [total_counts_tmp.cpu().numpy()]
            else:
                mu_list += [mu_tmp]
                var_list += [var_tmp]
                library_list += [library_tmp]
                # total_counts_list += [total_counts_tmp]
        mu_list = np.concatenate(mu_list, axis=-2)
        var_list = np.concatenate(var_list, axis=-2)
        library_list = np.concatenate(library_list, axis=-2)
        # total_counts_list = np.concatenate(total_counts_list, axis=-2)

        return_dict = {}
        return_dict["mu"] = mu_list
        return_dict["var"] = var_list
        return_dict["library"] = library_list
        # return_dict["total_counts"] = total_counts_list

        return return_dict
    
    
    @torch.inference_mode()
    def get_batch_corrected_expression(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        transform_batch: Optional[Sequence[Union[Number, str]]] = None,
        gene_list: Optional[Sequence[str]] = None,
        library_size: Union[float, Literal["latent"]] = 1,
        n_samples: int = 1,
        n_samples_overall: int = None,
        weights: Optional[Literal["uniform", "importance"]] = None,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        r"""Returns the normalized (decoded) gene expression.

        This is denoted as :math:`\rho_n` in the scVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        transform_batch
            Batch to condition on.
            If transform_batch is:

            - None, then real observed batch is used.
            - int, then batch transform_batch is used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        library_size
            Scale the expression frequencies to a common library size.
            This allows gene expression levels to be interpreted on a common scale of relevant
            magnitude. If set to `"latent"`, use the latent library size.
        n_samples
            Number of posterior samples to use for estimation.
        n_samples_overall
            Number of posterior samples to use for estimation. Overrides `n_samples`.
        weights
            Weights to use for sampling. If `None`, defaults to `"uniform"`.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.
        importance_weighting_kwargs
            Keyword arguments passed into :meth:`~scvi.model.base.RNASeqMixin._get_importance_weights`.

        Returns
        -------
        If `n_samples` is provided and `return_mean` is False,
        this method returns a 3d tensor of shape (n_samples, n_cells, n_genes).
        If `n_samples` is provided and `return_mean` is True, it returns a 2d tensor
        of shape (n_cells, n_genes).
        In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        Otherwise, the method expects `n_samples_overall` to be provided and returns a 2d tensor
        of shape (n_samples_overall, n_genes).
        """
        adata = self._validate_anndata(adata)

        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            assert n_samples == 1  # default value
            n_samples = n_samples_overall // len(indices) + 1
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        transform_batch = _get_batch_code_from_category(
            self.get_anndata_manager(adata, required=True), transform_batch
        )

        gene_mask = (
            slice(None) if gene_list is None else adata.var_names.isin(gene_list)
        )

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "`return_numpy` must be `True` if `n_samples > 1` and `return_mean` "
                    "is`False`, returning an `np.ndarray`.",
                    UserWarning,
                    stacklevel=settings.warnings_stacklevel,
                )
            return_numpy = True
        if library_size == "latent":
            generative_output_key = "mu"
            scaling = 1
        else:
            generative_output_key = "scale"
            scaling = library_size

        store_distributions = weights == "importance"
        if store_distributions and len(transform_batch) > 1:
            raise NotImplementedError(
                "Importance weights cannot be computed when expression levels are averaged across batches."
            )

        # exprs = []
        mus = []
        vars = []
        # zs = []
        # qz_store = DistributionConcatenator()
        # px_store = DistributionConcatenator()
        for tensors in scdl:
            # per_batch_exprs = []
            per_batch_mus = []
            per_batch_vars = []
            for batch in transform_batch:
                generative_kwargs = self._get_transform_batch_gen_kwargs(batch)
                inference_kwargs = {"n_samples": n_samples}
                inference_outputs, generative_outputs = self.module.forward(
                    tensors=tensors,
                    inference_kwargs=inference_kwargs,
                    generative_kwargs=generative_kwargs,
                    compute_loss=False,
                )
                mu_tmp = generative_outputs["mu"]
                var_tmp = generative_outputs["var"]
                # exp_ = getattr(generative_outputs["px"], generative_output_key)
                # exp_ = exp_[..., gene_mask]
                # exp_ *= scaling
                # per_batch_mus.append(exp_[None].cpu())
                per_batch_mus.append(mu_tmp[None].cpu())
                per_batch_vars.append(var_tmp[None].cpu())
                # if store_distributions:
                #     qz_store.store_distribution(inference_outputs["qz"])
                #     px_store.store_distribution(generative_outputs["px"])

            # zs.append(inference_outputs["z"].cpu())
            # per_batch_exprs = torch.cat(per_batch_exprs, dim=0).mean(0).numpy()
            per_batch_mus = torch.cat(per_batch_mus, dim=0).mean(0).numpy()
            per_batch_vars = torch.cat(per_batch_vars, dim=0).mean(0).numpy()
            # exprs.append(per_batch_exprs)
            mus.append(per_batch_mus)
            vars.append(per_batch_vars)

        cell_axis = 1 if n_samples > 1 else 0
        # exprs = np.concatenate(exprs, axis=cell_axis)
        mus = np.concatenate(mus, axis=cell_axis)
        vars = np.concatenate(vars, axis=cell_axis)
        # zs = torch.concat(zs, dim=cell_axis)

        # if n_samples_overall is not None:
        #     # Converts the 3d tensor to a 2d tensor
        #     exprs = exprs.reshape(-1, exprs.shape[-1])
        #     n_samples_ = exprs.shape[0]
        #     if (weights is None) or weights == "uniform":
        #         p = None
        #     else:
        #         qz = qz_store.get_concatenated_distributions(axis=0)
        #         x_axis = 0 if n_samples == 1 else 1
        #         px = px_store.get_concatenated_distributions(axis=x_axis)
        #         p = self._get_importance_weights(
        #             adata,
        #             indices,
        #             qz=qz,
        #             px=px,
        #             zs=zs,
        #             **importance_weighting_kwargs,
        #         )
        #     ind_ = np.random.choice(n_samples_, n_samples_overall, p=p, replace=True)
        #     exprs = exprs[ind_]
        if n_samples > 1 and return_mean:
            # exprs = exprs.mean(0)
            mus = mus.mean(0)
            vars = vars.mean(0)

        if return_numpy is None or return_numpy is False:
            # return pd.DataFrame(
            #     exprs,
            #     columns=adata.var_names[gene_mask],
            #     index=adata.obs_names[indices],
            # )
            return pd.DataFrame(
                mus,
                columns=adata.var_names[gene_mask],
                index=adata.obs_names[indices],
            ), pd.DataFrame(
                vars,
                columns=adata.var_names[gene_mask],
                index=adata.obs_names[indices],
            )
        else:
            # return exprs
            return mus, vars
