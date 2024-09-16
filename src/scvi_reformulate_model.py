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

from scvi import REGISTRY_KEYS
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

from scvi.model.base import ArchesMixin, BaseMinifiedModeModelClass, RNASeqMixin, VAEMixin
from scvi.nn import FCLayers
from scvi_reformulate_module \
    import VAENoiseVelo
import torch
import pandas as pd

class SCVINoiseVelo(
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

    _module_cls = VAENoiseVelo
    # self.device_ = "cpu"

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        nclones: int = 4,
        n_neighbors: int = 10,
        burstF: torch.Tensor = None,
        burstB: torch.Tensor = None,
        mean_up: torch.Tensor = None, 
        var_up: torch.Tensor = None,
        mean_down: torch.Tensor = None, 
        var_down: torch.Tensor = None,
        lib_full: torch.Tensor = None,
        tmax: float = 20,
        burst_B_gene: bool = False,
        gene_clusters = None,
        edge_index: Iterable[torch.Tensor] = None,
        edge_attr: Iterable[torch.Tensor] = None,
        clusters_exclusive: bool = False,
        decoder_type: Literal["mu_var", "burst"] = "mu_var",
        direction_up: torch.Tensor = None,
        dropout_rate: float = 0.4,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
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
            n_neighbors=n_neighbors,
            nclones=nclones,
            edge_index=edge_index,
            edge_attr=edge_attr,
            tmax=tmax,
            gene_clusters=gene_clusters,
            direction_up=direction_up,
            clusters_exclusive=clusters_exclusive,
            mean_up=mean_up, 
            var_up=var_up,
            mean_down=mean_down,
            var_down=var_down,
            lib_full=lib_full,
            decoder_type=decoder_type,
            burst_B_gene=burst_B_gene,
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
            "SCVI Model with the following params: \nn_hidden: {}, n_latent: {}, n_layers: {}, dropout_rate: "
            "{}, dispersion: {}, gene_likelihood: {}, latent_distribution: {}"
        ).format(
            n_hidden,
            n_latent,
            n_layers,
            dropout_rate,
            dispersion,
            gene_likelihood,
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

        mean_list = []
        dispersion_list = []
        burst_B_list = []
        burst_F_list = []
        mu_list = []
        var_list = []
        library_list = []
        # copy_num_all = self.module.copy_num.detach().numpy()
        # pi_list = []
        # gamma_mRNA_list = []

        for tensors in scdl:
            inference_kwargs = {"n_samples": n_samples}
            _, generative_outputs = self.module.forward(
                tensors=tensors,
                inference_kwargs=inference_kwargs,
                compute_loss=False,
            )
            px = generative_outputs["px"]
            # px_r = px.theta
            # px_rate = px.mu
            burst_F_tmp_ = generative_outputs['px_r_final']
            burst_B_tmp_ = generative_outputs['burst_B']
            # pi_tmp_ = generative_outputs['pi_up_down']
#             copy_num_tmp_ = generative_outputs['copy_num']
#             px_r_ = generative_outputs["px_r"]
#             px_rate_ = generative_outputs["px_rate"]
            mu_tmp = generative_outputs["mu"]
            var_tmp = generative_outputs["var"]
            library_tmp = generative_outputs["library"]

#             n_batch = px_rate.size(0) if n_samples == 1 else px_rate.size(1)

            # px_r = px_r.cpu().numpy()
            # if len(px_r.shape) == 1:
            #     dispersion_list += [np.repeat(px_r[np.newaxis, :], n_batch, axis=0)]
            # else:
            #     dispersion_list += [px_r]
            #     px_r_list += [px_r_]
                
            if (self.device_=='cuda') | (self.device_=='cuda:0'):
                # mean_list += [px_rate.cpu().numpy()]
                burst_B_list += [burst_B_tmp_.cpu().numpy()]
                burst_F_list += [burst_F_tmp_.cpu().numpy()]
                # pi_list += [pi_tmp_.cpu().numpy()]
#                 copy_num_list += [copy_num_tmp_.cpu().numpy()]
                mu_list += [mu_tmp.cpu().numpy()]
                var_list += [var_tmp.cpu().numpy()]
                library_list += [library_tmp.cpu().numpy()]
            else:
                # mean_list += [px_rate]
                burst_B_list += [burst_B_tmp_]
                burst_F_list += [burst_F_tmp_]
                # pi_list += [pi_tmp_]
#                 copy_num_list += [copy_num_tmp_]
                mu_list += [mu_tmp]
                var_list += [var_tmp]
                library_list += [library_tmp]
        # means = np.concatenate(mean_list, axis=-2)
        # dispersions = np.concatenate(dispersion_list, axis=-2)
        burst_B_mat = np.concatenate(burst_B_list, axis=-2)
        burst_F_mat = np.concatenate(burst_F_list, axis=-2)
        # pi_all = np.concatenate(pi_list, axis=-3)
        mu_list = np.concatenate(mu_list, axis=-2)
        var_list = np.concatenate(var_list, axis=-2)
        library_list = np.concatenate(library_list, axis=-2)

        return_dict = {}
        # return_dict["mean"] = means
        return_dict["burst_B"] = burst_B_mat
        return_dict["burst_F"] = burst_F_mat
        # return_dict['pi'] = pi_all
        return_dict["mu"] = mu_list
        return_dict["var"] = var_list
        return_dict["library"] = library_list

#         if self.module.gene_likelihood == "zinb":
#             return_dict["dropout"] = dropout
# #             return_dict["dispersions"] = dispersions
#         if self.module.gene_likelihood == "nb":
#             return_dict["dispersions"] = dispersions

        return return_dict




    @torch.inference_mode()
    def get_state_assignment(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        gene_list: Optional[Sequence[str]] = None,
        hard_assignment: bool = False,
        n_samples: int = 20,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
    ) -> Tuple[Union[np.ndarray, pd.DataFrame], List[str]]:
        """Returns cells by genes by states probabilities.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        hard_assignment
            Return a hard state assignment
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray",
                    stacklevel=2,
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        states = []
        for tensors in scdl:
            minibatch_samples = []
            for _ in range(n_samples):
                _, generative_outputs = self.module.forward(
                    tensors=tensors,
                    compute_loss=False,
                )
                output = generative_outputs["px_pi"]
                output = output[..., gene_mask, :]
                output = output.cpu().numpy()
                minibatch_samples.append(output)
            # samples by cells by genes by four
            states.append(np.stack(minibatch_samples, axis=0))
            if return_mean:
                states[-1] = np.mean(states[-1], axis=0)

        states = np.concatenate(states, axis=0)
        state_cats = [
            "induction",
            "induction_steady",
            "repression",
            "repression_steady",
        ]
        if hard_assignment and return_mean:
            hard_assign = states.argmax(-1)

            hard_assign = pd.DataFrame(
                data=hard_assign, index=adata.obs_names, columns=adata.var_names
            )
            for i, s in enumerate(state_cats):
                hard_assign = hard_assign.replace(i, s)

            states = hard_assign

        return states, state_cats

    @torch.inference_mode()
    def get_latent_time(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        gene_list: Optional[Sequence[str]] = None,
        time_statistic: Literal["mean", "max"] = "mean",
        n_samples: int = 1,
        n_samples_overall: Optional[int] = None,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Returns the cells by genes latent time.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        time_statistic
            Whether to compute expected time over states, or maximum a posteriori time over maximal
            probability state.
        n_samples
            Number of posterior samples to use for estimation.
        n_samples_overall
            Number of overall samples to return. Setting this forces n_samples=1.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray",
                    stacklevel=2,
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        times = []
        for tensors in scdl:
            minibatch_samples = []
            for _ in range(n_samples):
                _, generative_outputs = self.module.forward(
                    tensors=tensors,
                    compute_loss=False,
                )
                pi = generative_outputs["px_pi"]
                ind_prob = pi[..., 0]
                steady_prob = pi[..., 1]
                rep_prob = pi[..., 2]
                # rep_steady_prob = pi[..., 3]
#                 switch_time = F.softplus(self.module.switch_time_unconstr)
                switch_time = F.sigmoid(self.module.switch_time_unconstr) * self.module.tmax

                ind_time = generative_outputs["px_rho"] * switch_time
                rep_time = switch_time + (
                    generative_outputs["px_tau"] * (self.module.tmax - switch_time)
                )

                if time_statistic == "mean":
                    output = (
                        ind_prob * ind_time
                        + rep_prob * rep_time
                        + steady_prob * switch_time
                        # + rep_steady_prob * self.module.t_max
                    )
                else:
                    t = torch.stack(
                        [
                            ind_time,
                            switch_time.expand(ind_time.shape),
                            rep_time,
                            torch.zeros_like(ind_time),
                        ],
                        dim=2,
                    )
                    max_prob = torch.amax(pi, dim=-1)
                    max_prob = torch.stack([max_prob] * 4, dim=2)
                    max_prob_mask = pi.ge(max_prob)
                    output = (t * max_prob_mask).sum(dim=-1)

                output = output[..., gene_mask]
                output = output.cpu().numpy()
                minibatch_samples.append(output)
            # samples by cells by genes by four
            times.append(np.stack(minibatch_samples, axis=0))
            if return_mean:
                times[-1] = np.mean(times[-1], axis=0)

        if n_samples > 1:
            # The -2 axis correspond to cells.
            times = np.concatenate(times, axis=-2)
        else:
            times = np.concatenate(times, axis=0)

        if return_numpy is None or return_numpy is False:
            return pd.DataFrame(
                times,
                columns=adata.var_names[gene_mask],
                index=adata.obs_names[indices],
            )
        else:
            return times


    @torch.inference_mode()
    def get_velocity(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        gene_list: Optional[Sequence[str]] = None,
        n_samples: int = 1,
        n_samples_overall: Optional[int] = None,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
        velo_statistic: str = "mean",
        velo_mode: Literal["spliced", "unspliced"] = "spliced",
        clip: bool = True,
        device_: Literal["cpu", "cuda"] = "cpu",
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Returns cells by genes velocity estimates.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        gene_list
            Return velocities for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        n_samples
            Number of posterior samples to use for estimation for each cell.
        n_samples_overall
            Number of overall samples to return. Setting this forces n_samples=1.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.
        velo_statistic
            Whether to compute expected velocity over states, or maximum a posteriori velocity over maximal
            probability state.
        velo_mode
            Compute ds/dt or du/dt.
        clip
            Clip to minus spliced value

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        adata = self._validate_anndata(adata)
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
            n_samples = 1
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray",
                    stacklevel=2,
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)
            
        def get_nb_params(px_r_final, \
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
            theta_ = mu_ / ((var_ / mu_) - 1 + 1e-5)
            theta_[mu_ == 0] = 1e-2
            return mu_, theta_
        def get_velocity_local(mu_final, mu_, var_, \
                               gamma_mRNA_, burst_B):
        #             gamma_mRNA_tmp = torch.exp(gamma_mRNA.clone())
        #             gamma_mRNA_tmp = gamma_mRNA_tmp.repeat(px_init_rate.shape[0], 1)
        #             print(f'px_init_rate = {px_init_rate}')
        #             print(f'px_final_rate = {px_final_rate}')
            velo_mu_ = (mu_final - mu_) * gamma_mRNA_
            velo_var_ = gamma_mRNA_ * \
                ((2 * burst_B + 1) * mu_final + mu_ - 2 * var_)
            return velo_mu_, velo_var_
        
        def get_nb_params_down(mu_, theta_, velo_mu_, \
                               velo_var_, \
                            id_genes_all, id_cells_all, \
                            id_state, px_r_final, \
                            px_final_rate, \
                            gamma_mRNA_, t_s, \
                            px_rho, px_tau, device_):
            # mu_ = torch.zeros_like(px_r_final, \
            #                        dtype=torch.float, device=device_)
            # theta_ = torch.zeros_like(px_r_final, \
            #                           dtype=torch.float, device=device_)
            # down genes, down cells
            # mu_ = mu_0.clone()
            # theta_ = theta_0.clone()
            id_ = torch.nonzero(id_state == 2).flatten()
            id_cells = id_cells_all.clone()[id_].flatten()
            id_genes = id_genes_all.clone()
            if len(id_cells.shape) > 0:
                if id_genes.shape[0] > id_cells.shape[0] :
                    id_cells = id_cells[:, None]
                else:
                    id_genes = id_genes[:, None]
                mu_init = self.module.mean_upper[id_genes]
                mu_init = mu_init.repeat(id_cells.shape[0], 1)
                theta_init = \
                    self.module.mean_upper[id_genes] / \
                        (self.module.var_upper[id_genes] / \
                        self.module.mean_upper[id_genes] - 1 + 1e-5)
                # theta_init[self.module.mean_upper[id_genes] == 0] = 1
                theta_init = theta_init.repeat(id_cells.shape[0], 1)
                var_init = mu_init * (1 + mu_init / theta_init)
                # mu_init = F.sigmoid(self.module.mean_max[id_genes]) * \
                #     self.module.mean_upper[id_genes]
                # mu_init = mu_init.repeat(id_cells.shape[0], 1)
                # burstB = px_final_rate / px_r_final
                # var_init = (burstB[id_cells, id_genes] + 1) * mu_init
                mu_tmp, theta_tmp = \
                    get_nb_params(1e-2 * torch.ones_like(px_final_rate[id_cells, id_genes]), \
                                mu_init, \
                                1e-2 * torch.ones_like(px_final_rate[id_cells, id_genes]), \
                                t_s[id_cells, id_genes] * px_rho[id_cells, id_genes], \
                                var_init, \
                                gamma_mRNA_[id_cells, id_genes])
                # theta_tmp[theta_tmp == 0] = 1e-4
                # mu_tmp[mu_tmp < 0] = 1e-2
                mu_[id_cells, id_genes] = mu_tmp
                # theta_tmp[theta_tmp == 0] = 1
                theta_[id_cells, id_genes] = theta_tmp
                var_tmp = mu_tmp * (1 + mu_tmp / theta_tmp)
                velo_mu_tmp, velo_var_tmp = \
                    get_velocity_local(torch.zeros_like(px_final_rate[id_cells, id_genes]), \
                                       mu_tmp, \
                                       var_tmp, \
                                       gamma_mRNA_[id_cells, id_genes], \
                                        torch.zeros_like(px_final_rate[id_cells, id_genes]))

                velo_mu_[id_cells, id_genes] = velo_mu_tmp
                velo_var_[id_cells, id_genes] = velo_var_tmp
                # mu_tmp.detach()
                # theta_tmp.detach()
                # velo_mu_tmp.detach()
                # velo_var_tmp.detach()

            # down genes, down steady cells
            id_ = torch.nonzero(id_state == 3).flatten()
            id_cells = id_cells_all.clone()[id_].flatten()
            id_genes = id_genes_all.clone()
            if len(id_cells.shape) > 0:
                if id_genes.shape[0] > id_cells.shape[0] :
                    id_cells = id_cells[:, None]
                else:
                    id_genes = id_genes[:, None]
                # mu_init = self.mean_lower[id_genes]
                # mu_init = mu_init.repeat(id_cells.shape[0], 1)
                # theta_init = \
                #     self.mean_lower[id_genes] / \
                #         (self.var_lower[id_genes] / \
                #          self.mean_lower[id_genes] - 1 + 1e-5)
                # theta_init = theta_init.repeat(id_cells.shape[0], 1)
                # var_init = mu_init * (1 + mu_init / theta_init)
                mu_[id_cells, id_genes] = \
                    1e-2 * torch.ones_like(px_r_final[id_cells, id_genes])
                theta_[id_cells, id_genes] = \
                    1e-2 * torch.ones_like(px_r_final[id_cells, id_genes])

            # down genes, up cells
            id_ = torch.nonzero(id_state == 0).flatten()
            id_cells = id_cells_all.clone()[id_].flatten()
            id_genes = id_genes_all.clone()
            # end_penalty = None
            if len(id_cells.shape) > 0:
                if id_genes.shape[0] > id_cells.shape[0] :
                    id_cells = id_cells[:, None]
                else:
                    id_genes = id_genes[:, None]
                # mu_init = self.module.mean_upper[id_genes]
                # mu_init = mu_init.repeat(id_cells.shape[0], 1)
                # theta_init = \
                #     self.module.mean_upper[id_genes] / \
                #         (self.module.var_upper[id_genes] / \
                #          self.module.mean_upper[id_genes] - 1 + 1e-5)
                # theta_init = theta_init.repeat(id_cells.shape[0], 1)
                # var_init = mu_init * (1 + mu_init / theta_init)
                mu_tmp = 1e-2 * torch.ones_like(px_r_final[id_cells, id_genes])
                theta_tmp = 1e-2 * torch.ones_like(px_r_final[id_cells, id_genes])
                # mu_tmp, theta_tmp = \
                #     self.get_nb_params(px_r_final[id_cells, id_genes], \
                #                 mu_init, \
                #                 px_final_rate[id_cells, id_genes], \
                #                 t_s[id_cells, id_genes], \
                #                 var_init, \
                #                 gamma_mRNA_[id_cells, id_genes])
                var_tmp = mu_tmp * (1 + mu_tmp / theta_tmp)
                # end_penalty = \
                #     torch.log((torch.sum((mu_tmp).pow(2), dim=1)).mean() + \
                #                1e-12) + \
                #     torch.log((torch.sum((var_tmp).pow(2), dim=1)).mean() + \
                #                1e-12)
                mu_tmp, theta_tmp = \
                    get_nb_params(px_r_final[id_cells, id_genes], \
                                mu_tmp, \
                                px_final_rate[id_cells, id_genes], \
                                (self.module.tmax - t_s[id_cells, id_genes]) * \
                                    px_tau[id_cells, id_genes], \
                                var_tmp, \
                                gamma_mRNA_[id_cells, id_genes])
                # theta_tmp[theta_tmp == 0] = 1e-4
                # var_tmp = mu_tmp * (1 + mu_tmp / theta_tmp)
                mu_[id_cells, id_genes] = mu_tmp
                theta_[id_cells, id_genes] = theta_tmp

                var_tmp = mu_tmp * (1 + mu_tmp / theta_tmp)
                burst_B = px_final_rate[id_cells, id_genes] / \
                    px_r_final[id_cells, id_genes]
                velo_mu_tmp, velo_var_tmp = \
                    get_velocity_local(px_final_rate[id_cells, id_genes], \
                                       mu_tmp, \
                                       var_tmp, \
                                       gamma_mRNA_[id_cells, id_genes], \
                                       burst_B)

                velo_mu_[id_cells, id_genes] = velo_mu_tmp
                velo_var_[id_cells, id_genes] = velo_var_tmp
                # mu_tmp.detach()
                # theta_tmp.detach()

            # down genes, up steady cells
            id_ = torch.nonzero(id_state == 1).flatten()
            id_cells = id_cells_all.clone()[id_].flatten()
            id_genes = id_genes_all.clone()
            if len(id_cells.shape) > 0:
                if id_genes.shape[0] > id_cells.shape[0] :
                    id_cells = id_cells[:, None]
                else:
                    id_genes = id_genes[:, None]
                # mu_init = self.module.mean_upper[id_genes]
                # mu_init = mu_init.repeat(id_cells.shape[0], 1)
                # theta_init = \
                #     self.module.mean_upper[id_genes] / \
                #         (self.module.var_upper[id_genes] / \
                #          self.module.mean_upper[id_genes] - 1 + 1e-5)
                # theta_init = theta_init.repeat(id_cells.shape[0], 1)
                mu_[id_cells, id_genes] = px_final_rate[id_cells, id_genes]
                theta_[id_cells, id_genes] = px_r_final[id_cells, id_genes]
                
            return None
        
        def get_nb_params_up(mu_, theta_, \
                            velo_mu_, velo_var_, \
                            id_genes_all, id_cells_all, \
                            id_state, px_r_final, \
                            px_final_rate, \
                            gamma_mRNA_, t_s, \
                            px_rho, px_tau, device_):
            # mu_ = mu_0.clone()
            # theta_ = theta_0.clone()
            # mu_ = torch.zeros_like(px_r_final, \
            #                        dtype=torch.float, device=device_)
            # theta_ = torch.zeros_like(px_r_final, \
            #                           dtype=torch.float, device=device_)
            # up genes, up cells
            id_ = torch.nonzero(id_state == 0).flatten()
            id_cells = id_cells_all.clone()[id_].flatten()
            id_genes = id_genes_all.clone()
            if len(id_cells.shape) > 0:
                if id_genes.shape[0] > id_cells.shape[0] :
                    id_cells = id_cells[:, None]
                else:
                    id_genes = id_genes[:, None]
                # mu_init = self.mean_lower[id_genes]
                # mu_init = mu_init.repeat(id_cells.shape[0], 1)
                # theta_init = \
                #     self.mean_lower[id_genes] / \
                #         (self.var_lower[id_genes] / \
                #          self.mean_lower[id_genes] - 1 + 1e-5)
                # theta_init = theta_init.repeat(id_cells.shape[0], 1)
                # var_init = mu_init * (1 + mu_init / theta_init)
                mu_init = 1e-2 * torch.ones_like(px_r_final[id_cells, id_genes], \
                                dtype=torch.float, device=device_) + \
                                0
                var_init = 2e-2 * torch.ones_like(px_r_final[id_cells, id_genes], \
                                dtype=torch.float, device=device_) + \
                                0
                # mu_init = F.sigmoid(self.module.mean_min[id_genes]) * \
                #     self.module.mean_upper[id_genes]
                # mu_init = mu_init.repeat(id_cells.shape[0], 1)
                # burstB = px_final_rate / px_r_final
                # var_init = (burstB[id_cells, id_genes] + 1) * mu_init
                mu_tmp, theta_tmp = \
                    get_nb_params(px_r_final[id_cells, id_genes], \
                                        mu_init, \
                                        px_final_rate[id_cells, id_genes], \
                                        t_s[id_cells, id_genes] * px_rho[id_cells, id_genes], \
                                        var_init, \
                                        gamma_mRNA_[id_cells, id_genes])
                mu_[id_cells, id_genes] = mu_tmp
                # theta_tmp[theta_tmp == 0] = 1e-4
                theta_[id_cells, id_genes] = theta_tmp
                var_tmp = mu_tmp * (1 + mu_tmp / theta_tmp)
                burst_B = px_final_rate[id_cells, id_genes] / \
                    px_r_final[id_cells, id_genes]
                velo_mu_tmp, velo_var_tmp = \
                    get_velocity_local(px_final_rate[id_cells, id_genes], \
                                       mu_tmp, \
                                       var_tmp, \
                                       gamma_mRNA_[id_cells, id_genes], \
                                       burst_B)

                velo_mu_[id_cells, id_genes] = velo_mu_tmp
                velo_var_[id_cells, id_genes] = velo_var_tmp
                # mu_tmp.detach()
                # theta_tmp.detach()

            # up genes, up steady cells
            id_ = torch.nonzero(id_state == 1).flatten()
            id_cells = id_cells_all.clone()[id_].flatten()
            id_genes = id_genes_all.clone()
            if len(id_cells.shape) > 0:
                if id_genes.shape[0] > id_cells.shape[0] :
                    id_cells = id_cells[:, None]
                else:
                    id_genes = id_genes[:, None]
                mu_[id_cells, id_genes] = px_final_rate[id_cells, id_genes]
                theta_[id_cells, id_genes] = px_r_final[id_cells, id_genes]

            # up genes, down cells
            id_ = torch.nonzero(id_state == 2).flatten()
            id_cells = id_cells_all.clone()[id_].flatten()
            id_genes = id_genes_all.clone()
            end_penalty = None
            if len(id_cells.shape) > 0:
                if id_genes.shape[0] > id_cells.shape[0] :
                    id_cells = id_cells[:, None]
                else:
                    id_genes = id_genes[:, None]
                # mu_init = self.mean_lower[id_genes]
                # mu_init = mu_init.repeat(id_cells.shape[0], 1)
                # theta_init = \
                #     self.mean_lower[id_genes] / \
                #         (self.var_lower[id_genes] / \
                #          self.mean_lower[id_genes] - 1 + 1e-5)
                # theta_init = theta_init.repeat(id_cells.shape[0], 1)
                # var_init = mu_init * (1 + mu_init / theta_init)
                mu_init = 1e-2 * torch.ones_like(px_r_final[id_cells, id_genes], \
                                dtype=torch.float, device=device_) + \
                                0
                var_init = 2e-2 * torch.ones_like(px_r_final[id_cells, id_genes], \
                                dtype=torch.float, device=device_) + \
                                0
                mu_tmp, theta_tmp = \
                    get_nb_params(px_r_final[id_cells, id_genes], \
                                mu_init, \
                                px_final_rate[id_cells, id_genes], \
                                t_s[id_cells, id_genes], \
                                var_init, \
                                gamma_mRNA_[id_cells, id_genes])
                # theta_tmp[theta_tmp == 0] = 1e-4
                var_tmp = mu_tmp * (1 + mu_tmp / theta_tmp)
                mu_tmp, theta_tmp = \
                    get_nb_params(torch.ones_like(theta_tmp), \
                                mu_tmp, \
                                1e-2 * torch.ones_like(theta_tmp), \
                                (self.module.tmax - t_s[id_cells, id_genes]) * \
                                    px_tau[id_cells, id_genes], \
                                var_tmp, \
                                gamma_mRNA_[id_cells, id_genes])
                # theta_tmp[theta_tmp == 0] = 1e-4
                mu_[id_cells, id_genes] = mu_tmp
                theta_[id_cells, id_genes] = theta_tmp
                var_tmp = mu_tmp * (1 + mu_tmp / theta_tmp)
                # burst_B = px_final_rate[id_cells, id_genes] / \
                #     px_r_final[id_cells, id_genes]
                velo_mu_tmp, velo_var_tmp = \
                    get_velocity_local(torch.zeros_like(theta_tmp), \
                                       mu_tmp, \
                                       var_tmp, \
                                       gamma_mRNA_[id_cells, id_genes], \
                                       torch.zeros_like(theta_tmp))

                velo_mu_[id_cells, id_genes] = velo_mu_tmp
                velo_var_[id_cells, id_genes] = velo_var_tmp

                # mu_tmp.detach()
                # theta_tmp.detach()

            # up genes, down steady cells
            id_ = torch.nonzero(id_state == 3).flatten()
            id_cells = id_cells_all.clone()[id_].flatten()
            id_genes = id_genes_all.clone()
            
            if len(id_cells.shape) > 0:
                if id_genes.shape[0] > id_cells.shape[0]:
                    id_cells = id_cells[:, None]
                else:
                    id_genes = id_genes[:, None]
                # mu_init = self.mean_lower[id_genes]
                # mu_init = mu_init.repeat(id_cells.shape[0], 1)
                # theta_init = \
                #     self.mean_lower[id_genes] / \
                #         (self.var_lower[id_genes] / \
                #          self.mean_lower[id_genes] - 1 + 1e-5)
                # theta_init = theta_init.repeat(id_cells.shape[0], 1)
                # var_init = mu_init * (1 + mu_init / theta_init)

                mu_[id_cells, id_genes] = \
                    1e-2 * torch.ones_like(px_r_final[id_cells, id_genes])
                theta_[id_cells, id_genes] = \
                    1e-2 * torch.ones_like(px_r_final[id_cells, id_genes])
                
            return None

        velo_vars = []
        vars_ = []
        mus_ = []
        velo_mus = []
        thetas_ = []
        probs_states = []
        # pi_up_down = F.softmax()
        for tensors in scdl:
            minibatch_samples_mus = []
            minibatch_samples_vars = []
            minibatch_samples_thetas = []
            minibatch_samples_velo_mus = []
            minibatch_samples_velo_vars = []
            minibatch_probs_states = []
            for _ in range(n_samples):
                inference_outputs, generative_outputs = self.module.forward(
                    tensors=tensors,
                    compute_loss=False,
                )
                px_r_final = generative_outputs["px_r_final"]
                px_final_rate = generative_outputs["px_final_rate"]
                gamma_mRNA_tmp = generative_outputs["gamma_mRNA"]
                px_tau = generative_outputs["px_tau"]
                px_rho = generative_outputs["px_rho"]
                y_clone = generative_outputs["y_clone"]
                y_state = generative_outputs["y_state"]
                mu_f = generative_outputs["mu"]
                theta_f = generative_outputs["theta"]
                var_f = mu_f * (1 + mu_f / theta_f)
                t_s = generative_outputs["t_s"]
                logits_states = generative_outputs["logits_state"]
                prob_state_tmp = F.sigmoid(logits_states)

                mu_ = torch.zeros_like(px_r_final, \
                                    dtype=torch.float, device=device_)
                theta_ = torch.zeros_like(px_r_final, \
                                        dtype=torch.float, device=device_)
                velo_mu_ = torch.zeros_like(px_r_final, \
                                    dtype=torch.float, device=device_)
                velo_var_ = torch.zeros_like(px_r_final, \
                                        dtype=torch.float, device=device_)
                state_vals = torch.quantile(y_state, 0.5, dim=0)
                if state_vals[0] > 0.5:
                    state_vals[0] = 1
                    state_vals[1] = 0
                else:
                    state_vals[0] = 0
                    state_vals[1] = 1
                # id_clust_up = torch.argmax(y_state, dim=1).flatten()
                # id_clust_down = 1 - id_clust_up
                # print(torch.unique(id_clust_up))
                # cells where gene cluster 1 up, gene cluster 2 down
                id_up_down = torch.arange(px_r_final.shape[0], \
                                            device=device_, \
                                            dtype=torch.long)
                id_state = torch.argmax(y_clone, dim=1)
                if state_vals[0] > state_vals[1]:
                    # id_up_down = torch.nonzero(id_clust_up == 0).flatten()
                    # print(y_state[id_up_down, :].shape)
                    end_penalty = None
                    if id_up_down.shape[0] > 0:
                        # print(f'1 = {id_up_down.shape}')
                        id_genes_up = self.gene_clusters[0]
                        id_genes_down = self.gene_clusters[1]
                        _ = \
                            get_nb_params_up(mu_, theta_, \
                                             velo_mu_, velo_var_, \
                                                id_genes_up, id_up_down, \
                                                id_state, px_r_final, \
                                                px_final_rate, \
                                                gamma_mRNA_tmp, t_s, \
                                                px_rho, px_tau, device_) 
                        # mu_ += mu_tmp
                        # theta_ += theta_tmp
                        _ = \
                            get_nb_params_down(mu_, theta_, \
                                                velo_mu_, velo_var_, \
                                                id_genes_down, id_up_down, \
                                                id_state, px_r_final, \
                                                px_final_rate, \
                                                gamma_mRNA_tmp, t_s, \
                                                px_rho, px_tau, device_) 

                else:
                    # cells where gene cluster 2 up, gene cluster 1 down
                    id_genes_up = self.gene_clusters[1]
                    id_genes_down = self.gene_clusters[0]
                    if id_up_down.shape[0] > 0:
                        # print(f'2 = {id_up_down.shape}')
                        _ = \
                            get_nb_params_up(mu_, theta_, \
                                                velo_mu_, velo_var_, \
                                                id_genes_up, id_up_down, \
                                                id_state, px_r_final, \
                                                px_final_rate, \
                                                gamma_mRNA_tmp, t_s, \
                                                px_rho, px_tau, device_) 
                        # mu_ += mu_tmp
                        # theta_ += theta_tmp
                        # if end_penalty_tmp is not None:
                        #     if end_penalty is None:
                        #         end_penalty = end_penalty_tmp
                        #     else:
                        #         end_penalty += end_penalty_tmp
                        _ = \
                            get_nb_params_down(mu_, theta_, \
                                                velo_mu_, velo_var_, \
                                                id_genes_down, id_up_down, \
                                                id_state, px_r_final, \
                                                px_final_rate, \
                                                gamma_mRNA_tmp, t_s, \
                                                px_rho, px_tau, device_)

                theta_[theta_ == 0] = 1e-4
                # expectation
                var_ = mu_ * (1 + mu_ / theta_)
                output_mu = (mu_)
                output_theta = (theta_)
                output_velo_mu = (velo_mu_)
                output_var = (var_)
                output_velo_var = (velo_var_)
                output_prob_state = (prob_state_tmp)
                # if velo_statistic == "mean":
                #     output = (
                #         ind_prob * velo_ind
                #         + rep_prob * velo_rep
                #         + steady_prob * velo_steady
                #     )
                #     output_var = (
                #         ind_prob * var_ind
                #         + rep_prob * var_rep
                #         + steady_prob * var_steady
                #     )
                # # maximum
                # else:
                #     v = torch.stack(
                #         [
                #             velo_ind,
                #             velo_steady.expand(velo_ind.shape),
                #             velo_rep,
                #             torch.zeros_like(velo_rep),
                #         ],
                #         dim=2,
                #     )
                #     max_prob = torch.amax(pi, dim=-1)
                #     max_prob = torch.stack([max_prob] * 4, dim=2)
                #     max_prob_mask = pi.ge(max_prob)
                #     output = (v * max_prob_mask).sum(dim=-1)

                output_mu = output_mu[..., gene_mask]
                output_mu = output_mu.cpu().numpy()
                minibatch_samples_mus.append(output_mu)
                output_velo_mu = output_velo_mu[..., gene_mask]
                output_velo_mu = output_velo_mu.cpu().numpy()
                minibatch_samples_velo_mus.append(output_velo_mu)
                output_var = output_var[..., gene_mask]
                output_var = output_var.cpu().numpy()
                minibatch_samples_vars.append(output_var)
                output_velo_var = output_velo_var[..., gene_mask]
                output_velo_var = output_velo_var.cpu().numpy()
                minibatch_samples_velo_vars.append(output_velo_var)
                output_theta = output_theta[..., gene_mask]
                output_theta = output_theta.cpu().numpy()
                minibatch_samples_thetas.append(output_theta)
                output_prob_state = output_prob_state[..., gene_mask]
                output_prob_state = output_prob_state.cpu().numpy()
                minibatch_probs_states.append(output_prob_state)
            # samples by cells by genes
            velo_mus.append(np.stack(minibatch_samples_velo_mus, axis=0))
            mus_.append(np.stack(minibatch_samples_mus, axis=0))
            vars_.append(np.stack(minibatch_samples_vars, axis=0))
            velo_vars.append(np.stack(minibatch_samples_velo_vars, axis=0))
            thetas_.append(np.stack(minibatch_samples_thetas, axis=0))
            probs_states.append(np.stack(minibatch_probs_states, axis=0))
            if return_mean:
                # mean over samples axis
                velo_mus[-1] = np.mean(velo_mus[-1], axis=0)
                velo_vars[-1] = np.mean(velo_vars[-1], axis=0)
                mus_[-1] = np.mean(mus_[-1], axis=0)
                vars_[-1] = np.mean(vars_[-1], axis=0)
                thetas_[-1] = np.mean(thetas_[-1], axis=0)
                probs_states[-1] = np.mean(probs_states[-1], axis=0)

        if n_samples > 1:
            # The -2 axis correspond to cells.
            mus_ = np.concatenate(mus_, axis=-2)
            vars_ = np.concatenate(vars_, axis=-2)
            thetas_ = np.concatenate(thetas_, axis=-2)
            velo_mus = np.concatenate(velo_mus, axis=-2)
            velo_vars = np.concatenate(velo_vars, axis=-2)
            probs_states = np.concatenate(probs_states, axis=-2)
        else:
            mus_ = np.concatenate(mus_, axis=0)
            vars_ = np.concatenate(vars_, axis=0)
            thetas_ = np.concatenate(thetas_, axis=0)
            velo_mus = np.concatenate(velo_mus, axis=0)
            velo_vars = np.concatenate(velo_vars, axis=0)
            probs_states = np.concatenate(probs_states, axis=0)

#         spliced = self.adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)

#         if clip:
#             velos = np.clip(velos, -spliced[indices], None)

        # if return_numpy is None or return_numpy is False:
        #     return pd.DataFrame(
        #         velos,
        #         columns=adata.var_names[gene_mask],
        #         index=adata.obs_names[indices],
        #     )
        # else:
        return mus_, velo_mus, vars_, velo_vars, thetas_, probs_states

    @torch.inference_mode()
    def get_expression_fit(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        gene_list: Optional[Sequence[str]] = None,
        n_samples: int = 1,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
        restrict_to_latent_dim: Optional[int] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        r"""Returns the fitted spliced and unspliced abundance (s(t) and u(t)).

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        adata = self._validate_anndata(adata)

        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray",
                    stacklevel=2,
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        fits_s = []
        fits_u = []
        for tensors in scdl:
            minibatch_samples_s = []
            minibatch_samples_u = []
            for _ in range(n_samples):
                inference_outputs, generative_outputs = self.module.forward(
                    tensors=tensors,
                    compute_loss=False,
                    generative_kwargs={"latent_dim": restrict_to_latent_dim},
                )

                gamma = inference_outputs["gamma"]
                beta = inference_outputs["beta"]
                alpha = inference_outputs["alpha"]
                alpha_1 = inference_outputs["alpha_1"]
                lambda_alpha = inference_outputs["lambda_alpha"]
                px_pi = generative_outputs["px_pi"]
                scale = generative_outputs["scale"]
                px_rho = generative_outputs["px_rho"]
                px_tau = generative_outputs["px_tau"]

                (
                    mixture_dist_s,
                    mixture_dist_u,
                    _,
                ) = self.module.get_px(
                    px_pi,
                    px_rho,
                    px_tau,
                    scale,
                    gamma,
                    beta,
                    alpha,
                    alpha_1,
                    lambda_alpha,
                )
                fit_s = mixture_dist_s.mean
                fit_u = mixture_dist_u.mean

                fit_s = fit_s[..., gene_mask]
                fit_s = fit_s.cpu().numpy()
                fit_u = fit_u[..., gene_mask]
                fit_u = fit_u.cpu().numpy()

                minibatch_samples_s.append(fit_s)
                minibatch_samples_u.append(fit_u)

            # samples by cells by genes
            fits_s.append(np.stack(minibatch_samples_s, axis=0))
            if return_mean:
                # mean over samples axis
                fits_s[-1] = np.mean(fits_s[-1], axis=0)
            # samples by cells by genes
            fits_u.append(np.stack(minibatch_samples_u, axis=0))
            if return_mean:
                # mean over samples axis
                fits_u[-1] = np.mean(fits_u[-1], axis=0)

        if n_samples > 1:
            # The -2 axis correspond to cells.
            fits_s = np.concatenate(fits_s, axis=-2)
            fits_u = np.concatenate(fits_u, axis=-2)
        else:
            fits_s = np.concatenate(fits_s, axis=0)
            fits_u = np.concatenate(fits_u, axis=0)

        if return_numpy is None or return_numpy is False:
            df_s = pd.DataFrame(
                fits_s,
                columns=adata.var_names[gene_mask],
                index=adata.obs_names[indices],
            )
            df_u = pd.DataFrame(
                fits_u,
                columns=adata.var_names[gene_mask],
                index=adata.obs_names[indices],
            )
            return df_s, df_u
        else:
            return fits_s, fits_u

    @torch.inference_mode()
    def get_gene_likelihood(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        gene_list: Optional[Sequence[str]] = None,
        n_samples: int = 1,
        batch_size: Optional[int] = None,
        return_mean: bool = True,
        return_numpy: Optional[bool] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        r"""Returns the likelihood per gene. Higher is better.

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
            magnitude. If set to `"latent"`, use the latent libary size.
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame includes
            gene names as columns. If either `n_samples=1` or `return_mean=True`, defaults to `False`.
            Otherwise, it defaults to `True`.

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names
            gene_mask = [True if gene in gene_list else False for gene in all_genes]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "return_numpy must be True if n_samples > 1 and return_mean is False, returning np.ndarray",
                    stacklevel=2,
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        rls = []
        for tensors in scdl:
            minibatch_samples = []
            for _ in range(n_samples):
                inference_outputs, generative_outputs = self.module.forward(
                    tensors=tensors,
                    compute_loss=False,
                )
                x = tensors[REGISTRY_KEYS.X_KEY]
                # unspliced = tensors[REGISTRY_KEYS.U_KEY]

                # gamma = inference_outputs["gamma"]
                # beta = inference_outputs["beta"]
                # alpha = inference_outputs["alpha"]
                # alpha_1 = inference_outputs["alpha_1"]
                # lambda_alpha = inference_outputs["lambda_alpha"]
                # px_pi = generative_outputs["px_pi"]
                # scale = generative_outputs["scale"]
                # px_rho = generative_outputs["px_rho"]
                px = generative_outputs["px"]
                px_r_final = generative_outputs["px_r_final"]

                # (
                #     mixture_dist_s,
                #     mixture_dist_u,
                #     _,
                # ) = self.module.get_px(
                #     px_pi,
                #     px_rho,
                #     px_tau,
                #     scale,
                #     gamma,
                #     beta,
                #     alpha,
                #     alpha_1,
                #     lambda_alpha,
                # )
                device = px_r_final.device
                reconst_loss_ = -px.log_prob(x.to(device))
                # reconst_loss_u = -mixture_dist_u.log_prob(unspliced.to(device))
                output = -(reconst_loss_)
                output = output[..., gene_mask]
                output = output.cpu().numpy()
                minibatch_samples.append(output)
            # samples by cells by genes by four
            rls.append(np.stack(minibatch_samples, axis=0))
            if return_mean:
                rls[-1] = np.mean(rls[-1], axis=0)

        rls = np.concatenate(rls, axis=0)
        return rls
    
    @torch.inference_mode()
    def get_latent_representation_new(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        give_mean: bool = True,
        mc_samples: int = 5000,
        batch_size: Optional[int] = None,
        return_dist: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Return the latent representation for each cell.

        This is typically denoted as :math:`z_n`.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        give_mean
            Give mean of distribution or sample from it.
        mc_samples
            For distributions with no closed-form mean (e.g., `logistic normal`), how many Monte Carlo
            samples to take for computing mean.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_dist
            Return (mean, variance) of distributions instead of just the mean.
            If `True`, ignores `give_mean` and `mc_samples`. In the case of the latter,
            `mc_samples` is used to compute the mean of a transformed distribution.
            If `return_dist` is true the untransformed mean and variance are returned.

        Returns
        -------
        Low-dimensional representation for each cell or a tuple containing its mean and variance.
        """
        self._check_if_trained(warn=False)

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        latent = []
        latent_qzm = []
        latent_qzv = []
        # latent_y = []
        latent_logits = []
        latent_probs = []
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            if "qz" in outputs:
                qz = outputs["qz"]
            else:
                qz_m, qz_v = outputs["qz_m"], outputs["qz_v"]
                qz = torch.distributions.Normal(qz_m, qz_v.sqrt())
            if give_mean:
                # does each model need to have this latent distribution param?
                if self.module.latent_distribution == "ln":
                    samples = qz.sample([mc_samples])
                    z = torch.nn.functional.softmax(samples, dim=-1)
                    z = z.mean(dim=0)
                else:
                    z = qz.loc
            else:
                z = outputs["z"]

            latent += [z.cpu()]
            latent_qzm += [qz.loc.cpu()]
            latent_qzv += [qz.scale.square().cpu()]
            latent_probs += [outputs["prob"].cpu()]
            latent_logits += [outputs["logits"].cpu()]
        return (
            (torch.cat(latent_qzm).numpy(), torch.cat(latent_qzv).numpy())
            if return_dist
            else torch.cat(latent).numpy(), \
                torch.cat(latent_probs).numpy(), \
                torch.cat(latent_logits).numpy()
        )

