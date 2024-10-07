# noSpliceVelo infers RNA velocity without separating unspliced and spliced transcripts
This is the repository for the tool noSpliceVelo, which infers RNA velocity without separating unspliced and spliced transcripts as described in our manuscript [Mahajan, Tarun, and Sergei Maslov. "noSpliceVelo infers gene expression dynamics without separating unspliced and spliced transcripts." bioRxiv (2024): 2024-08.](https://doi.org/10.1101/2024.08.08.607261). noSpliceVelo models stochastic gene expression using an experimentally tested model of bursty gene expression. From the bursty model, we demonstrate that RNA velocity can be inferred by analyzing the relationship between the mean and variance of gene expression.

The central tenet of our method is that the temporal relationship between the variance and the mean of gene expression is such that variance always leads and the mean follows. Consequently, for bursty genes, for upregulation of gene expression, variance increases at a rate much faster than the mean, and the variance-vs-mean relationship has a negative curvature. Conversely, for downregulation of gene expression, the variance decreases at a rate much faster than the mean, and the variance-vs-mean relationship has a positive curvature. By separating the negative and positive curvatures, we can infer the direction of gene expression change and hence the RNA velocity.

Interestingly, splicing-based RNA velocity methods such as [scVelo](https://scvelo.readthedocs.io/en/stable/), and its modifications and extensions, also work by separating the uregulated and downregulated branches of gene expression by using their negative and positive curvatures, respectively. This is achieved using the causal relationship between unspliced and spliced transcripts, where unspliced abundance leads and spliced abundance follow.

Given the popularity of [scvi-tools](https://docs.scvi-tools.org/en/stable/) for count modeling of single-cell RNA sequencing (scRNA-seq) data, we have used its framework for implementing noSpliceVelo. Specifically, we use a deep neural network architecture inspired by the architecture introduced for the method veloVI in [Gayoso, Adam, et al. "Deep generative modeling of transcriptional dynamics for RNA velocity analysis in single cells." Nature methods 21.1 (2024): 50-59.](https://doi.org/10.1038/s41592-023-01994-w), which is also included in [scvi-tools](https://docs.scvi-tools.org/en/latest/api/reference/scvi.external.VELOVI.html). Despite some similarity in the architecture, noSpliceVelo does not require the separation of unspliced and spliced transcripts, which is a key feature of veloVI.

Next, we provide a brief overview of the environment setup, source code, reproducibility, and application to custom scRNA-seq data.

## Environment:
1. CPU:
    1. For Linux and Windows, create a conda environment using the yaml file `environmnent_cpu.yml` with the command:
  ```bash
  conda env create -f environment_cpu.yml
  ```
    2. For macOS, create a conda environment using the yaml file `environmnent_cpu_OSX.yml` with the command:
  ```bash
  conda env create -f environmnent_cpu_OSX.yml
  ```
2. GPU:
    1. For Linux and Windows, create a conda environment using the yaml file `environmnent_gpu.yml` with the command:
  ```bash
  conda env create -f environment_gpu.yml
  ```
    2. For macOS, create a conda environment using the yaml file `environmnent_gpu_OSX.yml` with the command:
  ```bash
  conda env create -f environmnent_gpu_OSX.yml
  ```
3. We recommend using a GPU; we tested with CUDA 12.1.

## Source code:
The source code for noSpliceVelo is available in the `src` directory. The main classes and functions are implemented in the `noSpliceVelo_model.py` and `noSpliceVelo_module.py` files. The `noSpliceVelo` class is inherited from the `scvi.model.SCVI` class, which is the base class for all models in [scvi-tools](https://docs.scvi-tools.org/en/stable/).

The main classes and functions for the first variational autoencoder, which is a modified version of scvi, are implemented in the `scvi_modified_capture_efficiency_module.py` and `scvi_modified_capture_efficiency_model.py` files.

## Reproducibility:
For reproducing the results in the manuscript, we have included extensive Jupyter notebooks in the `notebooks` directory. The following notebooks are available:

1. `pancreas_reproducibility.ipynb`: This notebook provides detailed steps for reproducing Figure 3 from the manuscript, which shows the results from the case study focusing on endocrinogenesis in mouse pancreas. The processed anndata object, with velocities estimated by noSpliceVelo, is available at [Mahajan, Tarun (2024). adata_pancrease_nosplicevelo.h5ad. figshare. Dataset.](https://doi.org/10.6084/m9.figshare.27021841.v1).
2. `mouse_erythroid_reproducibility.ipynb`: This notebook provides detailed steps for reproducing Figure 4 from the manuscript, which shows the results from the case study focusing on gastrulation in mouse erythroid development. The processed anndata object, with velocities estimated by noSpliceVelo, is available at [Mahajan, Tarun (2024). adata_mouse_erythroid. figshare. Dataset.](https://doi.org/10.6084/m9.figshare.27022324.v2).
3. `human_erythroid_reproducibility.ipynb`: This notebook provides detailed steps for reproducing Figure 4 from the manuscript, which shows the results from the case study focusing on human erythroid development. The processed anndata object, with velocities estimated by noSpliceVelo, is available at [Mahajan, Tarun (2024). adata_human_erythroid. figshare. Dataset.](https://doi.org/10.6084/m9.figshare.27022330.v1).
4. Coming soon: a python notebook for reproducing Figure 5.

## Custom data:
For your custom dataset or a publicly available dataset not already analyzed in our manuscript, refer to the notebook `pancreas_example_nosplicevelo.ipynb`