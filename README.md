# noSpliceVelo infers RNA velocity without separating unspliced and spliced transcripts
This is the repository for the tool noSpliceVelo, which infers RNA velocity without separating unspliced and spliced transcripts as described in our manuscript [Mahajan, Tarun, and Sergei Maslov. "noSpliceVelo infers gene expression dynamics without separating unspliced and spliced transcripts." bioRxiv (2024): 2024-08.](https://doi.org/10.1101/2024.08.08.607261). noSpliceVelo models stochastic gene expression using an experimentally tested model of bursty gene expression. From the bursty model, we demonstrate that RNA velocity can be inferred by analyzing the relationship between the mean and variance of gene expression.

The central tenet of our method is that the temporal relationship between the variance and the mean of gene expression is such that variance always leads and the mean follows. Consequently, for bursty genes, for upregulation of gene expression, variances increases at a rate much faster than the mean, and the variance-vs-mean relationship has a negative curvature. Conversely, for downregulation of gene expression, the variance decreases at a rate much faster than the mean, and the variance-vs-mean relationship has a positive curvature. By separating the negative and positive curvatures, we can infer the direction of gene expression change and hence the RNA velocity.

Interestingly, splicing-based RNA velocity methods such as [scVelo](https://scvelo.readthedocs.io/en/stable/), and its modifications and extensions, also work by separating the uregulated and downregulated branches of gene expression by using their negative and positive curvatures, respectively. This is achieved using the causal relationship between unspliced and spliced transcripts, where unspliced abundance leads and spliced abundance follow.

The tool uses a deep neural network to predict the velocity of the cells from single-cell transcriptomics data. The tool is described in 

Given the popularity of [scvi-tools](https://docs.scvi-tools.org/en/stable/) for count modeling of single-cell RNA sequencing (scRNA-seq) data, we have used its framework for implementing noSpliceVelo. Specifically, we use a deep neural network architecture inspired by the architecture introduced for the method veloVI in [Gayoso, Adam, et al. "Deep generative modeling of transcriptional dynamics for RNA velocity analysis in single cells." Nature methods 21.1 (2024): 50-59.](https://doi.org/10.1038/s41592-023-01994-w), which is also included in [scvi-tools](https://docs.scvi-tools.org/en/latest/api/reference/scvi.external.VELOVI.html). Despite the similarity in the architecture, noSpliceVelo does not require the separation of unspliced and spliced transcripts, which is a key feature of veloVI.

## Environment:
1. Create a new conda environment and install the required packages:
```bash
conda create -n noSpliceVelo_env python=3.10.13
conda activate noSpliceVelo_env
pip install numpy==1.24.3
pip install pandas==2.0.1
pip install matplotlib==3.7.1
pip install seaborn==0.12.2
pip install anndata==0.8.0
pip install scanpy==1.9.3
pip install scvelo==0.2.5
```
2. Also install [PyTorch](https://pytorch.org/get-started/previous-versions/); we used version 2.3.1. We recommend using a GPU; we tested with CUDA 12.1.
3. Additionally, install [scvi-tools](https://docs.scvi-tools.org/en/stable/installation.html). We used version 1.0.4.


