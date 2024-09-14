# noSpliceVelo infers RNA velocity without separating unspliced and spliced transcripts
noSpliceVelo infers RNA velocity without separating unspliced and spliced transcripts

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
2. Also install [PyTorch](https://pytorch.org/get-started/previous-versions/); we used version 2.3.1. We recommend using a GPU for faster training. We tested wuth CUDA 12.1.
3. Additionally, install [scvi-tools](https://docs.scvi-tools.org/en/stable/installation.html). We used version 1.0.4.


