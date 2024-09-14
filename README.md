# noSpliceVelo infers RNA velocity without separating unspliced and spliced transcripts
noSpliceVelo infers RNA velocity without separating unspliced and spliced transcripts

## Environment:
```bash
conda create -n noSpliceVelo python=3.10.13
conda activate noSpliceVelo
pip install numpy==1.24.3
pip install pandas==2.0.1
pip install matplotlib==3.7.1
pip install seaborn==0.12.2
# Uncomment for PyTorch with CUDA 12.1, Linux and Windows
# pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
# Uncomment for PyTorch with CPU only, Linux and Windows
# pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu
# Uncomment for PyTorch with OSX
# pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
pip install anndata==0.8.0
pip install scanpy==1.9.3
pip install scvelo==0.2.5
```

