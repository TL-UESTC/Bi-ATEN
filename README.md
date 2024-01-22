# Bi-ATEN
Pytorch implementation of AAAI2024 paper Agile Multi-Source-Free Domain Adaptation.

# Additional figures
We provide figures with full legends in `figs-with-full-legends/` folder, with additional examples of Figure 5.

# Dataset preparation
Put the DomainNet dataset under `dataset/`

# Source pretrained weights
Download pretrained source models [here](https://drive.google.com/file/d/14C5EWnYax7LjzxSriaUttpYNxOQKszsY/view?usp=sharing). Decompress and put it under main directory.

# Necessary packages
- python==3.8
- pytorch==1.13
- timm
- termcolor
- Apex
```
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./
https://github.com/NVIDIA/apex/issues/1227
```

# How to run
Run `run.sh`. Logs are under `my/`.

# Key files
`main.py` contains main training code, and `models/model.py` contains model design.
