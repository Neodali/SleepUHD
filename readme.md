# SleepUHD: FCA-Enhanced Variational Autoencoder for Ultra-High-Definition Image Restoration

> **[Presentation slides with inference examples and comparisons](./presentazione.pdf)**

## Introduction

This repository contains **SleepUHD**, a research project that originates from an in-depth study of the original [DreamUHD](https://ojs.aaai.org/index.php/AAAI/article/view/32640) paper ("DreamUHD: Frequency Enhanced Variational Autoencoder for Ultra-High-Definition Image Restoration", AAAI 2025).

By carefully analysing the FE-VAE encoder architecture we identified a critical issue in the **SFAD (Spectral Feature Aggregation and Distribution)** block. Inside the encoder, SFAD performs a channel-wise low-pass filtering operation: it aggregates spectral features across channels and redistributes a smoothed version of them. While this improves global consistency, it has an unwanted side-effect — it **softens the inter-channel variations**, radically altering the individual contribution of each channel and ultimately **compromising the colour variations** that each channel was specifically capturing.

To address this, we replaced the SFAD block with an **FCA (Frequency Channel Attention)** block, as proposed in:

> Qilong Wang, Banggu Wu, Pengfei Zhu, Peihua Li, Wangmeng Zuo, and Qinghua Hu.  
> *"FcaNet: Frequency Channel Attention Networks."*  
> IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

The FCA block leverages Discrete Cosine Transform (DCT)-based multi-spectral channel attention, allowing each channel to retain its distinctive frequency information while still performing meaningful channel recalibration — without the destructive averaging that SFAD introduced.

### Encoder Architecture

The figure below zooms into the FE-VAE encoder, highlighting the position of the original SFAD block:

![Zoom-in on SFAD in the encoder](./figs/zoominsfad.svg)

### SFAD → FCA Replacement

The following diagram illustrates how the SFAD block was replaced by the FCA block inside the encoder:

![SFAD replaced by FCA](./figs/sostittuzionesfadconfca.svg)

### Visual Comparison: SFAD vs FCA

A side-by-side visual comparison of the outputs produced by the two variants is shown below:

![Comparison SFAD vs FCA](./figs/confronttoSFADFCA.svg)

### Quantitative Results

![Results](./figs/results.png)

> **Note:** The results presented above were obtained **exclusively on the low-light enhancement task**. Furthermore, due to GPU memory constraints, all experiments were run with a **3K crop** rather than full-resolution UHD images. Results on other tasks and at full resolution may differ.

---

## Key Features (inherited from DreamUHD)
- VAE-Based UHD Image Restoration: To the best of our knowledge, this is the first work to introduce VAEs into the domain of UHD image restoration. By operating in the compact latent space of a VAE, our framework enhances restoration consistency and significantly reduces computational overhead.
- Frequency-Enhanced VAE (FE-VAE): We propose a novel Fourier-based, frequency-enhanced VAE that is both lightweight and powerful. By incorporating the global perceptual capabilities of the Fourier domain, FE-VAE achieves a substantial reduction in parameter count and computational cost without compromising its representational power.
- Wavelet Transform-based Adapter (WTA): A wavelet-based adapter is introduced to supplement the high-frequency details essential for high-fidelity image restoration. This module effectively bridges the domain gap between the pre-trained VAE and degraded images by combining spatial and frequency information.
- **FCA Block (SleepUHD contribution):** Replaces the SFAD block in the FE-VAE encoder with a DCT-based frequency channel attention module that preserves per-channel colour fidelity.



## Framework Overview
The SleepUHD framework builds on DreamUHD's three main components:
- A frozen, pre-trained FE-VAE (with SFAD replaced by FCA): This serves as the backbone of our model, providing an efficient and compact latent space for image representation. 
- A Wavelet Transform-based Adapter (WTA): This module works in tandem with the FE-VAE to inject high-frequency information and mitigate the domain gap.
- An arbitrary restoration network (IRNet): This is a lightweight network that performs the actual restoration task within the latent space. 

![Framework](./figs/framework.png)


## Environment Setup
- Python 3.8+
- PyTorch (CUDA recommended)
- Recommended packages (see scripts):

Quick install (mirror example):
```bash
pip3 install PyWavelets packaging lpips opencv-python einops scikit-image torchmetrics omegaconf tensorboard thop lmdb matplotlib timm openai-clip pandas facexlib sentencepiece future icecream imgaug accelerate addict transformers==4.37.2 pyiqa
```



## Repository Structure
- `basicsr/`: training/validation pipelines, models, architectures, losses, metrics, data
- `options/`: YAML configs for VAE pretraining and DreamUHD tasks
- `inference.py`: single/multi-image inference entry
- `train.sh`, `test.sh`, `test_VAE.sh`: usage examples
- `metrics.py`, `calculate_psnr_ssim.py`: evaluation utilities
- `weight/`: place pretrained weights here (see below)


## Datasets
Update the dataset paths in the YAMLs under `options/` to your local locations.
- Low-light example (`options/DreamUHD_LL.yml`):
  - Train: `./data/UHD_LL/training_set/{input,gt}`
  - Val/Test: `./data/UHD_LL/test/{input,gt}`
- Dehaze example (`options/DreamUHD_haze.yml`):
  - Train: `./data/UHD_haze/train/{input,gt}`
  - Test: `./data/UHD_haze/test/{input,gt}`
- Deblur example (`options/DreamUHD_bulr.yml`):
  - Train: `./data/UHD_deblur/train/{input_new,gt_new}`
  - Test: `./data/UHD_deblur/test/...`

If you pretrain the FE-VAE, point `vae_weight` and `config` to the VAE checkpoint and config.


## Pretrained Weights
Place weights in `weight/` or provide relative paths:
- FE-VAE for low-light: `weight/VAE_LL.pth` (update in YAML: `network_g.vae_weight`)
- Task weights:
  - Low-light: `weight/DreamUHD_lowlight.pth`
  - Dehaze: `weight/DreamUHD_haze.pth`


## Training
Use `basicsr/train.py` with a task YAML in `options/`.
Example (low-light):
```bash
python ./basicsr/train.py -opt ./options/DreamUHD_LL.yml
```
Notes:
- Adjust `datasets.*` paths and `logger` settings in the YAML.
- EMA: some configs use `param_key: params_ema`; ensure your weights match.

VAE pretraining (optional):
```bash
python ./basicsr/train.py -opt ./options/VAE_LL.yml
```


## Inference
Single-folder inference with a task config and weight:
```bash
python ./inference.py \
  --config ./options/DreamUHD_haze.yml \
  --input  ./data/UHD_haze/test/input \
  --weight ./weight/DreamUHD_haze.pth \
  --output ./exp/haze
```
Common flags:
- `--test_tile`: enable tiled inference
- `--max_size`: max image size for whole-image inference (otherwise tiles)

Example (low-light):
```bash
python ./inference.py \
  --config ./options/DreamUHD_LL.yml \
  --input ./data/UHD_LL/test/input \
  --weight ./weight/DreamUHD_lowlight.pth \
  --output ./exp/LL
```


## Evaluation
Compute PSNR/SSIM on results:
```bash
python ./calculate_psnr_ssim.py \
  --gt_path ./data/UHD_haze/test/gt \
  --results_path ./exp/haze \
  --test_log ./test.log
```


## Results

See the [introduction section](#introduction) above for the visual comparison and quantitative results.

> Results are limited to the **low-light enhancement** task with a **3K crop** due to GPU memory constraints.


## Configuration
Key fields in `options/DreamUHD_*.yml`:
- `datasets`: paired datasets (`PairedImageDataset`) with `dataroot_lq`/`dataroot_gt`
- `network_g`: `type: DreamUHD` with VAE settings
  - `vae_weight`, `config`: path to FE-VAE checkpoint and its config
  - `param_key`: e.g., `params_ema` when using EMA weights
- `train`: optimizers, schedulers, and loss weights (L1, SSIM, FFT, LPIPS, GAN)
- `logger`: frequencies, TensorBoard/W&B

VAE configs (`options/VAE_*.yml`) use `model_type: VAEModel` and `network_g: AutoencoderKL_freup2`.


## Reproducing Paper Results
- Train or download FE-VAE weights, set `network_g.vae_weight` and `config` in your task YAML.
- Train task-specific DreamUHD model with corresponding YAML.
- Run inference and evaluate with the provided scripts.


## Citation
If you find this work helpful, please cite the original DreamUHD paper and the FcaNet paper:
```bibtex
@inproceedings{liu2025dreamuhd,
  title={DreamUHD: Frequency Enhanced Variational Autoencoder for Ultra-High-Definition Image Restoration},
  author={Liu, Yidi and Li, Dong and Xiao, Jie and Bao, Yuanfei and Xu, Senyan and Fu, Xueyang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={6},
  pages={5712--5720},
  year={2025}
}

@inproceedings{wang2021fcanet,
  title={FcaNet: Frequency Channel Attention Networks},
  author={Wang, Qilong and Wu, Banggu and Zhu, Pengfei and Li, Peihua and Zuo, Wangmeng and Hu, Qinghua},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages={783--792},
  year={2021}
}
```


## License
This repository is for research purposes. Please check the license terms of any upstream components in `basicsr/`.





