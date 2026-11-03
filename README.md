# FSR-Net: Wavelet-Domain Dehazing with Mamba and Asymmetric Detail Enhancement

FSR-Net is a two-stage single-image dehazing project built on WDMamba. The current manuscript refers to the mask-guided architecture as LDM-Net; FSR-Net is the repository and code name.

---

> **Abstract:** *State space models (SSMs), especially Mamba, have shown strong potential in image restoration due to their linear complexity and long-range modeling capability. However, haze degradation exhibits different characteristics across frequency and spatial domains: large-scale veiling effects mainly lie in low-frequency components, while residual detail degradation after reconstruction is spatially non-uniform. To address this, we propose LDM-Net, a two-stage wavelet-domain dehazing network. In the first stage, the input hazy image is decomposed by discrete wavelet transform (DWT), and a Mamba-based Low-Frequency Restoration Network (LRN) restores the low-frequency component to remove global haze and recover coarse structures. The restored low-frequency component is then recombined with the original high-frequency sub-bands through inverse wavelet transform (IWT) to obtain an intermediate result. In the second stage, a lightweight Detail Enhancement Network predicts a spatial mask from the intermediate result and adaptively fuses local-detail and global-context restoration branches. Extensive experiments on synthetic and real-world datasets demonstrate that LDM-Net achieves competitive dehazing performance with favorable computational efficiency.*

---

## Method

The first stage applies a discrete wavelet transform to the hazy image. A Mamba-based low-frequency restoration network processes the low-frequency component, which is recombined with the high-frequency components by inverse wavelet transform to produce a coarse image. Direction-alternating state-space modeling and local–channel interaction combine long-range and local information.

The second stage refines the coarse image with a lightweight detail enhancement network. A spatial mask guides the fusion of local-detail and global-context branches.

Training scripts and experiment configurations will be released after publication. Datasets and trained weights are not included in this repository.

## Environment and inference

The project was developed on Ubuntu with NVIDIA GPUs. A previously tested environment used Python 3.10, PyTorch 2.6.0 with CUDA 12.4 and `mamba_ssm` 2.2.5. The Mamba and causal-convolution extensions require versions compatible with your GPU, CUDA, PyTorch and compiler.

`inference.py` is a legacy script: it constructs `WaveMamba`, expects a compatible checkpoint containing `params`, assumes dataset-specific ground-truth pairing, and sets GPU 0 in the script. It is not a general inference entry point for all architecture variants; verify the model and checkpoint compatibility before use. The current repository does not include pretrained weights.

## Training results

The FSR-Net training results are available via [Baidu Netdisk: FSR-Net_Results](https://pan.baidu.com/s/1EKZZ_ZpodAxar64uG6TTJg?pwd=iw5y). Extraction code: `iw5y`.

## Acknowledgments

This project builds on [WDMamba](https://github.com/SunJ000/WDMamba), [BasicSR](https://github.com/XPixelGroup/BasicSR), and [Wave-Mamba](https://github.com/AlexZou14/Wave-Mamba). We thank their authors for making their work available.
