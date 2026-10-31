# FSR-Net: Wavelet-Domain Dehazing with Mamba and Asymmetric Detail Enhancement

FSR-Net is a two-stage single-image dehazing project built on WDMamba. The current manuscript refers to the mask-guided architecture as LDM-Net; FSR-Net is the repository and code name.

## Method

The first stage applies a discrete wavelet transform to the hazy image. A Mamba-based low-frequency restoration network processes the low-frequency component, which is recombined with the high-frequency components by inverse wavelet transform to produce a coarse image. Direction-alternating state-space modeling and local–channel interaction combine long-range and local information.

The second stage refines the coarse image with a lightweight detail enhancement network. A spatial mask guides the fusion of local-detail and global-context branches.

## Project files

- `basicsr/archs/`: model architectures and wavelet components.
- `basicsr/utils/`: supporting image and model utilities.
- `inference.py`: legacy inference entry point.
- `basicsr/models/` and `basicsr/losses/`: model wrappers and losses.
- `review_runs/`: snapshots of experimental model variants.
- `requirements.txt` and `setup.py`: dependencies and package setup.

Training scripts and experiment configurations are withheld while the paper is under review and are planned for release after publication. Model code and losses are retained. Datasets and trained weights are not included.

## Environment and inference

The project was developed on Ubuntu with NVIDIA GPUs. A previously tested environment used Python 3.10, PyTorch 2.6.0 with CUDA 12.4 and `mamba_ssm` 2.2.5. The Mamba and causal-convolution extensions require versions compatible with your GPU, CUDA, PyTorch and compiler.

`inference.py` is a legacy script: it constructs `WaveMamba`, expects a compatible checkpoint containing `params`, assumes dataset-specific ground-truth pairing, and sets GPU 0 in the script. It is not a general inference entry point for all architecture variants; verify the model and checkpoint compatibility before use. The current repository does not include pretrained weights.

## Acknowledgments

This project builds on [WDMamba](https://github.com/SunJ000/WDMamba), [BasicSR](https://github.com/XPixelGroup/BasicSR), and [Wave-Mamba](https://github.com/AlexZou14/Wave-Mamba). We thank their authors for making their work available.
