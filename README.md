# FSR-Net: A Coarse-to-Fine Image Dehazing Network with Frequency-Spatial Refinement

This is the office implementation of **FSR-Net: A Coarse-to-Fine Image Dehazing Network with Frequency-Spatial Refinement**.

Guangcheng Wang<sup>1</sup>, Xiaodong He<sup>1</sup>, Jifan Yang<sup>2</sup>, Ruifeng Gao<sup>1</sup>, Member, IEEE, Yipo Huang<sup>3</sup>, and Hancheng Zhu<sup>4</sup>, Member, IEEE

---

> **Abstract:** *State space models (SSMs), especially Mamba, have shown strong potential in image restoration due to their linear complexity and long-range modeling capability. However, haze degradation exhibits different characteristics across frequency and spatial domains: large-scale veiling effects mainly lie in low-frequency components, while residual detail degradation after reconstruction is spatially non-uniform. To address this, we propose LDM-Net, a two-stage wavelet-domain dehazing network. In the first stage, the input hazy image is decomposed by discrete wavelet transform (DWT), and a Mamba-based Low-Frequency Restoration Network (LRN) restores the low-frequency component to remove global haze and recover coarse structures. The restored low-frequency component is then recombined with the original high-frequency sub-bands through inverse wavelet transform (IWT) to obtain an intermediate result. In the second stage, a lightweight Detail Enhancement Network predicts a spatial mask from the intermediate result and adaptively fuses local-detail and global-context restoration branches. Extensive experiments on synthetic and real-world datasets demonstrate that LDM-Net achieves competitive dehazing performance with favorable computational efficiency.*

---

Training scripts and experiment configurations will be released after publication. Datasets and trained weights are not included in this repository.

## Dependencies and Installation

The project was developed on Ubuntu with NVIDIA GPUs. A previously tested environment used Python 3.10, PyTorch 2.6.0 with CUDA 12.4 and `mamba_ssm` 2.2.5. The Mamba and causal-convolution extensions require versions compatible with your GPU, CUDA, PyTorch and compiler.

```bash
git clone https://github.com/yoimiya2004/FSR-Net.git
cd FSR-Net
conda create -n fsrnet python=3.10
conda activate fsrnet
pip install -r requirements.txt
pip install -e .
```

## Train the model

The unified training script `train_fsrnet.sh` will be released after publication.

## Inference

```bash
python inference.py -i /path/to/hazy -g /path/to/gt -w /path/to/weights.pth -o /path/to/output
```

This legacy script requires a compatible WaveMamba checkpoint and dataset-specific image pairing.

## Training results

The FSR-Net training results are available via [Baidu Netdisk: FSR-Net_Results](https://pan.baidu.com/s/1EKZZ_ZpodAxar64uG6TTJg?pwd=iw5y). Extraction code: `iw5y`.

## Acknowledgments

This project builds on [WDMamba](https://github.com/SunJ000/WDMamba), [BasicSR](https://github.com/XPixelGroup/BasicSR), and [Wave-Mamba](https://github.com/AlexZou14/Wave-Mamba). We thank their authors for making their work available.
