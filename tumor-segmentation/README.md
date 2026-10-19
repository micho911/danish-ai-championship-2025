# Tumor Segmentation

**Task** ([full description](TASK.md)): segment tumours in whole-body PET Maximum Intensity Projection (MIP) images. The training set has 182 patient images with masks and 426 healthy controls, and submissions are scored by Dice coefficient.

## Approach

1. **Attention U-Net** ([`models/`](models/)): a U-Net with attention gates on an ImageNet-pretrained EfficientNet encoder (`segmentation_models_pytorch`), trained in **PyTorch Lightning** with a Dice + BCE loss, mixed precision and controls included as empty masks. Entry points are [`train_local.py`](train_local.py) and [`models/train.py`](models/train.py). [`README_TUMOR_SEGMENTATION.md`](README_TUMOR_SEGMENTATION.md) and [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md) have more detail.
2. **HPC training** ([`slurm/train_unet.slurm`](slurm/train_unet.slurm)): a SLURM batch job for the GPU nodes on UCloud.
3. **Hybrid pipeline** ([`ultimate.py`](ultimate.py)): ResNet-encoder U-Net variants plus the scaffolding to combine them with **nnU-Net**.
4. **nnU-Net v2 data prep** ([`test.py`](test.py)): converts the dataset into nnU-Net's `Dataset<ID>_<name>` layout (`imagesTr`/`labelsTr`/`dataset.json`), with binarised patient labels and all-zero labels for controls.

## Status

Unfinished. The training code is complete, but the competition `api.py` is still the template stub and no trained weights are included.

## What I learned

- Segmentation architectures (U-Net, attention gates, pretrained encoders) and loss design for small, imbalanced foreground regions.
- How nnU-Net expects data to be laid out, and why its self-configuring approach is such a strong baseline.
- Submitting and monitoring GPU training jobs on a shared HPC cluster.
