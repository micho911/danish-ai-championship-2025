# Hybrid CNN Training Guide

This modified `ultimate.py` file now supports training on your own tumor segmentation data with simplified local paths.

## Features Added

1. **Training Mode**: Train the hybrid CNN model on your 2D MIP-PET data
2. **Testing Mode**: Test the trained model and compute metrics
3. **Prediction Mode**: Predict tumor segmentation on individual images
4. **Inference Mode**: Original inference pipeline with simplified paths
5. **Sample Creation**: Create sample input files for testing
6. **Simplified Paths**: All paths are now relative and work locally

## Directory Structure

After running the script, the following directories will be created automatically:

```
tumor-segmentation/
├── data/                    # Your training data (as before)
│   ├── patients/
│   │   ├── imgs/
│   │   └── labels/
│   └── controls/
│       └── imgs/
├── input/                   # Input files for inference mode
├── output/                  # Output files from inference
├── checkpoints/             # Pretrained model weights (if available)
├── temp/                    # Temporary processing files
├── results/                 # Processing results
└── hybrid_cnn_weights.pt    # Your trained model weights
```

## Installation

```bash
pip install -r requirements_ultimate.txt
```

## Usage

### 1. Training
```bash
python ultimate.py --mode train --data_root data --epochs 50 --batch_size 4 --lr 1e-4 --val_split 0.2
```

### 2. Training without healthy controls
```bash
python ultimate.py --mode train --data_root data --epochs 50 --batch_size 4 --no_controls
```

### 3. Testing trained model
```bash
python ultimate.py --mode test --data_root data
```

### 4. Predicting on a single image
```bash
python ultimate.py --mode predict --image_path path/to/image.png --output_path prediction.png
```

### 5. Create sample input files for testing inference
```bash
python ultimate.py --mode create_sample --image_path data/patients/imgs/patient_000.png
```

### 6. Run inference mode (processes files in input/ directory)
```bash
python ultimate.py --mode inference
```

## Path Simplifications Made

All paths have been simplified for local use:

- **Input**: `input/` (instead of `/input/`)
- **Output**: `output/` (instead of `/output/images/automated-petct-lesion-segmentation/`)
- **Checkpoints**: `checkpoints/` (instead of `/opt/algorithm/checkpoints/hybrid_cnn/`)
- **Temp files**: `temp/` (instead of complex nnUNet paths)
- **Results**: `results/` (instead of complex nnUNet paths)

The script automatically creates these directories when needed.

## Training Parameters

- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size for training (default: 4)
- `--lr`: Learning rate (default: 1e-4)
- `--val_split`: Validation split ratio (default: 0.2)
- `--no_controls`: Don't use control images in training
- `--data_root`: Root directory containing the data (default: 'data')

## Model Architecture

The model uses a two-stage approach:

1. **Stage 1**: ResNet encoder + U-Net decoder for initial segmentation
2. **Stage 2**: U-Net that refines the prediction using:
   - Original CT/PET channels
   - Stage 1 probability maps
   - Stage 1 predictions

## Output

- Trained model weights are saved as `hybrid_cnn_weights.pt`
- Training progress is printed to console
- Best model is saved based on validation Dice score
- Test results show mean Dice score and statistics

## Notes

- The model adapts the original 3D medical imaging approach to work with 2D MIP-PET images
- Images are automatically padded to be divisible by 8 for proper U-Net downsampling
- The dataset class handles variable image sizes
- Training includes both BCE loss and Dice loss for better segmentation performance
