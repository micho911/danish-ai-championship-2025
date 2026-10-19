#!/bin/bash
#SBATCH --job-name=tumor_segmentation
#SBATCH --output=logs/tumor_segmentation_%j.out
#SBATCH --error=logs/tumor_segmentation_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your.email@institution.edu

# Load required modules (adjust based on your HPC system)
module load cuda/11.8
module load python/3.9
module load anaconda3

# Activate conda environment (if using conda)
# conda activate tumor_segmentation

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:${PWD}"

# Create necessary directories
mkdir -p logs
mkdir -p models/weights
mkdir -p data/controls/masks

# Generate synthetic black masks for controls (if not already present)
echo "Generating synthetic masks for control data..."
python -c "
import os
import numpy as np
import cv2
from pathlib import Path

control_img_dir = Path('data/controls/imgs')
control_mask_dir = Path('data/controls/masks')

if control_img_dir.exists() and not control_mask_dir.exists():
    control_mask_dir.mkdir(parents=True, exist_ok=True)
    
    for img_file in control_img_dir.glob('*.png'):
        # Load image to get dimensions
        img = cv2.imread(str(img_file))
        if img is not None:
            # Create black mask with same dimensions
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            
            # Save mask
            mask_file = control_mask_dir / img_file.name
            cv2.imwrite(str(mask_file), mask)
            print(f'Created mask: {mask_file}')
"

# Install requirements (if not already installed)
echo "Installing requirements..."
pip install -r requirements.txt

# Start training
echo "Starting tumor segmentation training..."

python models/train.py \
    --data_dir data \
    --batch_size 8 \
    --num_workers 4 \
    --image_size 400 991 \
    --include_controls \
    --control_weight 0.3 \
    --val_split 0.2 \
    --max_epochs 100 \
    --learning_rate 1e-4 \
    --weight_decay 1e-5 \
    --dice_weight 0.6 \
    --bce_weight 0.4 \
    --patience 15 \
    --accelerator gpu \
    --devices 1 \
    --precision 16-mixed \
    --save_dir models/weights \
    --log_dir logs \
    --experiment_name tumor_segmentation \
    --seed 42

echo "Training completed!"

# Optional: Run inference test
echo "Testing inference..."
python models/infer.py \
    --model_path models/weights/best_metric_model_segmentation2d_dict.ckpt \
    --image_path data/patients/imgs/patient_000.png \
    --output_path test_prediction.png \
    --benchmark \
    --num_runs 10

echo "Job completed successfully!" 