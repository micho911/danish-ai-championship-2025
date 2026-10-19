#!/usr/bin/env python3
"""
Local training script for tumor segmentation model.
This script can be run locally to train the model before deploying.
"""

import os
import sys
from pathlib import Path

# Add models directory to path
sys.path.append(str(Path(__file__).parent / "models"))

import torch
import pytorch_lightning as pl
from unet_backbone import AttentionUNet
from lightning_module import train_model
from dataset import create_data_loaders

# Optimize for Tensor Cores on NVIDIA GPUs
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium')


def main():
    """Main training function."""
    print("Starting local training for tumor segmentation model...")
    
    # Set random seed
    pl.seed_everything(42)
    
    # Create output directories
    os.makedirs("models/weights", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Check if data directory exists
    data_dir = "data"
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} does not exist")
        return
    
    # Create model
    print("Creating model...")
    model = AttentionUNet(
        encoder_name="efficientnet-b0",
        encoder_weights="imagenet",
        decoder_attention_type="scse",
        in_channels=3,
        classes=1,
        activation=None,  # No activation during training to avoid double sigmoid
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        data_dir=data_dir,
        batch_size=4,  # Smaller batch size for local training
        num_workers=2,  # Fewer workers for local training
        image_size=(400, 991),
        include_controls=True,
        control_weight=0.3,
        val_split=0.2,
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Check for data leakage
    print("\nChecking for data leakage...")
    train_paths = set(s['img_path'] for s in train_loader.dataset.patient_samples)
    val_paths = set(s['img_path'] for s in val_loader.dataset.patient_samples)
    overlap = train_paths.intersection(val_paths)
    
    if overlap:
        print(f"ERROR: Data leakage detected! {len(overlap)} overlapping samples:")
        for path in list(overlap)[:5]:
            print(f"  {path}")
        print("Aborting training due to data leakage.")
        return
    else:
        print("✓ No data leakage detected")
    
    # Print sample paths for verification
    print(f"\nTraining samples (first 3):")
    for i, sample in enumerate(train_loader.dataset.patient_samples[:3]):
        print(f"  {sample['img_path']}")
    
    print(f"Validation samples (first 3):")
    for i, sample in enumerate(val_loader.dataset.patient_samples[:3]):
        print(f"  {sample['img_path']}")
    
    print(f"Train dataset total: {len(train_loader.dataset)} samples")
    print(f"Val dataset total: {len(val_loader.dataset)} samples")
    
    # Train model
    print("Starting training...")
    trainer = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=50,  # Fewer epochs for local training
        learning_rate=3e-4,  # Higher learning rate for faster convergence
        weight_decay=1e-5,
        dice_weight=0.6,
        bce_weight=0.4,
        save_dir="models/weights",
        log_dir="logs",
        experiment_name="tumor_segmentation_local",
        patience=10,
        accelerator="auto",
        devices=1,
        precision="16-mixed" if torch.cuda.is_available() else "32",
    )
    
    print("Training completed!")
    
    # Print best model path
    best_model_path = trainer.checkpoint_callback.best_model_path
    if best_model_path:
        print(f"Best model saved to: {best_model_path}")
    else:
        print("No best model found")
    
    # Test inference
    print("Testing inference...")
    try:
        from models.infer import load_model, predict_single_image
        
        # Load the trained model
        model = load_model(best_model_path)
        
        # Test on a sample image
        test_image = "data/patients/imgs/patient_000.png"
        if os.path.exists(test_image):
            predict_single_image(model, test_image, "test_prediction.png")
            print("Inference test completed!")
        else:
            print("Test image not found, skipping inference test")
            
    except Exception as e:
        print(f"Error during inference test: {e}")


if __name__ == "__main__":
    main() 