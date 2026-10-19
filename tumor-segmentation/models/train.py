#!/usr/bin/env python3
"""
Training script for tumor segmentation model.
"""

import argparse
import os
import sys
from pathlib import Path

# Add the models directory to the path
sys.path.append(str(Path(__file__).parent))

import torch
import pytorch_lightning as pl
from unet_backbone import AttentionUNet
from lightning_module import train_model
from dataset import create_data_loaders

# Optimize for Tensor Cores on NVIDIA GPUs
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train tumor segmentation model")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--image_size", type=int, nargs=2, default=[400, 991], help="Image size (height, width)")
    parser.add_argument("--include_controls", action="store_true", help="Include control data in training")
    parser.add_argument("--control_weight", type=float, default=0.3, help="Weight for control samples")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    
    # Model arguments
    parser.add_argument("--encoder_name", type=str, default="efficientnet-b0", help="Encoder name")
    parser.add_argument("--encoder_weights", type=str, default="imagenet", help="Encoder weights")
    parser.add_argument("--decoder_attention_type", type=str, default="scse", help="Decoder attention type")
    
    # Training arguments
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate (recommended: 3e-4 for faster convergence)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--dice_weight", type=float, default=0.6, help="Weight for Dice loss")
    parser.add_argument("--bce_weight", type=float, default=0.4, help="Weight for BCE loss")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    
    # Hardware arguments
    parser.add_argument("--accelerator", type=str, default="auto", help="Training accelerator")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices")
    parser.add_argument("--precision", type=str, default="16-mixed", help="Training precision")
    
    # Output arguments
    parser.add_argument("--save_dir", type=str, default="models/weights", help="Directory to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs")
    parser.add_argument("--experiment_name", type=str, default="tumor_segmentation", help="Experiment name")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic training")
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    pl.seed_everything(args.seed)
    
    # Create output directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data directory {args.data_dir} does not exist")
    
    print(f"Training configuration:")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Image size: {args.image_size}")
    print(f"  Include controls: {args.include_controls}")
    print(f"  Max epochs: {args.max_epochs}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Devices: {args.devices}")
    print(f"  Precision: {args.precision}")
    print()
    
    # Create model
    print("Creating model...")
    model = AttentionUNet(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        decoder_attention_type=args.decoder_attention_type,
        in_channels=3,
        classes=1,
        activation=None,  # No activation during training to avoid double sigmoid
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print()
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=tuple(args.image_size),
        include_controls=args.include_controls,
        control_weight=args.control_weight,
        val_split=args.val_split,
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print()

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
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dice_weight=args.dice_weight,
        bce_weight=args.bce_weight,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        experiment_name=args.experiment_name,
        patience=args.patience,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
    )

    print("Training completed!")

    # Print best model path
    best_model_path = trainer.checkpoint_callback.best_model_path
    if best_model_path:
        print(f"Best model saved to: {best_model_path}")
    else:
        print("No best model found")


if __name__ == "__main__":
    main() 