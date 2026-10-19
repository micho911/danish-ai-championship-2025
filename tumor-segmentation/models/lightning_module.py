import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, NAdam
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import segmentation_models_pytorch as smp
from typing import Dict, Any, Optional
import numpy as np
import cv2
import sys
from pathlib import Path


class DiceLoss(nn.Module):
    """Dice Loss implementation."""
    
    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        
        # Flatten predictions and targets
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        # Calculate Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """Combined loss: 0.6 * DiceLoss + 0.4 * BCEWithLogitsLoss."""
    
    def __init__(self, dice_weight: float = 0.6, bce_weight: float = 0.4):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        dice_loss = self.dice_loss(pred, target)
        bce_loss = self.bce_loss(pred, target)
        
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


class TumorSegmentationLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for tumor segmentation training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 3e-4,  # Higher learning rate for faster convergence
        weight_decay: float = 1e-5,
        dice_weight: float = 0.6,
        bce_weight: float = 0.4,
        threshold: float = 0.5,
    ):
        super().__init__()
        
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.threshold = threshold
        
        # Loss function
        self.criterion = CombinedLoss(dice_weight=dice_weight, bce_weight=bce_weight)
        
        # Metrics
        self.train_dice_scores = []
        self.val_dice_scores = []
        
        # Save hyperparameters
        self.save_hyperparameters(ignore=['model'])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Model returns logits (no sigmoid applied)
        return self.model(x)
    
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        images = batch['image']
        masks = batch['mask']
        
        # Forward pass
        logits = self(images)
        
        # Calculate loss
        loss = self.criterion(logits, masks)
        
        # Calculate Dice score
        pred_masks = torch.sigmoid(logits) > self.threshold
        dice_score = self._calculate_dice_score_torch(pred_masks, masks)
        
        # Collect dice score for epoch end
        self.train_dice_scores.append(dice_score)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_dice', dice_score, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        images = batch['image']
        masks = batch['mask']
        
        # Forward pass
        logits = self(images)
        
        # Calculate loss
        loss = self.criterion(logits, masks)
        
        # Calculate Dice score
        pred_masks = torch.sigmoid(logits) > self.threshold
        dice_score = self._calculate_dice_score_torch(pred_masks, masks)
        
        # DEBUG: Print actual values for first few batches
        if batch_idx < 3:  # First 3 batches
            print(f"DEBUG - Validation Batch {batch_idx}:")
            print(f"  Image shape: {images.shape}")
            print(f"  Mask shape: {masks.shape}")
            print(f"  Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
            print(f"  Sigmoid range: [{torch.sigmoid(logits).min().item():.4f}, {torch.sigmoid(logits).max().item():.4f}]")
            print(f"  Prediction sum: {pred_masks.sum().item()}")
            print(f"  Target sum: {masks.sum().item()}")
            print(f"  Dice score: {dice_score.item():.4f}")
            
            # Check if this is a control sample (all zeros)
            if masks.sum().item() == 0:
                print(f"  -> Control sample detected (no tumor)")
            else:
                print(f"  -> Patient sample detected (has tumor)")
        
        # Collect dice score for epoch end
        self.val_dice_scores.append(dice_score)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_dice', dice_score, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def _calculate_dice_score_torch(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate Dice score with proper handling of edge cases - bypassing buggy utils function."""
        # Convert to float and flatten for calculation
        pred_flat = pred.float().view(-1)
        target_flat = target.float().view(-1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        # Handle edge cases
        if union == 0:
            # Both masks are empty - perfect agreement for background-only images
            dice = torch.tensor(1.0, device=pred.device)
        else:
            dice = (2.0 * intersection) / union
        
        # Clamp to valid range [0, 1]
        dice = torch.clamp(dice, 0.0, 1.0)
        
        return dice
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Use NAdam optimizer with higher learning rate for faster convergence
        optimizer = NAdam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Use CosineAnnealingLR which doesn't need steps_per_epoch
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.learning_rate * 0.01  # Minimum learning rate
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"  # CosineAnnealingLR updates per epoch
            }
        }
    
    def on_train_epoch_end(self):
        """Log epoch-level metrics."""
        if self.train_dice_scores:
            avg_dice = torch.stack(self.train_dice_scores).mean()
            self.log('train_dice_epoch', avg_dice, on_epoch=True)
        self.train_dice_scores.clear()
    
    def on_validation_epoch_end(self):
        """Log epoch-level metrics."""
        if self.val_dice_scores:
            avg_dice = torch.stack(self.val_dice_scores).mean()
            self.log('val_dice_epoch', avg_dice, on_epoch=True)
        self.val_dice_scores.clear()


def create_callbacks(
    save_dir: str = "models/weights",
    monitor: str = "val_dice",
    mode: str = "max",
    patience: int = 15,
) -> list:
    """
    Create training callbacks.
    
    Args:
        save_dir: Directory to save checkpoints
        monitor: Metric to monitor
        mode: 'min' or 'max'
        patience: Early stopping patience
    
    Returns:
        List of callbacks
    """
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="best_metric_model_segmentation2d_dict",
        monitor=monitor,
        mode=mode,
        save_top_k=1,
        save_last=True,
        verbose=True
    )
    
    # Early stopping callback
    early_stopping_callback = EarlyStopping(
        monitor=monitor,
        mode=mode,
        patience=patience,
        verbose=True
    )
    
    return [checkpoint_callback, early_stopping_callback]


def create_logger(
    log_dir: str = "logs",
    experiment_name: str = "tumor_segmentation"
) -> TensorBoardLogger:
    """
    Create TensorBoard logger.
    
    Args:
        log_dir: Directory to save logs
        experiment_name: Name of the experiment
    
    Returns:
        TensorBoard logger
    """
    return TensorBoardLogger(
        save_dir=log_dir,
        name=experiment_name,
        version=None
    )


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    max_epochs: int = 100,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    dice_weight: float = 0.6,
    bce_weight: float = 0.4,
    save_dir: str = "models/weights",
    log_dir: str = "logs",
    experiment_name: str = "tumor_segmentation",
    patience: int = 15,
    accelerator: str = "auto",
    devices: int = 1,
    precision: str = "16-mixed",
) -> pl.Trainer:
    """
    Train the tumor segmentation model.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        max_epochs: Maximum number of epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        dice_weight: Weight for Dice loss
        bce_weight: Weight for BCE loss
        save_dir: Directory to save checkpoints
        log_dir: Directory to save logs
        experiment_name: Name of the experiment
        patience: Early stopping patience
        accelerator: Training accelerator
        devices: Number of devices
        precision: Training precision
    
    Returns:
        Trained trainer object
    """
    # Create Lightning module
    lightning_module = TumorSegmentationLightningModule(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dice_weight=dice_weight,
        bce_weight=bce_weight
    )
    
    # Create callbacks
    callbacks = create_callbacks(save_dir, patience=patience)
    
    # Create logger
    logger = create_logger(log_dir, experiment_name)
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        val_check_interval=0.5,
        gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        deterministic=False,
        enable_progress_bar=True,
        enable_model_summary=True,
        enable_checkpointing=True,
    )
    
    # Train the model
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    
    return trainer


if __name__ == "__main__":
    # Test the Lightning module
    from unet_backbone import AttentionUNet
    
    # Create model
    model = AttentionUNet()
    
    # Create Lightning module
    lightning_module = TumorSegmentationLightningModule(model)
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 400, 991)
    dummy_target = torch.randn(2, 1, 400, 991)
    
    output = lightning_module(dummy_input)
    loss = lightning_module.criterion(output, dummy_target)
    
    print(f"Output shape: {output.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Test optimizer configuration
    optimizer = lightning_module.configure_optimizers()
    print(f"Optimizer: {type(optimizer['optimizer']).__name__}")
    print(f"Scheduler: {type(optimizer['lr_scheduler']['scheduler']).__name__}")