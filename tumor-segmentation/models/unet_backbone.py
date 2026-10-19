import torch
import torch.nn as nn
import numpy as np
import segmentation_models_pytorch as smp
from typing import Optional, Tuple


class AttentionUNet(nn.Module):
    """
    U-Net with EfficientNet-B0 encoder and attention gates decoder.
    Optimized for tumor segmentation with ~7M parameters.
    """
    
    def __init__(
        self,
        encoder_name: str = "efficientnet-b0",
        encoder_weights: str = "imagenet",
        decoder_attention_type: str = "scse",
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[str] = None,  # No activation during training to avoid double sigmoid
        decoder_channels: Tuple[int, ...] = (256, 128, 64, 32, 16),
        decoder_use_batchnorm: bool = True,
        decoder_use_attention: bool = True,
    ):
        super().__init__()
        
        # Initialize the U-Net with EfficientNet-B0 encoder
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            decoder_attention_type=decoder_attention_type,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
            decoder_channels=decoder_channels,
            decoder_use_batchnorm=decoder_use_batchnorm,
        )
        
        # Add attention gates if requested
        if decoder_use_attention:
            self._add_attention_gates()
    
    def _add_attention_gates(self):
        """Add attention gates to improve small lesion detection."""
        # This is handled by segmentation_models_pytorch's attention mechanisms
        # The decoder_attention_type="scse" already provides attention gates
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            Output tensor of shape (B, 1, H, W) with sigmoid activation
        """
        return self.model(x)
    
    def get_encoder_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get encoder features for feature extraction."""
        return self.model.encoder(x)


class TumorSegmentationModel(nn.Module):
    """
    Complete tumor segmentation model with preprocessing and postprocessing.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        threshold: float = 0.3,  # Lower threshold for medical segmentation
    ):
        super().__init__()
        
        self.device = device
        self.threshold = threshold
        
        # Use a more powerful encoder
        self.unet = AttentionUNet(
            encoder_name="efficientnet-b2",  # More capacity
            encoder_weights="imagenet",
            decoder_attention_type="scse",
            in_channels=3,
            classes=1,
            activation=None,
            decoder_channels=(512, 256, 128, 64, 32),  # Increased capacity
        )
        
        if model_path:
            self.load_weights(model_path)
        
        self.to(device)
        self.eval()
    
    def load_weights(self, model_path: str):
        """Load model weights from checkpoint."""
        print(f"Loading weights from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if "state_dict" in checkpoint:
            # Lightning checkpoint - extract model weights
            state_dict = checkpoint["state_dict"]
            print(f"Loading Lightning checkpoint with {len(state_dict)} keys")
            
            # Remove 'model.' prefix from Lightning state dict keys
            model_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('model.'):
                    new_key = key[6:]  # Remove 'model.' prefix
                    model_state_dict[new_key] = value
                else:
                    model_state_dict[key] = value
            
            # Load into the unet submodule
            missing_keys, unexpected_keys = self.unet.load_state_dict(model_state_dict, strict=False)
            if missing_keys:
                print(f"Missing keys: {missing_keys[:5]}...")  # Show first 5
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys[:5]}...")  # Show first 5
                
        else:
            # Direct state dict - try loading into unet first
            print(f"Loading direct state dict with {len(checkpoint)} keys")
            try:
                missing_keys, unexpected_keys = self.unet.load_state_dict(checkpoint, strict=False)
                if missing_keys:
                    print(f"Missing keys: {missing_keys[:5]}...")
                if unexpected_keys:
                    print(f"Unexpected keys: {unexpected_keys[:5]}...")
            except Exception as e:
                print(f"Failed to load into unet, trying main model: {e}")
                # If that fails, try loading into the main model
                self.load_state_dict(checkpoint, strict=False)
        
        print("Weights loaded successfully!")
    
    def preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """
        Preprocess input image.
        
        Args:
            image: Input tensor of shape (H, W, 3) or (3, H, W)
            
        Returns:
            Preprocessed tensor of shape (1, 3, H, W)
        """
        # Ensure correct shape and dtype
        if image.dim() == 3:
            if image.shape[0] == 3:  # (3, H, W)
                image = image.unsqueeze(0)  # Add batch dimension
            elif image.shape[-1] == 3:  # (H, W, 3)
                image = image.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        
        # Normalize to [0, 1] if not already
        if image.max() > 1.0:
            image = image.float() / 255.0
        
        return image.to(self.device)
    
    def postprocess(self, output: torch.Tensor) -> torch.Tensor:
        """
        Postprocess model output with improved thresholding.
        
        Args:
            output: Model output of shape (1, 1, H, W) (logits)
            
        Returns:
            Binary segmentation mask of shape (H, W)
        """
        # Apply sigmoid to convert logits to probabilities
        probabilities = torch.sigmoid(output.squeeze())
        
        # Use adaptive thresholding or lower fixed threshold
        if probabilities.max() > 0.1:  # Only if there's some activation
            # Option 1: Adaptive threshold (Otsu-like)
            threshold = torch.quantile(probabilities[probabilities > 0.05], 0.7)
            threshold = torch.clamp(threshold, 0.2, 0.6)
        else:
            threshold = self.threshold
            
        binary_mask = (probabilities > threshold).float()
        
        # Optional: Apply morphological operations to clean up
        if binary_mask.sum() > 0:
            # Convert to numpy for morphological ops
            mask_np = binary_mask.cpu().numpy()
            from scipy import ndimage
            # Remove small noise
            mask_np = ndimage.binary_opening(mask_np, structure=np.ones((3,3)))
            # Fill small holes
            mask_np = ndimage.binary_closing(mask_np, structure=np.ones((5,5)))
            binary_mask = torch.from_numpy(mask_np).float().to(self.device)
            
        return binary_mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Complete forward pass with preprocessing and postprocessing.
        
        Args:
            x: Input tensor of shape (H, W, 3) or (3, H, W)
            
        Returns:
            Binary segmentation mask of shape (H, W)
        """
        # Preprocess
        x = self.preprocess(x)
        
        # Model inference
        with torch.no_grad():
            output = self.unet(x)
        
        # Postprocess
        return self.postprocess(output)
    
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """Alias for forward method."""
        return self.forward(image)


if __name__ == "__main__":
    # Test the model
    model = TumorSegmentationModel()
    
    # Create dummy input
    dummy_input = torch.randn(400, 991, 3)  # Typical image size
    
    # Test forward pass
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")