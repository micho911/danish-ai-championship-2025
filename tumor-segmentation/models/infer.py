#!/usr/bin/env python3
"""
Inference script for tumor segmentation model.
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import numpy as np
import cv2
from PIL import Image
import time

# Add the models directory to the path
sys.path.append(str(Path(__file__).parent))

from unet_backbone import TumorSegmentationModel


def load_model(model_path: str, device: str = "auto") -> TumorSegmentationModel:
    """
    Load the trained tumor segmentation model.
    
    Args:
        model_path: Path to the trained model checkpoint
        device: Device to load the model on ('auto', 'cuda', 'cpu')
    
    Returns:
        Loaded model
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {model_path}")
    print(f"Using device: {device}")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load model
    model = TumorSegmentationModel(
        model_path=model_path,
        device=device,
        threshold=0.5
    )
    
    print("Model loaded successfully!")
    return model


def preprocess_image(image_path: str, target_size: tuple = (400, 991)) -> torch.Tensor:
    """
    Preprocess image for inference - MUST match training pipeline.
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Use SAME transforms as validation (without augmentations)
    val_transforms = A.Compose([
        A.Resize(height=target_size[0], width=target_size[1]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])
    
    # Apply transforms
    transformed = val_transforms(image=image)
    image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
    
    return image_tensor


def predict_single_image(
    model: TumorSegmentationModel,
    image_path: str,
    output_path: str = None,
    save_overlay: bool = True
) -> np.ndarray:
    """
    Predict segmentation for a single image.
    
    Args:
        model: Loaded model
        image_path: Path to input image
        output_path: Path to save output (optional)
        save_overlay: Whether to save overlay image
    
    Returns:
        Segmentation mask as numpy array
    """
    print(f"Processing image: {image_path}")
    
    # Preprocess image
    image_tensor = preprocess_image(image_path)
    
    # Make prediction
    start_time = time.time()
    with torch.no_grad():
        segmentation = model.predict(image_tensor)
    inference_time = time.time() - start_time
    
    print(f"Inference time: {inference_time:.3f} seconds")
    
    # Convert to numpy array
    segmentation_np = segmentation.cpu().numpy().astype(np.uint8)
    
    # Save results if output path is provided
    if output_path:
        # Save binary mask
        mask_path = output_path.replace('.png', '_mask.png')
        cv2.imwrite(mask_path, segmentation_np * 255)
        print(f"Mask saved to: {mask_path}")
        
        # Save overlay if requested
        if save_overlay:
            # Load original image
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            # Resize segmentation mask to match original image size
            original_height, original_width = original_image.shape[:2]
            segmentation_resized = cv2.resize(segmentation_np, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
            
            # Create overlay
            overlay = original_image.copy()
            # Create boolean mask and ensure it has the right shape for indexing
            mask = segmentation_resized > 0
            # Ensure mask has the same shape as the image for proper indexing
            if mask.shape != overlay.shape[:2]:
                mask = cv2.resize(mask.astype(np.uint8), (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            overlay[mask] = [255, 0, 0]  # Red overlay for tumor regions
            
            # Blend overlay with original image
            alpha = 0.3
            blended = cv2.addWeighted(original_image, 1 - alpha, overlay, alpha, 0)
            
            # Save overlay
            overlay_path = output_path.replace('.png', '_overlay.png')
            cv2.imwrite(overlay_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
            print(f"Overlay saved to: {overlay_path}")
    
    return segmentation_np


def predict_batch(
    model: TumorSegmentationModel,
    image_dir: str,
    output_dir: str = None,
    file_pattern: str = "*.png"
) -> None:
    """
    Predict segmentation for a batch of images.
    
    Args:
        model: Loaded model
        image_dir: Directory containing input images
        output_dir: Directory to save outputs (optional)
        file_pattern: File pattern to match
    """
    import glob
    
    # Get list of image files
    image_files = glob.glob(os.path.join(image_dir, file_pattern))
    image_files.sort()
    
    print(f"Found {len(image_files)} images to process")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Process each image
    for i, image_path in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        
        try:
            if output_dir:
                output_path = os.path.join(output_dir, f"pred_{os.path.basename(image_path)}")
            else:
                output_path = None
            
            predict_single_image(model, image_path, output_path)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    print("Batch processing completed!")


def benchmark_model(
    model: TumorSegmentationModel,
    image_path: str,
    num_runs: int = 100
) -> dict:
    """
    Benchmark model performance.
    
    Args:
        model: Loaded model
        image_path: Path to test image
        num_runs: Number of runs for benchmarking
    
    Returns:
        Dictionary with benchmark results
    """
    print(f"Benchmarking model with {num_runs} runs...")
    
    # Preprocess image once
    image_tensor = preprocess_image(image_path)
    
    # Warm up
    for _ in range(10):
        with torch.no_grad():
            _ = model.predict(image_tensor)
    
    # Benchmark
    times = []
    for i in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            _ = model.predict(image_tensor)
        end_time = time.time()
        times.append(end_time - start_time)
    
    # Calculate statistics
    times = np.array(times)
    results = {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'median_time': np.median(times),
        'fps': 1.0 / np.mean(times)
    }
    
    print(f"Benchmark results:")
    print(f"  Mean inference time: {results['mean_time']:.3f} ± {results['std_time']:.3f} seconds")
    print(f"  Min/Max time: {results['min_time']:.3f} / {results['max_time']:.3f} seconds")
    print(f"  Median time: {results['median_time']:.3f} seconds")
    print(f"  FPS: {results['fps']:.1f}")
    
    return results


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Inference for tumor segmentation model")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--image_path", type=str, help="Path to input image")
    parser.add_argument("--image_dir", type=str, help="Directory containing input images")
    parser.add_argument("--output_path", type=str, help="Path to save output")
    parser.add_argument("--output_dir", type=str, help="Directory to save outputs")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    parser.add_argument("--num_runs", type=int, default=100, help="Number of benchmark runs")
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model_path, args.device)
    
    # Run inference
    if args.benchmark:
        if not args.image_path:
            print("Error: --image_path is required for benchmarking")
            return
        benchmark_model(model, args.image_path, args.num_runs)
    
    elif args.image_path:
        # Single image inference
        predict_single_image(model, args.image_path, args.output_path)
    
    elif args.image_dir:
        # Batch inference
        predict_batch(model, args.image_dir, args.output_dir)
    
    else:
        print("Error: Please specify either --image_path or --image_dir")


if __name__ == "__main__":
    main()