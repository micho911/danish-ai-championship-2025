import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import List, Tuple, Optional, Dict, Any
import cv2
import random


class RandomTumourCrop(A.DualTransform):
    """
    Random crop that ensures tumor pixels are included in the crop.
    This dramatically increases the fraction of positive pixels in each batch.
    """
    
    def __init__(self, crop_size=(96, 96), p=1.0):
        super().__init__(p)
        self.crop_h, self.crop_w = crop_size
    
    def apply(self, img, mask=None, **params):
        """Apply crop to image."""
        # Get crop coordinates
        y1, x1 = self._get_crop_coordinates(img, mask)
        return img[y1:y1+self.crop_h, x1:x1+self.crop_w]
    
    def apply_to_mask(self, mask, **params):
        """Apply same crop to mask."""
        # Get crop coordinates (mask will be passed to _get_crop_coordinates)
        y1, x1 = self._get_crop_coordinates(mask, mask)
        return mask[y1:y1+self.crop_h, x1:x1+self.crop_w]
    
    def _get_crop_coordinates(self, img, mask):
        """Get crop coordinates based on mask content."""
        h, w = img.shape[:2]
        
        if mask is not None:
            # Find tumor pixels
            ys, xs = np.where(mask > 0)
            if len(xs) > 0:
                # Find a random tumor pixel and crop around it
                idx = random.randint(0, len(xs) - 1)
                yc, xc = ys[idx], xs[idx]
                y1 = np.clip(yc - self.crop_h // 2, 0, h - self.crop_h)
                x1 = np.clip(xc - self.crop_w // 2, 0, w - self.crop_w)
                return y1, x1
        
        # Fall back to random crop if no tumor pixels found
        y1 = random.randint(0, h - self.crop_h)
        x1 = random.randint(0, w - self.crop_w)
        return y1, x1


class TumorSegmentationDataset(Dataset):
    """
    Dataset for tumor segmentation with support for both patient and control data.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        include_controls: bool = True,
        control_weight: float = 0.3,
        samples: Optional[List[Dict[str, str]]] = None,  # NEW: explicit samples
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Path to data directory
            split: 'train', 'val', or 'test'
            transform: Albumentations transforms
            include_controls: Whether to include healthy control data
            control_weight: Weight for control samples in training (e.g., 0.3 means 30% of patient count)
            samples: Optional pre-defined list of samples (for proper train/val splits)
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.include_controls = include_controls
        self.control_weight = control_weight
        
        # Get patient data - use provided samples or scan directory
        if samples is not None:
            self.patient_samples = samples
        else:
            self.patient_samples = self._get_patient_samples()
        
        # Get control data if requested
        self.control_samples = []
        if include_controls and split == "train":
            all_control_samples = self._get_control_samples()
            # Balance control samples based on control_weight
            num_patients = len(self.patient_samples)
            num_controls_to_include = int(num_patients * control_weight)
            
            # Randomly sample control images to avoid overwhelming the dataset
            import random
            random.seed(42)  # For reproducibility
            self.control_samples = random.sample(all_control_samples, min(num_controls_to_include, len(all_control_samples)))
        
        # Combine samples
        self.samples = self.patient_samples + self.control_samples
        
        print(f"Dataset initialized with {len(self.patient_samples)} patient samples")
        if self.control_samples:
            print(f"and {len(self.control_samples)} control samples (control_weight={control_weight})")
            total_samples = len(self.patient_samples) + len(self.control_samples)
            patient_ratio = len(self.patient_samples) / total_samples
            control_ratio = len(self.control_samples) / total_samples
            print(f"Balance: {patient_ratio:.1%} patients, {control_ratio:.1%} controls")
    
    def _get_patient_samples(self) -> List[Dict[str, str]]:
        """Get patient image and mask paths."""
        patient_img_dir = os.path.join(self.data_dir, "patients", "imgs")
        patient_mask_dir = os.path.join(self.data_dir, "patients", "labels")
        
        samples = []
        if os.path.exists(patient_img_dir) and os.path.exists(patient_mask_dir):
            for img_name in sorted(os.listdir(patient_img_dir)):
                if img_name.endswith('.png'):
                    # Extract patient number
                    patient_num = img_name.replace('patient_', '').replace('.png', '')
                    mask_name = f"segmentation_{patient_num}.png"
                    
                    img_path = os.path.join(patient_img_dir, img_name)
                    mask_path = os.path.join(patient_mask_dir, mask_name)
                    
                    if os.path.exists(mask_path):
                        samples.append({
                            'img_path': img_path,
                            'mask_path': mask_path,
                            'type': 'patient'
                        })
        
        return samples
    
    def _get_control_samples(self) -> List[Dict[str, str]]:
        """Get control image paths (masks will be synthetic black masks)."""
        control_img_dir = os.path.join(self.data_dir, "controls", "imgs")
        
        samples = []
        if os.path.exists(control_img_dir):
            for img_name in sorted(os.listdir(control_img_dir)):
                if img_name.endswith('.png'):
                    img_path = os.path.join(control_img_dir, img_name)
                    samples.append({
                        'img_path': img_path,
                        'mask_path': None,  # Will create synthetic black mask
                        'type': 'control'
                    })
        
        return samples
    
    def _load_image(self, img_path: str) -> np.ndarray:
        """Load image as RGB numpy array."""
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def _load_mask(self, mask_path: Optional[str], img_shape: Tuple[int, int]) -> np.ndarray:
        """Load mask or create synthetic black mask for controls."""
        if mask_path is None:
            # Create synthetic black mask for controls
            return np.zeros(img_shape[:2], dtype=np.uint8)
        
        # Try to load as RGBA first, then fall back to grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")
        
        # Handle different mask formats
        if mask.ndim == 3 and mask.shape[2] == 4:
            # RGBA mask - extract alpha channel or convert to binary
            alpha_channel = mask[:, :, 3]
            # Convert to binary: any non-zero alpha value becomes 1
            mask = (alpha_channel > 0).astype(np.uint8) * 255
        elif mask.ndim == 3 and mask.shape[2] == 3:
            # RGB mask - convert to grayscale
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        elif mask.ndim == 2:
            # Already grayscale
            pass
        else:
            raise ValueError(f"Unexpected mask format: {mask.shape}")
        
        return mask
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        image = self._load_image(sample['img_path'])
        mask = self._load_mask(sample['mask_path'], image.shape)

        # --- Fix orientation: always make width >= height before transforms ---
        if image.shape[1] < image.shape[0]:
            image = np.transpose(image, (1, 0, 2))
            mask = np.transpose(mask, (1, 0))
        # ---------------------------------------------------------------------

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).float() / 255.0
        
        # Ensure mask is float and properly normalized
        if mask.dtype != torch.float32:
            mask = mask.float()
        
        # Normalize mask to [0, 1] if it's in [0, 255] range
        if mask.max() > 1.0:
            mask = mask / 255.0
        
        # Add channel dimension to mask if it doesn't have one
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)  # Add channel dimension
        
        return {
            'image': image,
            'mask': mask,
            'type': sample['type'],
            'img_path': sample['img_path']
        }


def get_transforms(
    split: str = "train",
    image_size: Tuple[int, int] = (400, 991),
    p_augment: float = 0.5,
) -> A.Compose:
    """
    Get transforms for the dataset.
    """
    # Always use image_size[0] as height, image_size[1] as width
    if split == "train":
        return A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.HorizontalFlip(p=0.5),
            #A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(p=0.2),
            # Add your tumor-aware cropping here when ready:
            # RandomTumourCrop(crop_size=(512, 512), p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ])


def create_data_loaders(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (400, 991),
    include_controls: bool = True,
    control_weight: float = 0.3,
    val_split: float = 0.2,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.
    """
    # Get all patient samples
    patient_samples = []
    patient_img_dir = os.path.join(data_dir, "patients", "imgs")
    patient_mask_dir = os.path.join(data_dir, "patients", "labels")
    
    if os.path.exists(patient_img_dir) and os.path.exists(patient_mask_dir):
        for img_name in sorted(os.listdir(patient_img_dir)):
            if img_name.endswith('.png'):
                patient_num = img_name.replace('patient_', '').replace('.png', '')
                mask_name = f"segmentation_{patient_num}.png"
                
                img_path = os.path.join(patient_img_dir, img_name)
                mask_path = os.path.join(patient_mask_dir, mask_name)
                
                if os.path.exists(mask_path):
                    patient_samples.append({
                        'img_path': img_path,
                        'mask_path': mask_path,
                        'type': 'patient'
                    })
    
    # Split patient samples
    num_val = int(len(patient_samples) * val_split)
    train_patients = patient_samples[num_val:]
    val_patients = patient_samples[:num_val]
    
    # Create datasets with explicit samples
    train_transforms = get_transforms("train", image_size)
    val_transforms = get_transforms("val", image_size)
    
    # Pass train_patients explicitly to avoid leakage
    train_dataset = TumorSegmentationDataset(
        data_dir=data_dir,
        split="train",
        transform=train_transforms,
        include_controls=include_controls,
        control_weight=control_weight,
        samples=train_patients  # FIXED: explicit train split
    )
    
    # Pass val_patients explicitly
    val_dataset = TumorSegmentationDataset(
        data_dir=data_dir,
        split="val", 
        transform=val_transforms,
        include_controls=False,  # No controls in validation
        samples=val_patients  # FIXED: explicit val split
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    data_dir = "data"
    
    # Test dataset creation
    dataset = TumorSegmentationDataset(
        data_dir=data_dir,
        split="train",
        transform=get_transforms("train"),
        include_controls=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test a sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Image shape: {sample['image'].shape}")
        print(f"Mask shape: {sample['mask'].shape}")
        print(f"Type: {sample['type']}")
    
    # Test data loaders
    train_loader, val_loader = create_data_loaders(
        data_dir=data_dir,
        batch_size=4,
        include_controls=True
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test a batch
    for batch in train_loader:
        print(f"Batch image shape: {batch['image'].shape}")
        print(f"Batch mask shape: {batch['mask'].shape}")
        break