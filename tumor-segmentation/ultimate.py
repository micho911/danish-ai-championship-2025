
import os
import time
import glob
import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2

from scipy.special import softmax


class TumorDataset2D(Dataset):
    """Dataset class for 2D MIP-PET tumor segmentation"""
    
    def __init__(self, data_root, use_controls=True):
        self.data_root = Path(data_root)
        self.use_controls = use_controls
        
        # Get patient data (images + segmentations)
        self.patient_imgs = sorted(glob.glob(str(self.data_root / 'patients' / 'imgs' / '*.png')))
        self.patient_labels = sorted(glob.glob(str(self.data_root / 'patients' / 'labels' / '*.png')))
        
        # Ensure matching pairs
        assert len(self.patient_imgs) == len(self.patient_labels), \
            f"Mismatch: {len(self.patient_imgs)} images vs {len(self.patient_labels)} labels"
        
        self.samples = []
        
        # Add patient samples
        for img_path, label_path in zip(self.patient_imgs, self.patient_labels):
            self.samples.append((img_path, label_path, 'patient'))
        
        # Add control samples (healthy, no tumor)
        if use_controls:
            control_imgs = sorted(glob.glob(str(self.data_root / 'controls' / 'imgs' / '*.png')))
            for img_path in control_imgs:
                self.samples.append((img_path, None, 'control'))
        
        print(f"Dataset created with {len(self.samples)} samples "
              f"({len(self.patient_imgs)} patients, "
              f"{len(control_imgs) if use_controls else 0} controls)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label_path, sample_type = self.samples[idx]
        
        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Convert to 3-channel (simulate CT + PET channels)
        # For MIP-PET data, we'll duplicate the channel to simulate multi-modal input
        img_3ch = np.stack([img, img, img], axis=0).astype(np.float32) / 255.0
        
        if sample_type == 'patient':
            # Load segmentation mask
            mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not load mask: {label_path}")
            
            # Convert mask to binary (0 = background, 1 = tumor)
            mask = (mask > 127).astype(np.float32)
        else:
            # Control sample - create empty mask (no tumors)
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
        
        # Convert to torch tensors
        img_tensor = torch.from_numpy(img_3ch)  # Shape: (3, H, W)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)  # Shape: (1, H, W)
        
        return img_tensor, mask_tensor



class Hybrid_cnn():
    def __init__(self):
        """
        Write your own input validators here
        Initialize your model etc.
        """
        # set some paths and parameters (simplified for local use)
        self.input_path = 'input/'  # local input directory
        self.output_path = 'output/'  # local output directory
        self.pretrained_weights_path = 'checkpoints/'  # local checkpoints directory
        self.temp_path = 'temp/'  # temporary files directory
        self.result_path = 'results/'  # results directory
        self.nii_seg_file = 'prediction.nii.gz'
        self.npz_seg_file = 'prediction.npz'
        
        # Training specific paths
        self.data_root = 'data'
        self.weights_save_path = 'hybrid_cnn_weights.pt'
        
        # Create directories if they don't exist
        os.makedirs(self.input_path, exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.pretrained_weights_path, exist_ok=True)
        os.makedirs(self.temp_path, exist_ok=True)
        os.makedirs(self.result_path, exist_ok=True)

    def convert_mha_to_nii(self, mha_input_path, nii_out_path):  #nnUNet specific
        img = sitk.ReadImage(mha_input_path)
        sitk.WriteImage(img, nii_out_path, True)

    def convert_nii_to_mha(self, nii_input_path, mha_out_path):  #nnUNet specific
        img = sitk.ReadImage(nii_input_path)
        sitk.WriteImage(img, mha_out_path, True)

    def check_gpu(self):
        """
        Check if GPU is available
        """
        print('Checking GPU availability')
        is_available = torch.cuda.is_available()
        print('Available: ' + str(is_available))
        print(f'Device count: {torch.cuda.device_count()}')
        if is_available:
            print(f'Current device: {torch.cuda.current_device()}')
            print('Device name: ' + torch.cuda.get_device_name(0))
            print('Device memory: ' + str(torch.cuda.get_device_properties(0).total_memory))

    def load_inputs(self):
        """
        Read from input/ directory (simplified for local use)
        """
        # Look for CT and PET files in the input directory
        ct_files = glob.glob(os.path.join(self.input_path, '*ct*.mha')) + \
                   glob.glob(os.path.join(self.input_path, '*CT*.mha')) + \
                   glob.glob(os.path.join(self.input_path, 'images/ct/*.mha'))
        
        pet_files = glob.glob(os.path.join(self.input_path, '*pet*.mha')) + \
                    glob.glob(os.path.join(self.input_path, '*PET*.mha')) + \
                    glob.glob(os.path.join(self.input_path, 'images/pet/*.mha'))
        
        if not ct_files or not pet_files:
            raise ValueError(f"Could not find CT and PET files in {self.input_path}")
        
        ct_mha = ct_files[0]  # Take the first found file
        pet_mha = pet_files[0]
        uuid = os.path.splitext(os.path.basename(ct_mha))[0]

        # Convert to temporary nii files for processing
        temp_pet_path = os.path.join(self.temp_path, 'temp_pet.nii.gz')
        temp_ct_path = os.path.join(self.temp_path, 'temp_ct.nii.gz')
        
        self.convert_mha_to_nii(ct_mha, temp_ct_path)
        self.convert_mha_to_nii(pet_mha, temp_pet_path)
        
        return uuid

    def write_outputs(self, uuid):
        """
        Write to output/ directory (simplified for local use)
        """
        os.makedirs(self.output_path, exist_ok=True)
        output_file = os.path.join(self.output_path, f"{uuid}_segmentation.mha")
        
        result_file = os.path.join(self.result_path, self.nii_seg_file)
        if os.path.exists(result_file):
            self.convert_nii_to_mha(result_file, output_file)
            print(f'Output written to: {output_file}')
        else:
            print(f'Warning: Result file {result_file} not found')
            # Create a dummy output for testing
            dummy_output = os.path.join(self.output_path, f"{uuid}_segmentation_dummy.txt")
            with open(dummy_output, 'w') as f:
                f.write("Dummy output - processing completed but no segmentation file found")
            print(f'Dummy output written to: {dummy_output}')

    def predict_ssl(self):
        """
        Your algorithm goes here (updated for simplified paths)
        """
        print("ssl segmentation starting!")

        # Load images from temporary location
        temp_pet_path = os.path.join(self.temp_path, 'temp_pet.nii.gz')
        temp_ct_path = os.path.join(self.temp_path, 'temp_ct.nii.gz')
        
        if not os.path.exists(temp_pet_path) or not os.path.exists(temp_ct_path):
            raise ValueError("Temporary CT/PET files not found. Run load_inputs() first.")

        # Load volumes        
        img_pet = sitk.ReadImage(temp_pet_path)
        pet_volume = sitk.GetArrayFromImage(img_pet)

        img_ct = sitk.ReadImage(temp_ct_path)
        ct_volume = sitk.GetArrayFromImage(img_ct)

        ct_volume_normalized = (ct_volume - (-800)) / (400 - (-800))
        ct_volume_normalized[ct_volume_normalized > 1] = 1.
        ct_volume_normalized[ct_volume_normalized < 0] = 0.

        pet_volume_normalized = (pet_volume - 0) / (0.95 * 15 - 0)
        pet_volume_normalized[pet_volume_normalized > 1] = 1.
        pet_volume_normalized[pet_volume_normalized < 0] = 0.

        start = ct_volume_normalized.shape[1] // 2 - 224 // 2
        end = start + 224
        pet_cropped = pet_volume_normalized[:, start:end, start:end]
        ct_cropped = ct_volume_normalized[:, start:end, start:end]

        model = ResNet(Bottleneck, [3, 4, 6, 3])
        num_channels = model.layer4[2].bn3.weight.shape[0]
        seg_decoder = Unet_Decoder(n_channels=num_channels, n_classes=2)

        model_stage_2 = UNet(n_channels=5, n_classes=2, bilinear=False)

        result = torch.empty([3, pet_cropped.shape[0], 2, 224, 224])
        
        # Check if pretrained weights exist
        if not os.path.exists(self.pretrained_weights_path):
            print(f"Warning: Pretrained weights directory {self.pretrained_weights_path} not found.")
            print("Using randomly initialized models (for testing purposes)")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            seg_decoder.to(device)
            model_stage_2.to(device)
            
            # Create dummy prediction for testing
            final_pred = torch.zeros([pet_cropped.shape[0], 224, 224])
            pred_result = final_pred.cpu().data.numpy()
        else:
            for fold in range(3):
                fold_weights = os.path.join(self.pretrained_weights_path, f'fold_{fold+1}_best_checkpoint.pth.tar')
                fold_weights_stage2 = os.path.join(self.pretrained_weights_path, f'fold_{fold+1}_best_checkpoint_2nd_stage.pth.tar')
                
                if not os.path.exists(fold_weights) or not os.path.exists(fold_weights_stage2):
                    print(f"Warning: Fold {fold+1} weights not found, skipping...")
                    continue

                # Load checkpoint files
                checkpoint = torch.load(fold_weights, map_location="cpu")
                msg_1 = model.load_state_dict(checkpoint['en_state_dict'], strict=False)
                msg_2 = seg_decoder.load_state_dict(checkpoint['de_state_dict'], strict=False)

                checkpoint = torch.load(fold_weights_stage2, map_location="cpu")
                msg_3 = model_stage_2.load_state_dict(checkpoint['state_dict'], strict=False)

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.to(device)
                seg_decoder.to(device)
                model_stage_2.to(device)
                model.eval()
                seg_decoder.eval()
                model_stage_2.eval()

                pred_volume = torch.empty([pet_cropped.shape[0], 2, 224, 224])

                for i in range(pet_cropped.shape[0]):
                    ct_slice = ct_cropped[i, :, :].astype(np.float32)
                    ct_slice = (ct_slice -  0.2617) /  0.3239
                    ct_slice = ct_slice.reshape((1,) + ct_slice.shape)
                    ct_slice = torch.from_numpy(ct_slice.astype(np.float32)).to(device)
                    ct_slice = ct_slice.repeat(3, 1, 1)

                    pet_slice = pet_cropped[i, :, :].astype(np.float32)
                    pet_slice = (pet_slice -  0.0456) /  0.0855
                    pet_slice = pet_slice.reshape((1,) + pet_slice.shape)
                    pet_slice = torch.from_numpy(pet_slice.astype(np.float32)).to(device)
                    pet_slice = pet_slice.repeat(3, 1, 1)
                    
                    input_images = torch.cat((torch.unsqueeze(ct_slice, dim=0), torch.unsqueeze(pet_slice, dim=0)), dim=1)[:,2:5,:,:]

                    with torch.no_grad():
                        encoder_4_layers_features = model(input_images)
                        output = seg_decoder(input_images, encoder_4_layers_features)
                        _, pred_result = torch.max(output, dim=1)

                        input_images = torch.cat((torch.unsqueeze(ct_slice, dim=0), torch.unsqueeze(pet_slice, dim=0), output, torch.unsqueeze(pred_result, dim=0)), dim=1)[:,[2,5,6,7,8],:,:]
                        pred_prob = model_stage_2(input_images)

                    pred_volume[i] = pred_prob[0]
        
                result[fold] = pred_volume
        
            final_pred = torch.softmax(torch.mean(result,dim=0), dim=1)
            pred_result = final_pred.cpu().data.numpy()        
            pred_result[pet_cropped.shape[0] - 15:,:,:] = 0 # remove the last few slices that are in the brain region
            pred_result[:15,:,:] = 0 # remove the first few slices that are in the brain region
        
        pred_pad_volume = np.transpose(np.pad(pred_result, ((0,0), (0,0), (start, ct_volume_normalized.shape[1] - end), (start, ct_volume_normalized.shape[1] - end)), 'constant'), (1, 0, 2, 3))
        
        # Skip nnUNet combination for now (requires nnUNet installation)
        pred_sum_result = np.argmax(pred_pad_volume, axis=0).astype(np.uint8)

        pred_save_image = sitk.GetImageFromArray(pred_sum_result)
        pred_save_image.SetSpacing(img_pet.GetSpacing())
        pred_save_image.SetOrigin(img_pet.GetOrigin())
        pred_save_image.SetDirection(img_pet.GetDirection())
        
        os.makedirs(self.result_path, exist_ok=True)
        sitk.WriteImage(pred_save_image, os.path.join(self.result_path, self.nii_seg_file))

        print("ssl segmentation done!")

    def train(self, epochs=50, batch_size=4, lr=1e-4, val_split=0.2, use_controls=True):
        """
        Train the hybrid CNN model on your own data
        """
        print("Starting training...")
        
        # Create dataset
        dataset = TumorDataset2D(self.data_root, use_controls=use_controls)
        
        # Split dataset
        total_size = len(dataset)
        val_size = int(val_split * total_size)
        train_size = total_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=self._pad_collate
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=self._pad_collate
        )
        
        # Initialize models
        encoder = ResNet(Bottleneck, [3, 4, 6, 3])
        num_channels = encoder.layer4[2].bn3.weight.shape[0]
        decoder = Unet_Decoder(n_channels=num_channels, n_classes=2)
        stage2_model = UNet(n_channels=5, n_classes=2, bilinear=False)
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder.to(device)
        decoder.to(device)
        stage2_model.to(device)
        
        # Optimizers
        optimizer_stage1 = torch.optim.AdamW(
            list(encoder.parameters()) + list(decoder.parameters()), 
            lr=lr
        )
        optimizer_stage2 = torch.optim.AdamW(stage2_model.parameters(), lr=lr)
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        best_dice = 0.0
        
        for epoch in range(epochs):
            # Stage 1 training
            encoder.train()
            decoder.train()
            stage2_model.eval()
            
            epoch_loss_stage1 = 0.0
            train_dice_stage1 = []
            
            for batch_idx, (images, masks) in enumerate(train_loader):
                images = images.to(device)
                masks = masks.to(device).long()
                
                optimizer_stage1.zero_grad()
                
                # Forward pass stage 1
                encoder_features = encoder(images)
                stage1_output = decoder(images, encoder_features)
                
                loss_stage1 = criterion(stage1_output, masks.squeeze(1))
                loss_stage1.backward()
                optimizer_stage1.step()
                
                epoch_loss_stage1 += loss_stage1.item()
                
                # Calculate dice
                with torch.no_grad():
                    pred_stage1 = torch.argmax(stage1_output, dim=1)
                    dice = self._calculate_dice(pred_stage1, masks.squeeze(1))
                    train_dice_stage1.append(dice)
                
                if batch_idx % 10 == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}, Stage1 Loss: {loss_stage1.item():.4f}')
            
            # Stage 2 training (using stage 1 predictions)
            encoder.eval()
            decoder.eval()
            stage2_model.train()
            
            epoch_loss_stage2 = 0.0
            train_dice_stage2 = []
            
            for batch_idx, (images, masks) in enumerate(train_loader):
                images = images.to(device)
                masks = masks.to(device).long()
                
                optimizer_stage2.zero_grad()
                
                # Get stage 1 predictions
                with torch.no_grad():
                    encoder_features = encoder(images)
                    stage1_output = decoder(images, encoder_features)
                    stage1_pred = torch.argmax(stage1_output, dim=1).unsqueeze(1).float()
                
                # Prepare input for stage 2: [CT, PET, stage1_prob0, stage1_prob1, stage1_pred]
                stage2_input = torch.cat([
                    images[:, :2, :, :],  # CT and PET channels
                    stage1_output,        # Stage 1 probabilities
                    stage1_pred           # Stage 1 predictions
                ], dim=1)
                
                # Forward pass stage 2
                stage2_output = stage2_model(stage2_input)
                
                loss_stage2 = criterion(stage2_output, masks.squeeze(1))
                loss_stage2.backward()
                optimizer_stage2.step()
                
                epoch_loss_stage2 += loss_stage2.item()
                
                # Calculate dice
                with torch.no_grad():
                    pred_stage2 = torch.argmax(stage2_output, dim=1)
                    dice = self._calculate_dice(pred_stage2, masks.squeeze(1))
                    train_dice_stage2.append(dice)
            
            # Validation
            val_dice_stage1, val_dice_stage2 = self._validate(
                val_loader, encoder, decoder, stage2_model, device
            )
            
            avg_train_dice_stage1 = np.mean(train_dice_stage1)
            avg_train_dice_stage2 = np.mean(train_dice_stage2)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Stage1 - Train Loss: {epoch_loss_stage1/len(train_loader):.4f}, '
                  f'Train Dice: {avg_train_dice_stage1:.4f}, Val Dice: {val_dice_stage1:.4f}')
            print(f'  Stage2 - Train Loss: {epoch_loss_stage2/len(train_loader):.4f}, '
                  f'Train Dice: {avg_train_dice_stage2:.4f}, Val Dice: {val_dice_stage2:.4f}')
            
            # Save best model
            if val_dice_stage2 > best_dice:
                best_dice = val_dice_stage2
                self._save_models(encoder, decoder, stage2_model, epoch, best_dice)
                print(f'  New best model saved! Dice: {best_dice:.4f}')
        
        print(f"Training completed! Best validation dice: {best_dice:.4f}")

    def _validate(self, val_loader, encoder, decoder, stage2_model, device):
        """Validation loop"""
        encoder.eval()
        decoder.eval()
        stage2_model.eval()
        
        val_dice_stage1 = []
        val_dice_stage2 = []
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device).long()
                
                # Stage 1 forward
                encoder_features = encoder(images)
                stage1_output = decoder(images, encoder_features)
                pred_stage1 = torch.argmax(stage1_output, dim=1)
                dice1 = self._calculate_dice(pred_stage1, masks.squeeze(1))
                val_dice_stage1.append(dice1)
                
                # Stage 2 forward
                stage1_pred = pred_stage1.unsqueeze(1).float()
                stage2_input = torch.cat([
                    images[:, :2, :, :],
                    stage1_output,
                    stage1_pred
                ], dim=1)
                
                stage2_output = stage2_model(stage2_input)
                pred_stage2 = torch.argmax(stage2_output, dim=1)
                dice2 = self._calculate_dice(pred_stage2, masks.squeeze(1))
                val_dice_stage2.append(dice2)
        
        return np.mean(val_dice_stage1), np.mean(val_dice_stage2)

    def _calculate_dice(self, pred, target, smooth=1e-6):
        """Calculate Dice coefficient"""
        pred_flat = pred.view(-1).float()
        target_flat = target.view(-1).float()
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        return dice.item()

    def _pad_collate(self, batch):
        """Collate function to handle variable sized images"""
        images, masks = zip(*batch)
        
        # Find max dimensions
        max_h = max(img.shape[1] for img in images)
        max_w = max(img.shape[2] for img in images)
        
        # Pad to multiples of 8 for proper downsampling
        max_h = ((max_h + 7) // 8) * 8
        max_w = ((max_w + 7) // 8) * 8
        
        padded_images = []
        padded_masks = []
        
        for img, mask in zip(images, masks):
            # Pad image
            pad_h = max_h - img.shape[1]
            pad_w = max_w - img.shape[2]
            padded_img = F.pad(img, (0, pad_w, 0, pad_h))
            padded_images.append(padded_img)
            
            # Pad mask
            padded_mask = F.pad(mask, (0, pad_w, 0, pad_h))
            padded_masks.append(padded_mask)
        
        return torch.stack(padded_images), torch.stack(padded_masks)

    def _save_models(self, encoder, decoder, stage2_model, epoch, dice_score):
        """Save model checkpoints"""
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'stage2_state_dict': stage2_model.state_dict(),
            'dice_score': dice_score
        }
        torch.save(checkpoint, self.weights_save_path)
        print(f"Models saved to {self.weights_save_path}")

    def load_trained_models(self):
        """Load trained model weights"""
        if not os.path.exists(self.weights_save_path):
            print(f"No trained weights found at {self.weights_save_path}")
            return None, None, None
        
        checkpoint = torch.load(self.weights_save_path, map_location='cpu')
        
        encoder = ResNet(Bottleneck, [3, 4, 6, 3])
        num_channels = encoder.layer4[2].bn3.weight.shape[0]
        decoder = Unet_Decoder(n_channels=num_channels, n_classes=2)
        stage2_model = UNet(n_channels=5, n_classes=2, bilinear=False)
        
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        stage2_model.load_state_dict(checkpoint['stage2_state_dict'])
        
        print(f"Loaded models from {self.weights_save_path} (Dice: {checkpoint['dice_score']:.4f})")
        
        return encoder, decoder, stage2_model

    def predict_with_trained_model(self, image_path, output_path=None):
        """
        Predict tumor segmentation using trained models on a single 2D image
        """
        # Load trained models
        encoder, decoder, stage2_model = self.load_trained_models()
        if encoder is None:
            print("No trained models available. Please train first.")
            return None
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder.to(device)
        decoder.to(device)
        stage2_model.to(device)
        
        encoder.eval()
        decoder.eval()
        stage2_model.eval()
        
        # Load and preprocess image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        original_shape = img.shape
        
        # Convert to 3-channel and normalize
        img_3ch = np.stack([img, img, img], axis=0).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_3ch).unsqueeze(0).to(device)  # Add batch dimension
        
        with torch.no_grad():
            # Stage 1 prediction
            encoder_features = encoder(img_tensor)
            stage1_output = decoder(img_tensor, encoder_features)
            stage1_pred = torch.argmax(stage1_output, dim=1).unsqueeze(1).float()
            
            # Stage 2 prediction
            stage2_input = torch.cat([
                img_tensor[:, :2, :, :],  # First 2 channels
                stage1_output,            # Stage 1 probabilities
                stage1_pred               # Stage 1 predictions
            ], dim=1)
            
            stage2_output = stage2_model(stage2_input)
            final_pred = torch.argmax(stage2_output, dim=1)
            
            # Convert to numpy and resize back to original shape
            pred_mask = final_pred.cpu().numpy()[0]
            
            if pred_mask.shape != original_shape:
                pred_mask = cv2.resize(pred_mask.astype(np.uint8), 
                                     (original_shape[1], original_shape[0]), 
                                     interpolation=cv2.INTER_NEAREST)
            
            # Convert to 0/255 format
            result_mask = (pred_mask * 255).astype(np.uint8)
            
            if output_path:
                cv2.imwrite(output_path, result_mask)
                print(f"Prediction saved to {output_path}")
            
            return result_mask

    def test_trained_model(self, test_data_root=None):
        """
        Test the trained model on a test dataset and compute metrics
        """
        if test_data_root is None:
            test_data_root = self.data_root
        
        # Load trained models
        encoder, decoder, stage2_model = self.load_trained_models()
        if encoder is None:
            print("No trained models available. Please train first.")
            return
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoder.to(device)
        decoder.to(device)
        stage2_model.to(device)
        
        encoder.eval()
        decoder.eval()
        stage2_model.eval()
        
        # Create test dataset
        test_dataset = TumorDataset2D(test_data_root, use_controls=False)  # Only test on patients
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                                collate_fn=self._pad_collate)
        
        dice_scores = []
        
        print("Testing trained model on dataset...")
        with torch.no_grad():
            for i, (images, masks) in enumerate(test_loader):
                images = images.to(device)
                masks = masks.to(device).long()
                
                # Stage 1 prediction
                encoder_features = encoder(images)
                stage1_output = decoder(images, encoder_features)
                stage1_pred = torch.argmax(stage1_output, dim=1).unsqueeze(1).float()
                
                # Stage 2 prediction
                stage2_input = torch.cat([
                    images[:, :2, :, :],
                    stage1_output,
                    stage1_pred
                ], dim=1)
                
                stage2_output = stage2_model(stage2_input)
                final_pred = torch.argmax(stage2_output, dim=1)
                
                # Calculate dice score
                dice = self._calculate_dice(final_pred, masks.squeeze(1))
                dice_scores.append(dice)
                
                if i % 10 == 0:
                    print(f"Processed {i+1}/{len(test_loader)} samples, "
                          f"Current Dice: {dice:.4f}")
        
        mean_dice = np.mean(dice_scores)
        std_dice = np.std(dice_scores)
        
        print(f"\n=== Test Results ===")
        print(f"Mean Dice Score: {mean_dice:.4f} ± {std_dice:.4f}")
        print(f"Min Dice Score: {np.min(dice_scores):.4f}")
        print(f"Max Dice Score: {np.max(dice_scores):.4f}")
        print(f"Samples tested: {len(dice_scores)}")
        
        return dice_scores

    def create_sample_input(self, patient_image_path=None):
        """
        Create sample input files for testing the inference pipeline
        """
        if patient_image_path is None:
            # Use first patient image if available
            patient_imgs = glob.glob(os.path.join(self.data_root, 'patients', 'imgs', '*.png'))
            if not patient_imgs:
                print("No patient images found for creating sample input")
                return
            patient_image_path = patient_imgs[0]
        
        print(f"Creating sample input from: {patient_image_path}")
        
        # Load the 2D image
        img_2d = cv2.imread(patient_image_path, cv2.IMREAD_GRAYSCALE)
        if img_2d is None:
            print(f"Could not load image: {patient_image_path}")
            return
        
        # Create a fake 3D volume by stacking the 2D image
        # This simulates a simple 3D CT/PET volume for testing
        num_slices = 50
        fake_3d_volume = np.stack([img_2d] * num_slices, axis=0)
        
        # Create SimpleITK images
        ct_image = sitk.GetImageFromArray(fake_3d_volume.astype(np.float32))
        pet_image = sitk.GetImageFromArray((fake_3d_volume * 0.8).astype(np.float32))  # Slightly different for PET
        
        # Set some basic metadata
        ct_image.SetSpacing([1.0, 1.0, 1.0])
        ct_image.SetOrigin([0.0, 0.0, 0.0])
        pet_image.SetSpacing([1.0, 1.0, 1.0])
        pet_image.SetOrigin([0.0, 0.0, 0.0])
        
        # Save as .mha files in the input directory
        ct_path = os.path.join(self.input_path, 'sample_ct.mha')
        pet_path = os.path.join(self.input_path, 'sample_pet.mha')
        
        sitk.WriteImage(ct_image, ct_path)
        sitk.WriteImage(pet_image, pet_path)
        
        print(f"Sample CT saved to: {ct_path}")
        print(f"Sample PET saved to: {pet_path}")
        print("You can now run inference mode to test the pipeline!")

    def process(self):
        """
        Read inputs from /input, process with your algorithm and write to /output
        """
        # process function will be called once for each test sample
        self.check_gpu()
        print('Start processing')
        uuid = self.load_inputs()

        print('Start prediction')
        self.predict_ssl()      # get ssl output

        print('Start output writing')
        self.write_outputs(uuid)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)
        x1 = self.maxpool(x0)

        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        # we return the output tokens from the encoder blocks
        return [x0, x2, x3, x4, x5]


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.outc = OutConv(64, n_classes)
        self.softmax = nn.Softmax(dim=1)

        self._init_weight()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x_prob = self.softmax(x)
        return x_prob
 
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Unet_Decoder(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(Unet_Decoder, self).__init__()
        self.bilinear = bilinear
        factor = 2 if bilinear else 1
        self.up1 = Up(n_channels, (n_channels // 2) // factor, bilinear)
        self.up2 = Up(n_channels // 2, (n_channels // 4) // factor, bilinear)
        self.up3 = Up(n_channels // 4, (n_channels // 8) // factor, bilinear)
        self.up4_1 = nn.ConvTranspose2d(n_channels // 8, n_channels // 16, kernel_size=2, stride=2)
        self.up4_2 = nn.Sequential(
            nn.Conv2d(n_channels // 16 + 64, n_channels // 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_channels // 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels // 16, n_channels // 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_channels // 16),
            nn.ReLU(inplace=True)
        )
        self.up5_1 = nn.ConvTranspose2d(n_channels // 16, n_channels // 32, kernel_size=2, stride=2)
        self.up5_2 = nn.Sequential(
            nn.Conv2d(n_channels // 32 + 3, n_channels // 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_channels // 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_channels // 32, n_channels // 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(n_channels // 32),
            nn.ReLU(inplace=True)
        )
            
        self.outc = OutConv(n_channels // 32, n_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, encoder_feas):
        x = self.up1(encoder_feas[4], encoder_feas[3])
        x = self.up2(x, encoder_feas[2])
        x = self.up3(x, encoder_feas[1])
        x = self.up4_1(x)
        x = torch.cat([encoder_feas[0], x], dim=1)
        x = self.up4_2(x)
        x = self.up5_1(x)
        x = torch.cat([input, x], dim=1)
        x = self.up5_2(x)
        x = self.outc(x)
        x_prob = self.softmax(x)
        return x_prob


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        ###### Usually the bias is removed in conv layers before a batch norm layer, as the batch norm’s beta parameter (bias of nn.BatchNorm) will have the same effect 
        ###### and the bias of the conv layer might be canceled out by the mean subtraction.
        ###### https://discuss.pytorch.org/t/any-purpose-to-set-bias-false-in-densenet-torchvision/22067/2
        ###### http://proceedings.mlr.press/v37/ioffe15.html 
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hybrid CNN for tumor segmentation')
    parser.add_argument('--mode', choices=['train', 'test', 'predict', 'inference', 'create_sample'], 
                        default='inference',
                        help='Mode: train, test, predict, inference, or create_sample')
    parser.add_argument('--data_root', type=str, default='data',
                        help='Root directory for training data')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--no_controls', action='store_true',
                        help='Do not use control images in training')
    parser.add_argument('--image_path', type=str,
                        help='Path to image for prediction mode')
    parser.add_argument('--output_path', type=str,
                        help='Output path for prediction result')
    
    args = parser.parse_args()
    
    hybrid_cnn = Hybrid_cnn()
    hybrid_cnn.data_root = args.data_root
    
    if args.mode == 'train':
        print("Starting training mode...")
        hybrid_cnn.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            val_split=args.val_split,
            use_controls=not args.no_controls
        )
    elif args.mode == 'test':
        print("Starting test mode...")
        hybrid_cnn.test_trained_model()
    elif args.mode == 'predict':
        if not args.image_path:
            print("Error: --image_path is required for prediction mode")
        else:
            print(f"Predicting on image: {args.image_path}")
            result = hybrid_cnn.predict_with_trained_model(
                args.image_path, 
                args.output_path
            )
            if result is not None:
                print("Prediction completed successfully!")
    elif args.mode == 'create_sample':
        print("Creating sample input files...")
        hybrid_cnn.create_sample_input(args.image_path)
    else:
        print("Starting inference mode...")
        hybrid_cnn.process()