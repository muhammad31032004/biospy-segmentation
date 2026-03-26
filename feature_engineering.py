"""
Feature Engineering & Data Pipeline for Biopsy Image Segmentation
=================================================================
- Augmentation pipeline (heavy augmentation for limited 1800-image dataset)
- Custom Dataset class
- Preprocessing utilities
"""

import os
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Augmentation pipelines
# ---------------------------------------------------------------------------

def get_train_transforms(img_size=256):
    """Heavy augmentation to compensate for small dataset (1800 images)."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.2, rotate_limit=30,
            border_mode=cv2.BORDER_REFLECT, p=0.5
        ),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120*0.05, p=0.3),
            A.GridDistortion(p=0.3),
            A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.3),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.MedianBlur(blur_limit=5, p=0.2),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.CLAHE(clip_limit=4.0, p=0.3),
            A.HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.3
            ),
        ], p=0.4),
        A.CoarseDropout(
            max_holes=8, max_height=img_size // 16, max_width=img_size // 16,
            min_holes=2, fill_value=0, mask_fill_value=0, p=0.2
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_val_transforms(img_size=256):
    """Validation / test transforms -- resize + normalize only."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_tta_transforms(img_size=256):
    """Test-Time Augmentation variants for better predictions."""
    base_norm = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    hflip = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    vflip = A.Compose([
        A.Resize(img_size, img_size),
        A.VerticalFlip(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    rot90 = A.Compose([
        A.Resize(img_size, img_size),
        A.RandomRotate90(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return [base_norm, hflip, vflip, rot90]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SegmentationDataset(Dataset):
    """
    Loads biopsy images and corresponding binary masks.
    Images: .jpg  |  Masks: .png (0/255 binary)
    """
    def __init__(self, image_dir, mask_dir=None, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Read image as RGB
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.mask_dir is not None:
            # Mask filename: same stem but .png extension
            stem = os.path.splitext(img_name)[0]
            mask_path = os.path.join(self.mask_dir, stem + '.png')
            if not os.path.exists(mask_path):
                mask_path = os.path.join(self.mask_dir, img_name)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.float32)  # binarize to 0/1

            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask'].unsqueeze(0)  # (1, H, W)
            return image, mask, img_name
        else:
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented['image']
            return image, img_name


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def compute_dataset_stats(image_dir, sample_size=200):
    """Compute mean and std of the dataset (for custom normalization if needed)."""
    files = sorted(os.listdir(image_dir))[:sample_size]
    pixels = []
    for f in files:
        img = cv2.imread(os.path.join(image_dir, f), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        pixels.append(img.reshape(-1, 3))
    pixels = np.concatenate(pixels, axis=0)
    return pixels.mean(axis=0), pixels.std(axis=0)


def analyze_masks(mask_dir, sample_size=100):
    """Analyze mask statistics: foreground ratio, etc."""
    files = sorted(os.listdir(mask_dir))[:sample_size]
    ratios = []
    for f in files:
        mask = cv2.imread(os.path.join(mask_dir, f), cv2.IMREAD_GRAYSCALE)
        fg_ratio = (mask > 127).sum() / mask.size
        ratios.append(fg_ratio)
    ratios = np.array(ratios)
    print(f"Foreground ratio -- mean: {ratios.mean():.4f}, "
          f"std: {ratios.std():.4f}, min: {ratios.min():.4f}, max: {ratios.max():.4f}")
    return ratios


if __name__ == "__main__":
    base = os.path.join(os.path.dirname(__file__),
                        "Segmentation-20260326T063949Z-1-001", "Segmentation")
    print("=== Dataset Stats ===")
    mean, std = compute_dataset_stats(os.path.join(base, "training/images"))
    print(f"Mean: {mean}, Std: {std}")
    print("\n=== Mask Analysis ===")
    analyze_masks(os.path.join(base, "training/masks"))
