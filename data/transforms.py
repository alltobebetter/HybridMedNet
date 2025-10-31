"""
Data transformations and augmentations for medical images
"""

import torch
import torchvision.transforms as transforms
import numpy as np


def get_train_transforms(image_size=224):
    """
    Get training data transformations with augmentation
    
    Args:
        image_size: Target image size
    
    Returns:
        transforms.Compose object
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_transforms(image_size=224):
    """
    Get validation/test data transformations (no augmentation)
    
    Args:
        image_size: Target image size
    
    Returns:
        transforms.Compose object
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_test_transforms(image_size=224):
    """
    Get test data transformations (same as validation)
    
    Args:
        image_size: Target image size
    
    Returns:
        transforms.Compose object
    """
    return get_val_transforms(image_size)


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize image tensor for visualization
    
    Args:
        tensor: Normalized image tensor [C, H, W]
        mean: Mean used for normalization
        std: Std used for normalization
    
    Returns:
        Denormalized tensor
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    
    return tensor * std + mean


def get_test_time_augmentation_transforms(image_size=224, n_transforms=5):
    """
    Get multiple transformations for test-time augmentation
    
    Args:
        image_size: Target image size
        n_transforms: Number of augmented versions
    
    Returns:
        List of transform objects
    """
    base_transform = [
        transforms.Resize((image_size, image_size)),
    ]
    
    augmentations = [
        [],  # Original
        [transforms.RandomHorizontalFlip(p=1.0)],  # Flip
        [transforms.RandomRotation(5)],  # Rotate 5
        [transforms.RandomRotation(-5)],  # Rotate -5
        [transforms.ColorJitter(brightness=0.1)],  # Brightness
    ]
    
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
    
    transform_list = []
    for aug in augmentations[:n_transforms]:
        transform_list.append(
            transforms.Compose(base_transform + aug + normalize)
        )
    
    return transform_list


if __name__ == '__main__':
    # Test transforms
    from PIL import Image
    
    # Create dummy image
    img = Image.new('RGB', (256, 256), color='red')
    
    # Test train transforms
    train_transform = get_train_transforms(224)
    train_img = train_transform(img)
    print(f"Train transform output shape: {train_img.shape}")
    print(f"Train transform output range: [{train_img.min():.3f}, {train_img.max():.3f}]")
    
    # Test val transforms
    val_transform = get_val_transforms(224)
    val_img = val_transform(img)
    print(f"Val transform output shape: {val_img.shape}")
    print(f"Val transform output range: [{val_img.min():.3f}, {val_img.max():.3f}]")
    
    # Test denormalize
    denorm_img = denormalize(val_img)
    print(f"Denormalized range: [{denorm_img.min():.3f}, {denorm_img.max():.3f}]")
    
    # Test TTA
    tta_transforms = get_test_time_augmentation_transforms(224, n_transforms=3)
    print(f"\nNumber of TTA transforms: {len(tta_transforms)}")
