"""
Chest X-ray Dataset for medical image classification
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class ChestXrayDataset(Dataset):
    """
    Chest X-ray dataset for multi-label classification
    
    Supports:
    - CSV labels file with multi-label format
    - Folder structure for single-label classification
    """
    
    def __init__(self, data_dir, labels_file=None, transform=None, split='train'):
        """
        Args:
            data_dir: Directory containing images
            labels_file: CSV file with labels (optional)
            transform: Image transformations
            split: 'train', 'val', or 'test'
        """
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        
        # Default class names for ChestX-ray14
        self.class_names = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
            'Pleural_Thickening', 'Hernia'
        ]
        
        self.num_classes = len(self.class_names)
        
        # Load data
        if labels_file and os.path.exists(labels_file):
            self._load_from_csv(labels_file)
        else:
            self._load_from_folder()
    
    def _load_from_csv(self, labels_file):
        """Load data from CSV file"""
        df = pd.read_csv(labels_file)
        
        # Filter by split if column exists
        if 'split' in df.columns:
            df = df[df['split'] == self.split]
        
        self.image_paths = []
        self.labels = []
        
        for _, row in df.iterrows():
            img_path = os.path.join(self.data_dir, row['image'])
            if os.path.exists(img_path):
                self.image_paths.append(img_path)
                
                # Parse labels
                label_str = row['labels']
                label_vector = np.zeros(self.num_classes, dtype=np.float32)
                
                if label_str != 'No Finding':
                    labels = label_str.split('|')
                    for label in labels:
                        if label in self.class_names:
                            idx = self.class_names.index(label)
                            label_vector[idx] = 1.0
                
                self.labels.append(label_vector)
        
        self.labels = np.array(self.labels)
    
    def _load_from_folder(self):
        """Load data from folder structure"""
        self.image_paths = []
        self.labels = []
        
        # Get all image files
        for fname in os.listdir(self.data_dir):
            if fname.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(self.data_dir, fname)
                self.image_paths.append(img_path)
                
                # Random labels for demo
                label_vector = np.random.randint(0, 2, self.num_classes).astype(np.float32)
                self.labels.append(label_vector)
        
        self.labels = np.array(self.labels)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = torch.FloatTensor(self.labels[idx])
        
        return image, label


def create_sample_dataset(save_dir='./data/sample_chest_xray', num_samples=200):
    """
    Create a sample dataset for testing
    
    Args:
        save_dir: Directory to save sample data
        num_samples: Number of sample images to create
    
    Returns:
        data_dir: Path to image directory
        labels_file: Path to labels CSV file
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Creating {num_samples} sample images in {save_dir}...")
    
    # Class names
    class_names = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
        'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
        'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
        'Pleural_Thickening', 'Hernia'
    ]
    
    # Create sample images and labels
    data = []
    
    for i in range(num_samples):
        # Create random image
        img = Image.new('RGB', (224, 224), 
                       color=(np.random.randint(0, 255),
                              np.random.randint(0, 255),
                              np.random.randint(0, 255)))
        
        # Save image
        img_name = f'sample_{i:04d}.jpg'
        img_path = os.path.join(save_dir, img_name)
        img.save(img_path)
        
        # Random labels (1-3 diseases per image)
        num_diseases = np.random.randint(0, 4)
        if num_diseases == 0:
            labels_str = 'No Finding'
        else:
            selected = np.random.choice(class_names, num_diseases, replace=False)
            labels_str = '|'.join(selected)
        
        # Assign split
        if i < num_samples * 0.7:
            split = 'train'
        elif i < num_samples * 0.85:
            split = 'val'
        else:
            split = 'test'
        
        data.append({
            'image': img_name,
            'labels': labels_str,
            'split': split
        })
    
    # Save labels CSV
    df = pd.DataFrame(data)
    labels_file = os.path.join(save_dir, 'labels.csv')
    df.to_csv(labels_file, index=False)
    
    print(f"Created {num_samples} sample images")
    print(f"Labels saved to: {labels_file}")
    print(f"Train: {len(df[df['split']=='train'])}, "
          f"Val: {len(df[df['split']=='val'])}, "
          f"Test: {len(df[df['split']=='test'])}")
    
    return save_dir, labels_file


if __name__ == '__main__':
    # Test dataset creation
    data_dir, labels_file = create_sample_dataset(num_samples=50)
    
    # Test dataset loading
    from transforms import get_train_transforms
    dataset = ChestXrayDataset(data_dir, labels_file, 
                               transform=get_train_transforms(224),
                               split='train')
    
    print(f"\nDataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    
    # Test loading one sample
    image, label = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Label shape: {label.shape}")
    print(f"Label sum: {label.sum()}")
