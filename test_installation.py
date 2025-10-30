#!/usr/bin/env python
"""
测试HybridMedNet框架是否正确安装
"""

import sys
import torch
import torchvision
from types import SimpleNamespace

def test_imports():
    """测试所有必要的导入"""
    print("Testing imports...")
    try:
        from models.hybrid_med_net import HybridMedNet
        from models.feature_extraction import MultiScaleFeatureExtractor
        from models.attention import MultiScaleAttention
        from models.fusion import DynamicFeatureFusion
        from models.classifier import HierarchicalClassifier, MultiLabelClassifier
        from data.chest_xray_dataset import ChestXrayDataset
        from data.transforms import get_train_transforms, get_val_transforms
        from utils.metrics import calculate_metrics
        from utils.visualization import plot_training_curves
        from configs.default_config import Config
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_model_creation():
    """测试模型创建"""
    print("\nTesting model creation...")
    try:
        config = SimpleNamespace()
        config.MODEL = {
            'backbone': 'resnet50',
            'pretrained': False,
            'fpn_channels': 256,
            'fusion_channels': 512,
            'num_coarse_classes': 3,
            'num_classes': 14,
            'hierarchical': False,
            'dropout': 0.5
        }
        
        model = HybridMedNet(config)
        
        x = torch.randn(2, 3, 224, 224)
        output = model(x)
        
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print("✓ Model creation successful")
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_transforms():
    """测试数据变换"""
    print("\nTesting transforms...")
    try:
        from PIL import Image
        import numpy as np
        from data.transforms import get_train_transforms, get_val_transforms
        
        img = Image.new('RGB', (256, 256), color='red')
        
        train_transform = get_train_transforms(image_size=224, use_albumentations=False)
        val_transform = get_val_transforms(image_size=224, use_albumentations=False)
        
        train_img = train_transform(img)
        val_img = val_transform(img)
        
        print(f"  Train transform output: {train_img.shape}")
        print(f"  Val transform output: {val_img.shape}")
        print("✓ Transforms successful")
        return True
    except Exception as e:
        print(f"✗ Transforms failed: {e}")
        return False


def test_dataset():
    """测试数据集创建"""
    print("\nTesting dataset creation...")
    try:
        from data.chest_xray_dataset import create_sample_dataset, ChestXrayDataset
        from data.transforms import get_val_transforms
        
        data_dir, labels_file = create_sample_dataset(
            save_dir='./test_data',
            num_samples=10
        )
        
        transform = get_val_transforms(image_size=224, use_albumentations=False)
        
        dataset = ChestXrayDataset(
            data_dir=data_dir,
            labels_file=labels_file,
            transform=transform,
            mode='train'
        )
        
        print(f"  Dataset size: {len(dataset)}")
        
        image, labels = dataset[0]
        print(f"  Sample image shape: {image.shape}")
        print(f"  Sample labels shape: {labels.shape}")
        print("✓ Dataset creation successful")
        
        import shutil
        shutil.rmtree('./test_data')
        
        return True
    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics():
    """测试指标计算"""
    print("\nTesting metrics...")
    try:
        import numpy as np
        from utils.metrics import calculate_metrics
        
        y_true = np.random.randint(0, 2, (20, 14))
        y_pred = np.random.randint(0, 2, (20, 14))
        y_scores = np.random.rand(20, 14)
        
        metrics = calculate_metrics(y_true, y_pred, y_scores)
        
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  AUC: {metrics.get('auc', 0):.4f}")
        print("✓ Metrics calculation successful")
        return True
    except Exception as e:
        print(f"✗ Metrics calculation failed: {e}")
        return False


def main():
    print("="*80)
    print("HybridMedNet Installation Test")
    print("="*80)
    
    print(f"\nPython version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print("\n" + "="*80)
    
    tests = [
        test_imports,
        test_model_creation,
        test_transforms,
        test_dataset,
        test_metrics,
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    print("="*80)
    print("Test Summary")
    print("="*80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed! HybridMedNet is ready to use.")
        print("\nNext steps:")
        print("  1. Run: python train.py")
        print("  2. Check: ./checkpoints/best_model.pth")
        print("  3. Evaluate: python evaluate.py --model_path checkpoints/best_model.pth")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
