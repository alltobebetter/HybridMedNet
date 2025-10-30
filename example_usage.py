#!/usr/bin/env python
"""
HybridMedNet使用示例
展示如何使用框架进行医学影像分析
"""

import torch
from types import SimpleNamespace

def example_1_create_model():
    """示例1: 创建模型"""
    print("="*80)
    print("示例1: 创建HybridMedNet模型")
    print("="*80)
    
    from models.hybrid_med_net import HybridMedNet
    
    config = SimpleNamespace()
    config.MODEL = {
        'backbone': 'resnet50',
        'pretrained': True,
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
    
    print(f"✓ 模型创建成功")
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {output.shape}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    print()


def example_2_load_data():
    """示例2: 加载数据"""
    print("="*80)
    print("示例2: 创建数据集")
    print("="*80)
    
    from data.chest_xray_dataset import create_sample_dataset, ChestXrayDataset
    from data.transforms import get_train_transforms
    
    data_dir, labels_file = create_sample_dataset(
        save_dir='./temp_data',
        num_samples=50
    )
    
    transform = get_train_transforms(image_size=224, use_albumentations=False)
    
    dataset = ChestXrayDataset(
        data_dir=data_dir,
        labels_file=labels_file,
        transform=transform,
        mode='train'
    )
    
    image, labels = dataset[0]
    
    print(f"✓ 数据集创建成功")
    print(f"  数据集大小: {len(dataset)}")
    print(f"  图像形状: {image.shape}")
    print(f"  标签形状: {labels.shape}")
    print(f"  正样本数: {labels.sum().item()}")
    
    import shutil
    shutil.rmtree('./temp_data')
    print()


def example_3_training_loop():
    """示例3: 简单的训练循环"""
    print("="*80)
    print("示例3: 训练循环示例")
    print("="*80)
    
    from models.hybrid_med_net import HybridMedNet
    from data.chest_xray_dataset import create_sample_dataset, ChestXrayDataset
    from data.transforms import get_train_transforms
    from torch.utils.data import DataLoader
    import torch.nn as nn
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    config = SimpleNamespace()
    config.MODEL = {
        'backbone': 'resnet50',
        'pretrained': False,
        'fpn_channels': 256,
        'fusion_channels': 512,
        'num_classes': 14,
        'hierarchical': False,
        'dropout': 0.5
    }
    
    model = HybridMedNet(config).to(device)
    
    data_dir, labels_file = create_sample_dataset(
        save_dir='./temp_data',
        num_samples=20
    )
    
    transform = get_train_transforms(image_size=224, use_albumentations=False)
    dataset = ChestXrayDataset(data_dir, labels_file, transform, mode='train')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    print("\n开始训练（演示2个batch）...")
    model.train()
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        if batch_idx >= 2:
            break
        
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        print(f"  Batch {batch_idx+1}: Loss = {loss.item():.4f}")
    
    print("✓ 训练循环演示完成")
    
    import shutil
    shutil.rmtree('./temp_data')
    print()


def example_4_metrics():
    """示例4: 计算评估指标"""
    print("="*80)
    print("示例4: 计算评估指标")
    print("="*80)
    
    import numpy as np
    from utils.metrics import calculate_metrics, print_metrics
    
    y_true = np.random.randint(0, 2, (100, 14))
    y_pred = np.random.randint(0, 2, (100, 14))
    y_scores = np.random.rand(100, 14)
    
    metrics = calculate_metrics(y_true, y_pred, y_scores)
    
    print("✓ 指标计算完成")
    print_metrics(metrics)
    print()


def example_5_visualization():
    """示例5: 可视化"""
    print("="*80)
    print("示例5: 训练曲线可视化")
    print("="*80)
    
    from utils.visualization import plot_training_curves
    import matplotlib
    matplotlib.use('Agg')
    
    history = {
        'train_loss': [1.0, 0.8, 0.6, 0.5, 0.4],
        'val_loss': [1.1, 0.9, 0.7, 0.6, 0.55],
        'val_accuracy': [0.5, 0.6, 0.7, 0.75, 0.8],
        'val_auc': [0.6, 0.7, 0.75, 0.8, 0.82]
    }
    
    fig = plot_training_curves(history, metrics=['loss', 'accuracy'])
    fig.savefig('example_curves.png')
    
    print("✓ 训练曲线已保存到 example_curves.png")
    print()


def main():
    """运行所有示例"""
    print("\n" + "="*80)
    print("HybridMedNet 使用示例")
    print("="*80 + "\n")
    
    examples = [
        example_1_create_model,
        example_2_load_data,
        example_3_training_loop,
        example_4_metrics,
        example_5_visualization,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"✗ 示例失败: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("="*80)
    print("所有示例运行完成！")
    print("="*80)
    print("\n接下来可以:")
    print("  1. 查看完整文档: README_NEW.md")
    print("  2. 快速开始: QUICKSTART.md")
    print("  3. 运行训练: python train.py")
    print()


if __name__ == '__main__':
    main()
