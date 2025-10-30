#!/usr/bin/env python
"""
数据集下载和准备脚本
"""

import os
import argparse
import requests
from tqdm import tqdm
import pandas as pd


def download_file(url, save_path):
    """下载文件并显示进度条"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as file, tqdm(
        desc=os.path.basename(save_path),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)


def prepare_nih_chestxray14(data_dir='./data/ChestX-ray14'):
    """
    准备NIH ChestX-ray14数据集
    
    注意：由于数据集较大（~40GB），此脚本只提供下载链接和准备标签文件的说明
    """
    print("="*80)
    print("NIH ChestX-ray14 数据集准备")
    print("="*80)
    
    print("\n📦 数据集信息:")
    print("  - 名称: NIH ChestX-ray14")
    print("  - 大小: ~40 GB")
    print("  - 图像数: 112,120张")
    print("  - 类别数: 14种病理")
    
    print("\n📥 下载步骤:")
    print("  1. 访问: https://nihcc.app.box.com/v/ChestXray-NIHCC")
    print("  2. 下载以下文件:")
    print("     - images_001.tar.gz 到 images_012.tar.gz")
    print("     - Data_Entry_2017_v2020.csv")
    print("  3. 解压图像到:", data_dir)
    
    print("\n📋 标签文件格式:")
    print("  CSV包含以下列:")
    print("  - Image Index: 图像文件名")
    print("  - Finding Labels: 疾病标签（用|分隔）")
    print("  - Follow-up #")
    print("  - Patient ID")
    print("  - Patient Age")
    print("  - Patient Gender")
    
    print("\n⚙️ 配置示例:")
    print("  修改 configs/default_config.py:")
    print("  DATA = {")
    print(f"      'data_dir': '{data_dir}/images',")
    print(f"      'labels_file': '{data_dir}/Data_Entry_2017_v2020.csv',")
    print("      ...")
    print("  }")
    
    os.makedirs(data_dir, exist_ok=True)
    
    readme_path = os.path.join(data_dir, 'README.txt')
    with open(readme_path, 'w') as f:
        f.write("NIH ChestX-ray14 Dataset\n")
        f.write("="*50 + "\n\n")
        f.write("Download from: https://nihcc.app.box.com/v/ChestXray-NIHCC\n\n")
        f.write("Directory structure:\n")
        f.write("ChestX-ray14/\n")
        f.write("├── images/\n")
        f.write("│   ├── 00000001_000.png\n")
        f.write("│   ├── 00000001_001.png\n")
        f.write("│   └── ...\n")
        f.write("└── Data_Entry_2017_v2020.csv\n")
    
    print(f"\n✓ 已创建说明文件: {readme_path}")


def prepare_sample_dataset(data_dir='./data/sample_chest_xray', num_samples=200):
    """创建示例数据集（用于快速测试）"""
    print("="*80)
    print("创建示例数据集")
    print("="*80)
    
    from data.chest_xray_dataset import create_sample_dataset
    
    print(f"\n📦 生成 {num_samples} 张示例图像...")
    
    data_dir, labels_file = create_sample_dataset(
        save_dir=data_dir,
        num_samples=num_samples
    )
    
    print(f"\n✓ 示例数据集已创建:")
    print(f"  - 图像目录: {data_dir}")
    print(f"  - 标签文件: {labels_file}")
    
    print("\n📋 数据集统计:")
    df = pd.read_csv(labels_file)
    print(f"  - 总图像数: {len(df)}")
    print(f"  - 无病变: {len(df[df['labels'] == 'No Finding'])}")
    print(f"  - 有病变: {len(df[df['labels'] != 'No Finding'])}")


def prepare_kaggle_pneumonia(data_dir='./data/chest_xray_pneumonia'):
    """
    准备Kaggle胸部X光肺炎数据集
    """
    print("="*80)
    print("Kaggle肺炎数据集准备")
    print("="*80)
    
    print("\n📦 数据集信息:")
    print("  - 名称: Chest X-Ray Images (Pneumonia)")
    print("  - 大小: ~2 GB")
    print("  - 图像数: 5,863张")
    print("  - 类别: Normal, Pneumonia")
    
    print("\n📥 下载步骤:")
    print("  1. 访问: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia")
    print("  2. 下载数据集")
    print("  3. 解压到:", data_dir)
    
    print("\n📂 预期目录结构:")
    print("  chest_xray_pneumonia/")
    print("  ├── train/")
    print("  │   ├── NORMAL/")
    print("  │   └── PNEUMONIA/")
    print("  ├── test/")
    print("  └── val/")
    
    os.makedirs(data_dir, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description='下载和准备医学影像数据集')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['sample', 'nih', 'kaggle'],
                       help='要准备的数据集')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='数据保存目录')
    parser.add_argument('--num_samples', type=int, default=200,
                       help='示例数据集的样本数')
    
    args = parser.parse_args()
    
    if args.dataset == 'sample':
        data_dir = args.data_dir or './data/sample_chest_xray'
        prepare_sample_dataset(data_dir, args.num_samples)
        
    elif args.dataset == 'nih':
        data_dir = args.data_dir or './data/ChestX-ray14'
        prepare_nih_chestxray14(data_dir)
        
    elif args.dataset == 'kaggle':
        data_dir = args.data_dir or './data/chest_xray_pneumonia'
        prepare_kaggle_pneumonia(data_dir)
    
    print("\n" + "="*80)
    print("准备完成!")
    print("="*80)
    print("\n下一步:")
    print("  1. 修改 configs/default_config.py 中的数据路径")
    print("  2. 运行: python train.py")


if __name__ == '__main__':
    main()
