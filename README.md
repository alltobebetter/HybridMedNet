# HybridMedNet - 医学影像深度学习诊断框架

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

**HybridMedNet** 是一个完整、可运行的医学影像深度学习诊断框架。它结合了多尺度特征提取、注意力机制和特征融合技术，专为医学影像多标签分类任务设计。

## 主要特性

- **🆕 现代架构**: 支持 ConvNeXt (2022), Swin Transformer (2021), EfficientNetV2, Vision Mamba (2024)
- **真实可用**: 完整实现，开箱即用，支持训练、评估和预测
- **多尺度架构**: 多种 Backbone + FPN 金字塔特征提取
- **先进注意力**: CBAM / 多头自注意力 / 跨尺度注意力
- **动态融合**: 自适应多尺度特征融合
- **多标签分类**: 支持医学影像的多疾病诊断
- **完整工具链**: 训练、评估、预测、可视化一应俱全
- **丰富可视化**: ROC曲线、训练曲线、预测可视化
- **高性能**: 支持GPU加速、混合精度训练
- **灵活配置**: 轻松切换不同架构，向后兼容

## 架构设计

### 🆕 现代架构（推荐）

```
输入图像 (3×224×224)
    ↓
[现代 Backbone]
ConvNeXt / Swin Transformer / EfficientNetV2
    ↓
[多尺度特征提取]
层次化特征金字塔
    ↓
[现代注意力机制]
多头自注意力 + 跨尺度注意力
    ↓
[动态特征融合]
自适应权重融合
    ↓
[多标签分类器]
14类疾病预测
    ↓
输出: 14维概率向量
```

### 经典架构（向后兼容）

```
输入图像 (3×224×224)
    ↓
[ResNet Backbone]
ResNet34/50/101 + FPN
    ↓
[CBAM 注意力]
通道 + 空间注意力
    ↓
[特征融合]
自适应融合
    ↓
[分类器]
多标签预测
```

### 核心组件

#### 现代架构
1. **ModernFeatureExtractor**: ConvNeXt/Swin/EfficientNet 特征提取
2. **ModernMultiScaleAttention**: 多头自注意力 + 跨尺度注意力
3. **DynamicFeatureFusion**: 动态特征融合
4. **MultiLabelClassifier**: 多标签分类器

#### 经典架构
1. **PyramidFeatureExtractor**: 基于ResNet的多尺度特征提取
2. **FeaturePyramidNetwork (FPN)**: 特征金字塔网络
3. **MultiScaleAttention**: 多尺度CBAM注意力
4. **DynamicFeatureFusion**: 动态特征融合
5. **MultiLabelClassifier**: 多标签分类器

## 安装

### 环境要求

- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.0 (可选，用于GPU加速)

### 快速安装

```bash
# 克隆仓库
git clone https://github.com/alltobebetter/HybridMedNet.git
cd HybridMedNet

# 安装依赖
pip install -r requirements.txt

# 验证安装
python test_installation.py

# 🆕 测试现代架构
python test_modern_models.py
```

### 🆕 现代架构依赖

使用现代架构需要额外安装：

```bash
pip install timm einops

# 可选: Vision Mamba (实验性)
pip install mamba-ssm
```

## 快速开始

### 1. 使用示例数据训练

```bash
# 框架会自动创建示例数据集
python train.py
```

### 2. 使用真实数据集

支持的数据集格式：

#### 格式1: CSV标签文件（推荐）

```
data/
├── image_001.jpg
├── image_002.jpg
└── labels.csv
```

`labels.csv`:
```csv
image,labels
image_001.jpg,Pneumonia|Effusion
image_002.jpg,No Finding
```

#### 格式2: 文件夹结构（单标签）

```
data/
├── train/
│   ├── normal/
│   └── pneumonia/
└── val/
```

修改 `configs/default_config.py`:

```python
DATA = {
    'data_dir': './data/your_dataset',
    'labels_file': './data/your_dataset/labels.csv',
    'image_size': 224,
    'num_workers': 4,
}
```

### 3. 训练模型

```bash
# 默认配置训练
python train.py

# 监控训练（可选）
tensorboard --logdir=./logs
```

训练输出:
- `checkpoints/best_model.pth` - 最佳模型
- `logs/training_curves.png` - 训练曲线
- `logs/roc_curves.png` - ROC曲线

### 4. 评估模型

```bash
python evaluate.py \
    --model_path checkpoints/best_model.pth \
    --save_results
```

### 5. 预测新图像

```bash
# 单张图像
python predict.py \
    --image_path test_image.jpg \
    --model_path checkpoints/best_model.pth \
    --visualize

# 批量预测
python predict.py \
    --image_path ./test_images/ \
    --model_path checkpoints/best_model.pth \
    --visualize \
    --save_dir ./predictions
```

## 支持的数据集

### NIH ChestX-ray14

112,120张胸部X光图像，14种病理标签：

- Atelectasis (肺不张)
- Cardiomegaly (心脏肥大)
- Effusion (积液)
- Infiltration (浸润)
- Mass (肿块)
- Nodule (结节)
- Pneumonia (肺炎)
- Pneumothorax (气胸)
- Consolidation (实变)
- Edema (水肿)
- Emphysema (肺气肿)
- Fibrosis (纤维化)
- Pleural_Thickening (胸膜增厚)
- Hernia (疝气)

**下载**: [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC)

### 其他支持的数据集

- **ISIC**: 皮肤病变分类
- **COVID-19 CT**: 新冠肺炎CT诊断
- **RSNA Pneumonia**: 肺炎检测

## 配置说明

### 模型配置

#### 🆕 现代架构（推荐）

```python
MODEL = {
    # 现代 Backbone
    'backbone': 'convnext_base',  # convnext_tiny/small/base/large
                                   # swin_tiny/small/base
                                   # efficientnetv2_s/m/l
    
    'pretrained': True,            # 使用ImageNet预训练
    'num_classes': 14,             # 疾病类别数
    'fusion_channels': 512,        # 融合后通道数
    'dropout': 0.5,                # Dropout率
    
    # 现代注意力配置
    'use_modern_attention': True,  # 使用现代注意力机制
    'use_cross_scale_attention': True,  # 跨尺度注意力
    'num_heads': 8,                # 注意力头数
}
```

#### 经典架构

```python
MODEL = {
    'backbone': 'resnet50',        # resnet34, resnet50, resnet101
    'pretrained': True,
    'num_classes': 14,
    'fpn_channels': 256,           # FPN通道数
    'fusion_channels': 512,
    'dropout': 0.5,
    'hierarchical': False,         # 是否层次化分类
}
```

### 训练配置

```python
TRAIN = {
    'batch_size': 32,            # 批次大小
    'epochs': 50,                # 训练轮数
    'learning_rate': 1e-4,       # 学习率
    'weight_decay': 1e-4,        # 权重衰减
    'lr_scheduler': 'cosine',    # 学习率调度
    'mixed_precision': True,     # 混合精度训练
}
```

## 性能指标

在NIH ChestX-ray14数据集上的性能（示例）：

| 类别 | AUC | Precision | Recall | F1 |
|------|-----|-----------|--------|-----|
| Atelectasis | 0.82 | 0.78 | 0.75 | 0.76 |
| Cardiomegaly | 0.91 | 0.88 | 0.86 | 0.87 |
| Effusion | 0.88 | 0.84 | 0.82 | 0.83 |
| Pneumonia | 0.76 | 0.72 | 0.70 | 0.71 |
| **Mean** | **0.84** | **0.81** | **0.78** | **0.79** |

*注：实际性能取决于数据质量和训练配置*

## 项目结构

```
HybridMedNet/
├── configs/                    # 配置文件
│   ├── __init__.py
│   └── default_config.py
├── models/                     # 模型定义
│   ├── __init__.py
│   ├── hybrid_med_net.py      # 主模型
│   ├── feature_extraction.py  # 特征提取
│   ├── attention.py           # 注意力机制
│   ├── fusion.py              # 特征融合
│   └── classifier.py          # 分类器
├── data/                       # 数据处理
│   ├── __init__.py
│   ├── chest_xray_dataset.py  # 数据集
│   └── transforms.py          # 数据增强
├── utils/                      # 工具函数
│   ├── __init__.py
│   ├── metrics.py             # 评估指标
│   └── visualization.py       # 可视化
├── train.py                    # 训练脚本
├── evaluate.py                 # 评估脚本
├── predict.py                  # 预测脚本
├── test_installation.py        # 安装测试
├── requirements.txt            # 依赖列表
├── QUICKSTART.md              # 快速开始
└── README.md                   # 本文件
```

## 高级功能

### 1. 层次化分类

设置 `hierarchical: True` 启用两级分类：

```python
MODEL = {
    'hierarchical': True,
    'num_coarse_classes': 3,  # 粗粒度类别（如：正常/异常/严重）
    'num_fine_classes': 14,   # 细粒度类别（具体疾病）
}
```

### 2. 测试时增强 (TTA)

```python
from data.transforms import get_test_time_augmentation_transforms

tta_transforms = get_test_time_augmentation_transforms(n_transforms=5)
# 对同一图像应用多个变换，取平均预测
```

### 3. 类别权重

处理数据不平衡：

```python
from utils.metrics import compute_class_weights

class_weights = compute_class_weights(train_labels, num_classes=14)
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
```

### 4. CAM可视化

```python
from utils.visualization import visualize_attention

attention_maps = model.get_attention_maps(image)
visualization = visualize_attention(image, attention_maps[0])
```

## 故障排除

### CUDA内存不足

```python
# 减小batch size
TRAIN = {'batch_size': 16}  # 或更小

# 使用梯度累积
TRAIN = {
    'batch_size': 16,
    'gradient_accumulation_steps': 2  # 等效于batch_size=32
}
```

### 收敛缓慢

```python
# 调整学习率
TRAIN = {'learning_rate': 3e-4}  # 增大学习率

# 使用warmup
TRAIN = {'warmup_epochs': 5}
```

### 过拟合

```python
# 增加dropout
MODEL = {'dropout': 0.6}

# 增强数据增强
DATA = {'use_albumentations': True}
```

## 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 致谢

- [PyTorch](https://pytorch.org/) - 深度学习框架
- [torchvision](https://pytorch.org/vision/) - 计算机视觉库
- [timm](https://github.com/rwightman/pytorch-image-models) - 预训练模型
- NIH Clinical Center - ChestX-ray14数据集

## 联系方式

- 作者: 苏淋
- Email: me@supage.eu.org
- 项目链接: https://github.com/alltobebetter/HybridMedNet

## 引用

如果您在研究中使用了HybridMedNet，请引用：

```bibtex
@software{hybridmednet2024,
  title={HybridMedNet: A Multi-scale Medical Image Diagnosis Framework},
  author={Su, Lin},
  year={2024},
  url={https://github.com/alltobebetter/HybridMedNet}
}
```

## 路线图

- [ ] 支持更多backbone（EfficientNet, Vision Transformer）
- [ ] 3D医学影像支持（CT, MRI）
- [ ] 模型压缩和量化
- [ ] Web部署接口
- [ ] 移动端支持
- [ ] 联邦学习
