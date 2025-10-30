# HybridMedNet 快速开始指南

本指南将帮助您在5分钟内快速开始使用HybridMedNet进行医学影像诊断。

## 环境准备

### 1. 安装依赖

```bash
# 创建虚拟环境（推荐）
conda create -n hybridmed python=3.8
conda activate hybridmed

# 安装依赖
pip install -r requirements.txt
```

### 2. 验证安装

```bash
python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
python -c "import torchvision; print(f'torchvision {torchvision.__version__} installed')"
```

## 快速训练

### 使用示例数据集

框架会自动创建一个示例数据集用于测试：

```bash
python train.py
```

这将：
1. 自动生成200张示例图像和标签
2. 使用ResNet50作为backbone
3. 训练14类多标签分类模型
4. 保存最佳模型到 `./checkpoints/best_model.pth`

### 使用自己的数据集

#### 方法1：使用标签CSV文件（推荐）

准备数据格式：
```
data/
├── image_001.jpg
├── image_002.jpg
└── labels.csv
```

labels.csv格式：
```csv
image,labels
image_001.jpg,Pneumonia|Effusion
image_002.jpg,No Finding
image_003.jpg,Mass
```

然后修改 `configs/default_config.py`:

```python
DATA = {
    'data_dir': './data/your_dataset',
    'labels_file': './data/your_dataset/labels.csv',
    ...
}
```

#### 方法2：使用文件夹结构（单标签分类）

```
data/
├── train/
│   ├── normal/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── pneumonia/
│       ├── img3.jpg
│       └── img4.jpg
└── val/
    └── ...
```

## 评估模型

```bash
python evaluate.py --model_path checkpoints/best_model.pth --save_results
```

这将输出：
- 准确率、精确率、召回率、F1分数
- 每个类别的AUC分数
- ROC曲线图

## 预测新图像

### 单张图像预测

```bash
python predict.py \
    --image_path path/to/image.jpg \
    --model_path checkpoints/best_model.pth \
    --visualize \
    --save_dir ./predictions
```

### 批量预测

```bash
python predict.py \
    --image_path path/to/images/ \
    --model_path checkpoints/best_model.pth \
    --visualize \
    --save_dir ./predictions
```

## 配置说明

主要配置位于 `configs/default_config.py`:

### 模型配置
```python
MODEL = {
    'backbone': 'resnet50',      # 可选: resnet34, resnet101
    'num_classes': 14,           # 疾病类别数
    'hierarchical': False,       # 是否使用层次化分类
    'pretrained': True,          # 是否使用预训练权重
}
```

### 训练配置
```python
TRAIN = {
    'batch_size': 32,            # 根据GPU内存调整
    'epochs': 50,
    'learning_rate': 1e-4,
}
```

## 训练监控

### TensorBoard

```bash
tensorboard --logdir=./logs
```

然后在浏览器中打开 `http://localhost:6006`

### 训练输出

训练过程中会自动保存：
- `logs/training_curves.png` - 训练曲线
- `logs/roc_curves.png` - ROC曲线
- `checkpoints/best_model.pth` - 最佳模型

## 使用真实医学数据集

### NIH ChestX-ray14 数据集

1. 下载数据集：
```bash
# 下载链接: https://nihcc.app.box.com/v/ChestXray-NIHCC
```

2. 准备标签文件（已提供在数据集中）

3. 修改配置：
```python
DATA = {
    'data_dir': './data/ChestX-ray14/images',
    'labels_file': './data/ChestX-ray14/Data_Entry_2017.csv',
    ...
}
```

### ISIC皮肤病变数据集

```bash
# 下载: https://challenge.isic-archive.com/
```

### COVID-19胸部CT数据集

```bash
# 下载: https://github.com/UCSD-AI4H/COVID-CT
```

## 常见问题

### 1. CUDA内存不足

减小batch_size：
```python
TRAIN = {
    'batch_size': 16,  # 或更小
}
```

### 2. 训练速度慢

- 使用更少的workers：`num_workers: 2`
- 禁用数据增强（仅测试用）：`use_albumentations: False`

### 3. 准确率低

- 增加训练轮数
- 使用更大的backbone（resnet101）
- 调整学习率
- 检查数据标签是否正确

## 更多信息

- 完整文档：`README.md`
- 技术细节：`details.md`
- API文档：查看各模块的docstring

## 下一步

1. 尝试不同的backbone网络
2. 调整数据增强策略
3. 实验层次化分类（设置 `hierarchical: True`）
4. 集成CAM热力图可视化
5. 部署为Web API服务

祝您使用愉快！
