import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class HierarchicalClassifier(nn.Module):
    """
    层次化分类器 - 实现粗粒度和细粒度的两级分类
    
    医学影像诊断通常需要先判断大类（如正常/异常），再进行细分类（具体疾病类型）
    这种层次化的策略可以提高诊断准确性
    """
    def __init__(self, in_channels: int, num_coarse_classes: int, num_fine_classes: int, 
                 dropout: float = 0.5):
        super().__init__()
        
        self.in_channels = in_channels
        self.num_coarse_classes = num_coarse_classes
        self.num_fine_classes = num_fine_classes
        
        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 全局最大池化
        self.global_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # 粗粒度分类头（如：正常/异常）
        self.coarse_classifier = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.BatchNorm1d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_channels // 2, num_coarse_classes)
        )
        
        # 细粒度分类头（具体疾病类型）
        self.fine_classifier = nn.Sequential(
            nn.Linear(in_channels + num_coarse_classes, in_channels // 2),
            nn.BatchNorm1d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_channels // 2, num_fine_classes)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Feature maps [B, C, H, W]
        
        Returns:
            coarse_pred: Coarse-grained predictions [B, num_coarse_classes]
            fine_pred: Fine-grained predictions [B, num_fine_classes]
        """
        batch_size = x.size(0)
        
        # 双池化策略
        avg_pooled = self.global_pool(x).view(batch_size, -1)  # [B, C]
        max_pooled = self.global_maxpool(x).view(batch_size, -1)  # [B, C]
        
        # 融合平均池化和最大池化
        fused_features = torch.cat([avg_pooled, max_pooled], dim=1)  # [B, 2C]
        fused_features = self.feature_fusion(fused_features)  # [B, C]
        
        # 粗粒度分类
        coarse_pred = self.coarse_classifier(fused_features)  # [B, num_coarse]
        
        # 细粒度分类（利用粗粒度预测的信息）
        fine_input = torch.cat([fused_features, coarse_pred], dim=1)  # [B, C + num_coarse]
        fine_pred = self.fine_classifier(fine_input)  # [B, num_fine]
        
        return coarse_pred, fine_pred


class SimpleClassifier(nn.Module):
    """
    简单的单层分类器 - 用于不需要层次化分类的场景
    """
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.BatchNorm1d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(in_channels // 2, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature maps [B, C, H, W]
        
        Returns:
            pred: Predictions [B, num_classes]
        """
        batch_size = x.size(0)
        pooled = self.global_pool(x).view(batch_size, -1)
        pred = self.classifier(pooled)
        return pred


class MultiLabelClassifier(nn.Module):
    """
    多标签分类器 - 用于一张图像可能包含多个疾病的场景
    例如：胸部X光可能同时存在肺炎和积液
    """
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_maxpool = nn.AdaptiveMaxPool2d((1, 1))
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # 为每个类别创建独立的分类器
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channels, in_channels // 4),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout / 2),
                nn.Linear(in_channels // 4, 1)
            )
            for _ in range(num_classes)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature maps [B, C, H, W]
        
        Returns:
            pred: Multi-label predictions [B, num_classes]
        """
        batch_size = x.size(0)
        
        # 双池化
        avg_pooled = self.global_pool(x).view(batch_size, -1)
        max_pooled = self.global_maxpool(x).view(batch_size, -1)
        
        # 融合特征
        fused_features = torch.cat([avg_pooled, max_pooled], dim=1)
        fused_features = self.feature_fusion(fused_features)
        
        # 独立分类
        outputs = []
        for classifier in self.classifiers:
            out = classifier(fused_features)
            outputs.append(out)
        
        pred = torch.cat(outputs, dim=1)  # [B, num_classes]
        
        return pred


if __name__ == '__main__':
    # 测试代码
    print("Testing HierarchicalClassifier...")
    classifier = HierarchicalClassifier(in_channels=512, num_coarse_classes=3, num_fine_classes=14)
    x = torch.randn(4, 512, 7, 7)
    coarse, fine = classifier(x)
    print(f"Input: {x.shape}, Coarse: {coarse.shape}, Fine: {fine.shape}")
    
    print("\nTesting MultiLabelClassifier...")
    ml_classifier = MultiLabelClassifier(in_channels=512, num_classes=14)
    pred = ml_classifier(x)
    print(f"Input: {x.shape}, Output: {pred.shape}")
