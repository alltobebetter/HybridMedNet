import torch
import torch.nn as nn
from .feature_extraction import PyramidFeatureExtractor
from .attention import MultiScaleAttention
from .fusion import AdaptiveFeatureFusion
from .classifier import HierarchicalClassifier

class HybridMedNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 特征提取
        self.feature_extractor = PyramidFeatureExtractor(
            backbone=config.MODEL['backbone'],
            pretrained=config.MODEL['pretrained']
        )
        
        # 多尺度注意力
        self.attention = MultiScaleAttention(
            in_channels=config.MODEL['feature_dims']
        )
        
        # 特征融合
        self.fusion = AdaptiveFeatureFusion(
            in_channels=sum(config.MODEL['feature_dims']),
            out_channels=512
        )
        
        # 分类器
        self.classifier = HierarchicalClassifier(
            in_channels=512,
            num_classes=config.MODEL['num_classes']
        )
        
    def forward(self, x):
        # 多尺度特征提取
        multi_scale_features = self.feature_extractor(x)
        
        # 注意力增强
        attended_features = self.attention(multi_scale_features)
        
        # 特征融合
        fused_features = self.fusion(attended_features)
        
        # 层次化分类
        coarse_pred, fine_pred = self.classifier(fused_features)
        
        return coarse_pred, fine_pred
