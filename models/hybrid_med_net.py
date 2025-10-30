import torch
import torch.nn as nn
from .feature_extraction import MultiScaleFeatureExtractor
from .attention import MultiScaleAttention
from .fusion import DynamicFeatureFusion
from .classifier import HierarchicalClassifier, MultiLabelClassifier

# 导入现代化模块
try:
    from .modern_backbones import ModernFeatureExtractor
    from .modern_attention import ModernMultiScaleAttention
    MODERN_AVAILABLE = True
except ImportError:
    MODERN_AVAILABLE = False
    print("Warning: Modern modules not available, using classic architecture only")


class HybridMedNet(nn.Module):
    """
    HybridMedNet主模型 - 医学影像诊断深度网络
    
    架构：
    1. 多尺度特征提取（支持经典和现代 Backbone）
    2. 多尺度注意力机制（CBAM / Modern Attention）
    3. 自适应特征融合
    4. 层次化分类器
    
    支持的 Backbone:
    - 经典: resnet34, resnet50, resnet101
    - 现代: convnext_tiny/small/base, swin_tiny/small/base, efficientnetv2_s/m/l
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        backbone_name = config.MODEL['backbone']
        
        # 判断使用经典还是现代架构
        modern_backbones = [
            'convnext', 'swin', 'efficientnet', 'vim'
        ]
        self.use_modern = any(name in backbone_name for name in modern_backbones)
        
        if self.use_modern and not MODERN_AVAILABLE:
            raise ImportError(
                "Modern backbones requested but dependencies not installed. "
                "Please install: pip install timm einops"
            )
        
        # 特征提取器
        if self.use_modern:
            self.feature_extractor = ModernFeatureExtractor(
                backbone=backbone_name,
                pretrained=config.MODEL['pretrained']
            )
        else:
            # 经典架构
            from .feature_extraction import FeaturePyramidNetwork
            self.feature_extractor = MultiScaleFeatureExtractor(
                backbone=backbone_name,
                pretrained=config.MODEL['pretrained'],
                fpn_out_channels=config.MODEL.get('fpn_channels', 256)
            )
        
        # 获取特征维度
        feature_dims = self.feature_extractor.get_feature_dims()
        
        # 多尺度注意力
        if self.use_modern and config.MODEL.get('use_modern_attention', True):
            self.attention = ModernMultiScaleAttention(
                channels_list=feature_dims,
                num_heads=config.MODEL.get('num_heads', 8),
                use_cross_scale=config.MODEL.get('use_cross_scale_attention', True)
            )
        else:
            self.attention = MultiScaleAttention(
                in_channels=feature_dims
            )
        
        # 动态特征融合
        self.fusion = DynamicFeatureFusion(
            channels_list=feature_dims,
            out_channels=config.MODEL.get('fusion_channels', 512)
        )
        
        # 分类器
        fusion_channels = config.MODEL.get('fusion_channels', 512)
        
        if config.MODEL.get('hierarchical', True):
            # 层次化分类器
            self.classifier = HierarchicalClassifier(
                in_channels=fusion_channels,
                num_coarse_classes=config.MODEL.get('num_coarse_classes', 3),
                num_fine_classes=config.MODEL['num_classes'],
                dropout=config.MODEL.get('dropout', 0.5)
            )
            self.is_hierarchical = True
        else:
            # 多标签分类器
            self.classifier = MultiLabelClassifier(
                in_channels=fusion_channels,
                num_classes=config.MODEL['num_classes'],
                dropout=config.MODEL.get('dropout', 0.5)
            )
            self.is_hierarchical = False
        
    def forward(self, x):
        """
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            如果是层次化分类：
                coarse_pred: [B, num_coarse_classes]
                fine_pred: [B, num_fine_classes]
            如果是多标签分类：
                pred: [B, num_classes]
        """
        # 1. 多尺度特征提取（Backbone + FPN）
        multi_scale_features = self.feature_extractor(x)
        
        # 2. 注意力增强
        attended_features = self.attention(multi_scale_features)
        
        # 3. 特征融合
        fused_features = self.fusion(attended_features)
        
        # 4. 分类
        if self.is_hierarchical:
            coarse_pred, fine_pred = self.classifier(fused_features)
            return coarse_pred, fine_pred
        else:
            pred = self.classifier(fused_features)
            return pred
    
    def get_attention_maps(self, x):
        """
        获取注意力图用于可视化
        """
        with torch.no_grad():
            multi_scale_features = self.feature_extractor(x)
            # 这里可以进一步提取注意力权重
            return multi_scale_features


if __name__ == '__main__':
    # 测试代码
    from types import SimpleNamespace
    
    config = SimpleNamespace()
    config.MODEL = {
        'backbone': 'resnet50',
        'pretrained': False,
        'fpn_channels': 256,
        'fusion_channels': 512,
        'num_coarse_classes': 3,
        'num_classes': 14,
        'hierarchical': True,
        'dropout': 0.5
    }
    
    model = HybridMedNet(config)
    x = torch.randn(2, 3, 224, 224)
    
    coarse, fine = model(x)
    print(f"Input: {x.shape}")
    print(f"Coarse prediction: {coarse.shape}")
    print(f"Fine prediction: {fine.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
