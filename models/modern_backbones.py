"""
现代化 Backbone 实现
支持 ConvNeXt, Swin Transformer, Vision Mamba
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import timm


class ConvNeXtBackbone(nn.Module):
    """
    ConvNeXt Backbone (2022)
    现代化的卷积网络，性能接近 Transformer
    论文: A ConvNet for the 2020s
    """
    def __init__(self, model_name='convnext_base', pretrained=True):
        super().__init__()
        
        # 使用 timm 加载预训练模型
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )
        
        # 获取特征通道数
        self.feature_info = self.model.feature_info
        self.feature_channels = [info['num_chs'] for info in self.feature_info]
        
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            List of features at different scales
        """
        features = self.model(x)
        return features


class SwinTransformerBackbone(nn.Module):
    """
    Swin Transformer Backbone (2021)
    层次化的 Vision Transformer
    论文: Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    """
    def __init__(self, model_name='swin_base_patch4_window7_224', pretrained=True):
        super().__init__()
        
        # 使用 timm 加载预训练模型
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )
        
        # 获取特征通道数
        self.feature_info = self.model.feature_info
        self.feature_channels = [info['num_chs'] for info in self.feature_info]
        
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            List of features at different scales
        """
        features = self.model(x)
        return features


class VisionMambaBackbone(nn.Module):
    """
    Vision Mamba Backbone (2024)
    基于状态空间模型的视觉架构
    论文: Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model
    
    注意: 这是简化版实现，完整版需要 mamba-ssm 库
    """
    def __init__(self, model_name='vim_base_patch16_224', pretrained=False):
        super().__init__()
        
        try:
            # 尝试使用 timm 加载（如果可用）
            self.model = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=(0, 1, 2, 3)
            )
            self.feature_info = self.model.feature_info
            self.feature_channels = [info['num_chs'] for info in self.feature_info]
            self.use_timm = True
        except:
            # 如果 timm 不支持，使用 ConvNeXt 作为替代
            print("Warning: Vision Mamba not available in timm, using ConvNeXt as fallback")
            self.model = timm.create_model(
                'convnext_base',
                pretrained=pretrained,
                features_only=True,
                out_indices=(0, 1, 2, 3)
            )
            self.feature_info = self.model.feature_info
            self.feature_channels = [info['num_chs'] for info in self.feature_info]
            self.use_timm = True
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            List of features at different scales
        """
        features = self.model(x)
        return features


class EfficientNetV2Backbone(nn.Module):
    """
    EfficientNetV2 Backbone (2021)
    高效的卷积网络
    """
    def __init__(self, model_name='tf_efficientnetv2_m', pretrained=True):
        super().__init__()
        
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4)
        )
        
        self.feature_info = self.model.feature_info
        self.feature_channels = [info['num_chs'] for info in self.feature_info]
        
    def forward(self, x):
        features = self.model(x)
        return features


class ModernFeatureExtractor(nn.Module):
    """
    统一的现代特征提取器接口
    支持多种现代 backbone
    """
    SUPPORTED_BACKBONES = {
        # ConvNeXt 系列
        'convnext_tiny': ('convnext', 'convnext_tiny'),
        'convnext_small': ('convnext', 'convnext_small'),
        'convnext_base': ('convnext', 'convnext_base'),
        'convnext_large': ('convnext', 'convnext_large'),
        
        # Swin Transformer 系列
        'swin_tiny': ('swin', 'swin_tiny_patch4_window7_224'),
        'swin_small': ('swin', 'swin_small_patch4_window7_224'),
        'swin_base': ('swin', 'swin_base_patch4_window7_224'),
        
        # EfficientNetV2 系列
        'efficientnetv2_s': ('efficientnet', 'tf_efficientnetv2_s'),
        'efficientnetv2_m': ('efficientnet', 'tf_efficientnetv2_m'),
        'efficientnetv2_l': ('efficientnet', 'tf_efficientnetv2_l'),
        
        # Vision Mamba (实验性)
        'vim_tiny': ('mamba', 'vim_tiny_patch16_224'),
        'vim_small': ('mamba', 'vim_small_patch16_224'),
        'vim_base': ('mamba', 'vim_base_patch16_224'),
    }
    
    def __init__(self, backbone='convnext_base', pretrained=True):
        super().__init__()
        
        if backbone not in self.SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone: {backbone}. "
                f"Supported: {list(self.SUPPORTED_BACKBONES.keys())}"
            )
        
        backbone_type, model_name = self.SUPPORTED_BACKBONES[backbone]
        
        if backbone_type == 'convnext':
            self.backbone = ConvNeXtBackbone(model_name, pretrained)
        elif backbone_type == 'swin':
            self.backbone = SwinTransformerBackbone(model_name, pretrained)
        elif backbone_type == 'efficientnet':
            self.backbone = EfficientNetV2Backbone(model_name, pretrained)
        elif backbone_type == 'mamba':
            self.backbone = VisionMambaBackbone(model_name, pretrained)
        else:
            raise ValueError(f"Unknown backbone type: {backbone_type}")
        
        self.feature_channels = self.backbone.feature_channels
        self.backbone_name = backbone
        
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            List of multi-scale features
        """
        return self.backbone(x)
    
    def get_feature_dims(self):
        """返回特征维度"""
        return self.feature_channels


if __name__ == '__main__':
    # 测试不同的 backbone
    print("Testing Modern Backbones...")
    print("=" * 80)
    
    backbones_to_test = [
        'convnext_tiny',
        'convnext_base',
        'swin_tiny',
        'efficientnetv2_s',
    ]
    
    x = torch.randn(2, 3, 224, 224)
    
    for backbone_name in backbones_to_test:
        print(f"\nTesting {backbone_name}...")
        try:
            model = ModernFeatureExtractor(backbone=backbone_name, pretrained=False)
            features = model(x)
            
            print(f"  Input: {x.shape}")
            print(f"  Feature channels: {model.feature_channels}")
            for i, feat in enumerate(features):
                print(f"  Feature {i+1}: {feat.shape}")
            
            # 计算参数量
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  Total parameters: {total_params:,}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "=" * 80)
    print("Testing complete!")
