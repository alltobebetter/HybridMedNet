import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Dict


class PyramidFeatureExtractor(nn.Module):
    """
    金字塔特征提取器 - 多尺度特征提取
    使用预训练的ResNet/EfficientNet作为backbone
    """
    def __init__(self, backbone='resnet50', pretrained=True):
        super().__init__()
        self.backbone_name = backbone
        
        # 加载预训练的backbone
        if 'resnet' in backbone.lower():
            if backbone == 'resnet50':
                base_model = models.resnet50(pretrained=pretrained)
            elif backbone == 'resnet34':
                base_model = models.resnet34(pretrained=pretrained)
            elif backbone == 'resnet101':
                base_model = models.resnet101(pretrained=pretrained)
            else:
                raise ValueError(f"Unsupported backbone: {backbone}")
            
            # 提取ResNet的不同层作为特征金字塔
            self.conv1 = nn.Sequential(
                base_model.conv1,
                base_model.bn1,
                base_model.relu,
                base_model.maxpool
            )
            self.layer1 = base_model.layer1  # 64 channels (ResNet50: 256)
            self.layer2 = base_model.layer2  # 128 channels (ResNet50: 512)
            self.layer3 = base_model.layer3  # 256 channels (ResNet50: 1024)
            self.layer4 = base_model.layer4  # 512 channels (ResNet50: 2048)
            
            # 定义特征通道数
            if backbone == 'resnet50' or backbone == 'resnet101':
                self.feature_channels = [256, 512, 1024, 2048]
            else:  # resnet34
                self.feature_channels = [64, 128, 256, 512]
        
        else:
            raise ValueError(f"Backbone {backbone} not implemented yet")
        
    def forward(self, x):
        """
        Args:
            x: Input tensor [B, 3, H, W]
        
        Returns:
            List of feature maps at different scales
        """
        features = []
        
        # Stage 0: Initial convolution
        x = self.conv1(x)  # [B, C, H/4, W/4]
        
        # Stage 1-4: ResNet blocks
        x1 = self.layer1(x)   # [B, 256, H/4, W/4]
        features.append(x1)
        
        x2 = self.layer2(x1)  # [B, 512, H/8, W/8]
        features.append(x2)
        
        x3 = self.layer3(x2)  # [B, 1024, H/16, W/16]
        features.append(x3)
        
        x4 = self.layer4(x3)  # [B, 2048, H/32, W/32]
        features.append(x4)
        
        return features


class FeaturePyramidNetwork(nn.Module):
    """
    特征金字塔网络 (FPN) - 用于特征融合
    """
    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        super().__init__()
        
        # 1x1卷积用于降维到统一通道数
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels_list
        ])
        
        # 3x3卷积用于消除上采样的混叠效应
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels_list
        ])
        
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: List of feature maps from backbone [f1, f2, f3, f4]
                     从低分辨率到高分辨率
        
        Returns:
            List of FPN feature maps
        """
        # 从最高层开始，自顶向下融合
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]
        
        # 自顶向下路径
        for i in range(len(laterals) - 1, 0, -1):
            # 上采样高层特征
            upsampled = F.interpolate(
                laterals[i],
                size=laterals[i-1].shape[-2:],
                mode='nearest'
            )
            # 逐元素相加
            laterals[i-1] = laterals[i-1] + upsampled
        
        # 应用3x3卷积
        outputs = [conv(lateral) for conv, lateral in zip(self.output_convs, laterals)]
        
        return outputs


class MultiScaleFeatureExtractor(nn.Module):
    """
    完整的多尺度特征提取模块
    结合Backbone + FPN
    """
    def __init__(self, backbone='resnet50', pretrained=True, fpn_out_channels=256):
        super().__init__()
        
        # Backbone特征提取器
        self.backbone = PyramidFeatureExtractor(backbone=backbone, pretrained=pretrained)
        
        # 特征金字塔网络
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=self.backbone.feature_channels,
            out_channels=fpn_out_channels
        )
        
        self.out_channels = fpn_out_channels
        
    def forward(self, x):
        """
        Args:
            x: Input image [B, 3, H, W]
        
        Returns:
            List of multi-scale features after FPN
        """
        # 提取多尺度特征
        features = self.backbone(x)
        
        # FPN融合
        fpn_features = self.fpn(features)
        
        return fpn_features
    
    def get_feature_dims(self):
        """返回每个尺度的特征维度"""
        return [self.out_channels] * len(self.backbone.feature_channels)


if __name__ == '__main__':
    # 测试代码
    model = MultiScaleFeatureExtractor(backbone='resnet50', pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    features = model(x)
    
    print(f"Input shape: {x.shape}")
    for i, feat in enumerate(features):
        print(f"Feature {i+1} shape: {feat.shape}")
