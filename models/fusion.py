import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveFeatureFusion(nn.Module):
    """
    自适应特征融合模块 - 智能融合多尺度特征
    使用注意力机制动态学习每个尺度特征的权重
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 1x1卷积降维
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        
        # 注意力权重生成
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, features):
        """
        Args:
            features: List of feature maps [B, C, H, W]
        
        Returns:
            Fused features [B, out_channels, H, W]
        """
        # 将所有特征调整到相同的空间尺寸（使用第一个特征的尺寸）
        target_size = features[0].shape[-2:]
        resized_features = []
        
        for feat in features:
            if feat.shape[-2:] != target_size:
                resized = F.interpolate(
                    feat,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )
            else:
                resized = feat
            resized_features.append(resized)
        
        # 特征拼接
        concat_features = torch.cat(resized_features, dim=1)
        
        # 生成注意力权重
        attention_weights = self.attention(concat_features)
        
        # 加权融合
        weighted_features = concat_features * attention_weights
        
        # 降维并归一化
        output = self.conv1x1(weighted_features)
        output = self.bn(output)
        
        return output


class DynamicFeatureFusion(nn.Module):
    """
    动态特征融合 - 为每个尺度学习独立的权重
    """
    def __init__(self, channels_list, out_channels):
        super().__init__()
        self.num_scales = len(channels_list)
        self.out_channels = out_channels
        
        # 为每个尺度创建独立的卷积
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for ch in channels_list
        ])
        
        # 动态权重生成器
        self.weight_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(sum(channels_list), self.num_scales, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, features):
        """
        Args:
            features: List of feature maps at different scales
        
        Returns:
            Dynamically fused features
        """
        # 调整所有特征到相同尺寸
        target_size = features[0].shape[-2:]
        resized_features = []
        
        for feat in features:
            if feat.shape[-2:] != target_size:
                resized = F.interpolate(
                    feat,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )
            else:
                resized = feat
            resized_features.append(resized)
        
        # 拼接用于生成权重
        concat_for_weights = torch.cat(resized_features, dim=1)
        weights = self.weight_generator(concat_for_weights)  # [B, num_scales, 1, 1]
        
        # 将每个尺度映射到统一通道数
        transformed_features = []
        for feat, conv in zip(resized_features, self.scale_convs):
            transformed = conv(feat)
            transformed_features.append(transformed)
        
        # 动态加权求和
        output = 0
        for i, feat in enumerate(transformed_features):
            weight = weights[:, i:i+1, :, :]
            output = output + feat * weight
        
        return output


class HierarchicalFeatureFusion(nn.Module):
    """
    层次化特征融合 - 从低层到高层逐步融合
    """
    def __init__(self, channels_list, out_channels):
        super().__init__()
        
        # 创建层次化融合路径
        self.fusion_blocks = nn.ModuleList()
        
        current_channels = channels_list[0]
        for i in range(1, len(channels_list)):
            fusion_block = nn.Sequential(
                nn.Conv2d(current_channels + channels_list[i], out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.fusion_blocks.append(fusion_block)
            current_channels = out_channels
        
    def forward(self, features):
        """
        Args:
            features: List of feature maps from low to high level
        
        Returns:
            Hierarchically fused features
        """
        # 从最低层开始逐步融合
        fused = features[0]
        target_size = features[0].shape[-2:]
        
        for i, (feat, fusion_block) in enumerate(zip(features[1:], self.fusion_blocks)):
            # 调整当前特征的尺寸
            if feat.shape[-2:] != target_size:
                feat_resized = F.interpolate(
                    feat,
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )
            else:
                feat_resized = feat
            
            # 拼接并融合
            concat = torch.cat([fused, feat_resized], dim=1)
            fused = fusion_block(concat)
        
        return fused


if __name__ == '__main__':
    # 测试代码
    print("Testing AdaptiveFeatureFusion...")
    fusion = AdaptiveFeatureFusion(in_channels=256*4, out_channels=512)
    features = [
        torch.randn(2, 256, 56, 56),
        torch.randn(2, 256, 28, 28),
        torch.randn(2, 256, 14, 14),
        torch.randn(2, 256, 7, 7)
    ]
    output = fusion(features)
    print(f"Output shape: {output.shape}")
    
    print("\nTesting DynamicFeatureFusion...")
    channels = [256, 512, 1024, 2048]
    dynamic_fusion = DynamicFeatureFusion(channels, out_channels=512)
    features = [
        torch.randn(2, 256, 56, 56),
        torch.randn(2, 512, 28, 28),
        torch.randn(2, 1024, 14, 14),
        torch.randn(2, 2048, 7, 7)
    ]
    output = dynamic_fusion(features)
    print(f"Output shape: {output.shape}")
