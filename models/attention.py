import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialAttention(nn.Module):
    """
    空间注意力模块 - 关注图像中的关键区域
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # 沿通道维度进行平均池化和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 拼接两个池化结果
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        # 卷积 + Sigmoid
        attention = self.conv(x_cat)
        return self.sigmoid(attention)


class ChannelAttention(nn.Module):
    """
    通道注意力模块 - 学习不同特征通道的重要性
    """
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    结合通道注意力和空间注意力
    """
    def __init__(self, in_channels, ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        # 通道注意力
        x = x * self.channel_attention(x)
        # 空间注意力
        x = x * self.spatial_attention(x)
        return x


class MultiScaleAttention(nn.Module):
    """
    多尺度注意力模块 - 对多个尺度的特征分别应用注意力机制
    """
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        # 为单一通道数创建CBAM
        if isinstance(in_channels, int):
            self.cbam = CBAM(in_channels, ratio)
            self.is_multi_channel = False
        # 为多个通道数创建多个CBAM
        elif isinstance(in_channels, (list, tuple)):
            self.cbam_modules = nn.ModuleList([
                CBAM(ch, ratio) for ch in in_channels
            ])
            self.is_multi_channel = True
        else:
            raise ValueError("in_channels must be int or list of ints")
        
    def forward(self, features):
        """
        Args:
            features: 单个特征图 [B, C, H, W] 或特征图列表 [[B, C1, H1, W1], ...]
        
        Returns:
            注意力增强后的特征
        """
        if not self.is_multi_channel:
            # 单个特征图
            return self.cbam(features)
        else:
            # 多个特征图
            if not isinstance(features, (list, tuple)):
                raise ValueError("Expected list of features for multi-channel attention")
            
            attended_features = []
            for feat, cbam in zip(features, self.cbam_modules):
                attended = cbam(feat)
                attended_features.append(attended)
            
            return attended_features


class CrossScaleAttention(nn.Module):
    """
    跨尺度注意力 - 让不同尺度的特征相互增强
    """
    def __init__(self, channels_list):
        super().__init__()
        self.num_scales = len(channels_list)
        
        # 创建跨尺度的注意力权重生成器
        self.attention_weights = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(ch, self.num_scales, 1),
                nn.Softmax(dim=1)
            )
            for ch in channels_list
        ])
        
    def forward(self, features):
        """
        Args:
            features: List of feature maps at different scales
        
        Returns:
            Enhanced features with cross-scale attention
        """
        # 为每个尺度计算注意力权重
        weights = []
        for feat, weight_gen in zip(features, self.attention_weights):
            w = weight_gen(feat)  # [B, num_scales, 1, 1]
            weights.append(w)
        
        # 使用注意力权重增强特征
        enhanced_features = []
        for i, feat in enumerate(features):
            # 收集所有尺度对当前特征的贡献
            weighted_sum = 0
            for j, (other_feat, w) in enumerate(zip(features, weights)):
                # 将其他尺度的特征调整到当前尺度
                if i != j:
                    resized = F.interpolate(
                        other_feat,
                        size=feat.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                else:
                    resized = other_feat
                
                # 加权求和
                weight = w[:, i:i+1, :, :]  # 第i个权重
                weighted_sum = weighted_sum + resized * weight
            
            enhanced_features.append(weighted_sum)
        
        return enhanced_features


if __name__ == '__main__':
    # 测试代码
    print("Testing MultiScaleAttention...")
    channels = [256, 512, 1024, 2048]
    attention = MultiScaleAttention(channels)
    
    features = [
        torch.randn(2, 256, 56, 56),
        torch.randn(2, 512, 28, 28),
        torch.randn(2, 1024, 14, 14),
        torch.randn(2, 2048, 7, 7)
    ]
    
    attended = attention(features)
    print(f"Number of scales: {len(attended)}")
    for i, feat in enumerate(attended):
        print(f"Scale {i+1}: {feat.shape}")
