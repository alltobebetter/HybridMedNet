import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()
        
    def forward(self, features):
        b, c, h, w = features[0].shape
        attended_features = []
        
        for feat in features:
            # 通道注意力
            channel_weight = self.channel_attention(feat)
            channel_refined = feat * channel_weight
            
            # 空间注意力
            spatial_weight = self.spatial_attention(channel_refined)
            refined = channel_refined * spatial_weight
            
            attended_features.append(refined)
            
        return attended_features

class ChannelAttention(nn.Module):
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
