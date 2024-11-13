import torch
import torch.nn as nn

class AdaptiveFeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, features):
        # 特征拼接
        concat_features = torch.cat(features, dim=1)
        
        # 注意力权重
        attention_weights = self.attention(concat_features)
        
        # 加权融合
        weighted_features = concat_features * attention_weights
        
        # 降维
        output = self.conv1x1(weighted_features)
        
        return output
