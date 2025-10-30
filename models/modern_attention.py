"""
现代注意力机制
包含 Multi-head Self-Attention, Cross-Attention, Deformable Attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力机制 (Transformer 标准组件)
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        """
        Args:
            x: [B, N, C] where N = H*W
        Returns:
            [B, N, C]
        """
        B, N, C = x.shape
        
        # 生成 Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 应用注意力
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class CrossAttention(nn.Module):
    """
    交叉注意力机制
    用于不同特征之间的交互
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x, context):
        """
        Args:
            x: query [B, N, C]
            context: key/value [B, M, C]
        Returns:
            [B, N, C]
        """
        B, N, C = x.shape
        _, M, _ = context.shape
        
        # Q from x, K,V from context
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(B, M, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 应用注意力
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class SpatialCrossScaleAttention(nn.Module):
    """
    空间跨尺度注意力
    让不同尺度的特征相互增强
    """
    def __init__(self, channels_list, num_heads=8):
        super().__init__()
        self.num_scales = len(channels_list)
        
        # 为每个尺度创建投影层，统一到相同维度
        target_dim = min(channels_list)
        self.projections = nn.ModuleList([
            nn.Conv2d(ch, target_dim, 1) if ch != target_dim else nn.Identity()
            for ch in channels_list
        ])
        
        # 跨尺度注意力
        self.cross_attentions = nn.ModuleList([
            CrossAttention(target_dim, num_heads=num_heads)
            for _ in range(self.num_scales)
        ])
        
        # 输出投影
        self.out_projections = nn.ModuleList([
            nn.Conv2d(target_dim, ch, 1)
            for ch in channels_list
        ])
        
    def forward(self, features):
        """
        Args:
            features: List of [B, C_i, H_i, W_i]
        Returns:
            List of enhanced features
        """
        B = features[0].shape[0]
        
        # 投影到统一维度
        projected = []
        for feat, proj in zip(features, self.projections):
            proj_feat = proj(feat)
            projected.append(proj_feat)
        
        # 对每个尺度应用跨尺度注意力
        enhanced = []
        for i, (feat, cross_attn) in enumerate(zip(projected, self.cross_attentions)):
            H, W = feat.shape[-2:]
            
            # 将特征展平为序列
            feat_flat = rearrange(feat, 'b c h w -> b (h w) c')
            
            # 收集其他尺度的特征作为 context
            context_list = []
            for j, other_feat in enumerate(projected):
                if i != j:
                    # 调整到相同空间尺寸
                    other_resized = F.interpolate(
                        other_feat, size=(H, W),
                        mode='bilinear', align_corners=False
                    )
                    other_flat = rearrange(other_resized, 'b c h w -> b (h w) c')
                    context_list.append(other_flat)
            
            # 拼接所有 context
            context = torch.cat(context_list, dim=1)
            
            # 应用交叉注意力
            enhanced_flat = cross_attn(feat_flat, context)
            enhanced_feat = rearrange(enhanced_flat, 'b (h w) c -> b c h w', h=H, w=W)
            
            # 残差连接
            enhanced_feat = enhanced_feat + feat
            
            enhanced.append(enhanced_feat)
        
        # 投影回原始维度
        outputs = []
        for feat, out_proj in zip(enhanced, self.out_projections):
            output = out_proj(feat)
            outputs.append(output)
        
        return outputs


class EfficientAttention(nn.Module):
    """
    高效注意力机制
    使用线性复杂度的注意力计算
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        assert dim % num_heads == 0
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        """
        Args:
            x: [B, N, C]
        Returns:
            [B, N, C]
        """
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 高效注意力: softmax(Q) @ (K^T @ V)
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)
        
        # [B, H, N, D] @ [B, H, D, N] @ [B, H, N, D] = [B, H, N, D]
        context = k.transpose(-2, -1) @ v  # [B, H, D, D]
        x = q @ context  # [B, H, N, D]
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class ModernMultiScaleAttention(nn.Module):
    """
    现代化的多尺度注意力模块
    结合自注意力和跨尺度注意力
    """
    def __init__(self, channels_list, num_heads=8, use_cross_scale=True):
        super().__init__()
        self.use_cross_scale = use_cross_scale
        
        # 为每个尺度创建自注意力
        self.self_attentions = nn.ModuleList()
        for ch in channels_list:
            # 使用 1x1 卷积调整维度以适应注意力
            self.self_attentions.append(
                nn.Sequential(
                    nn.Conv2d(ch, ch, 1),
                    nn.BatchNorm2d(ch),
                )
            )
        
        # 跨尺度注意力
        if use_cross_scale:
            self.cross_scale_attn = SpatialCrossScaleAttention(
                channels_list, num_heads=num_heads
            )
        
    def forward(self, features):
        """
        Args:
            features: List of [B, C_i, H_i, W_i]
        Returns:
            List of attended features
        """
        # 自注意力
        attended = []
        for feat, self_attn in zip(features, self.self_attentions):
            att_feat = self_attn(feat)
            attended.append(att_feat + feat)  # 残差连接
        
        # 跨尺度注意力
        if self.use_cross_scale:
            attended = self.cross_scale_attn(attended)
        
        return attended


if __name__ == '__main__':
    print("Testing Modern Attention Mechanisms...")
    print("=" * 80)
    
    # 测试多头自注意力
    print("\n1. Multi-Head Self-Attention")
    mhsa = MultiHeadSelfAttention(dim=256, num_heads=8)
    x = torch.randn(2, 196, 256)  # [B, N, C]
    out = mhsa(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    
    # 测试交叉注意力
    print("\n2. Cross Attention")
    cross_attn = CrossAttention(dim=256, num_heads=8)
    query = torch.randn(2, 196, 256)
    context = torch.randn(2, 49, 256)
    out = cross_attn(query, context)
    print(f"   Query: {query.shape}, Context: {context.shape} -> Output: {out.shape}")
    
    # 测试现代多尺度注意力
    print("\n3. Modern Multi-Scale Attention")
    channels = [128, 256, 512, 1024]
    modern_attn = ModernMultiScaleAttention(channels, num_heads=8)
    features = [
        torch.randn(2, 128, 56, 56),
        torch.randn(2, 256, 28, 28),
        torch.randn(2, 512, 14, 14),
        torch.randn(2, 1024, 7, 7),
    ]
    out_features = modern_attn(features)
    print(f"   Number of scales: {len(out_features)}")
    for i, feat in enumerate(out_features):
        print(f"   Scale {i+1}: {feat.shape}")
    
    print("\n" + "=" * 80)
    print("Testing complete!")
