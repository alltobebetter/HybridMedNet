"""
测试现代化模型架构
验证所有新增的 backbone 和注意力机制
"""
import torch
import sys
from configs.default_config import Config
from models.hybrid_med_net import HybridMedNet


def test_backbone(backbone_name, input_size=(224, 224)):
    """测试单个 backbone"""
    print(f"\n{'='*80}")
    print(f"Testing: {backbone_name}")
    print(f"{'='*80}")
    
    try:
        # 创建配置
        config = Config()
        config.MODEL['backbone'] = backbone_name
        config.MODEL['pretrained'] = False  # 测试时不加载预训练权重
        config.MODEL['hierarchical'] = False
        
        # 创建模型
        model = HybridMedNet(config)
        model.eval()
        
        # 测试前向传播
        batch_size = 2
        x = torch.randn(batch_size, 3, *input_size)
        
        with torch.no_grad():
            output = model(x)
        
        # 打印信息
        print(f"✓ Model created successfully")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")
        
        # 测试梯度
        loss = output.sum()
        loss.backward()
        print(f"✓ Backward pass successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_architectures():
    """比较不同架构的性能"""
    print("\n" + "="*80)
    print("Architecture Comparison")
    print("="*80)
    
    architectures = {
        'Classic': [
            'resnet34',
            'resnet50',
        ],
        'Modern CNN': [
            'convnext_tiny',
            'convnext_base',
        ],
        'Transformer': [
            'swin_tiny',
            'swin_base',
        ],
        'Efficient': [
            'efficientnetv2_s',
            'efficientnetv2_m',
        ]
    }
    
    results = {}
    
    for category, backbones in architectures.items():
        print(f"\n{category}:")
        print("-" * 80)
        
        for backbone in backbones:
            success = test_backbone(backbone)
            results[backbone] = success
    
    # 总结
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    
    successful = [name for name, success in results.items() if success]
    failed = [name for name, success in results.items() if not success]
    
    print(f"\n✓ Successful: {len(successful)}/{len(results)}")
    for name in successful:
        print(f"  - {name}")
    
    if failed:
        print(f"\n✗ Failed: {len(failed)}/{len(results)}")
        for name in failed:
            print(f"  - {name}")
    
    return len(failed) == 0


def test_attention_mechanisms():
    """测试不同的注意力机制"""
    print("\n" + "="*80)
    print("Testing Attention Mechanisms")
    print("="*80)
    
    config = Config()
    config.MODEL['backbone'] = 'convnext_base'
    config.MODEL['pretrained'] = False
    
    # 测试经典注意力
    print("\n1. Classic CBAM Attention")
    config.MODEL['use_modern_attention'] = False
    try:
        model = HybridMedNet(config)
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        print(f"   ✓ Output shape: {output.shape}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # 测试现代注意力
    print("\n2. Modern Multi-Scale Attention")
    config.MODEL['use_modern_attention'] = True
    config.MODEL['use_cross_scale_attention'] = False
    try:
        model = HybridMedNet(config)
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        print(f"   ✓ Output shape: {output.shape}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # 测试跨尺度注意力
    print("\n3. Cross-Scale Attention")
    config.MODEL['use_modern_attention'] = True
    config.MODEL['use_cross_scale_attention'] = True
    try:
        model = HybridMedNet(config)
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        print(f"   ✓ Output shape: {output.shape}")
    except Exception as e:
        print(f"   ✗ Error: {e}")


def main():
    print("="*80)
    print("HybridMedNet Modern Architecture Testing")
    print("="*80)
    
    # 检查依赖
    print("\nChecking dependencies...")
    try:
        import timm
        print(f"✓ timm version: {timm.__version__}")
    except ImportError:
        print("✗ timm not installed. Install with: pip install timm")
        return
    
    try:
        import einops
        print(f"✓ einops installed")
    except ImportError:
        print("✗ einops not installed. Install with: pip install einops")
        return
    
    # 测试所有架构
    all_success = compare_architectures()
    
    # 测试注意力机制
    test_attention_mechanisms()
    
    # 最终结果
    print("\n" + "="*80)
    if all_success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed. Check the output above.")
    print("="*80)


if __name__ == '__main__':
    main()
