"""
快速验证升级是否成功
不需要安装 PyTorch，只检查代码结构
"""
import os
import sys


def check_file_exists(filepath):
    """检查文件是否存在"""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"  {status} {filepath}")
    return exists


def check_imports():
    """检查模块导入"""
    print("\n检查模块导入...")
    
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        
        # 检查新模块
        print("\n  现代架构模块:")
        from models import modern_backbones
        print("    ✓ modern_backbones")
        
        from models import modern_attention
        print("    ✓ modern_attention")
        
        # 检查主模型
        print("\n  主模型:")
        from models import hybrid_med_net
        print("    ✓ hybrid_med_net (已更新)")
        
        # 检查配置
        print("\n  配置文件:")
        from configs import default_config
        print("    ✓ default_config (已更新)")
        
        return True
        
    except Exception as e:
        print(f"    ✗ 导入失败: {e}")
        return False


def check_architecture_support():
    """检查支持的架构"""
    print("\n检查支持的架构...")
    
    try:
        from models.modern_backbones import ModernFeatureExtractor
        
        supported = ModernFeatureExtractor.SUPPORTED_BACKBONES
        
        print(f"\n  支持 {len(supported)} 种现代架构:")
        
        categories = {
            'ConvNeXt': [],
            'Swin Transformer': [],
            'EfficientNetV2': [],
            'Vision Mamba': []
        }
        
        for name in supported.keys():
            if 'convnext' in name:
                categories['ConvNeXt'].append(name)
            elif 'swin' in name:
                categories['Swin Transformer'].append(name)
            elif 'efficientnet' in name:
                categories['EfficientNetV2'].append(name)
            elif 'vim' in name:
                categories['Vision Mamba'].append(name)
        
        for category, models in categories.items():
            if models:
                print(f"\n  {category}:")
                for model in models:
                    print(f"    - {model}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 检查失败: {e}")
        return False


def main():
    print("="*80)
    print("HybridMedNet 升级验证")
    print("="*80)
    
    # 检查新文件
    print("\n1. 检查新增文件...")
    files_to_check = [
        'models/modern_backbones.py',
        'models/modern_attention.py',
        'test_modern_models.py',
        'MODERN_ARCHITECTURE_GUIDE.md',
    ]
    
    all_exist = all(check_file_exists(f) for f in files_to_check)
    
    # 检查更新的文件
    print("\n2. 检查更新的文件...")
    updated_files = [
        'models/hybrid_med_net.py',
        'configs/default_config.py',
        'requirements.txt',
        'README.md',
    ]
    
    all_updated = all(check_file_exists(f) for f in updated_files)
    
    # 检查导入
    print("\n3. 检查模块导入...")
    imports_ok = check_imports()
    
    # 检查架构支持
    print("\n4. 检查架构支持...")
    arch_ok = check_architecture_support()
    
    # 总结
    print("\n" + "="*80)
    print("验证结果")
    print("="*80)
    
    results = {
        "新增文件": all_exist,
        "更新文件": all_updated,
        "模块导入": imports_ok,
        "架构支持": arch_ok,
    }
    
    for name, status in results.items():
        symbol = "✓" if status else "✗"
        print(f"{symbol} {name}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ 升级成功！所有检查通过。")
        print("\n下一步:")
        print("1. 安装依赖: pip install timm einops")
        print("2. 测试模型: python test_modern_models.py")
        print("3. 查看文档: MODERN_ARCHITECTURE_GUIDE.md")
    else:
        print("✗ 升级未完成，请检查上述错误。")
    print("="*80)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
