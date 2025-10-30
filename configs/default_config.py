class Config:
    # 模型配置
    MODEL = {
        'name': 'HybridMedNet',
        'input_size': (224, 224),
        'num_classes': 14,
        'num_coarse_classes': 3,
        
        # Backbone 选择
        # 经典: 'resnet34', 'resnet50', 'resnet101'
        # 现代: 'convnext_tiny', 'convnext_small', 'convnext_base',
        #       'swin_tiny', 'swin_small', 'swin_base',
        #       'efficientnetv2_s', 'efficientnetv2_m'
        'backbone': 'convnext_base',  # 默认使用现代架构
        
        'pretrained': True,
        'fpn_channels': 256,
        'fusion_channels': 512,
        'dropout': 0.5,
        'hierarchical': False,
        
        # 现代架构专用配置
        'use_modern_attention': True,  # 使用现代注意力机制
        'use_cross_scale_attention': True,  # 使用跨尺度注意力
        'num_heads': 8,  # 注意力头数
    }
    
    # 训练配置
    TRAIN = {
        'batch_size': 32,
        'epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'lr_scheduler': 'cosine',
        'warmup_epochs': 5,
        'gradient_accumulation_steps': 1,
        'mixed_precision': True,
        'early_stopping_patience': 10,
    }
    
    # 数据配置
    DATA = {
        'data_dir': './data/sample_chest_xray',
        'labels_file': './data/sample_chest_xray/labels.csv',
        'image_size': 224,
        'num_workers': 4,
        'train_split': 0.7,
        'val_split': 0.15,
        'test_split': 0.15,
        'use_albumentations': False,
    }
    
    # 损失函数配置
    LOSS = {
        'type': 'bce',
        'pos_weight': None,
        'coarse_weight': 0.4,
        'fine_weight': 0.6,
        'consistency_weight': 0.2,
    }
    
    # 其他配置
    CHECKPOINT = {
        'save_dir': './checkpoints',
        'save_freq': 5,
        'save_best': True,
    }
    
    LOGGING = {
        'log_dir': './logs',
        'tensorboard': True,
        'print_freq': 10,
    }
