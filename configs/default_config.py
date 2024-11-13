class Config:
    # 模型配置
    MODEL = {
        'name': 'HybridMedNet',
        'input_size': (224, 224),
        'num_classes': 20,
        'backbone': 'resnet50',
        'pretrained': True,
        'feature_dims': [256, 512, 1024],
    }
    
    # 训练配置
    TRAIN = {
        'batch_size': 32,
        'epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'lr_scheduler': 'cosine',
        'warmup_epochs': 5,
    }
    
    # 数据配置
    DATA = {
        'train_path': './data/train',
        'val_path': './data/val',
        'test_path': './data/test',
        'num_workers': 4,
    }
    
    # 损失函数配置
    LOSS = {
        'coarse_weight': 0.4,
        'fine_weight': 0.6,
        'consistency_weight': 0.2,
    }
