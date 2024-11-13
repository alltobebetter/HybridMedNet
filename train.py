import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from configs.default_config import Config
from models.hybrid_med_net import HybridMedNet
from utils.metrics import calculate_metrics

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (data, coarse_target, fine_target) in enumerate(train_loader):
        data = data.to(device)
        coarse_target = coarse_target.to(device)
        fine_target = fine_target.to(device)
        
        optimizer.zero_grad()
        coarse_pred, fine_pred = model(data)
        
        # 计算损失
        coarse_loss = criterion(coarse_pred, coarse_target)
        fine_loss = criterion(fine_pred, fine_target)
        consistency_loss = F.kl_div(
            F.log_softmax(fine_pred, dim=1),
            F.softmax(coarse_pred, dim=1)
        )
        
        loss = (Config.LOSS['coarse_weight'] * coarse_loss + 
                Config.LOSS['fine_weight'] * fine_loss +
                Config.LOSS['consistency_weight'] * consistency_loss)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(train_loader)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridMedNet(Config).to(device)
    
    # 初始化优化器和学习率调度器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.TRAIN['learning_rate'],
        weight_decay=Config.TRAIN['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Config.TRAIN['epochs']
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    for epoch in range(Config.TRAIN['epochs']):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        
        print(f'Epoch {epoch+1}/{Config.TRAIN["epochs"]}, Loss: {train_loss:.4f}')

if __name__ == '__main__':
    main()
