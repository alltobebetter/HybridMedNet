import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from configs.default_config import Config
from models.hybrid_med_net import HybridMedNet
from data.chest_xray_dataset import ChestXrayDataset, create_sample_dataset
from data.transforms import get_train_transforms, get_val_transforms
from utils.metrics import calculate_metrics, compute_auc_scores, print_metrics, AverageMeter
from utils.visualization import plot_training_curves, plot_roc_curves


def train_epoch(model, dataloader, criterion, optimizer, device, config, epoch):
    """训练一个epoch"""
    model.train()
    
    losses = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config.TRAIN["epochs"]} [Train]')
    
    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        if config.MODEL.get('hierarchical', False):
            coarse_pred, fine_pred = model(images)
            loss = criterion(fine_pred, labels)
        else:
            pred = model(images)
            loss = criterion(pred, labels)
        
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item(), images.size(0))
        
        pbar.set_postfix({'loss': f'{losses.avg:.4f}'})
        
        if batch_idx % config.LOGGING['print_freq'] == 0 and batch_idx > 0:
            print(f'  Batch [{batch_idx}/{len(dataloader)}] Loss: {losses.avg:.4f}')
    
    return losses.avg


def validate_epoch(model, dataloader, criterion, device, config, epoch):
    """验证一个epoch"""
    model.eval()
    
    losses = AverageMeter()
    all_preds = []
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config.TRAIN["epochs"]} [Val]')
        
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            if config.MODEL.get('hierarchical', False):
                coarse_pred, fine_pred = model(images)
                pred = fine_pred
            else:
                pred = model(images)
            
            loss = criterion(pred, labels)
            
            losses.update(loss.item(), images.size(0))
            
            scores = torch.sigmoid(pred)
            preds = (scores > 0.5).float()
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_scores.append(scores.cpu())
            
            pbar.set_postfix({'loss': f'{losses.avg:.4f}'})
    
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_scores = torch.cat(all_scores, dim=0).numpy()
    
    metrics = calculate_metrics(all_labels, all_preds, all_scores, average='macro')
    
    return losses.avg, metrics, all_labels, all_scores


def main(args):
    print("="*80)
    print("HybridMedNet Training")
    print("="*80)
    
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    os.makedirs(config.CHECKPOINT['save_dir'], exist_ok=True)
    os.makedirs(config.LOGGING['log_dir'], exist_ok=True)
    
    data_dir = config.DATA['data_dir']
    labels_file = config.DATA['labels_file']
    
    if not os.path.exists(data_dir):
        print(f"\nData directory not found. Creating sample dataset...")
        data_dir, labels_file = create_sample_dataset(
            save_dir=data_dir,
            num_samples=200
        )
        config.DATA['data_dir'] = data_dir
        config.DATA['labels_file'] = labels_file
    
    print(f"\n{'='*80}")
    print("Loading Dataset")
    print(f"{'='*80}")
    
    train_transform = get_train_transforms(
        image_size=config.DATA['image_size'],
        use_albumentations=config.DATA['use_albumentations']
    )
    val_transform = get_val_transforms(
        image_size=config.DATA['image_size'],
        use_albumentations=config.DATA['use_albumentations']
    )
    
    full_dataset = ChestXrayDataset(
        data_dir=data_dir,
        labels_file=labels_file,
        transform=None,
        mode='train'
    )
    
    total_size = len(full_dataset)
    train_size = int(config.DATA['train_split'] * total_size)
    val_size = int(config.DATA['val_split'] * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    test_dataset.dataset.transform = val_transform
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.TRAIN['batch_size'],
        shuffle=True,
        num_workers=config.DATA['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.TRAIN['batch_size'],
        shuffle=False,
        num_workers=config.DATA['num_workers'],
        pin_memory=True
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {train_size} samples")
    print(f"  Val:   {val_size} samples")
    print(f"  Test:  {test_size} samples")
    
    print(f"\n{'='*80}")
    print("Building Model")
    print(f"{'='*80}")
    
    model = HybridMedNet(config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {config.MODEL['name']}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Backbone: {config.MODEL['backbone']}")
    print(f"Hierarchical: {config.MODEL['hierarchical']}")
    
    if config.LOSS['type'] == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.TRAIN['learning_rate'],
        weight_decay=config.TRAIN['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.TRAIN['epochs'],
        eta_min=1e-6
    )
    
    if config.LOGGING['tensorboard']:
        writer = SummaryWriter(config.LOGGING['log_dir'])
    
    print(f"\n{'='*80}")
    print("Training")
    print(f"{'='*80}")
    
    best_val_auc = 0.0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_auc': [],
    }
    
    for epoch in range(config.TRAIN['epochs']):
        print(f"\nEpoch {epoch+1}/{config.TRAIN['epochs']}")
        print("-" * 80)
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, config, epoch)
        val_loss, val_metrics, val_labels, val_scores = validate_epoch(
            model, val_loader, criterion, device, config, epoch
        )
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_auc'].append(val_metrics.get('auc', 0.0))
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print_metrics(val_metrics, prefix='  Val ')
        
        if config.LOGGING['tensorboard']:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Metrics/accuracy', val_metrics['accuracy'], epoch)
            writer.add_scalar('Metrics/auc', val_metrics.get('auc', 0.0), epoch)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        if val_metrics.get('auc', 0.0) > best_val_auc:
            best_val_auc = val_metrics['auc']
            if config.CHECKPOINT['save_best']:
                save_path = os.path.join(config.CHECKPOINT['save_dir'], 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_auc': best_val_auc,
                    'config': config
                }, save_path)
                print(f"  Saved best model (AUC: {best_val_auc:.4f})")
        
        if (epoch + 1) % config.CHECKPOINT['save_freq'] == 0:
            save_path = os.path.join(config.CHECKPOINT['save_dir'], f'model_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config
            }, save_path)
            print(f"  Saved checkpoint: {save_path}")
    
    print(f"\n{'='*80}")
    print("Training Complete!")
    print(f"{'='*80}")
    print(f"Best validation AUC: {best_val_auc:.4f}")
    
    if config.LOGGING['tensorboard']:
        writer.close()
    
    fig = plot_training_curves(history, metrics=['loss', 'accuracy', 'auc'])
    fig.savefig(os.path.join(config.LOGGING['log_dir'], 'training_curves.png'), dpi=300, bbox_inches='tight')
    print(f"\nTraining curves saved to {config.LOGGING['log_dir']}/training_curves.png")
    
    try:
        plot_roc_curves(
            val_labels,
            val_scores,
            class_names=ChestXrayDataset.DISEASE_LABELS,
            save_path=os.path.join(config.LOGGING['log_dir'], 'roc_curves.png')
        )
    except Exception as e:
        print(f"Warning: Could not plot ROC curves - {e}")
    
    print("\nDone!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train HybridMedNet')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    main(args)
