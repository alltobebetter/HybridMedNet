import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from configs.default_config import Config
from models.hybrid_med_net import HybridMedNet
from data.chest_xray_dataset import ChestXrayDataset
from data.transforms import get_val_transforms
from utils.metrics import calculate_metrics, compute_auc_scores, print_metrics
from utils.visualization import plot_roc_curves, plot_confusion_matrix


def evaluate(model, dataloader, device, config):
    """评估模型"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_scores = []
    
    print("Evaluating...")
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            
            if config.MODEL.get('hierarchical', False):
                coarse_pred, fine_pred = model(images)
                pred = fine_pred
            else:
                pred = model(images)
            
            scores = torch.sigmoid(pred)
            preds = (scores > 0.5).float()
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_scores.append(scores.cpu())
    
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_scores = torch.cat(all_scores, dim=0).numpy()
    
    return all_preds, all_labels, all_scores


def main(args):
    print("="*80)
    print("HybridMedNet Evaluation")
    print("="*80)
    
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    print(f"\nLoading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if 'config' in checkpoint:
        config = checkpoint['config']
    
    model = HybridMedNet(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Model loaded successfully!")
    
    print(f"\n{'='*80}")
    print("Loading Test Dataset")
    print(f"{'='*80}")
    
    test_transform = get_val_transforms(
        image_size=config.DATA['image_size'],
        use_albumentations=config.DATA.get('use_albumentations', False)
    )
    
    test_dataset = ChestXrayDataset(
        data_dir=config.DATA['data_dir'],
        labels_file=config.DATA['labels_file'],
        transform=test_transform,
        mode='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.TRAIN['batch_size'],
        shuffle=False,
        num_workers=config.DATA['num_workers'],
        pin_memory=True
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    preds, labels, scores = evaluate(model, test_loader, device, config)
    
    print(f"\n{'='*80}")
    print("Results")
    print(f"{'='*80}")
    
    metrics = calculate_metrics(labels, preds, scores, average='macro')
    print_metrics(metrics, prefix='Test ')
    
    auc_scores = compute_auc_scores(
        labels, scores,
        class_names=ChestXrayDataset.DISEASE_LABELS
    )
    
    print("\nAUC per class:")
    for class_name, auc in auc_scores.items():
        print(f"  {class_name:20s}: {auc:.4f}")
    
    mean_auc = np.mean(list(auc_scores.values()))
    print(f"\n  Mean AUC: {mean_auc:.4f}")
    
    if args.save_results:
        output_dir = os.path.dirname(args.model_path)
        
        try:
            plot_roc_curves(
                labels, scores,
                class_names=ChestXrayDataset.DISEASE_LABELS,
                save_path=os.path.join(output_dir, 'test_roc_curves.png')
            )
        except Exception as e:
            print(f"\nWarning: Could not plot ROC curves - {e}")
        
        results_file = os.path.join(output_dir, 'test_results.txt')
        with open(results_file, 'w') as f:
            f.write("Test Results\n")
            f.write("="*80 + "\n\n")
            f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall:    {metrics['recall']:.4f}\n")
            f.write(f"F1 Score:  {metrics['f1_score']:.4f}\n")
            f.write(f"AUC:       {metrics.get('auc', 0.0):.4f}\n\n")
            f.write("AUC per class:\n")
            for class_name, auc in auc_scores.items():
                f.write(f"  {class_name:20s}: {auc:.4f}\n")
        
        print(f"\nResults saved to {results_file}")
    
    print("\nEvaluation complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate HybridMedNet')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--save_results', action='store_true',
                       help='Save evaluation results')
    
    args = parser.parse_args()
    main(args)
