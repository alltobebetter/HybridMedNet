import os
import argparse
import torch
from PIL import Image
import numpy as np

from configs.default_config import Config
from models.hybrid_med_net import HybridMedNet
from data.chest_xray_dataset import ChestXrayDataset
from data.transforms import get_val_transforms, denormalize
from utils.visualization import visualize_attention

import matplotlib.pyplot as plt


def predict_single_image(model, image_path, transform, device, config, threshold=0.5):
    """预测单张图像"""
    model.eval()
    
    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)
    
    if config.DATA.get('use_albumentations', False):
        transformed = transform(image=original_image)
        image_tensor = transformed['image'].unsqueeze(0)
    else:
        image_tensor = transform(image).unsqueeze(0)
    
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        if config.MODEL.get('hierarchical', False):
            coarse_pred, fine_pred = model(image_tensor)
            pred = fine_pred
        else:
            pred = model(image_tensor)
        
        scores = torch.sigmoid(pred).cpu().numpy()[0]
    
    predicted_classes = []
    for i, score in enumerate(scores):
        if score > threshold:
            predicted_classes.append({
                'class': ChestXrayDataset.DISEASE_LABELS[i],
                'score': float(score)
            })
    
    predicted_classes = sorted(predicted_classes, key=lambda x: x['score'], reverse=True)
    
    return predicted_classes, scores, original_image


def visualize_prediction(image, predicted_classes, scores, class_names, save_path=None):
    """可视化预测结果"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title('Input Image', fontsize=14, fontweight='bold')
    
    if predicted_classes:
        title = "Predicted Diseases:\n" + "\n".join(
            [f"{p['class']}: {p['score']:.3f}" for p in predicted_classes[:5]]
        )
    else:
        title = "No significant findings"
    
    ax1.text(
        0.5, -0.1, title,
        ha='center', va='top',
        transform=ax1.transAxes,
        fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )
    
    colors = ['green' if s > 0.5 else 'gray' for s in scores]
    bars = ax2.barh(range(len(class_names)), scores, color=colors, alpha=0.7)
    ax2.set_yticks(range(len(class_names)))
    ax2.set_yticklabels(class_names, fontsize=9)
    ax2.set_xlabel('Confidence Score', fontsize=11)
    ax2.set_title('All Class Predictions', fontsize=14, fontweight='bold')
    ax2.axvline(x=0.5, color='red', linestyle='--', linewidth=1, label='Threshold')
    ax2.set_xlim([0, 1])
    ax2.legend()
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    return fig


def main(args):
    print("="*80)
    print("HybridMedNet Prediction")
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
    model.eval()
    
    print("Model loaded successfully!")
    
    transform = get_val_transforms(
        image_size=config.DATA['image_size'],
        use_albumentations=config.DATA.get('use_albumentations', False)
    )
    
    if os.path.isfile(args.image_path):
        image_paths = [args.image_path]
    elif os.path.isdir(args.image_path):
        image_paths = [
            os.path.join(args.image_path, f)
            for f in os.listdir(args.image_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
    else:
        raise ValueError(f"Invalid image path: {args.image_path}")
    
    print(f"\nFound {len(image_paths)} images to predict")
    
    for idx, img_path in enumerate(image_paths):
        print(f"\n{'='*80}")
        print(f"Image {idx+1}/{len(image_paths)}: {os.path.basename(img_path)}")
        print(f"{'='*80}")
        
        try:
            predicted_classes, scores, original_image = predict_single_image(
                model, img_path, transform, device, config, threshold=args.threshold
            )
            
            if predicted_classes:
                print("\nPredicted diseases:")
                for pred in predicted_classes:
                    print(f"  {pred['class']:20s}: {pred['score']:.4f}")
            else:
                print("\nNo significant findings (all scores below threshold)")
            
            if args.visualize:
                save_path = None
                if args.save_dir:
                    os.makedirs(args.save_dir, exist_ok=True)
                    save_name = os.path.splitext(os.path.basename(img_path))[0] + '_prediction.png'
                    save_path = os.path.join(args.save_dir, save_name)
                
                fig = visualize_prediction(
                    original_image,
                    predicted_classes,
                    scores,
                    ChestXrayDataset.DISEASE_LABELS,
                    save_path=save_path
                )
                
                if not args.save_dir:
                    plt.show()
                
                plt.close(fig)
        
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    print(f"\n{'='*80}")
    print("Prediction complete!")
    print(f"{'='*80}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict with HybridMedNet')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to image or directory of images')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold (default: 0.5)')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize predictions')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    main(args)
