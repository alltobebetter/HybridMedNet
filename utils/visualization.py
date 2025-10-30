import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
import cv2


def visualize_attention(image, attention_map, alpha=0.5, cmap='jet'):
    """
    可视化注意力图
    
    Args:
        image: 原始图像 [H, W, 3] 或 [3, H, W]
        attention_map: 注意力图 [H', W']
        alpha: 叠加透明度
        cmap: 颜色映射
    
    Returns:
        叠加后的图像
    """
    # 转换为numpy
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    if torch.is_tensor(attention_map):
        attention_map = attention_map.cpu().numpy()
    
    # 调整图像维度 [3, H, W] -> [H, W, 3]
    if image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    
    # 归一化图像到[0, 1]
    if image.max() > 1.0:
        image = image / 255.0
    
    # 调整注意力图大小以匹配图像
    if attention_map.shape != image.shape[:2]:
        attention_map = cv2.resize(
            attention_map,
            (image.shape[1], image.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
    
    # 归一化注意力图
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    
    # 应用颜色映射
    cmap_fn = plt.get_cmap(cmap)
    attention_colored = cmap_fn(attention_map)[:, :, :3]
    
    # 叠加
    overlayed = alpha * attention_colored + (1 - alpha) * image
    overlayed = np.clip(overlayed, 0, 1)
    
    return overlayed


def plot_roc_curves(y_true, y_scores, class_names=None, save_path=None):
    """
    绘制ROC曲线（多类别）
    
    Args:
        y_true: 真实标签 [N, C] (one-hot) 或 [N]
        y_scores: 预测概率 [N, C]
        class_names: 类别名称
        save_path: 保存路径
    """
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_scores):
        y_scores = y_scores.cpu().numpy()
    
    # 处理单标签转one-hot
    if len(y_true.shape) == 1:
        num_classes = y_scores.shape[1]
        y_true_onehot = np.zeros((len(y_true), num_classes))
        y_true_onehot[np.arange(len(y_true)), y_true] = 1
        y_true = y_true_onehot
    
    num_classes = y_true.shape[1]
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]
    
    plt.figure(figsize=(10, 8))
    
    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, num_classes))
    
    for i in range(num_classes):
        # 检查是否该类别存在
        if len(np.unique(y_true[:, i])) > 1:
            fpr, tpr, _ = roc_curve(y_true[:, i], y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(
                fpr, tpr,
                color=colors[i],
                lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc:.3f})'
            )
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Multi-class Classification', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")
    
    plt.tight_layout()
    return plt


def plot_training_curves(history, metrics=['loss', 'accuracy'], save_path=None):
    """
    绘制训练曲线
    
    Args:
        history: 训练历史字典，包含 train_* 和 val_* 的指标
        metrics: 要绘制的指标列表
        save_path: 保存路径
    """
    num_metrics = len(metrics)
    fig, axes = plt.subplots(1, num_metrics, figsize=(6*num_metrics, 5))
    
    if num_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        train_key = f'train_{metric}'
        val_key = f'val_{metric}'
        
        if train_key in history:
            epochs = range(1, len(history[train_key]) + 1)
            ax.plot(epochs, history[train_key], 'b-', label=f'Train {metric}', linewidth=2)
        
        if val_key in history:
            epochs = range(1, len(history[val_key]) + 1)
            ax.plot(epochs, history[val_key], 'r-', label=f'Val {metric}', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(metric.capitalize(), fontsize=12)
        ax.set_title(f'Training {metric.capitalize()}', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    return fig


def plot_confusion_matrix(cm, class_names, normalize=False, save_path=None):
    """
    绘制混淆矩阵
    
    Args:
        cm: 混淆矩阵
        class_names: 类别名称
        normalize: 是否归一化
        save_path: 保存路径
    """
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
    )
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''), fontsize=14)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.tight_layout()
    return plt


def visualize_batch_predictions(images, labels, predictions, class_names=None, n_samples=8):
    """
    可视化一批预测结果
    
    Args:
        images: 图像张量 [B, 3, H, W]
        labels: 真实标签 [B]
        predictions: 预测标签 [B]
        class_names: 类别名称
        n_samples: 显示样本数量
    """
    if torch.is_tensor(images):
        images = images.cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    
    n_samples = min(n_samples, len(images))
    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = axes.flatten() if n_samples > 1 else [axes]
    
    for i in range(n_samples):
        ax = axes[i]
        
        # 转换图像
        img = images[i].transpose(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        ax.imshow(img)
        
        # 获取标签名称
        true_label = class_names[labels[i]] if class_names else f'{labels[i]}'
        pred_label = class_names[predictions[i]] if class_names else f'{predictions[i]}'
        
        # 设置标题颜色
        color = 'green' if labels[i] == predictions[i] else 'red'
        ax.set_title(f'True: {true_label}\nPred: {pred_label}', color=color, fontsize=10)
        ax.axis('off')
    
    # 隐藏多余的子图
    for i in range(n_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


def plot_class_distribution(labels, class_names=None, save_path=None):
    """
    绘制类别分布
    
    Args:
        labels: 标签数组
        class_names: 类别名称
        save_path: 保存路径
    """
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    # 统计每个类别的数量
    unique, counts = np.unique(labels, return_counts=True)
    
    if class_names is None:
        class_names = [f'Class {i}' for i in unique]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(unique)), counts, color='skyblue', edgecolor='navy')
    
    # 添加数值标签
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}',
                ha='center', va='bottom', fontsize=10)
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Class Distribution', fontsize=14)
    plt.xticks(range(len(unique)), class_names, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class distribution saved to {save_path}")
    
    plt.tight_layout()
    return plt


if __name__ == '__main__':
    print("Testing visualization utilities...")
    
    # 测试训练曲线
    history = {
        'train_loss': [0.8, 0.6, 0.4, 0.3, 0.2],
        'val_loss': [0.9, 0.7, 0.5, 0.4, 0.35],
        'train_accuracy': [0.6, 0.7, 0.8, 0.85, 0.9],
        'val_accuracy': [0.55, 0.65, 0.75, 0.8, 0.83]
    }
    
    fig = plot_training_curves(history, metrics=['loss', 'accuracy'])
    plt.show()
    
    print("Visualization tests completed!")
