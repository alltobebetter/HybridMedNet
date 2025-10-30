import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


def calculate_metrics(y_true, y_pred, y_scores=None, average='macro'):
    """
    计算分类指标
    
    Args:
        y_true: 真实标签 [N,] or [N, C]
        y_pred: 预测标签 [N,] or [N, C]
        y_scores: 预测概率 [N, C] (可选，用于计算AUC)
        average: 多类别平均方式 ('macro', 'micro', 'weighted')
    
    Returns:
        dict: 包含各种指标的字典
    """
    # 转换为numpy
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    if y_scores is not None and torch.is_tensor(y_scores):
        y_scores = y_scores.cpu().numpy()
    
    # 如果是多标签（二维），需要特殊处理
    is_multilabel = len(y_true.shape) > 1 and y_true.shape[1] > 1
    
    metrics = {}
    
    if is_multilabel:
        # 多标签分类
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # 计算每个类别的AUC
        if y_scores is not None:
            try:
                auc_scores = []
                for i in range(y_true.shape[1]):
                    if len(np.unique(y_true[:, i])) > 1:
                        auc = roc_auc_score(y_true[:, i], y_scores[:, i])
                        auc_scores.append(auc)
                metrics['auc'] = np.mean(auc_scores) if auc_scores else 0.0
                metrics['auc_per_class'] = auc_scores
            except Exception as e:
                print(f"Warning: Could not compute AUC - {e}")
                metrics['auc'] = 0.0
    else:
        # 单标签多类别分类
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # 混淆矩阵
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # AUC
        if y_scores is not None:
            try:
                num_classes = y_scores.shape[1] if len(y_scores.shape) > 1 else 2
                if num_classes == 2:
                    metrics['auc'] = roc_auc_score(y_true, y_scores[:, 1])
                else:
                    metrics['auc'] = roc_auc_score(
                        y_true, y_scores, 
                        multi_class='ovr',
                        average=average
                    )
            except Exception as e:
                print(f"Warning: Could not compute AUC - {e}")
                metrics['auc'] = 0.0
    
    return metrics


def compute_auc_scores(y_true, y_scores, class_names=None):
    """
    计算每个类别的AUC分数（用于多标签分类）
    
    Args:
        y_true: 真实标签 [N, C]
        y_scores: 预测概率 [N, C]
        class_names: 类别名称列表
    
    Returns:
        dict: 每个类别的AUC分数
    """
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_scores):
        y_scores = y_scores.cpu().numpy()
    
    num_classes = y_true.shape[1]
    auc_scores = {}
    
    for i in range(num_classes):
        class_name = class_names[i] if class_names else f'Class_{i}'
        
        # 检查是否该类别在数据中存在
        if len(np.unique(y_true[:, i])) > 1:
            try:
                auc = roc_auc_score(y_true[:, i], y_scores[:, i])
                auc_scores[class_name] = auc
            except Exception as e:
                print(f"Warning: Could not compute AUC for {class_name} - {e}")
                auc_scores[class_name] = 0.0
        else:
            auc_scores[class_name] = 0.0
    
    return auc_scores


def print_metrics(metrics, prefix=''):
    """
    打印指标
    
    Args:
        metrics: 指标字典
        prefix: 前缀字符串
    """
    print(f"\n{prefix}Metrics:")
    print(f"  Accuracy:  {metrics.get('accuracy', 0.0):.4f}")
    print(f"  Precision: {metrics.get('precision', 0.0):.4f}")
    print(f"  Recall:    {metrics.get('recall', 0.0):.4f}")
    print(f"  F1 Score:  {metrics.get('f1_score', 0.0):.4f}")
    if 'auc' in metrics:
        print(f"  AUC:       {metrics['auc']:.4f}")


def compute_class_weights(labels, num_classes):
    """
    计算类别权重（用于不平衡数据集）
    
    Args:
        labels: 标签数组
        num_classes: 类别数量
    
    Returns:
        torch.Tensor: 类别权重
    """
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    
    # 统计每个类别的样本数
    if len(labels.shape) > 1:
        # 多标签
        counts = labels.sum(axis=0)
    else:
        # 单标签
        counts = np.bincount(labels, minlength=num_classes)
    
    # 避免除零
    counts = np.maximum(counts, 1)
    
    # 计算权重：反比于类别频率
    total = len(labels)
    weights = total / (num_classes * counts)
    
    return torch.FloatTensor(weights)


def top_k_accuracy(output, target, k=5):
    """
    计算Top-K准确率
    
    Args:
        output: 模型输出 [B, C]
        target: 真实标签 [B]
        k: Top-K
    
    Returns:
        float: Top-K准确率
    """
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        return correct_k.mul_(100.0 / batch_size).item()


class AverageMeter:
    """
    计算并存储平均值和当前值
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    # 测试代码
    print("Testing metrics...")
    
    # 单标签分类
    y_true = np.array([0, 1, 2, 1, 0, 2, 1, 0])
    y_pred = np.array([0, 1, 2, 1, 0, 1, 1, 0])
    y_scores = np.random.rand(8, 3)
    
    metrics = calculate_metrics(y_true, y_pred, y_scores, average='macro')
    print_metrics(metrics, prefix='Single-label ')
    
    # 多标签分类
    y_true_ml = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1]
    ])
    y_pred_ml = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    y_scores_ml = np.random.rand(4, 3)
    
    metrics_ml = calculate_metrics(y_true_ml, y_pred_ml, y_scores_ml)
    print_metrics(metrics_ml, prefix='Multi-label ')
    
    # AUC per class
    auc_scores = compute_auc_scores(y_true_ml, y_scores_ml, ['Class A', 'Class B', 'Class C'])
    print("\nAUC per class:")
    for name, score in auc_scores.items():
        print(f"  {name}: {score:.4f}")
