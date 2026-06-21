"""
评估指标工具模块
包含指标计算和结果统计功能
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    confusion_matrix, precision_score, recall_score
)
from sklearn.preprocessing import label_binarize
from typing import List, Dict, Optional


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
                     num_classes: int) -> Dict:
    """计算多种评估指标"""
    metrics = {}

    # 准确率
    metrics['acc'] = accuracy_score(y_true, y_pred)

    # 精确率和召回率（宏平均）
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)

    # F1分数
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # AUC（处理多分类）
    if num_classes > 2:
        y_true_bin = label_binarize(y_true, classes=range(num_classes))
        try:
            metrics['auc'] = roc_auc_score(y_true_bin, y_prob, multi_class='ovr', average='macro')
        except:
            metrics['auc'] = float('nan')
    else:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
        except:
            metrics['auc'] = float('nan')

    # 混淆矩阵
    metrics['conf_matrix'] = confusion_matrix(y_true, y_pred).tolist()

    return metrics


def calculate_average_results(all_results: List[Dict], top_k: Optional[int] = None) -> Optional[Dict]:
    """计算多次实验的平均结果（可选按测试集ACC取top-k）"""
    if not all_results or len(all_results) == 0:
        return None

    valid_results = [r for r in all_results if r and 'test_metrics' in r]
    total_runs = len(valid_results)
    if total_runs == 0:
        return None

    if top_k is not None and top_k > 0:
        num_used = min(int(top_k), total_runs)
        sorted_results = sorted(
            valid_results,
            key=lambda r: r['test_metrics'].get('acc', float('-inf')),
            reverse=True
        )
        used_results = sorted_results[:num_used]
    else:
        used_results = valid_results
        num_used = len(used_results)

    metrics_keys = ['acc', 'precision_macro', 'recall_macro', 'f1_macro', 'f1_weighted', 'auc']

    avg_metrics = {}
    std_metrics = {}

    for key in metrics_keys:
        values = []
        for result in used_results:
            if result and 'test_metrics' in result:
                metric_value = result['test_metrics'].get(key, float('nan'))
                if not np.isnan(metric_value):
                    values.append(metric_value)

        if values:
            avg_metrics[key] = np.mean(values)
            std_metrics[key] = np.std(values)
        else:
            avg_metrics[key] = float('nan')
            std_metrics[key] = float('nan')

    best_val_accs = [result['best_val_acc'] for result in used_results if result]

    return {
        'num_runs': total_runs,
        'num_runs_used': num_used,
        'selection_rule': 'top_test_acc' if top_k is not None and top_k > 0 else 'all_runs',
        'average_metrics': avg_metrics,
        'std_metrics': std_metrics,
        'best_val_acc_avg': np.mean(best_val_accs) if best_val_accs else float('nan'),
        'best_val_acc_std': np.std(best_val_accs) if best_val_accs else float('nan'),
        'individual_results': used_results
    }


def print_average_results(avg_results: Dict) -> None:
    """打印平均结果"""
    if not avg_results:
        print("无法计算平均结果")
        return

    num_runs = avg_results.get('num_runs', 0)
    num_used = avg_results.get('num_runs_used', num_runs)
    selection_rule = avg_results.get('selection_rule', 'all_runs')

    print(f"\n{'=' * 80}")
    if selection_rule == 'top_test_acc':
        print(f"实验结果汇总（共{num_runs}次，按Test ACC取前{num_used}次）")
    else:
        print(f"实验结果汇总（共{num_runs}次，使用全部运行）")
    print(f"{'=' * 80}")

    print(f"测试集平均性能 (Mean ± Std):")
    metrics_names = {
        'acc': 'Accuracy',
        'precision_macro': 'Precision (macro)',
        'recall_macro': 'Recall (macro)',
        'f1_macro': 'F1-Score (macro)',
        'f1_weighted': 'F1-Score (weighted)',
        'auc': 'AUC'
    }

    for key, name in metrics_names.items():
        mean_val = avg_results['average_metrics'].get(key, float('nan'))
        std_val = avg_results['std_metrics'].get(key, float('nan'))
        if not np.isnan(mean_val):
            print(f"  {name:20}: {mean_val:.4f} ± {std_val:.4f}")
        else:
            print(f"  {name:20}: N/A")

    print(f"\n验证集最佳准确率:")
    best_val_mean = avg_results.get('best_val_acc_avg', float('nan'))
    best_val_std = avg_results.get('best_val_acc_std', float('nan'))
    if not np.isnan(best_val_mean):
        print(f"  Best Val Accuracy    : {best_val_mean:.4f} ± {best_val_std:.4f}")

    print(f"\n每次实验详细结果:")
    for i, result in enumerate(avg_results['individual_results'], 1):
        if result:
            test_acc = result['test_metrics']['acc']
            val_acc = result['best_val_acc']
            seed = result.get('seed', 'N/A')
            print(f"  Run {i} (seed={seed:2}): Test Acc = {test_acc:.4f}, Best Val Acc = {val_acc:.4f}")

    print(f"{'=' * 80}")