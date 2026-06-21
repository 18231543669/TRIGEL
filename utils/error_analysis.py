"""
错误样本分析模块
用于分析模型在多次实验中的错误分类模式
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict, Counter


class MisclassificationAnalyzer:
    """错误分类分析器
    
    功能：
    1. 跟踪每个样本在多次实验中的分类情况
    2. 识别频繁被错误分类的样本
    3. 分析错误模式（少数类 vs 多数类）
    4. 提供改进建议
    
    Attributes:
        predictions_history: 每次实验的预测结果历史
        labels_history: 每次实验的真实标签历史
        indices_history: 每次实验的样本索引历史
        num_classes: 类别数量
    """
    
    def __init__(self):
        """初始化分析器"""
        self.predictions_history = []  # [(run_id, predictions, probabilities)]
        self.labels_history = []       # [(run_id, labels)]
        self.indices_history = []      # [(run_id, indices)]
        self.num_classes = None
        
    def record_prediction(self, run_id: int, indices: np.ndarray, 
                         labels: np.ndarray, predictions: np.ndarray,
                         probabilities: np.ndarray) -> None:
        """记录一次实验的预测结果
        
        Args:
            run_id: 实验运行ID
            indices: 样本索引
            labels: 真实标签
            predictions: 预测标签
            probabilities: 预测概率
        """
        self.predictions_history.append({
            'run_id': run_id,
            'indices': indices.copy(),
            'labels': labels.copy(),
            'predictions': predictions.copy(),
            'probabilities': probabilities.copy()
        })
        
        if self.num_classes is None:
            self.num_classes = len(np.unique(labels))
    
    def analyze(self, y_full: np.ndarray, train_mask: np.ndarray) -> Dict:
        """分析错误分类模式
        
        Args:
            y_full: 完整标签数组
            train_mask: 训练集掩码
            
        Returns:
            分析结果字典
        """
        if len(self.predictions_history) == 0:
            return {'error': '没有预测历史记录'}
        
        # 1. 统计每个样本的错误次数
        sample_errors = self._count_sample_errors()
        
        # 2. 分析类别分布
        class_analysis = self._analyze_class_distribution(y_full, train_mask)
        
        # 3. 识别频繁错误样本
        frequent_errors = self._identify_frequent_errors(sample_errors, y_full)
        
        # 4. 分析错误模式
        error_patterns = self._analyze_error_patterns(sample_errors, y_full, 
                                                       class_analysis['minority_classes'])
        
        # 5. 提供改进建议
        recommendations = self._generate_recommendations(error_patterns, class_analysis)
        
        return {
            'total_runs': len(self.predictions_history),
            'total_test_samples': len(sample_errors),
            'class_analysis': class_analysis,
            'sample_errors': sample_errors,
            'frequent_errors': frequent_errors,
            'error_patterns': error_patterns,
            'recommendations': recommendations
        }
    
    def _count_sample_errors(self) -> Dict[int, Dict]:
        """统计每个样本的错误次数"""
        sample_errors = defaultdict(lambda: {
            'error_count': 0,
            'total_runs': 0,
            'error_rate': 0.0,
            'true_label': None,
            'predicted_labels': [],
            'prediction_confidence': []
        })
        
        for record in self.predictions_history:
            indices = record['indices']
            labels = record['labels']
            predictions = record['predictions']
            probabilities = record['probabilities']
            
            for i, idx in enumerate(indices):
                sample_errors[idx]['total_runs'] += 1
                sample_errors[idx]['true_label'] = labels[i]
                
                if predictions[i] != labels[i]:
                    sample_errors[idx]['error_count'] += 1
                    sample_errors[idx]['predicted_labels'].append(predictions[i])
                
                # 记录预测置信度（预测类别的概率）
                pred_prob = probabilities[i][predictions[i]]
                sample_errors[idx]['prediction_confidence'].append(pred_prob)
        
        # 计算错误率
        for idx in sample_errors:
            error_count = sample_errors[idx]['error_count']
            total_runs = sample_errors[idx]['total_runs']
            sample_errors[idx]['error_rate'] = error_count / total_runs if total_runs > 0 else 0.0
        
        return dict(sample_errors)
    
    def _analyze_class_distribution(self, y_full: np.ndarray, 
                                    train_mask: np.ndarray) -> Dict:
        """分析类别分布"""
        # 基于训练集识别少数类
        train_labels = y_full[train_mask]
        label_counts = Counter(train_labels)
        total_train = len(train_labels)
        
        # 计算阈值
        ratios = [count / total_train for count in label_counts.values()]
        ratios.sort()
        threshold = ratios[0] * 1.5 if len(ratios) >= 2 else 0.3
        
        minority_classes = []
        for label, count in sorted(label_counts.items()):
            ratio = count / total_train
            if ratio < threshold and count >= 5:
                minority_classes.append(int(label))
        
        if not minority_classes and len(label_counts) > 1:
            min_label = min(label_counts.items(), key=lambda x: x[1])[0]
            minority_classes = [int(min_label)]
        
        majority_classes = [int(label) for label in label_counts.keys() 
                           if label not in minority_classes]
        
        return {
            'train_label_counts': dict(label_counts),
            'minority_classes': minority_classes,
            'majority_classes': majority_classes,
            'total_train_samples': total_train
        }
    
    def _identify_frequent_errors(self, sample_errors: Dict[int, Dict],
                                  y_full: np.ndarray, 
                                  threshold: float = 0.5) -> List[Dict]:
        """识别频繁错误的样本
        
        Args:
            sample_errors: 样本错误统计
            y_full: 完整标签
            threshold: 错误率阈值（默认50%）
        """
        frequent_errors = []
        
        for idx, error_info in sample_errors.items():
            if error_info['error_rate'] >= threshold:
                # 统计最常被预测为哪个类别
                if error_info['predicted_labels']:
                    pred_counter = Counter(error_info['predicted_labels'])
                    most_common_pred = pred_counter.most_common(1)[0]
                else:
                    most_common_pred = (None, 0)
                
                # 计算平均预测置信度
                avg_confidence = np.mean(error_info['prediction_confidence']) if error_info['prediction_confidence'] else 0.0
                
                frequent_errors.append({
                    'sample_idx': idx,
                    'true_label': int(error_info['true_label']),
                    'error_count': error_info['error_count'],
                    'total_runs': error_info['total_runs'],
                    'error_rate': error_info['error_rate'],
                    'most_common_prediction': int(most_common_pred[0]) if most_common_pred[0] is not None else None,
                    'prediction_frequency': most_common_pred[1],
                    'avg_confidence': avg_confidence
                })
        
        # 按错误率降序排序
        frequent_errors.sort(key=lambda x: x['error_rate'], reverse=True)
        
        return frequent_errors
    
    def _analyze_error_patterns(self, sample_errors: Dict[int, Dict],
                                y_full: np.ndarray,
                                minority_classes: List[int]) -> Dict:
        """分析错误模式"""
        # 按类别统计错误
        errors_by_class = defaultdict(lambda: {
            'total_samples': 0,
            'error_samples': 0,
            'avg_error_rate': 0.0,
            'is_minority': False
        })
        
        for idx, error_info in sample_errors.items():
            true_label = error_info['true_label']
            is_minority = true_label in minority_classes
            
            errors_by_class[true_label]['total_samples'] += 1
            errors_by_class[true_label]['is_minority'] = is_minority
            
            if error_info['error_count'] > 0:
                errors_by_class[true_label]['error_samples'] += 1
            
            errors_by_class[true_label]['avg_error_rate'] += error_info['error_rate']
        
        # 计算平均错误率
        for label in errors_by_class:
            total = errors_by_class[label]['total_samples']
            if total > 0:
                errors_by_class[label]['avg_error_rate'] /= total
        
        # 统计少数类 vs 多数类的整体错误情况
        minority_total_errors = 0
        minority_total_samples = 0
        majority_total_errors = 0
        majority_total_samples = 0
        
        for idx, error_info in sample_errors.items():
            true_label = error_info['true_label']
            is_minority = true_label in minority_classes
            
            if is_minority:
                minority_total_samples += 1
                minority_total_errors += error_info['error_count']
            else:
                majority_total_samples += 1
                majority_total_errors += error_info['error_count']
        
        total_runs = len(self.predictions_history)
        
        return {
            'errors_by_class': dict(errors_by_class),
            'minority_error_rate': (minority_total_errors / (minority_total_samples * total_runs) 
                                   if minority_total_samples > 0 else 0.0),
            'majority_error_rate': (majority_total_errors / (majority_total_samples * total_runs)
                                   if majority_total_samples > 0 else 0.0),
            'minority_total_samples': minority_total_samples,
            'majority_total_samples': majority_total_samples
        }
    
    def _generate_recommendations(self, error_patterns: Dict, 
                                 class_analysis: Dict) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        minority_error_rate = error_patterns['minority_error_rate']
        majority_error_rate = error_patterns['majority_error_rate']
        
        # 1. 少数类问题分析
        if minority_error_rate > 0.3:
            recommendations.append(
                f"⚠️ 少数类错误率较高 ({minority_error_rate:.2%})，建议："
            )
            recommendations.append(
                "   - 检查少数类增强参数：minority_drop_rate 是否设置为0"
            )
            recommendations.append(
                "   - 增加 max_new_edges 参数，为少数类添加更多边"
            )
            recommendations.append(
                "   - 提高 minority_edge_weight，增强少数类边的权重"
            )
        
        # 2. 多数类问题分析
        if majority_error_rate > 0.3:
            recommendations.append(
                f"⚠️ 多数类错误率较高 ({majority_error_rate:.2%})，建议："
            )
            recommendations.append(
                "   - 检查 majority_drop_rate 是否过大，导致多数类特征丢失过多"
            )
            recommendations.append(
                "   - 考虑降低 edge_add_prob 或 max_new_edges，减少图结构过度增强"
            )
        
        # 3. 类别不平衡分析
        errors_by_class = error_patterns['errors_by_class']
        worst_class = max(errors_by_class.items(), 
                         key=lambda x: x[1]['avg_error_rate'])
        
        if worst_class[1]['avg_error_rate'] > 0.4:
            recommendations.append(
                f"⚠️ 类别 {worst_class[0]} 的平均错误率最高 ({worst_class[1]['avg_error_rate']:.2%})，建议："
            )
            if worst_class[1]['is_minority']:
                recommendations.append(
                    "   - 针对此少数类增加专门的增强策略"
                )
            else:
                recommendations.append(
                    "   - 检查此多数类样本是否存在噪声或离群点"
                )
        
        # 4. 模型架构建议
        if minority_error_rate > majority_error_rate * 1.5:
            recommendations.append(
                "💡 少数类错误远高于多数类，建议："
            )
            recommendations.append(
                "   - 考虑关闭图融合模块 (--use_fusion=False)，使用简单拼接"
            )
            recommendations.append(
                "   - 增加 GCN 层数 (--gcn_layers 3)"
            )
            recommendations.append(
                "   - 调整类别权重损失函数"
            )
        
        # 5. 数据质量建议
        if minority_error_rate > 0.5 and majority_error_rate > 0.5:
            recommendations.append(
                "⚠️ 整体错误率过高，建议："
            )
            recommendations.append(
                "   - 检查数据质量，是否存在标签噪声"
            )
            recommendations.append(
                "   - 考虑使用特征选择或降维"
            )
            recommendations.append(
                "   - 尝试不同的图构建方法（调整 cosine_k 参数）"
            )
        
        if not recommendations:
            recommendations.append("✓ 模型性能良好，无明显问题")
        
        return recommendations
    
    def print_analysis(self, analysis_result: Dict) -> None:
        """打印分析结果到终端
        
        Args:
            analysis_result: 分析结果字典
        """
        print("\n" + "=" * 80)
        print("错误样本分析报告")
        print("=" * 80)
        
        # 1. 基本信息
        print(f"\n总实验次数: {analysis_result['total_runs']}")
        print(f"测试样本数: {analysis_result['total_test_samples']}")
        
        # 2. 类别分布
        class_analysis = analysis_result['class_analysis']
        print(f"\n类别分布（基于训练集）:")
        print(f"  少数类: {class_analysis['minority_classes']}")
        print(f"  多数类: {class_analysis['majority_classes']}")
        print(f"  训练集各类样本数: {class_analysis['train_label_counts']}")
        
        # 3. 错误模式
        error_patterns = analysis_result['error_patterns']
        print(f"\n错误率统计:")
        print(f"  少数类平均错误率: {error_patterns['minority_error_rate']:.2%} "
              f"({error_patterns['minority_total_samples']} 个测试样本)")
        print(f"  多数类平均错误率: {error_patterns['majority_error_rate']:.2%} "
              f"({error_patterns['majority_total_samples']} 个测试样本)")
        
        # 4. 各类别详细错误
        print(f"\n各类别错误详情:")
        print(f"{'类别':<8} {'测试样本数':<12} {'错误样本数':<12} {'平均错误率':<12} {'类型':<10}")
        print("-" * 70)
        
        for label, info in sorted(error_patterns['errors_by_class'].items()):
            class_type = "少数类" if info['is_minority'] else "多数类"
            print(f"{label:<8} {info['total_samples']:<12} {info['error_samples']:<12} "
                  f"{info['avg_error_rate']:<12.2%} {class_type:<10}")
        
        # 5. 频繁错误样本
        frequent_errors = analysis_result['frequent_errors']
        if frequent_errors:
            print(f"\n频繁错误样本 (错误率 ≥ 50%, 前10个):")
            print(f"{'样本ID':<10} {'真实类别':<10} {'错误次数':<10} {'错误率':<10} "
                  f"{'常预测为':<10} {'平均置信度':<12}")
            print("-" * 80)
            
            for error in frequent_errors[:10]:
                print(f"{error['sample_idx']:<10} {error['true_label']:<10} "
                      f"{error['error_count']}/{error['total_runs']:<8} "
                      f"{error['error_rate']:<10.2%} "
                      f"{error['most_common_prediction']:<10} "
                      f"{error['avg_confidence']:<12.2%}")
        else:
            print(f"\n✓ 没有频繁错误样本（所有样本错误率 < 50%）")
        
        # 6. 改进建议
        print(f"\n改进建议:")
        for rec in analysis_result['recommendations']:
            print(rec)
        
        print("=" * 80)
    
    def save_report(self, analysis_result: Dict, save_path: str) -> None:
        """保存分析报告到文件
        
        Args:
            analysis_result: 分析结果字典
            save_path: 保存路径
        """
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("错误样本分析报告\n")
            f.write("=" * 80 + "\n\n")
            
            # 1. 基本信息
            f.write(f"总实验次数: {analysis_result['total_runs']}\n")
            f.write(f"测试样本数: {analysis_result['total_test_samples']}\n\n")
            
            # 2. 类别分布
            class_analysis = analysis_result['class_analysis']
            f.write("类别分布（基于训练集）:\n")
            f.write(f"  少数类: {class_analysis['minority_classes']}\n")
            f.write(f"  多数类: {class_analysis['majority_classes']}\n")
            f.write(f"  训练集各类样本数: {class_analysis['train_label_counts']}\n\n")
            
            # 3. 错误模式
            error_patterns = analysis_result['error_patterns']
            f.write("错误率统计:\n")
            f.write(f"  少数类平均错误率: {error_patterns['minority_error_rate']:.2%} "
                   f"({error_patterns['minority_total_samples']} 个测试样本)\n")
            f.write(f"  多数类平均错误率: {error_patterns['majority_error_rate']:.2%} "
                   f"({error_patterns['majority_total_samples']} 个测试样本)\n\n")
            
            # 4. 各类别详细错误
            f.write("各类别错误详情:\n")
            f.write(f"{'类别':<8} {'测试样本数':<12} {'错误样本数':<12} {'平均错误率':<12} {'类型':<10}\n")
            f.write("-" * 70 + "\n")
            
            for label, info in sorted(error_patterns['errors_by_class'].items()):
                class_type = "少数类" if info['is_minority'] else "多数类"
                f.write(f"{label:<8} {info['total_samples']:<12} {info['error_samples']:<12} "
                       f"{info['avg_error_rate']:<12.2%} {class_type:<10}\n")
            
            # 5. 频繁错误样本
            frequent_errors = analysis_result['frequent_errors']
            f.write("\n频繁错误样本 (错误率 ≥ 50%):\n")
            if frequent_errors:
                f.write(f"{'样本ID':<10} {'真实类别':<10} {'错误次数':<10} {'错误率':<10} "
                       f"{'常预测为':<10} {'平均置信度':<12}\n")
                f.write("-" * 80 + "\n")
                
                for error in frequent_errors:
                    f.write(f"{error['sample_idx']:<10} {error['true_label']:<10} "
                           f"{error['error_count']}/{error['total_runs']:<8} "
                           f"{error['error_rate']:<10.2%} "
                           f"{error['most_common_prediction']:<10} "
                           f"{error['avg_confidence']:<12.2%}\n")
            else:
                f.write("✓ 没有频繁错误样本（所有样本错误率 < 50%）\n")
            
            # 6. 改进建议
            f.write("\n改进建议:\n")
            for rec in analysis_result['recommendations']:
                f.write(rec + "\n")
            
            f.write("\n" + "=" * 80 + "\n")
