"""
增强样本分析核心模块
精简版，用于集成到实验流程中自动运行

功能：
1. 误判类别相似度分析
2. t-SNE可视化 + LOF离群点检测
"""

import os
import numpy as np
from typing import Dict, List, Optional
from sklearn.manifold import TSNE
from sklearn.neighbors import LocalOutlierFactor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def analyze_misclassification_similarity(
    omics_data: List[np.ndarray],
    y_full: np.ndarray,
    train_mask: np.ndarray,
    error_samples: List[Dict]
) -> Dict:
    """
    分析错误样本与误判类别的相似度
    
    Args:
        omics_data: 各组学特征矩阵列表 [omics1, omics2, ...]
        y_full: 完整标签数组
        train_mask: 训练集掩码
        error_samples: 错误样本列表，每个元素包含:
            - sample_idx: 样本索引
            - true_label: 真实标签  
            - predicted_as: 预测标签（常预测为）
            - error_rate: 错误率
    
    Returns:
        各组学的相似度分析结果
    """
    print("  → 误判类别相似度分析...")
    
    # 只分析有预测标签的样本
    samples_with_pred = [s for s in error_samples 
                         if 'predicted_as' in s and s['predicted_as'] is not None]
    
    if not samples_with_pred:
        print("    ⚠️  没有误判类别信息，跳过相似度分析")
        return {'error': '没有误判类别信息'}
    
    results = {}
    num_omics = len(omics_data)
    
    for omics_idx, omics in enumerate(omics_data):
        omics_name = f'omics_{omics_idx+1}'
        
        # 计算各类别的中心（基于训练集）
        class_centers = {}
        for label in np.unique(y_full[train_mask]):
            class_mask = train_mask & (y_full == label)
            if class_mask.sum() > 0:
                class_centers[label] = omics[class_mask].mean(axis=0)
        
        similarity_analysis = []
        
        for sample_info in samples_with_pred:
            sample_idx = sample_info['sample_idx']
            true_label = sample_info['true_label']
            pred_label = sample_info['predicted_as']
            
            sample_vec = omics[sample_idx]
            
            # 计算到真实类别中心的距离
            dist_to_true = np.inf
            if true_label in class_centers:
                dist_to_true = np.linalg.norm(sample_vec - class_centers[true_label])
            
            # 计算到误判类别中心的距离
            dist_to_pred = np.inf
            if pred_label in class_centers:
                dist_to_pred = np.linalg.norm(sample_vec - class_centers[pred_label])
            
            # 计算到所有类别中心的距离
            distances_to_all = {}
            for label, center in class_centers.items():
                distances_to_all[int(label)] = float(np.linalg.norm(sample_vec - center))
            
            # 找到最近的类别
            closest_label = min(distances_to_all.items(), key=lambda x: x[1])[0]
            
            similarity_analysis.append({
                'sample_idx': sample_idx,
                'true_label': int(true_label),
                'predicted_as': int(pred_label),
                'error_rate': sample_info['error_rate'],
                'dist_to_true_class': float(dist_to_true),
                'dist_to_pred_class': float(dist_to_pred),
                'closer_to_pred': bool(dist_to_pred < dist_to_true),
                'distance_ratio': float(dist_to_pred / dist_to_true) if dist_to_true > 0 else 0,
                'closest_class': closest_label,
                'distances_to_all_classes': distances_to_all
            })
        
        # 统计
        closer_to_pred_count = sum(1 for s in similarity_analysis if s['closer_to_pred'])
        perfect_errors = [s for s in similarity_analysis if s['error_rate'] >= 0.99]
        perfect_closer_to_pred = sum(1 for s in perfect_errors if s['closer_to_pred'])
        
        results[omics_name] = {
            'total_samples_analyzed': len(similarity_analysis),
            'closer_to_predicted_class': closer_to_pred_count,
            'closer_to_predicted_ratio': f"{closer_to_pred_count/len(similarity_analysis)*100:.1f}%",
            'perfect_errors_analyzed': len(perfect_errors),
            'perfect_closer_to_pred': perfect_closer_to_pred,
            'perfect_closer_to_pred_ratio': f"{perfect_closer_to_pred/len(perfect_errors)*100:.1f}%" if perfect_errors else "N/A",
            'detailed_analysis': similarity_analysis
        }
        
        print(f"    {omics_name}: {closer_to_pred_count}/{len(similarity_analysis)} "
              f"样本更接近误判类别 ({results[omics_name]['closer_to_predicted_ratio']})")
    
    return results


def visualize_and_detect_outliers(
    omics_data: List[np.ndarray],
    y_full: np.ndarray,
    train_mask: np.ndarray,
    error_samples: List[Dict],
    output_dir: str
) -> Dict:
    """
    可视化样本分布并检测离群点
    
    Args:
        omics_data: 各组学特征矩阵列表
        y_full: 完整标签数组
        train_mask: 训练集掩码
        error_samples: 错误样本列表
        output_dir: 输出目录
    
    Returns:
        各组学的可视化和离群点检测结果
    """
    print("  → t-SNE可视化 + 离群点检测...")
    
    results = {}
    test_mask = ~train_mask
    test_indices = np.where(test_mask)[0]
    
    for omics_idx, omics in enumerate(omics_data):
        omics_name = f'omics_{omics_idx+1}'
        
        # 只对测试集进行可视化
        test_data = omics[test_mask]
        test_labels = y_full[test_mask]
        
        # t-SNE降维
        print(f"    {omics_name}: t-SNE降维中...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(test_data)-1))
        test_2d = tsne.fit_transform(test_data)
        
        # LOF离群点检测
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
        outlier_labels = lof.fit_predict(test_data)
        outlier_scores = -lof.negative_outlier_factor_
        
        outlier_indices = test_indices[outlier_labels == -1]
        outlier_info = []
        
        error_sample_indices = set(s['sample_idx'] for s in error_samples) if error_samples else set()
        
        for idx in outlier_indices:
            local_idx = np.where(test_indices == idx)[0][0]
            outlier_info.append({
                'sample_idx': int(idx),
                'true_label': int(y_full[idx]),
                'outlier_score': float(outlier_scores[local_idx]),
                'is_error_sample': idx in error_sample_indices
            })
        
        # 绘图
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 获取所有类别
        unique_labels = np.unique(test_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        # 绘制所有测试样本
        for label_idx, label in enumerate(unique_labels):
            mask = test_labels == label
            ax.scatter(test_2d[mask, 0], test_2d[mask, 1], 
                      c=[colors[label_idx]], 
                      label=f'Class {int(label)}',
                      alpha=0.6, s=50, edgecolors='none')
        
        # 标记离群点
        if len(outlier_indices) > 0:
            outlier_mask = outlier_labels == -1
            ax.scatter(test_2d[outlier_mask, 0], test_2d[outlier_mask, 1],
                      facecolors='none', edgecolors='red', 
                      s=200, linewidths=2, label='Outliers (LOF)', alpha=0.8)
        
        # 标记错误样本
        if error_samples:
            error_indices = [s['sample_idx'] for s in error_samples]
            error_mask = np.isin(test_indices, error_indices)
            
            if error_mask.any():
                ax.scatter(test_2d[error_mask, 0], test_2d[error_mask, 1],
                          marker='x', c='black', s=100, 
                          linewidths=2, label='Error Samples', alpha=0.9)
            
            # 特别标记100%错误样本
            perfect_indices = [s['sample_idx'] for s in error_samples if s['error_rate'] >= 0.99]
            perfect_mask = np.isin(test_indices, perfect_indices)
            
            if perfect_mask.any():
                ax.scatter(test_2d[perfect_mask, 0], test_2d[perfect_mask, 1],
                          marker='*', c='red', s=300, 
                          linewidths=2, label='100% Error Samples', 
                          edgecolors='darkred', alpha=1.0)
        
        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.set_title(f'{omics_name} - Test Sample Distribution (t-SNE)', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 保存图片
        output_path = os.path.join(output_dir, f'{omics_name}_tsne_visualization.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 统计离群点与错误样本的重叠
        outlier_and_error = set(outlier_indices) & error_sample_indices
        overlap_ratio = len(outlier_and_error) / len(outlier_indices) if len(outlier_indices) > 0 else 0
        
        results[omics_name] = {
            'total_test_samples': len(test_data),
            'outliers_detected': len(outlier_indices),
            'outlier_ratio': f"{len(outlier_indices)/len(test_data)*100:.1f}%",
            'outlier_details': outlier_info,
            'outliers_also_errors': len(outlier_and_error),
            'outlier_error_overlap_ratio': f"{overlap_ratio*100:.1f}%",
            'visualization_path': output_path
        }
        
        print(f"    {omics_name}: 检测到 {len(outlier_indices)} 个离群点 "
              f"({results[omics_name]['outlier_ratio']})")
    
    return results


def save_enhanced_analysis_report(
    similarity_results: Dict,
    visualization_results: Dict,
    output_path: str
) -> None:
    """
    保存增强分析报告
    
    Args:
        similarity_results: 相似度分析结果
        visualization_results: 可视化和离群点检测结果
        output_path: 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("增强样本分析报告\n")
        f.write("=" * 80 + "\n\n")
        
        # 1. 误判类别相似度分析
        f.write("=" * 80 + "\n")
        f.write("1. 误判类别相似度分析\n")
        f.write("=" * 80 + "\n")
        
        if 'error' in similarity_results:
            f.write(f"\n{similarity_results['error']}\n")
        else:
            for omics_name, result in similarity_results.items():
                f.write(f"\n{omics_name}:\n")
                f.write(f"  分析样本数: {result['total_samples_analyzed']}\n")
                f.write(f"  更接近误判类别的样本: {result['closer_to_predicted_class']} "
                       f"({result['closer_to_predicted_ratio']})\n")
                f.write(f"  100%错误样本中更接近误判类别: {result['perfect_closer_to_pred']} "
                       f"({result['perfect_closer_to_pred_ratio']})\n")
                
                # 详细分析表格
                f.write(f"\n  详细分析:\n")
                f.write(f"  {'样本ID':<10} {'真实':<6} {'预测':<6} {'错误率':<10} "
                       f"{'到真实类':<12} {'到预测类':<12} {'距离比':<10} {'结论'}\n")
                f.write(f"  {'-'*90}\n")
                
                for s in result['detailed_analysis']:
                    conclusion = "更像预测类" if s['closer_to_pred'] else "更像真实类"
                    f.write(f"  {s['sample_idx']:<10} {s['true_label']:<6} {s['predicted_as']:<6} "
                           f"{s['error_rate']*100:>6.1f}%   {s['dist_to_true_class']:>10.3f}  "
                           f"{s['dist_to_pred_class']:>10.3f}  {s['distance_ratio']:>8.3f}  {conclusion}\n")
        
        # 2. 可视化和离群点检测
        f.write(f"\n{'=' * 80}\n")
        f.write("2. 可视化和离群点检测结果\n")
        f.write(f"{'=' * 80}\n")
        
        for omics_name, result in visualization_results.items():
            f.write(f"\n{omics_name}:\n")
            f.write(f"  测试样本数: {result['total_test_samples']}\n")
            f.write(f"  检测到离群点: {result['outliers_detected']} ({result['outlier_ratio']})\n")
            f.write(f"  离群点中也是错误样本: {result['outliers_also_errors']} "
                   f"({result['outlier_error_overlap_ratio']})\n")
            f.write(f"  可视化图片: {result['visualization_path']}\n")
            
            if result['outlier_details']:
                f.write(f"\n  离群点详情:\n")
                f.write(f"  {'样本ID':<10} {'真实类别':<10} {'离群分数':<12} {'是否错误样本'}\n")
                f.write(f"  {'-'*50}\n")
                
                sorted_outliers = sorted(result['outlier_details'], 
                                        key=lambda x: x['outlier_score'], 
                                        reverse=True)
                for o in sorted_outliers:
                    is_error = "是" if o['is_error_sample'] else "否"
                    f.write(f"  {o['sample_idx']:<10} {o['true_label']:<10} "
                           f"{o['outlier_score']:>10.3f}  {is_error}\n")
        
        f.write("\n" + "=" * 80 + "\n")


def print_enhanced_analysis_summary(
    similarity_results: Dict,
    visualization_results: Dict
) -> None:
    """
    打印增强分析摘要
    
    Args:
        similarity_results: 相似度分析结果
        visualization_results: 可视化和离群点检测结果
    """
    print(f"\n{'=' * 60}")
    print("增强分析摘要")
    print(f"{'=' * 60}")
    
    # 相似度分析摘要
    if 'error' not in similarity_results:
        print("\n【误判类别相似度】")
        for omics_name, result in similarity_results.items():
            ratio = float(result['closer_to_predicted_ratio'].rstrip('%'))
            print(f"  {omics_name}: {result['closer_to_predicted_class']}/"
                  f"{result['total_samples_analyzed']} 样本更接近误判类别 "
                  f"({result['closer_to_predicted_ratio']})")
            
            if ratio > 70:
                print(f"    ⚠️  大部分错误样本确实更像误判类别")
                print(f"        → 可能是边界样本或标注错误")
            elif ratio < 30:
                print(f"    ℹ️  大部分错误样本仍更接近真实类别")
                print(f"        → 可能是模型问题而非数据问题")
    
    # 离群点检测摘要
    print("\n【离群点检测】")
    for omics_name, result in visualization_results.items():
        overlap = float(result['outlier_error_overlap_ratio'].rstrip('%'))
        print(f"  {omics_name}: 检测到 {result['outliers_detected']} 个离群点")
        print(f"    其中 {result['outliers_also_errors']} 个也是错误样本 "
              f"({result['outlier_error_overlap_ratio']})")
        
        if overlap > 50:
            print(f"    ⚠️  离群样本更容易被误判")
        
        print(f"    可视化图片: {result['visualization_path']}")
    
    print(f"\n{'=' * 60}")


def run_enhanced_analysis(
    omics_data: List[np.ndarray],
    y_full: np.ndarray,
    train_mask: np.ndarray,
    error_samples: List[Dict],
    output_dir: str
) -> None:
    """
    运行完整的增强分析（主入口函数）
    
    Args:
        omics_data: 各组学特征矩阵列表
        y_full: 完整标签数组
        train_mask: 训练集掩码
        error_samples: 错误样本列表
        output_dir: 输出目录
    """
    if not error_samples:
        print("  ⚠️  没有错误样本，跳过增强分析")
        return
    
    print(f"\n{'=' * 60}")
    print("开始增强样本分析")
    print(f"{'=' * 60}")
    
    # 1. 误判类别相似度分析
    similarity_results = analyze_misclassification_similarity(
        omics_data, y_full, train_mask, error_samples
    )
    
    # 2. 可视化和离群点检测
    visualization_results = visualize_and_detect_outliers(
        omics_data, y_full, train_mask, error_samples, output_dir
    )
    
    # 3. 保存报告
    report_path = os.path.join(output_dir, 'enhanced_analysis_report.txt')
    save_enhanced_analysis_report(similarity_results, visualization_results, report_path)
    print(f"\n  ✓ 增强分析报告已保存: {report_path}")
    
    # 4. 打印摘要
    print_enhanced_analysis_summary(similarity_results, visualization_results)
    
    print(f"\n{'=' * 60}")
    print("增强分析完成")
    print(f"{'=' * 60}")
