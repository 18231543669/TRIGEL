"""
样本特征检查模块 - 增强版
新增功能：
1. 误判类别相似度分析
2. t-SNE可视化 + 离群点检测
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.manifold import TSNE
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端


class SampleFeatureChecker:
    """样本特征检查器 - 增强版"""

    def __init__(self, result_dir: str):
        self.result_dir = result_dir
        self.omics_data = []
        self.y_full = None
        self.train_mask = None
        self.num_omics = 0
        self.error_samples = None

    def load_data(self) -> bool:
        """从结果目录加载数据"""
        try:
            data_info_path = os.path.join(self.result_dir, 'original_data_info.npz')
            if not os.path.exists(data_info_path):
                print(f"❌ 未找到原始数据信息: {data_info_path}")
                return False

            data_info = np.load(data_info_path)
            self.y_full = data_info['y_full']
            self.train_mask = data_info['train_mask']
            self.num_omics = int(data_info['num_omics'])

            print(f"✓ 加载数据信息: {len(self.y_full)} 个样本, {self.num_omics} 个组学")

            self.omics_data = []
            for i in range(self.num_omics):
                omics_path = os.path.join(self.result_dir, f'omics_{i+1}_features.npy')
                if not os.path.exists(omics_path):
                    print(f"❌ 未找到组学{i+1}特征矩阵: {omics_path}")
                    return False

                omics_features = np.load(omics_path)
                self.omics_data.append(omics_features)
                print(f"✓ 加载组学{i+1}: {omics_features.shape}")

            return True

        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def parse_error_report(self) -> bool:
        """解析错误分析报告，提取错误样本信息（包括误判类别）"""
        try:
            report_path = os.path.join(self.result_dir, 'misclassification_analysis.txt')
            if not os.path.exists(report_path):
                print(f"❌ 未找到错误分析报告: {report_path}")
                return False

            self.error_samples = []

            with open(report_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            in_error_section = False
            for line in lines:
                if '频繁错误样本' in line and '错误率 ≥ 50%' in line:
                    in_error_section = True
                    continue

                if in_error_section:
                    if line.strip().startswith('-'):
                        continue
                    if '改进建议' in line or '=' in line:
                        break

                    parts = line.strip().split()
                    if len(parts) >= 4 and parts[0].isdigit():
                        sample_idx = int(parts[0])
                        true_label = int(parts[1])
                        error_info = parts[2].split('/')
                        error_count = int(error_info[0])
                        total_runs = int(error_info[1])
                        error_rate = float(parts[3].rstrip('%')) / 100
                        
                        # 解析"常预测为"列（如果存在）
                        predicted_as = int(parts[4]) if len(parts) >= 5 and parts[4].isdigit() else None

                        self.error_samples.append({
                            'sample_idx': sample_idx,
                            'true_label': true_label,
                            'error_count': error_count,
                            'total_runs': total_runs,
                            'error_rate': error_rate,
                            'predicted_as': predicted_as
                        })

            if self.error_samples:
                print(f"✓ 解析到 {len(self.error_samples)} 个错误样本")
                
                # 统计有误判类别信息的样本
                with_pred = sum(1 for s in self.error_samples if s['predicted_as'] is not None)
                print(f"  其中 {with_pred} 个样本有误判类别信息")

                perfect_errors = [s for s in self.error_samples if s['error_rate'] >= 0.99]
                if perfect_errors:
                    print(f"  ⚠️  其中 {len(perfect_errors)} 个样本错误率100% (需重点关注)")

                return True
            else:
                print("✓ 没有找到频繁错误样本")
                return False

        except Exception as e:
            print(f"❌ 解析错误报告失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def analyze_misclassification_similarity(self) -> Dict:
        """【新功能1】分析错误样本与误判类别的相似度"""
        print("新功能1: 分析误判类别相似度...")

        if not self.error_samples:
            return {'error': '没有错误样本'}

        # 只分析有误判类别信息的样本
        samples_with_pred = [s for s in self.error_samples if s['predicted_as'] is not None]
        
        if not samples_with_pred:
            return {'error': '没有误判类别信息'}

        results = {}

        for omics_idx, omics in enumerate(self.omics_data):
            omics_name = f'omics_{omics_idx+1}'
            
            # 计算各类别的中心
            class_centers = {}
            for label in np.unique(self.y_full[self.train_mask]):
                class_mask = self.train_mask & (self.y_full == label)
                class_centers[label] = omics[class_mask].mean(axis=0)

            similarity_analysis = []

            for sample_info in samples_with_pred:
                sample_idx = sample_info['sample_idx']
                true_label = sample_info['true_label']
                pred_label = sample_info['predicted_as']
                
                sample_vec = omics[sample_idx]
                
                # 计算到真实类别中心的距离
                if true_label in class_centers:
                    dist_to_true = np.linalg.norm(sample_vec - class_centers[true_label])
                else:
                    dist_to_true = np.inf
                
                # 计算到误判类别中心的距离
                if pred_label in class_centers:
                    dist_to_pred = np.linalg.norm(sample_vec - class_centers[pred_label])
                else:
                    dist_to_pred = np.inf
                
                # 计算到所有其他类别中心的距离
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

        return results

    def visualize_and_detect_outliers(self, output_dir: Optional[str] = None) -> Dict:
        """【新功能2】可视化样本分布并检测离群点"""
        print("新功能2: t-SNE可视化 + 离群点检测...")

        if output_dir is None:
            output_dir = self.result_dir

        results = {}
        test_mask = ~self.train_mask
        test_indices = np.where(test_mask)[0]

        for omics_idx, omics in enumerate(self.omics_data):
            omics_name = f'omics_{omics_idx+1}'
            print(f"  处理 {omics_name}...")

            # 只对测试集进行可视化
            test_data = omics[test_mask]
            test_labels = self.y_full[test_mask]

            # t-SNE降维
            print(f"    - 执行t-SNE降维...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(test_data)-1))
            test_2d = tsne.fit_transform(test_data)

            # LOF离群点检测
            print(f"    - LOF离群点检测...")
            lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
            outlier_labels = lof.fit_predict(test_data)
            outlier_scores = -lof.negative_outlier_factor_
            
            outlier_indices = test_indices[outlier_labels == -1]
            outlier_info = []
            
            for idx in outlier_indices:
                local_idx = np.where(test_indices == idx)[0][0]
                outlier_info.append({
                    'sample_idx': int(idx),
                    'true_label': int(self.y_full[idx]),
                    'outlier_score': float(outlier_scores[local_idx]),
                    'is_error_sample': any(s['sample_idx'] == idx for s in self.error_samples) if self.error_samples else False
                })

            # 绘图
            print(f"    - 生成可视化图...")
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
            if self.error_samples:
                error_indices = [s['sample_idx'] for s in self.error_samples]
                error_mask = np.isin(test_indices, error_indices)
                
                if error_mask.any():
                    ax.scatter(test_2d[error_mask, 0], test_2d[error_mask, 1],
                              marker='x', c='black', s=100, 
                              linewidths=2, label='Error Samples', alpha=0.9)

                # 特别标记100%错误样本
                perfect_indices = [s['sample_idx'] for s in self.error_samples if s['error_rate'] >= 0.99]
                perfect_mask = np.isin(test_indices, perfect_indices)
                
                if perfect_mask.any():
                    ax.scatter(test_2d[perfect_mask, 0], test_2d[perfect_mask, 1],
                              marker='*', c='red', s=300, 
                              linewidths=2, label='100% Error Samples', 
                              edgecolors='darkred', alpha=1.0)

            ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
            ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
            ax.set_title(f'{omics_name} - Test Sample Distribution (t-SNE)', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)

            # 保存图片
            output_path = os.path.join(output_dir, f'{omics_name}_tsne_visualization.png')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"    ✓ 可视化已保存: {output_path}")

            # 统计离群点与错误样本的重叠
            if self.error_samples:
                error_sample_indices = set(s['sample_idx'] for s in self.error_samples)
                outlier_and_error = set(outlier_indices) & error_sample_indices
                overlap_ratio = len(outlier_and_error) / len(outlier_indices) if len(outlier_indices) > 0 else 0
            else:
                overlap_ratio = 0
                outlier_and_error = set()

            results[omics_name] = {
                'total_test_samples': len(test_data),
                'outliers_detected': len(outlier_indices),
                'outlier_ratio': f"{len(outlier_indices)/len(test_data)*100:.1f}%",
                'outlier_details': outlier_info,
                'outliers_also_errors': len(outlier_and_error),
                'outlier_error_overlap_ratio': f"{overlap_ratio*100:.1f}%",
                'visualization_path': output_path
            }

        return results

    def check_all_enhanced(self) -> Dict:
        """执行所有检查（包括新功能）"""
        if not self.error_samples:
            return {'error': '没有错误样本需要检查'}

        print(f"\n{'=' * 80}")
        print("开始增强版样本特征检查")
        print(f"{'=' * 80}\n")

        results = {
            'total_error_samples': len(self.error_samples),
            'perfect_error_samples': len([s for s in self.error_samples if s['error_rate'] >= 0.99]),
            
            # 新功能1: 误判类别相似度分析
            'misclassification_similarity': self.analyze_misclassification_similarity(),
            
            # 新功能2: 可视化和离群点检测
            'visualization_and_outliers': self.visualize_and_detect_outliers(),
        }

        return results

    def print_enhanced_report(self, check_results: Dict) -> None:
        """打印增强版检查报告"""
        print(f"\n{'=' * 80}")
        print("增强版样本特征检查报告")
        print(f"{'=' * 80}\n")

        print(f"总错误样本数: {check_results['total_error_samples']}")
        print(f"错误率100%样本数: {check_results['perfect_error_samples']}")

        # 误判类别相似度分析
        print(f"\n{'=' * 80}")
        print("1. 误判类别相似度分析")
        print(f"{'=' * 80}")
        
        sim_results = check_results['misclassification_similarity']
        if 'error' in sim_results:
            print(f"  {sim_results['error']}")
        else:
            for omics_name, result in sim_results.items():
                print(f"\n{omics_name}:")
                print(f"  分析样本数: {result['total_samples_analyzed']}")
                print(f"  更接近误判类别的样本: {result['closer_to_predicted_class']} ({result['closer_to_predicted_ratio']})")
                print(f"  100%错误样本中更接近误判类别: {result['perfect_closer_to_pred']} ({result['perfect_closer_to_pred_ratio']})")
                
                # 显示前5个最接近误判类别的样本
                sorted_samples = sorted(result['detailed_analysis'], 
                                       key=lambda x: x['distance_ratio'])
                print(f"\n  前5个最接近误判类别的样本:")
                print(f"  {'样本ID':<10} {'真实':<6} {'预测':<6} {'错误率':<10} {'距离比':<10} {'结论'}")
                print(f"  {'-'*70}")
                for s in sorted_samples[:5]:
                    conclusion = "更像预测类" if s['closer_to_pred'] else "更像真实类"
                    print(f"  {s['sample_idx']:<10} {s['true_label']:<6} {s['predicted_as']:<6} "
                          f"{s['error_rate']*100:>6.1f}%   {s['distance_ratio']:>8.3f}  {conclusion}")

        # 可视化和离群点检测
        print(f"\n{'=' * 80}")
        print("2. 可视化和离群点检测结果")
        print(f"{'=' * 80}")
        
        vis_results = check_results['visualization_and_outliers']
        for omics_name, result in vis_results.items():
            print(f"\n{omics_name}:")
            print(f"  测试样本数: {result['total_test_samples']}")
            print(f"  检测到离群点: {result['outliers_detected']} ({result['outlier_ratio']})")
            print(f"  离群点中也是错误样本: {result['outliers_also_errors']} ({result['outlier_error_overlap_ratio']})")
            print(f"  可视化图片: {result['visualization_path']}")
            
            # 显示前5个离群点
            if result['outlier_details']:
                print(f"\n  前5个离群点:")
                print(f"  {'样本ID':<10} {'真实类别':<10} {'离群分数':<12} {'是否错误样本'}")
                print(f"  {'-'*50}")
                sorted_outliers = sorted(result['outlier_details'], 
                                        key=lambda x: x['outlier_score'], 
                                        reverse=True)
                for o in sorted_outliers[:5]:
                    is_error = "是" if o['is_error_sample'] else "否"
                    print(f"  {o['sample_idx']:<10} {o['true_label']:<10} {o['outlier_score']:>10.3f}  {is_error}")

        print(f"\n{'=' * 80}\n")

    def save_enhanced_report(self, check_results: Dict, output_path: str) -> None:
        """保存增强版检查报告"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("增强版样本特征检查报告\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"总错误样本数: {check_results['total_error_samples']}\n")
            f.write(f"错误率100%样本数: {check_results['perfect_error_samples']}\n")

            # 误判类别相似度分析
            f.write(f"\n{'=' * 80}\n")
            f.write("1. 误判类别相似度分析\n")
            f.write(f"{'=' * 80}\n")
            
            sim_results = check_results['misclassification_similarity']
            if 'error' in sim_results:
                f.write(f"  {sim_results['error']}\n")
            else:
                for omics_name, result in sim_results.items():
                    f.write(f"\n{omics_name}:\n")
                    f.write(f"  分析样本数: {result['total_samples_analyzed']}\n")
                    f.write(f"  更接近误判类别的样本: {result['closer_to_predicted_class']} ({result['closer_to_predicted_ratio']})\n")
                    f.write(f"  100%错误样本中更接近误判类别: {result['perfect_closer_to_pred']} ({result['perfect_closer_to_pred_ratio']})\n")
                    
                    f.write(f"\n  详细分析（所有样本）:\n")
                    f.write(f"  {'样本ID':<10} {'真实':<6} {'预测':<6} {'错误率':<10} {'到真实类':<12} {'到预测类':<12} {'距离比':<10} {'结论'}\n")
                    f.write(f"  {'-'*90}\n")
                    for s in result['detailed_analysis']:
                        conclusion = "更像预测类" if s['closer_to_pred'] else "更像真实类"
                        f.write(f"  {s['sample_idx']:<10} {s['true_label']:<6} {s['predicted_as']:<6} "
                              f"{s['error_rate']*100:>6.1f}%   {s['dist_to_true_class']:>10.3f}  "
                              f"{s['dist_to_pred_class']:>10.3f}  {s['distance_ratio']:>8.3f}  {conclusion}\n")

            # 可视化和离群点检测
            f.write(f"\n{'=' * 80}\n")
            f.write("2. 可视化和离群点检测结果\n")
            f.write(f"{'=' * 80}\n")
            
            vis_results = check_results['visualization_and_outliers']
            for omics_name, result in vis_results.items():
                f.write(f"\n{omics_name}:\n")
                f.write(f"  测试样本数: {result['total_test_samples']}\n")
                f.write(f"  检测到离群点: {result['outliers_detected']} ({result['outlier_ratio']})\n")
                f.write(f"  离群点中也是错误样本: {result['outliers_also_errors']} ({result['outlier_error_overlap_ratio']})\n")
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
                        f.write(f"  {o['sample_idx']:<10} {o['true_label']:<10} {o['outlier_score']:>10.3f}  {is_error}\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"✓ 增强版报告已保存到: {output_path}")


def check_error_samples_enhanced(result_dir: str) -> None:
    """便捷函数：增强版错误样本检查"""
    checker = SampleFeatureChecker(result_dir)

    # 加载数据
    if not checker.load_data():
        return

    # 解析错误报告
    if not checker.parse_error_report():
        print("没有错误样本需要检查")
        return

    # 执行增强版检查
    try:
        check_results = checker.check_all_enhanced()

        # 打印报告
        checker.print_enhanced_report(check_results)

        # 保存报告
        output_path = os.path.join(result_dir, 'sample_feature_check_enhanced_report.txt')
        checker.save_enhanced_report(check_results, output_path)

        print(f"\n✅ 增强版检查完成！")
        print(f"   - 报告已保存: {output_path}")
        print(f"   - 可视化图片已保存在: {result_dir}")

    except Exception as e:
        print(f"\n⚠️  样本检查失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("用法: python sample_checker_enhanced.py <results_directory>")
        print("示例: python sample_checker_enhanced.py results/experiment_20250101_120000")
    else:
        result_dir = sys.argv[1]
        check_error_samples_enhanced(result_dir)
