"""
样本特征检查模块
对错误样本进行全面的特征质量检查和异常检测

功能：
1. 缺失值和异常值检测（NaN, Inf, 离群点）
2. 特征分布统计（偏态、峰度、方差）
3. 样本质量检查（稀疏度、全零特征）
4. 跨组学一致性检查
5. 样本相似度分析（是否是边界样本）
6. 错误样本 vs 正确样本的特征对比
7. 重点关注错误率100%的样本
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity


class SampleFeatureChecker:
    """样本特征检查器
    
    对错误样本的特征矩阵进行全面检查，识别可能的数据质量问题
    
    Attributes:
        result_dir: 结果目录路径
        omics_data: 各组学的特征矩阵列表
        y_full: 完整标签
        train_mask: 训练集掩码
        num_omics: 组学数量
        error_samples: 错误样本信息
    """
    
    def __init__(self, result_dir: str):
        """初始化样本检查器
        
        Args:
            result_dir: 结果目录路径（包含错误分析报告和原始数据）
        """
        self.result_dir = result_dir
        self.omics_data = []
        self.y_full = None
        self.train_mask = None
        self.num_omics = 0
        self.error_samples = None
        
    def load_data(self) -> bool:
        """从结果目录加载数据
        
        Returns:
            加载成功返回True，失败返回False
        """
        try:
            # 加载原始数据信息
            data_info_path = os.path.join(self.result_dir, 'original_data_info.npz')
            if not os.path.exists(data_info_path):
                print(f"❌ 未找到原始数据信息: {data_info_path}")
                return False
            
            data_info = np.load(data_info_path)
            self.y_full = data_info['y_full']
            self.train_mask = data_info['train_mask']
            self.num_omics = int(data_info['num_omics'])
            
            print(f"✓ 加载数据信息: {len(self.y_full)} 个样本, {self.num_omics} 个组学")
            
            # 加载各组学特征矩阵
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
        """解析错误分析报告，提取错误样本信息
        
        Returns:
            解析成功返回True，失败返回False
        """
        try:
            report_path = os.path.join(self.result_dir, 'misclassification_analysis.txt')
            if not os.path.exists(report_path):
                print(f"❌ 未找到错误分析报告: {report_path}")
                return False
            
            self.error_samples = []
            
            with open(report_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 查找频繁错误样本部分
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
                        
                        self.error_samples.append({
                            'sample_idx': sample_idx,
                            'true_label': true_label,
                            'error_count': error_count,
                            'total_runs': total_runs,
                            'error_rate': error_rate
                        })
            
            if self.error_samples:
                print(f"✓ 解析到 {len(self.error_samples)} 个错误样本")
                
                # 统计错误率100%的样本
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
    
    def check_all(self) -> Dict:
        """执行所有检查
        
        Returns:
            完整的检查结果字典
        """
        if not self.error_samples:
            return {'error': '没有错误样本需要检查'}
        
        print(f"\n{'=' * 80}")
        print("开始全面样本特征检查")
        print(f"{'=' * 80}\n")
        
        results = {
            'total_error_samples': len(self.error_samples),
            'perfect_error_samples': len([s for s in self.error_samples if s['error_rate'] >= 0.99]),
            'omics_info': self._get_omics_info(),
            'missing_values': self._check_missing_values(),
            'outliers': self._check_outliers(),
            'feature_distribution': self._check_feature_distribution(),
            'sample_quality': self._check_sample_quality(),
            'cross_omics_consistency': self._check_cross_omics_consistency(),
            'sample_similarity': self._check_sample_similarity(),
            'feature_comparison': self._compare_error_vs_correct(),
            'perfect_errors_analysis': self._analyze_perfect_errors(),
            'recommendations': []
        }
        
        # 生成建议
        results['recommendations'] = self._generate_recommendations(results)
        
        return results
    
    def _get_omics_info(self) -> Dict:
        """获取组学基本信息"""
        info = {}
        for i, omics in enumerate(self.omics_data):
            info[f'omics_{i+1}'] = {
                'shape': omics.shape,
                'num_features': omics.shape[1],
                'dtype': str(omics.dtype),
                'memory_mb': omics.nbytes / (1024**2)
            }
        return info
    
    def _check_missing_values(self) -> Dict:
        """检查缺失值和异常值"""
        print("检查1/8: 缺失值和异常值...")
        
        results = {}
        error_indices = [s['sample_idx'] for s in self.error_samples]
        
        for i, omics in enumerate(self.omics_data):
            omics_name = f'omics_{i+1}'
            error_omics = omics[error_indices]
            
            # 检查NaN
            nan_count = np.isnan(error_omics).sum()
            nan_samples = np.isnan(error_omics).any(axis=1).sum()
            
            # 检查Inf
            inf_count = np.isinf(error_omics).sum()
            inf_samples = np.isinf(error_omics).any(axis=1).sum()
            
            # 检查全局数据的NaN和Inf（用于对比）
            global_nan = np.isnan(omics).sum()
            global_inf = np.isinf(omics).sum()
            
            results[omics_name] = {
                'error_samples_nan_count': int(nan_count),
                'error_samples_nan_samples': int(nan_samples),
                'error_samples_inf_count': int(inf_count),
                'error_samples_inf_samples': int(inf_samples),
                'global_nan_count': int(global_nan),
                'global_inf_count': int(global_inf),
                'has_issues': nan_count > 0 or inf_count > 0
            }
        
        return results
    
    def _check_outliers(self) -> Dict:
        """检查离群点（z-score和IQR方法）"""
        print("检查2/8: 离群点检测...")
        
        results = {}
        error_indices = [s['sample_idx'] for s in self.error_samples]
        
        for i, omics in enumerate(self.omics_data):
            omics_name = f'omics_{i+1}'
            error_omics = omics[error_indices]
            
            # 方法1: Z-score (|z| > 3)
            z_scores = np.abs(stats.zscore(error_omics, axis=0, nan_policy='omit'))
            z_outliers = (z_scores > 3).sum(axis=1)  # 每个样本的离群特征数
            
            # 方法2: IQR
            q1 = np.percentile(error_omics, 25, axis=0)
            q3 = np.percentile(error_omics, 75, axis=0)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            iqr_outliers = ((error_omics < lower_bound) | (error_omics > upper_bound)).sum(axis=1)
            
            # 识别严重离群样本
            severe_outliers = []
            for j, (sample_info, z_out, iqr_out) in enumerate(zip(
                self.error_samples, z_outliers, iqr_outliers
            )):
                if z_out > error_omics.shape[1] * 0.1:  # 超过10%特征是离群点
                    severe_outliers.append({
                        'sample_idx': sample_info['sample_idx'],
                        'error_rate': sample_info['error_rate'],
                        'z_outlier_count': int(z_out),
                        'iqr_outlier_count': int(iqr_out),
                        'outlier_ratio': float(z_out / error_omics.shape[1])
                    })
            
            results[omics_name] = {
                'z_score_outliers_mean': float(z_outliers.mean()),
                'z_score_outliers_max': int(z_outliers.max()),
                'iqr_outliers_mean': float(iqr_outliers.mean()),
                'iqr_outliers_max': int(iqr_outliers.max()),
                'severe_outlier_samples': severe_outliers
            }
        
        return results
    
    def _check_feature_distribution(self) -> Dict:
        """检查特征分布（偏态、峰度、方差）"""
        print("检查3/8: 特征分布统计...")
        
        results = {}
        error_indices = [s['sample_idx'] for s in self.error_samples]
        
        for i, omics in enumerate(self.omics_data):
            omics_name = f'omics_{i+1}'
            error_omics = omics[error_indices]
            
            # 计算偏态和峰度（按特征）
            skewness = stats.skew(error_omics, axis=0, nan_policy='omit')
            kurtosis = stats.kurtosis(error_omics, axis=0, nan_policy='omit')
            variance = np.var(error_omics, axis=0)
            
            # 识别异常分布的特征
            high_skew_features = np.sum(np.abs(skewness) > 2)
            high_kurtosis_features = np.sum(np.abs(kurtosis) > 7)
            low_variance_features = np.sum(variance < 0.01)
            
            results[omics_name] = {
                'skewness_mean': float(np.nanmean(np.abs(skewness))),
                'skewness_max': float(np.nanmax(np.abs(skewness))),
                'high_skew_features': int(high_skew_features),
                'kurtosis_mean': float(np.nanmean(np.abs(kurtosis))),
                'kurtosis_max': float(np.nanmax(np.abs(kurtosis))),
                'high_kurtosis_features': int(high_kurtosis_features),
                'variance_mean': float(np.nanmean(variance)),
                'variance_min': float(np.nanmin(variance)),
                'low_variance_features': int(low_variance_features)
            }
        
        return results
    
    def _check_sample_quality(self) -> Dict:
        """检查样本质量（稀疏度、全零特征等）"""
        print("检查4/8: 样本质量评估...")
        
        results = {}
        error_indices = [s['sample_idx'] for s in self.error_samples]
        
        for i, omics in enumerate(self.omics_data):
            omics_name = f'omics_{i+1}'
            error_omics = omics[error_indices]
            
            quality_issues = []
            
            for j, sample_info in enumerate(self.error_samples):
                sample_features = error_omics[j]
                
                # 计算稀疏度（接近0的特征比例）
                zero_ratio = np.sum(np.abs(sample_features) < 1e-10) / len(sample_features)
                
                # 计算特征范围
                feature_range = sample_features.max() - sample_features.min()
                
                # 计算非零特征的标准差
                nonzero_features = sample_features[np.abs(sample_features) > 1e-10]
                nonzero_std = np.std(nonzero_features) if len(nonzero_features) > 0 else 0
                
                # 标记有质量问题的样本
                has_issue = False
                issues = []
                
                if zero_ratio > 0.9:
                    has_issue = True
                    issues.append(f"高稀疏度({zero_ratio:.1%})")
                
                if feature_range < 1e-6:
                    has_issue = True
                    issues.append(f"特征范围过小({feature_range:.2e})")
                
                if nonzero_std < 1e-6:
                    has_issue = True
                    issues.append(f"特征变异过小({nonzero_std:.2e})")
                
                if has_issue:
                    quality_issues.append({
                        'sample_idx': sample_info['sample_idx'],
                        'error_rate': sample_info['error_rate'],
                        'zero_ratio': float(zero_ratio),
                        'feature_range': float(feature_range),
                        'nonzero_std': float(nonzero_std),
                        'issues': issues
                    })
            
            results[omics_name] = {
                'samples_with_quality_issues': len(quality_issues),
                'quality_issues_details': quality_issues
            }
        
        return results
    
    def _check_cross_omics_consistency(self) -> Dict:
        """检查跨组学一致性

        通过比较样本在各个组学中相对于类中心的距离来评估一致性，
        而不是直接计算不同维度特征之间的相似度
        """
        print("检查5/8: 跨组学一致性...")

        if self.num_omics < 2:
            return {'message': '只有一个组学，跳过跨组学检查'}

        error_indices = [s['sample_idx'] for s in self.error_samples]

        # 对每个组学，计算错误样本到其真实类别中心的距离
        inconsistency_samples = []

        for j, sample_info in enumerate(self.error_samples):
            sample_idx = sample_info['sample_idx']
            true_label = sample_info['true_label']

            # 计算该样本在各个组学中的"异常程度"
            # （到类中心的标准化距离）
            distances = []

            for i, omics in enumerate(self.omics_data):
                # 获取该类别的训练样本
                same_class_mask = self.train_mask & (self.y_full == true_label)

                if same_class_mask.sum() == 0:
                    continue

                # 计算类中心
                class_center = omics[same_class_mask].mean(axis=0)

                # 计算类内标准差
                class_std = omics[same_class_mask].std(axis=0).mean()

                # 计算样本到类中心的欧氏距离
                sample_vec = omics[sample_idx]
                distance = np.linalg.norm(sample_vec - class_center)

                # 标准化距离（相对于类内标准差）
                normalized_distance = distance / (class_std + 1e-10)
                distances.append(normalized_distance)

            if len(distances) < 2:
                continue

            # 计算距离的变异系数（标准差/均值）
            # 如果各组学中样本的相对位置差异很大，说明不一致
            distance_mean = np.mean(distances)
            distance_std = np.std(distances)
            distance_cv = distance_std / (distance_mean + 1e-10)

            # 找出异常的组学（距离明显偏离平均值）
            outlier_omics = []
            for i, dist in enumerate(distances):
                z_score = abs(dist - distance_mean) / (distance_std + 1e-10)
                if z_score > 2:  # 超过2个标准差
                    outlier_omics.append({
                        'omics_idx': i + 1,
                        'distance': float(dist),
                        'z_score': float(z_score)
                    })

            # 标记跨组学不一致的样本
            # 判断标准：距离变异系数大 或 有组学距离明显异常
            if distance_cv > 0.5 or len(outlier_omics) > 0:
                inconsistency_samples.append({
                    'sample_idx': sample_info['sample_idx'],
                    'error_rate': sample_info['error_rate'],
                    'true_label': int(true_label),
                    'distance_cv': float(distance_cv),
                    'distances_to_class_center': [float(d) for d in distances],
                    'outlier_omics': outlier_omics
                })

        return {
            'inconsistent_samples': len(inconsistency_samples),
            'inconsistency_details': inconsistency_samples,
            'explanation': '通过比较样本在各组学中到类中心的距离来评估一致性'
        }

    def _check_sample_similarity(self) -> Dict:
        """检查样本相似度（是否是边界样本）"""
        print("检查6/8: 样本相似度分析...")

        results = {}
        error_indices = [s['sample_idx'] for s in self.error_samples]

        for i, omics in enumerate(self.omics_data):
            omics_name = f'omics_{i+1}'
            error_omics = omics[error_indices]

            # 计算错误样本与同类训练样本的平均相似度
            boundary_samples = []

            for j, sample_info in enumerate(self.error_samples):
                sample_vec = error_omics[j].reshape(1, -1)
                true_label = sample_info['true_label']

                # 找到同类的训练样本
                same_class_mask = self.train_mask & (self.y_full == true_label)
                if same_class_mask.sum() == 0:
                    continue

                same_class_samples = omics[same_class_mask]

                # 计算与同类样本的相似度
                similarities = cosine_similarity(sample_vec, same_class_samples)[0]
                avg_sim_same = similarities.mean()
                max_sim_same = similarities.max()

                # 计算与其他类样本的最大相似度
                other_class_mask = self.train_mask & (self.y_full != true_label)
                if other_class_mask.sum() > 0:
                    other_class_samples = omics[other_class_mask]
                    similarities_other = cosine_similarity(sample_vec, other_class_samples)[0]
                    max_sim_other = similarities_other.max()
                else:
                    max_sim_other = 0

                # 判断是否是边界样本
                is_boundary = (max_sim_other >= avg_sim_same) or (avg_sim_same < 0.5)

                if is_boundary:
                    boundary_samples.append({
                        'sample_idx': sample_info['sample_idx'],
                        'error_rate': sample_info['error_rate'],
                        'true_label': int(true_label),
                        'avg_similarity_same_class': float(avg_sim_same),
                        'max_similarity_other_class': float(max_sim_other),
                        'is_closer_to_other_class': bool(max_sim_other >= avg_sim_same)
                    })

            results[omics_name] = {
                'boundary_samples': len(boundary_samples),
                'boundary_sample_details': boundary_samples
            }

        return results

    def _compare_error_vs_correct(self) -> Dict:
        """对比错误样本 vs 正确样本的特征统计"""
        print("检查7/8: 错误样本 vs 正确样本对比...")

        results = {}
        error_indices = [s['sample_idx'] for s in self.error_samples]

        # 获取测试集中的正确样本
        test_mask = ~self.train_mask
        test_indices = np.where(test_mask)[0]
        correct_indices = [idx for idx in test_indices if idx not in error_indices]

        if len(correct_indices) == 0:
            return {'message': '没有正确分类的样本用于对比'}

        for i, omics in enumerate(self.omics_data):
            omics_name = f'omics_{i+1}'
            error_omics = omics[error_indices]
            correct_omics = omics[correct_indices]

            # 计算各项统计指标
            error_stats = {
                'mean': error_omics.mean(axis=0),
                'std': error_omics.std(axis=0),
                'median': np.median(error_omics, axis=0)
            }

            correct_stats = {
                'mean': correct_omics.mean(axis=0),
                'std': correct_omics.std(axis=0),
                'median': np.median(correct_omics, axis=0)
            }

            # 找出差异最大的特征
            mean_diff = np.abs(error_stats['mean'] - correct_stats['mean'])
            top_diff_features = np.argsort(mean_diff)[-10:][::-1]  # 前10个差异最大的特征

            results[omics_name] = {
                'error_samples_count': len(error_indices),
                'correct_samples_count': len(correct_indices),
                'error_mean_global': float(error_stats['mean'].mean()),
                'correct_mean_global': float(correct_stats['mean'].mean()),
                'error_std_global': float(error_stats['std'].mean()),
                'correct_std_global': float(correct_stats['std'].mean()),
                'top_diff_features': [int(f) for f in top_diff_features],
                'top_diff_values': [float(mean_diff[f]) for f in top_diff_features]
            }

        return results

    def _analyze_perfect_errors(self) -> Dict:
        """重点分析错误率100%的样本"""
        print("检查8/8: 重点分析错误率100%的样本...")

        perfect_errors = [s for s in self.error_samples if s['error_rate'] >= 0.99]

        if not perfect_errors:
            return {'message': '没有错误率100%的样本'}

        print(f"  发现 {len(perfect_errors)} 个错误率100%的样本，进行深度分析...")

        results = {
            'count': len(perfect_errors),
            'sample_indices': [s['sample_idx'] for s in perfect_errors],
            'per_omics_analysis': {}
        }

        perfect_indices = [s['sample_idx'] for s in perfect_errors]

        for i, omics in enumerate(self.omics_data):
            omics_name = f'omics_{i+1}'
            perfect_omics = omics[perfect_indices]

            # 详细分析
            analysis = {
                'samples': []
            }

            for j, sample_info in enumerate(perfect_errors):
                sample_features = perfect_omics[j]

                # 全面检查
                checks = {
                    'sample_idx': sample_info['sample_idx'],
                    'true_label': sample_info['true_label'],
                    'has_nan': bool(np.isnan(sample_features).any()),
                    'has_inf': bool(np.isinf(sample_features).any()),
                    'zero_ratio': float(np.sum(np.abs(sample_features) < 1e-10) / len(sample_features)),
                    'mean': float(sample_features.mean()),
                    'std': float(sample_features.std()),
                    'min': float(sample_features.min()),
                    'max': float(sample_features.max()),
                    'range': float(sample_features.max() - sample_features.min()),
                    'num_outliers': int(np.sum(np.abs(stats.zscore(sample_features)) > 3))
                }

                # 标记严重问题
                issues = []
                if checks['has_nan']:
                    issues.append("包含NaN")
                if checks['has_inf']:
                    issues.append("包含Inf")
                if checks['zero_ratio'] > 0.9:
                    issues.append(f"高稀疏度({checks['zero_ratio']:.1%})")
                if checks['range'] < 1e-6:
                    issues.append("特征无变化")
                if checks['num_outliers'] > len(sample_features) * 0.1:
                    issues.append(f"大量离群点({checks['num_outliers']}个)")

                checks['critical_issues'] = issues
                checks['has_critical_issues'] = len(issues) > 0

                analysis['samples'].append(checks)

            results['per_omics_analysis'][omics_name] = analysis

        return results

    def _generate_recommendations(self, check_results: Dict) -> List[str]:
        """根据检查结果生成改进建议"""
        recommendations = []

        # 1. 缺失值问题
        for omics_name, result in check_results['missing_values'].items():
            if result['has_issues']:
                recommendations.append(
                    f"⚠️  {omics_name} 存在缺失值或异常值："
                )
                if result['error_samples_nan_count'] > 0:
                    recommendations.append(
                        f"   - {result['error_samples_nan_samples']} 个错误样本包含NaN，"
                        f"共{result['error_samples_nan_count']}个NaN值"
                    )
                if result['error_samples_inf_count'] > 0:
                    recommendations.append(
                        f"   - {result['error_samples_inf_samples']} 个错误样本包含Inf，"
                        f"共{result['error_samples_inf_count']}个Inf值"
                    )
                recommendations.append("   建议: 使用插补或删除这些样本")

        # 2. 离群点问题
        for omics_name, result in check_results['outliers'].items():
            if result['severe_outlier_samples']:
                recommendations.append(
                    f"⚠️  {omics_name} 发现 {len(result['severe_outlier_samples'])} 个严重离群样本："
                )
                for s in result['severe_outlier_samples'][:3]:  # 只显示前3个
                    recommendations.append(
                        f"   - 样本{s['sample_idx']}: {s['outlier_ratio']:.1%}的特征是离群点"
                    )
                recommendations.append("   建议: 考虑移除或使用鲁棒的归一化方法")

        # 3. 特征分布问题
        for omics_name, result in check_results['feature_distribution'].items():
            num_features = check_results['omics_info'][omics_name]['num_features']
            if result['low_variance_features'] > num_features * 0.1:
                recommendations.append(
                    f"⚠️  {omics_name} 有 {result['low_variance_features']} 个低方差特征"
                )
                recommendations.append("   建议: 使用方差阈值进行特征选择")

        # 4. 样本质量问题
        for omics_name, result in check_results['sample_quality'].items():
            if result['samples_with_quality_issues'] > 0:
                recommendations.append(
                    f"⚠️  {omics_name} 有 {result['samples_with_quality_issues']} 个样本存在质量问题"
                )
                recommendations.append("   建议: 检查数据预处理流程，可能需要重新标准化")

        # 5. 跨组学一致性问题
        cross_omics = check_results['cross_omics_consistency']
        if 'message' not in cross_omics and cross_omics['inconsistent_samples'] > 0:
            recommendations.append(
                f"⚠️  发现 {cross_omics['inconsistent_samples']} 个跨组学不一致的样本"
            )
            recommendations.append("   这些样本在不同组学中表现出不同的相对位置")
            recommendations.append("   建议:")
            recommendations.append("       - 检查各组学数据预处理是否一致")
            recommendations.append("       - 使用跨组学标准化方法")
            recommendations.append("       - 考虑使用注意力机制来处理组学间的不一致性")

        # 6. 边界样本问题
        for omics_name, result in check_results['sample_similarity'].items():
            if result['boundary_samples'] > 0:
                recommendations.append(
                    f"⚠️  {omics_name} 发现 {result['boundary_samples']} 个边界样本"
                )
                recommendations.append("   建议: 这些样本本质上难以分类，考虑：")
                recommendations.append("       - 增加这些样本类别的训练样本")
                recommendations.append("       - 使用更复杂的模型")
                recommendations.append("       - 考虑集成学习方法")

        # 7. 错误率100%样本的特殊建议
        perfect_errors = check_results.get('perfect_errors_analysis', {})
        if perfect_errors.get('count', 0) > 0:
            recommendations.append(
                f"\n🔴 重点关注: {perfect_errors['count']} 个错误率100%的样本"
            )

            # 检查是否有严重问题
            has_critical = False
            for omics_name, analysis in perfect_errors.get('per_omics_analysis', {}).items():
                for sample in analysis['samples']:
                    if sample.get('has_critical_issues', False):
                        has_critical = True
                        break

            if has_critical:
                recommendations.append("   这些样本存在严重数据质量问题，强烈建议:")
                recommendations.append("   1. 仔细检查原始数据")
                recommendations.append("   2. 验证数据采集和预处理过程")
                recommendations.append("   3. 考虑标注是否正确")
                recommendations.append("   4. 如果确认数据无误，这些样本可能本质上无法分类")
            else:
                recommendations.append("   这些样本数据质量正常，但仍然错误率100%，可能原因:")
                recommendations.append("   1. 样本在特征空间中位于类别边界")
                recommendations.append("   2. 标注可能有误")
                recommendations.append("   3. 需要更多相似样本用于训练")

        if not recommendations:
            recommendations.append("✓ 未发现明显的数据质量问题")
            recommendations.append("  错误可能源于:")
            recommendations.append("  1. 模型容量不足")
            recommendations.append("  2. 训练样本不足")
            recommendations.append("  3. 类别本质上难以区分")

        return recommendations

    def print_report(self, check_results: Dict) -> None:
        """打印检查报告到终端"""
        print(f"\n{'=' * 80}")
        print("样本特征检查报告")
        print(f"{'=' * 80}\n")

        # 基本信息
        print(f"总错误样本数: {check_results['total_error_samples']}")
        print(f"错误率100%样本数: {check_results['perfect_error_samples']}")

        # 组学信息
        print(f"\n组学信息:")
        for omics_name, info in check_results['omics_info'].items():
            print(f"  {omics_name}: shape={info['shape']}, "
                  f"features={info['num_features']}, "
                  f"memory={info['memory_mb']:.2f}MB")

        # 缺失值检查
        print(f"\n1. 缺失值和异常值:")
        has_missing = False
        for omics_name, result in check_results['missing_values'].items():
            if result['has_issues']:
                has_missing = True
                print(f"  {omics_name}:")
                print(f"    错误样本 - NaN: {result['error_samples_nan_count']}, "
                      f"Inf: {result['error_samples_inf_count']}")
                print(f"    全局数据 - NaN: {result['global_nan_count']}, "
                      f"Inf: {result['global_inf_count']}")
        if not has_missing:
            print("  ✓ 未发现缺失值或异常值")

        # 离群点检查
        print(f"\n2. 离群点:")
        has_outliers = False
        for omics_name, result in check_results['outliers'].items():
            severe = result['severe_outlier_samples']
            if severe:
                has_outliers = True
                print(f"  {omics_name}: {len(severe)} 个严重离群样本")
                for s in severe[:3]:
                    print(f"    样本{s['sample_idx']}: "
                          f"离群特征比例={s['outlier_ratio']:.1%}, "
                          f"错误率={s['error_rate']:.1%}")
        if not has_outliers:
            print("  ✓ 未发现严重离群样本")

        # 特征分布
        print(f"\n3. 特征分布:")
        for omics_name, result in check_results['feature_distribution'].items():
            print(f"  {omics_name}:")
            print(f"    高偏态特征: {result['high_skew_features']}")
            print(f"    高峰度特征: {result['high_kurtosis_features']}")
            print(f"    低方差特征: {result['low_variance_features']}")

        # 样本质量
        print(f"\n4. 样本质量:")
        has_quality_issues = False
        for omics_name, result in check_results['sample_quality'].items():
            if result['samples_with_quality_issues'] > 0:
                has_quality_issues = True
                print(f"  {omics_name}: {result['samples_with_quality_issues']} 个样本有质量问题")
                for s in result['quality_issues_details'][:3]:
                    print(f"    样本{s['sample_idx']}: {', '.join(s['issues'])}")
        if not has_quality_issues:
            print("  ✓ 所有样本质量正常")

        # 跨组学一致性
        cross_omics = check_results['cross_omics_consistency']
        if 'message' not in cross_omics:
            print(f"\n5. 跨组学一致性:")
            if cross_omics['inconsistent_samples'] > 0:
                print(f"  发现 {cross_omics['inconsistent_samples']} 个跨组学不一致的样本")
                for s in cross_omics['inconsistency_details'][:3]:
                    print(f"    样本{s['sample_idx']}: "
                          f"距离变异系数={s['distance_cv']:.3f}")
            else:
                print("  ✓ 所有样本跨组学一致性良好")

        # 边界样本
        print(f"\n6. 边界样本:")
        total_boundary = 0
        for omics_name, result in check_results['sample_similarity'].items():
            total_boundary += result['boundary_samples']
            if result['boundary_samples'] > 0:
                print(f"  {omics_name}: {result['boundary_samples']} 个边界样本")
        if total_boundary == 0:
            print("  ✓ 未发现明显边界样本")

        # 错误vs正确对比
        if 'message' not in check_results['feature_comparison']:
            print(f"\n7. 错误样本 vs 正确样本:")
            for omics_name, result in check_results['feature_comparison'].items():
                print(f"  {omics_name}:")
                print(f"    错误样本平均值: {result['error_mean_global']:.4f}")
                print(f"    正确样本平均值: {result['correct_mean_global']:.4f}")
                print(f"    差异最大的前3个特征: {result['top_diff_features'][:3]}")

        # 错误率100%样本
        perfect = check_results['perfect_errors_analysis']
        if 'message' not in perfect:
            print(f"\n8. 错误率100%样本深度分析:")
            print(f"  共 {perfect['count']} 个样本")

            for omics_name, analysis in perfect['per_omics_analysis'].items():
                print(f"\n  {omics_name}:")
                for sample in analysis['samples'][:5]:  # 只显示前5个
                    print(f"    样本{sample['sample_idx']} (真实类别={sample['true_label']}):")
                    print(f"      稀疏度: {sample['zero_ratio']:.1%}, "
                          f"范围: {sample['range']:.2e}, "
                          f"离群点: {sample['num_outliers']}")
                    if sample['critical_issues']:
                        print(f"      ⚠️  严重问题: {', '.join(sample['critical_issues'])}")

        # 改进建议
        print(f"\n{'=' * 80}")
        print("改进建议:")
        print(f"{'=' * 80}")
        for rec in check_results['recommendations']:
            print(rec)

        print(f"\n{'=' * 80}\n")

    def save_report(self, check_results: Dict, output_path: str) -> None:
        """保存检查报告到文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("样本特征检查报告\n")
            f.write("=" * 80 + "\n\n")

            # 基本信息
            f.write(f"总错误样本数: {check_results['total_error_samples']}\n")
            f.write(f"错误率100%样本数: {check_results['perfect_error_samples']}\n")

            # 组学信息
            f.write(f"\n组学信息:\n")
            for omics_name, info in check_results['omics_info'].items():
                f.write(f"  {omics_name}: shape={info['shape']}, "
                       f"features={info['num_features']}, "
                       f"memory={info['memory_mb']:.2f}MB\n")

            # 各项检查结果
            f.write(f"\n1. 缺失值和异常值:\n")
            has_missing = False
            for omics_name, result in check_results['missing_values'].items():
                if result['has_issues']:
                    has_missing = True
                    f.write(f"  {omics_name}:\n")
                    f.write(f"    错误样本 - NaN: {result['error_samples_nan_count']}, "
                           f"Inf: {result['error_samples_inf_count']}\n")
            if not has_missing:
                f.write("  ✓ 未发现缺失值或异常值\n")

            f.write(f"\n2. 离群点:\n")
            has_outliers = False
            for omics_name, result in check_results['outliers'].items():
                severe = result['severe_outlier_samples']
                if severe:
                    has_outliers = True
                    f.write(f"  {omics_name}: {len(severe)} 个严重离群样本\n")
                    for s in severe:
                        f.write(f"    样本{s['sample_idx']}: "
                               f"离群特征比例={s['outlier_ratio']:.1%}\n")
            if not has_outliers:
                f.write("  ✓ 未发现严重离群样本\n")

            # 错误率100%样本
            perfect = check_results['perfect_errors_analysis']
            if 'message' not in perfect:
                f.write(f"\n8. 错误率100%样本详细信息:\n")
                f.write(f"  共 {perfect['count']} 个样本\n")

                for omics_name, analysis in perfect['per_omics_analysis'].items():
                    f.write(f"\n  {omics_name}:\n")
                    for sample in analysis['samples']:
                        f.write(f"    样本{sample['sample_idx']}:\n")
                        f.write(f"      真实类别: {sample['true_label']}\n")
                        f.write(f"      稀疏度: {sample['zero_ratio']:.1%}\n")
                        f.write(f"      特征范围: {sample['range']:.2e}\n")
                        f.write(f"      离群点数: {sample['num_outliers']}\n")
                        if sample['critical_issues']:
                            f.write(f"      ⚠️  问题: {', '.join(sample['critical_issues'])}\n")

            # 改进建议
            f.write(f"\n{'=' * 80}\n")
            f.write("改进建议:\n")
            f.write(f"{'=' * 80}\n")
            for rec in check_results['recommendations']:
                f.write(rec + "\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"✓ 详细报告已保存到: {output_path}")


def check_error_samples(result_dir: str) -> None:
    """便捷函数：检查错误样本

    Args:
        result_dir: 结果目录路径
    """
    checker = SampleFeatureChecker(result_dir)

    # 加载数据
    if not checker.load_data():
        return

    # 解析错误报告
    if not checker.parse_error_report():
        print("没有错误样本需要检查")
        return

    # 执行检查
    try:
        check_results = checker.check_all()

        # 打印报告
        checker.print_report(check_results)

        # 保存报告
        output_path = os.path.join(result_dir, 'sample_feature_check_report.txt')
        checker.save_report(check_results, output_path)

    except Exception as e:
        print(f"\n⚠️  样本检查失败: {e}")
        print("实验结果已保存，但样本检查未完成")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("用法: python sample_checker.py <results_directory>")
        print("示例: python sample_checker.py results/experiment_20250101_120000")
    else:
        result_dir = sys.argv[1]
        check_error_samples(result_dir)