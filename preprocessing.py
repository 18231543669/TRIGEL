"""
多组学数据预处理模块
提供标准化、特征选择、异常值处理等通用方法
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
from typing import List, Tuple, Optional, Dict
import warnings

warnings.filterwarnings("ignore")


class MultiOmicsPreprocessor:
    """多组学数据预处理器
    
    功能：
    1. 每个组学独立标准化（避免跨组学尺度差异）
    2. 移除低方差特征（减少噪声）
    3. 异常值处理（可选）
    4. 保持训练集和测试集一致性
    
    特点：
    - 通用：适用于任意多组学数据
    - 可配置：可根据数据特点调整参数
    - 可追溯：记录预处理过程
    """
    
    def __init__(self, 
                 scaler_type: str = 'standard',
                 variance_threshold: float = 0.0,
                 clip_outliers: bool = True,
                 outlier_std: float = 5.0,
                 verbose: bool = True):
        """初始化预处理器
        
        Args:
            scaler_type: 标准化方法 ('standard' 或 'robust')
                - 'standard': 标准正态分布 (mean=0, std=1)
                - 'robust': 使用中位数和四分位数（对异常值鲁棒）
            variance_threshold: 方差阈值，低于此值的特征被移除
            clip_outliers: 是否裁剪异常值
            outlier_std: 异常值定义（几倍标准差）
            verbose: 是否打印详细信息
        """
        self.scaler_type = scaler_type
        self.variance_threshold = variance_threshold
        self.clip_outliers = clip_outliers
        self.outlier_std = outlier_std
        self.verbose = verbose
        
        # 存储每个组学的预处理器
        self.scalers = []
        self.variance_selectors = []
        self.preprocessing_stats = []
        
    def fit_transform_omics(self, 
                           train_data: List[np.ndarray], 
                           test_data: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray], Dict]:
        """对多组学数据进行预处理（训练集+测试集）
        
        Args:
            train_data: 训练集数据列表 [omics1, omics2, ...]
            test_data: 测试集数据列表
            
        Returns:
            processed_train_data: 预处理后的训练集
            processed_test_data: 预处理后的测试集
            stats: 预处理统计信息
        """
        num_omics = len(train_data)
        processed_train = []
        processed_test = []
        
        if self.verbose:
            print("\n" + "=" * 70)
            print("多组学数据预处理")
            print("=" * 70)
        
        for i in range(num_omics):
            if self.verbose:
                print(f"\n处理组学 {i+1}/{num_omics}:")
                print(f"  原始维度: 训练集 {train_data[i].shape}, 测试集 {test_data[i].shape}")
            
            # 1. 异常值裁剪（可选）
            train_omics = train_data[i].copy()
            test_omics = test_data[i].copy()
            
            if self.clip_outliers:
                train_omics, test_omics = self._clip_outliers(train_omics, test_omics, i)
            
            # 2. 方差过滤
            if self.variance_threshold > 0:
                train_omics, test_omics = self._filter_low_variance(train_omics, test_omics, i)
            
            # 3. 标准化（最重要）
            train_omics, test_omics = self._standardize(train_omics, test_omics, i)
            
            processed_train.append(train_omics)
            processed_test.append(test_omics)
            
            # 记录统计信息
            self.preprocessing_stats.append({
                'omics_id': i + 1,
                'original_features': train_data[i].shape[1],
                'final_features': train_omics.shape[1],
                'features_removed': train_data[i].shape[1] - train_omics.shape[1],
                'train_samples': train_omics.shape[0],
                'test_samples': test_omics.shape[0],
                'train_mean': np.mean(train_omics),
                'train_std': np.std(train_omics),
                'train_min': np.min(train_omics),
                'train_max': np.max(train_omics)
            })
            
            if self.verbose:
                print(f"  最终维度: 训练集 {train_omics.shape}, 测试集 {test_omics.shape}")
                print(f"  移除特征: {train_data[i].shape[1] - train_omics.shape[1]}")
                print(f"  统计: mean={np.mean(train_omics):.4f}, std={np.std(train_omics):.4f}, "
                      f"min={np.min(train_omics):.4f}, max={np.max(train_omics):.4f}")
        
        # 汇总统计
        stats = self._generate_summary_stats(processed_train, processed_test)
        
        if self.verbose:
            self._print_summary(stats)
        
        return processed_train, processed_test, stats
    
    def _clip_outliers(self, train_data: np.ndarray, test_data: np.ndarray, 
                       omics_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """裁剪异常值"""
        # 基于训练集计算裁剪范围
        train_mean = np.mean(train_data, axis=0, keepdims=True)
        train_std = np.std(train_data, axis=0, keepdims=True)
        
        # 避免除零
        train_std[train_std == 0] = 1.0
        
        # 计算裁剪范围
        lower_bound = train_mean - self.outlier_std * train_std
        upper_bound = train_mean + self.outlier_std * train_std
        
        # 裁剪
        train_clipped = np.clip(train_data, lower_bound, upper_bound)
        test_clipped = np.clip(test_data, lower_bound, upper_bound)
        
        # 统计裁剪比例
        train_clipped_ratio = np.mean(train_data != train_clipped)
        test_clipped_ratio = np.mean(test_data != test_clipped)
        
        if self.verbose:
            print(f"  异常值裁剪: 训练集 {train_clipped_ratio:.2%}, 测试集 {test_clipped_ratio:.2%}")
        
        return train_clipped, test_clipped
    
    def _filter_low_variance(self, train_data: np.ndarray, test_data: np.ndarray,
                            omics_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """移除低方差特征"""
        selector = VarianceThreshold(threshold=self.variance_threshold)
        
        # 在训练集上拟合
        train_filtered = selector.fit_transform(train_data)
        
        # 应用到测试集
        test_filtered = selector.transform(test_data)
        
        # 保存选择器
        self.variance_selectors.append(selector)
        
        n_removed = train_data.shape[1] - train_filtered.shape[1]
        if self.verbose and n_removed > 0:
            print(f"  方差过滤: 移除 {n_removed} 个低方差特征 (阈值={self.variance_threshold})")
        
        return train_filtered, test_filtered
    
    def _standardize(self, train_data: np.ndarray, test_data: np.ndarray,
                    omics_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """标准化特征"""
        # 选择标准化器
        if self.scaler_type == 'standard':
            scaler = StandardScaler()
        elif self.scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")
        
        # 在训练集上拟合
        train_scaled = scaler.fit_transform(train_data)
        
        # 应用到测试集
        test_scaled = scaler.transform(test_data)
        
        # 保存标准化器
        self.scalers.append(scaler)
        
        if self.verbose:
            print(f"  标准化: 使用 {self.scaler_type} scaler")
        
        return train_scaled, test_scaled
    
    def _generate_summary_stats(self, train_data: List[np.ndarray], 
                                test_data: List[np.ndarray]) -> Dict:
        """生成汇总统计"""
        # 合并所有组学
        X_train_combined = np.concatenate(train_data, axis=1)
        X_test_combined = np.concatenate(test_data, axis=1)
        
        total_features = sum(d.shape[1] for d in train_data)
        
        return {
            'num_omics': len(train_data),
            'total_features': total_features,
            'train_samples': X_train_combined.shape[0],
            'test_samples': X_test_combined.shape[0],
            'combined_train_mean': np.mean(X_train_combined),
            'combined_train_std': np.std(X_train_combined),
            'combined_train_min': np.min(X_train_combined),
            'combined_train_max': np.max(X_train_combined),
            'omics_stats': self.preprocessing_stats
        }
    
    def _print_summary(self, stats: Dict) -> None:
        """打印汇总信息"""
        print("\n" + "=" * 70)
        print("预处理汇总")
        print("=" * 70)
        print(f"组学数量: {stats['num_omics']}")
        print(f"总特征数: {stats['total_features']}")
        print(f"训练样本: {stats['train_samples']}")
        print(f"测试样本: {stats['test_samples']}")
        print(f"\n合并后的训练集统计:")
        print(f"  Mean: {stats['combined_train_mean']:.6f}")
        print(f"  Std:  {stats['combined_train_std']:.6f}")
        print(f"  Min:  {stats['combined_train_min']:.6f}")
        print(f"  Max:  {stats['combined_train_max']:.6f}")
        print("=" * 70 + "\n")


def preprocess_multi_omics_data(train_data: List[np.ndarray], 
                                test_data: List[np.ndarray],
                                config: Optional[Dict] = None) -> Tuple[List[np.ndarray], List[np.ndarray], Dict]:
    """多组学数据预处理的便捷函数
    
    Args:
        train_data: 训练集数据列表
        test_data: 测试集数据列表
        config: 预处理配置（可选）
            - scaler_type: 'standard' 或 'robust'
            - variance_threshold: 方差阈值
            - clip_outliers: 是否裁剪异常值
            - outlier_std: 异常值定义（几倍标准差）
            - verbose: 是否打印详细信息
    
    Returns:
        processed_train_data: 预处理后的训练集
        processed_test_data: 预处理后的测试集
        stats: 预处理统计信息
    
    Example:
        >>> train_data = [omics1_train, omics2_train, omics3_train]
        >>> test_data = [omics1_test, omics2_test, omics3_test]
        >>> train_processed, test_processed, stats = preprocess_multi_omics_data(
        ...     train_data, test_data,
        ...     config={'scaler_type': 'standard', 'variance_threshold': 0.01}
        ... )
    """
    # 默认配置
    default_config = {
        'scaler_type': 'standard',      # 标准正态分布标准化
        'variance_threshold': 0.0,      # 不移除低方差特征（保守策略）
        'clip_outliers': True,          # 裁剪异常值
        'outlier_std': 5.0,             # 5倍标准差作为异常值
        'verbose': True                 # 打印详细信息
    }
    
    # 合并用户配置
    if config is not None:
        default_config.update(config)
    
    # 创建预处理器
    preprocessor = MultiOmicsPreprocessor(**default_config)
    
    # 执行预处理
    return preprocessor.fit_transform_omics(train_data, test_data)


# 推荐配置
RECOMMENDED_CONFIGS = {
    'conservative': {
        'scaler_type': 'standard',
        'variance_threshold': 0.0,
        'clip_outliers': True,
        'outlier_std': 5.0,
        'verbose': True
    },
    'aggressive': {
        'scaler_type': 'standard',
        'variance_threshold': 0.01,  # 移除方差<0.01的特征
        'clip_outliers': True,
        'outlier_std': 3.0,          # 更严格的异常值裁剪
        'verbose': True
    },
    'robust': {
        'scaler_type': 'robust',     # 使用RobustScaler（对异常值更鲁棒）
        'variance_threshold': 0.0,
        'clip_outliers': False,      # RobustScaler本身就对异常值鲁棒
        'outlier_std': 5.0,
        'verbose': True
    }
}
