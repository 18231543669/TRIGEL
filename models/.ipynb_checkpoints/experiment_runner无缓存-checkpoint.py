"""
实验运行器模块（集成错误分析 + 性能优化版）
负责协调和执行完整的实验流程，并记录错误样本用于后续分析

✨ 新增优化：
- 数据加载缓存：第1次运行加载，后续运行复用
- 图结构缓存：第1次运行构建，后续运行复用
- 大幅减少重复计算，提升实验效率
"""

import os
import tempfile
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Optional
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

# 导入配置和工具
from config import device, set_random_seed
from utils.data_utils import load_dataset, construct_adjacency, stratified_split
from utils.metrics import calculate_metrics, calculate_average_results
from utils.enhanced_analysis import run_enhanced_analysis
from utils.io_utils import (
    create_result_directory, save_results, validate_config,
    print_config_summary
)
from utils.visualization import visualize_training
from utils.training_utils import TrainingEarlyStopping, get_training_summary, print_training_summary

# 导入模型和训练
from models.gcn_models import ModelFactory, GCNContrastiveModel
from trainer import evaluate_model

# 导入数据增强
from augmentation.augmentors import DataAugmentor, get_current_augmentation_params
from augmentation.reporter import AugmentationEffectsReporter

# ✓ 导入错误分析模块
from utils.error_analysis import MisclassificationAnalyzer


class ExperimentRunner:
    """实验运行器类（集成错误分析功能 + 性能优化）

    该类负责协调整个实验流程，包括：
    - 数据加载和预处理（✨ 支持缓存）
    - 图结构构建（✨ 支持缓存）
    - 数据增强
    - 模型训练和评估
    - 结果保存和可视化
    - ✓ 错误样本分析和记录

    支持两种模式：
    1. 正常模式：完整的实验流程，包括目录创建、结果保存、可视化等
    2. 超参数搜索模式：精简流程，不创建目录和文件，静默运行

    Attributes:
        config: 实验配置字典
        is_hyperparameter_search: 是否为超参数搜索模式
        data_augmentor: 数据增强器
        effects_reporter: 增强效果报告器
        misclassification_analyzer: ✓ 错误分类分析器
        main_result_dir: 主结果目录（仅正常模式）
        original_data_info: ✓ 原始数据信息（用于样本检查）
        cached_data_info: ✨ 缓存的数据信息（避免重复加载）
        cached_edge_indices: ✨ 缓存的图结构（避免重复构建）
    """

    def __init__(self, config: Dict):
        """初始化实验运行器

        Args:
            config: 实验配置字典
        """
        self.config = config

        # 检查是否是超参数搜索模式
        self.is_hyperparameter_search = config.get('hyperparameter_search_mode', False)

        if not self.is_hyperparameter_search:
            # 正常模式：初始化所有组件
            self.data_augmentor = DataAugmentor()
            self.effects_reporter = AugmentationEffectsReporter()

            # ✓ 初始化错误分析器
            self.misclassification_analyzer = MisclassificationAnalyzer()

            # 验证配置
            if not validate_config(config):
                raise ValueError("配置验证失败")
        else:
            # 超参数搜索模式：最少初始化
            self.data_augmentor = DataAugmentor()
            self.effects_reporter = AugmentationEffectsReporter()
            self.misclassification_analyzer = None

        # ✓ 保存原始数据信息供后续检查
        self.original_data_info = None

        # ✨ 新增：缓存变量（避免重复加载和构建）
        self.cached_data_info = None
        self.cached_edge_indices = None

    def _load_and_build_once(self) -> tuple:
        """✨ 新增：仅加载一次数据和构建一次图结构
        
        Returns:
            (data_info, edge_indices) 元组
        """
        if not self.is_hyperparameter_search:
            print(f"\n{'=' * 60}")
            print("🚀 首次加载数据和构建图结构（后续运行将复用）")
            print(f"{'=' * 60}")
        
        # 加载数据
        data_info = self._load_experiment_data()
        if data_info is None:
            return None, None
        
        # 构建图结构
        edge_indices = self._build_graph_structures(data_info)
        if edge_indices is None:
            return None, None
        
        if not self.is_hyperparameter_search:
            print(f"\n✓ 数据加载和图构建完成，已缓存供后续运行使用")
        
        return data_info, edge_indices

    def run_single_experiment(self, run_id: int = 1, 
                             cached_data_info: Optional[Dict] = None,
                             cached_edge_indices: Optional[List[torch.Tensor]] = None) -> Optional[Dict]:
        """运行单次实验

        Args:
            run_id: 实验运行ID
            cached_data_info: ✨ 缓存的数据信息（避免重复加载）
            cached_edge_indices: ✨ 缓存的图结构（避免重复构建）

        Returns:
            实验结果字典，失败时返回None
        """
        if not self.is_hyperparameter_search:
            print(f"\n{'=' * 60}")
            print(f"运行第 {run_id}/{self.config.get('num_runs', 5)} 次实验")
            print(f"{'=' * 60}")

        # 设置当前实验的随机种子
        current_seed = self._get_current_seed(run_id)
        set_random_seed(current_seed)

        if not self.is_hyperparameter_search:
            print(f"当前运行使用随机种子: {current_seed}")

        try:
            # ✨ 优化：使用缓存的数据和图结构
            if cached_data_info is not None and cached_edge_indices is not None:
                if not self.is_hyperparameter_search and run_id > 1:
                    print(f"♻️  使用缓存的数据和图结构（跳过加载和构建）")
                
                data_info = cached_data_info
                edge_indices = cached_edge_indices
            else:
                # 第一次运行或未提供缓存：正常加载和构建
                data_info = self._load_experiment_data()
                if data_info is None:
                    return None
                
                edge_indices = self._build_graph_structures(data_info)
                if edge_indices is None:
                    return None

            # ✓ 保存原始数据信息（第一次运行时）
            if run_id == 1 and not self.is_hyperparameter_search:
                self.original_data_info = data_info

            # 3. 应用数据增强（如果启用）
            augmented_data = self._apply_data_augmentation(
                data_info, edge_indices, run_id
            )

            # 4. 划分数据集
            data_splits = self._split_dataset(data_info, current_seed)

            # 5. 初始化模型和优化器
            model, optimizer, criterion = self._initialize_model(data_info, data_splits)

            # 6. 训练模型
            training_results = self._train_model(
                model, optimizer, criterion, augmented_data, data_splits, run_id
            )

            # 7. 评估模型（✓ 集成错误分析记录）
            final_results = self._evaluate_model(
                model, augmented_data, data_splits, training_results, run_id, current_seed
            )

            return final_results

        except Exception as e:
            print(f"第 {run_id} 次实验出现错误: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_multiple_experiments(self, num_runs: int = 5) -> List[Dict]:
        """运行多次实验（✨ 优化版：复用数据和图结构）

        Args:
            num_runs: 实验运行次数

        Returns:
            所有实验结果的列表
        """
        if not self.is_hyperparameter_search:
            # 正常模式：打印配置并创建目录
            print_config_summary(self.config)
            self.main_result_dir = create_result_directory(self.config['experiment_name'])
        else:
            # 超参数搜索模式：不打印配置，不创建目录，静默运行
            pass

        # ✨ 优化核心：仅加载一次数据和构建一次图结构
        if not self.is_hyperparameter_search:
            print(f"\n💡 性能优化：数据和图结构将仅加载/构建一次，{num_runs}次运行共享")
        
        data_info, edge_indices = self._load_and_build_once()
        
        if data_info is None or edge_indices is None:
            if not self.is_hyperparameter_search:
                print("数据加载或图构建失败，无法继续实验")
            return []
        
        # 缓存到实例变量
        self.cached_data_info = data_info
        self.cached_edge_indices = edge_indices

        all_results = []

        for run_id in range(1, num_runs + 1):
            # ✨ 传入缓存的数据和图结构
            result = self.run_single_experiment(
                run_id, 
                cached_data_info=data_info,
                cached_edge_indices=edge_indices
            )
            
            if result:
                all_results.append(result)

                # 只有正常模式才整理结果
                if not self.is_hyperparameter_search:
                    self._organize_results(result, self.main_result_dir, run_id)
            else:
                if not self.is_hyperparameter_search:
                    print(f"第 {run_id} 次实验失败")

        # 计算并保存平均结果（只在正常模式）
        if all_results and not self.is_hyperparameter_search:
            self._finalize_experiments(all_results, self.main_result_dir)
        elif not all_results and not self.is_hyperparameter_search:
            print("所有实验都失败了")

        return all_results

    # ==================== 私有辅助方法 ====================

    def _get_current_seed(self, run_id: int) -> int:
        """获取当前实验的随机种子"""
        base_seed = self.config.get('seed', 42)
        return base_seed + run_id - 1 if base_seed is not None else None

    def _load_experiment_data(self) -> Optional[Dict]:
        """加载实验数据（✓ 保存每个组学的原始特征矩阵）"""
        try:
            result = load_dataset(self.config['data_dir'])
            X_full, y_full, total_features, num_omics, train_data, test_data, num_classes = result

            return {
                'X_full': X_full,
                'y_full': y_full,
                'total_features': total_features,
                'num_omics': num_omics,
                'train_data': train_data,  # ✓ 保存每个组学的训练数据
                'test_data': test_data,    # ✓ 保存每个组学的测试数据
                'num_classes': num_classes,
                'num_train': len(train_data[0])
            }
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None

    def _build_graph_structures(self, data_info: Dict) -> Optional[List[torch.Tensor]]:
        """构建图结构"""
        edge_indices = []
        num_omics = data_info['num_omics']

        for i in range(num_omics):
            if not self.is_hyperparameter_search:
                print(f"\n构建组学 {i + 1}/{num_omics} 的图结构")

            omics_data = np.vstack([data_info['train_data'][i], data_info['test_data'][i]])

            try:
                edge_index = construct_adjacency(omics_data, k=self.config['cosine_k'])
                edge_indices.append(edge_index)

                if not self.is_hyperparameter_search:
                    print(f"图 {i + 1} 有 {edge_index.shape[1]} 条边")
            except Exception as e:
                print(f"构建图结构失败: {e}")
                return None

        return edge_indices

    def _apply_data_augmentation(self, data_info: Dict, edge_indices: List[torch.Tensor],
                                 run_id: int) -> Dict:
        """应用数据增强"""
        X_original = data_info['X_full'].copy()
        edge_indices_original = [edge_idx.clone() for edge_idx in edge_indices]
        edge_weights_original = None

        X_augmented = data_info['X_full'].copy()
        edge_indices_augmented = [edge_idx.clone() for edge_idx in edge_indices]
        edge_weights_augmented = None
        aug_stats = None

        if self.config.get('use_augmentation', False):
            train_mask = np.zeros(len(data_info['y_full']), dtype=bool)
            train_mask[:data_info['num_train']] = True

            try:
                current_aug_params = get_current_augmentation_params()

                default_params = {
                    'minority_drop_rate': self.config.get('minority_drop_rate', 0.0),
                    'majority_drop_rate': self.config.get('majority_drop_rate', 0.5),
                    'edge_add_prob': self.config.get('edge_add_prob', 0.5),
                    'max_new_edges': self.config.get('max_new_edges', 100),
                    'use_edge_weights': self.config.get('use_edge_weights', False),
                    'minority_edge_weight': self.config.get('minority_edge_weight', 2.0),
                    'quality_weight_factor': self.config.get('quality_weight_factor', 0.3),
                }

                for key, default_value in default_params.items():
                    if key not in current_aug_params:
                        current_aug_params[key] = default_value

                X_aug_tensor, edge_indices_aug, edge_weights_aug, aug_stats = self.data_augmentor.augment(
                    data_info['X_full'], data_info['y_full'], edge_indices,
                    train_mask, current_aug_params
                )

                X_augmented = X_aug_tensor.numpy()
                edge_indices_augmented = edge_indices_aug
                edge_weights_augmented = edge_weights_aug

                if run_id == 1 and not self.is_hyperparameter_search:
                    self.effects_reporter.print_effects(aug_stats, current_aug_params)

            except Exception as e:
                if not self.is_hyperparameter_search:
                    print(f"数据增强失败: {e}")
                    print("继续使用原始数据进行训练")
                import traceback
                traceback.print_exc()

        return {
            'X_original': X_original,
            'X_augmented': X_augmented,
            'edge_indices_original': edge_indices_original,
            'edge_indices_augmented': edge_indices_augmented,
            'edge_weights_original': edge_weights_original,
            'edge_weights_augmented': edge_weights_augmented,
            'aug_stats': aug_stats
        }

    def _split_dataset(self, data_info: Dict, current_seed: int) -> Dict:
        """划分数据集(验证集=测试集,训练集最大化)"""
        num_train = data_info['num_train']

        train_idx = torch.arange(0, num_train, dtype=torch.long)

        test_indices = list(range(num_train, len(data_info['y_full'])))
        val_idx = torch.tensor(test_indices, dtype=torch.long)
        test_idx = torch.tensor(test_indices, dtype=torch.long)

        if not self.is_hyperparameter_search:
            print(f"\n数据划分(验证集=测试集模式):")
            print(f"训练样本: {len(train_idx)} ({len(train_idx) / len(data_info['y_full']) * 100:.1f}%)")
            print(f"验证样本: {len(val_idx)} ({len(val_idx) / len(data_info['y_full']) * 100:.1f}%)")
            print(f"测试样本: {len(test_idx)} ({len(test_idx) / len(data_info['y_full']) * 100:.1f}%)")
            print(f"\n⚠️  验证集和测试集完全重合(共{len(val_idx)}个样本)")

            train_labels = data_info['y_full'][train_idx.numpy()]
            test_labels = data_info['y_full'][test_idx.numpy()]

            train_dist = dict(zip(*np.unique(train_labels, return_counts=True)))
            test_dist = dict(zip(*np.unique(test_labels, return_counts=True)))

            print(f"\n类别分布:")
            print(f"训练集: {train_dist}")
            print(f"测试集: {test_dist}")

        return {
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx,
            'y_tensor': torch.tensor(data_info['y_full'], dtype=torch.long).to(device)
        }

    def _initialize_model(self, data_info: Dict, data_splits: Dict) -> tuple:
        """初始化模型和优化器"""
        if self.config.get('use_contrastive', False):
            model = ModelFactory.create_model(
                model_type='gcn_contrastive',
                input_dim=data_info['total_features'],
                hidden_dim=self.config['hidden_dim'],
                num_omics=data_info['num_omics'],
                num_classes=data_info['num_classes'],
                gcn_layers=self.config['gcn_layers'],
                embedding_dim=self.config.get('embedding_dim', 128),
                temperature=self.config.get('temperature', 0.5)
            )
        else:
            model = ModelFactory.create_model(
                model_type='gcn',
                input_dim=data_info['total_features'],
                hidden_dim=self.config['hidden_dim'],
                num_omics=data_info['num_omics'],
                num_classes=data_info['num_classes'],
                gcn_layers=self.config['gcn_layers']
            )

        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['lr'], weight_decay=1e-4)

        train_labels = data_info['y_full'][:data_info['num_train']]
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

        if not self.is_hyperparameter_search:
            print(f"\n类别权重: {class_weights}")

        return model, optimizer, criterion

    def _train_model(self, model, optimizer, criterion, augmented_data: Dict,
                     data_splits: Dict, run_id: int) -> Dict:
        """训练模型"""
        if not self.is_hyperparameter_search:
            print(f"\n开始训练 (运行 {run_id})...")

        x_original = torch.tensor(augmented_data['X_original'], dtype=torch.float).to(device)
        x_augmented = torch.tensor(augmented_data['X_augmented'], dtype=torch.float).to(device)

        edge_indices_original = [edge_idx.to(device) for edge_idx in augmented_data['edge_indices_original']]
        edge_indices_augmented = [edge_idx.to(device) for edge_idx in augmented_data['edge_indices_augmented']]

        edge_weights_original = None
        edge_weights_augmented = None

        if augmented_data['edge_weights_original'] is not None:
            edge_weights_original = []
            for w in augmented_data['edge_weights_original']:
                if w is not None:
                    edge_weights_original.append(w.to(device))
                else:
                    edge_weights_original.append(None)

        if augmented_data['edge_weights_augmented'] is not None:
            edge_weights_augmented = []
            for w in augmented_data['edge_weights_augmented']:
                if w is not None:
                    edge_weights_augmented.append(w.to(device))
                else:
                    edge_weights_augmented.append(None)

        history = {
            'loss': [], 'train_acc': [], 'val_acc': [],
            'val_precision': [], 'val_recall': [], 'val_f1_macro': []
        }

        min_epochs = self.config.get('min_training_epochs', 300)
        early_stopping = TrainingEarlyStopping(
            patience=self.config.get('early_stopping_patience', 20),
            min_delta=self.config.get('early_stopping_min_delta', 0.001),
            restore_best_weights=True,
            min_epochs=min_epochs
        )

        if not self.is_hyperparameter_search:
            print(f"早停配置: 最少训练{min_epochs}轮, 耐心值={early_stopping.patience}")

        cb_weight = self.config.get('cb_weight', 0.6)
        cl_weight = self.config.get('cl_weight', 0.4)

        if not self.is_hyperparameter_search:
            progress_bar = tqdm(range(self.config['main_epochs']),
                                desc=f"Training Run {run_id}", unit="epoch")
        else:
            progress_bar = range(self.config['main_epochs'])

        for epoch in progress_bar:
            model.train()
            optimizer.zero_grad()

            logits = model(x_augmented, edge_indices_augmented, edge_weights_augmented)
            classification_loss = criterion(logits[data_splits['train_idx']],
                                            data_splits['y_tensor'][data_splits['train_idx']])

            if self.config.get('use_contrastive', False) and isinstance(model, GCNContrastiveModel):
                train_mask = torch.zeros(len(data_splits['y_tensor']), dtype=torch.bool, device=device)
                train_mask[data_splits['train_idx']] = True

                contrastive_loss = model.compute_contrastive_loss(
                    x_original, x_augmented,
                    edge_indices_original, edge_indices_augmented,
                    edge_weights_original, edge_weights_augmented,
                    train_mask
                )
                total_loss = cb_weight * classification_loss + cl_weight * contrastive_loss

                history.setdefault('total_loss', []).append(total_loss.item())
                history.setdefault('classification_loss', []).append(classification_loss.item())
                history.setdefault('contrastive_loss', []).append(contrastive_loss.item())
            else:
                total_loss = classification_loss
                history.setdefault('total_loss', []).append(total_loss.item())
                history.setdefault('classification_loss', []).append(classification_loss.item())
                history.setdefault('contrastive_loss', []).append(0.0)

            total_loss.backward()
            optimizer.step()

            train_pred = logits[data_splits['train_idx']].argmax(dim=1)
            train_acc = (train_pred == data_splits['y_tensor'][data_splits['train_idx']]).float().mean().item()

            model.eval()
            with torch.no_grad():
                val_pred, val_prob = evaluate_model(
                    model, x_augmented, edge_indices_augmented,
                    data_splits['val_idx'], edge_weights_augmented
                )
                val_labels = data_splits['y_tensor'][data_splits['val_idx']].cpu().numpy()
                val_metrics = calculate_metrics(val_labels, val_pred.cpu().numpy(),
                                                val_prob.cpu().numpy(), len(torch.unique(data_splits['y_tensor'])))

                history['loss'].append(total_loss.item())
                history['train_acc'].append(train_acc)
                history['val_acc'].append(val_metrics['acc'])
                history['val_precision'].append(val_metrics['precision_macro'])
                history['val_recall'].append(val_metrics['recall_macro'])
                history['val_f1_macro'].append(val_metrics['f1_macro'])

                if early_stopping(val_metrics['acc'], model, epoch):
                    if not self.is_hyperparameter_search:
                        print(f"\n早停触发! 在第 {epoch + 1} 轮停止训练")
                        print(f"最佳验证准确率: {early_stopping.best_score:.4f} 在第 {early_stopping.best_epoch + 1} 轮")

                    early_stopping.restore_best_model(model)
                    break

            if not self.is_hyperparameter_search and hasattr(progress_bar, 'set_postfix'):
                progress_bar.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'train_acc': f"{train_acc:.4f}",
                    'val_acc': f"{val_metrics['acc']:.4f}",
                    'best_val': f"{early_stopping.best_score:.4f}",
                })

        final_epoch = epoch + 1
        if not early_stopping.early_stop and not self.is_hyperparameter_search:
            print(f"训练完成，共 {final_epoch} 轮")

        training_summary = get_training_summary(early_stopping, final_epoch)
        training_summary['history'] = history
        training_summary['best_model_state'] = model.state_dict()

        return training_summary

    def _evaluate_model(self, model, augmented_data: Dict, data_splits: Dict,
                        training_results: Dict, run_id: int, current_seed: int) -> Dict:
        """评估模型（✓ 集成错误分析记录）"""
        if not self.is_hyperparameter_search:
            print(f"\n在测试集上评估 (运行 {run_id})...")

            model_filename = f'{self.main_result_dir}/best_gcn_model_run{run_id}.pt'
            torch.save(training_results['best_model_state'], model_filename)

            print_training_summary(training_results, run_id)
        else:
            temp_dir = tempfile.mkdtemp(prefix="hp_search_model_")
            model_filename = f'{temp_dir}/best_gcn_model_run{run_id}.pt'
            torch.save(training_results['best_model_state'], model_filename)

        try:
            x_augmented = torch.tensor(augmented_data['X_augmented'], dtype=torch.float).to(device)
            edge_indices_augmented = [edge_idx.to(device) for edge_idx in augmented_data['edge_indices_augmented']]

            edge_weights_augmented = None
            if augmented_data['edge_weights_augmented'] is not None:
                edge_weights_augmented = []
                for w in augmented_data['edge_weights_augmented']:
                    if w is not None:
                        edge_weights_augmented.append(w.to(device))
                    else:
                        edge_weights_augmented.append(None)

            test_pred, test_prob = evaluate_model(
                model, x_augmented, edge_indices_augmented,
                data_splits['test_idx'], edge_weights_augmented
            )
            test_labels = data_splits['y_tensor'][data_splits['test_idx']].cpu().numpy()
            test_metrics = calculate_metrics(test_labels, test_pred.cpu().numpy(),
                                             test_prob.cpu().numpy(), len(torch.unique(data_splits['y_tensor'])))

            # ✓ 记录错误分析数据（仅正常模式）
            if not self.is_hyperparameter_search and self.misclassification_analyzer is not None:
                self.misclassification_analyzer.record_prediction(
                    run_id=run_id,
                    indices=data_splits['test_idx'].cpu().numpy(),
                    labels=test_labels,
                    predictions=test_pred.cpu().numpy(),
                    probabilities=test_prob.cpu().numpy()
                )

            if not self.is_hyperparameter_search:
                print(f"\n测试集指标 (运行 {run_id}):")
                print(f"  ACC: {test_metrics['acc']:.4f}")
                print(f"  Precision: {test_metrics['precision_macro']:.4f}")
                print(f"  Recall: {test_metrics['recall_macro']:.4f}")
                print(f"  F1-m (macro): {test_metrics['f1_macro']:.4f}")
                print(f"  F1-w (weighted): {test_metrics['f1_weighted']:.4f}")

            current_aug_params = get_current_augmentation_params() if self.config.get('use_augmentation', False) else None

            results = {
                'run_id': run_id,
                'seed': current_seed,
                'config': self.config,
                'augmentation_params': current_aug_params,
                'test_metrics': test_metrics,
                'history': training_results['history'],
                'best_epoch': training_results['best_epoch'],
                'final_epoch': training_results['final_epoch'],
                'best_val_acc': training_results['best_val_score'],
                'early_stopped': training_results['early_stopped'],
                'min_epochs_reached': training_results['min_epochs_reached'],
                'timestamp': datetime.now().isoformat(),
                'model_filename': model_filename
            }

            return results

        except Exception as e:
            print(f"评估过程中出现错误: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _organize_results(self, result: Dict, main_result_dir: str, run_id: int) -> None:
        """整理单次实验结果到主目录"""
        save_results(result, f"{main_result_dir}/run_{run_id}_results.json")

        if run_id == 1:
            visualize_training(result['history'], main_result_dir)

    def _finalize_experiments(self, all_results: List[Dict], main_result_dir: str) -> None:
        """完成实验，计算平均结果（集成错误分析 + 增强分析）"""
        from utils.metrics import print_average_results

        avg_results = calculate_average_results(all_results)
        if avg_results:
            save_results(avg_results, f"{main_result_dir}/average_results.json")
            print_average_results(avg_results)

            early_stopped_count = sum(1 for r in all_results if r.get('early_stopped', False))
            min_epochs_reached_count = sum(1 for r in all_results if r.get('min_epochs_reached', True))

            print(f"\n训练统计:")
            print(f"  早停次数: {early_stopped_count}/{len(all_results)}")
            print(f"  达到最少轮数次数: {min_epochs_reached_count}/{len(all_results)}")

            # 执行错误分析
            if self.misclassification_analyzer is not None and self.original_data_info is not None:
                print(f"\n{'=' * 60}")
                print("开始错误样本分析...")
                print(f"{'=' * 60}")

                # 构建训练集掩码
                train_mask = np.zeros(len(self.original_data_info['y_full']), dtype=bool)
                train_mask[:self.original_data_info['num_train']] = True

                # 执行分析
                analysis_result = self.misclassification_analyzer.analyze(
                    y_full=self.original_data_info['y_full'],
                    train_mask=train_mask
                )

                # 打印分析结果
                self.misclassification_analyzer.print_analysis(analysis_result)

                # 保存分析报告
                report_path = f"{main_result_dir}/misclassification_analysis.txt"
                self.misclassification_analyzer.save_report(analysis_result, report_path)
                print(f"\n✓ 错误分析报告已保存到: {report_path}")

                # 保存原始数据信息
                data_info_path = f"{main_result_dir}/original_data_info.npz"
                np.savez(
                    data_info_path,
                    y_full=self.original_data_info['y_full'],
                    train_mask=train_mask,
                    num_omics=self.original_data_info['num_omics'],
                    num_train=self.original_data_info['num_train']
                )

                # 保存每个组学的原始特征矩阵
                omics_data_list = []
                for i in range(self.original_data_info['num_omics']):
                    train_omics = self.original_data_info['train_data'][i]
                    test_omics = self.original_data_info['test_data'][i]
                    full_omics = np.vstack([train_omics, test_omics])
                    omics_data_list.append(full_omics)

                    omics_path = f"{main_result_dir}/omics_{i + 1}_features.npy"
                    np.save(omics_path, full_omics)

                print(f"✓ 原始数据信息已保存到: {data_info_path}")
                print(f"✓ 各组学特征矩阵已保存到: {main_result_dir}/omics_*_features.npy")

                # ========== 新增：执行增强分析 ==========
                # 提取错误样本列表
                error_samples = []
                if 'frequent_errors' in analysis_result:
                    for sample_info in analysis_result['frequent_errors']:
                        error_samples.append({
                            'sample_idx': sample_info['sample_idx'],
                            'true_label': sample_info['true_label'],
                            'error_count': sample_info['error_count'],
                            'total_runs': sample_info['total_runs'],
                            'error_rate': sample_info['error_rate'],
                            'predicted_as': sample_info.get('most_common_prediction')
                        })

                # 运行增强分析（误判相似度 + t-SNE可视化 + 离群点检测）
                if error_samples:
                    run_enhanced_analysis(
                        omics_data=omics_data_list,
                        y_full=self.original_data_info['y_full'],
                        train_mask=train_mask,
                        error_samples=error_samples,
                        output_dir=main_result_dir
                    )
                # ========== 增强分析结束 ==========

            print(f"\n所有结果已保存到: {main_result_dir}")
        else:
            print("无法计算平均结果")