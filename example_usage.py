"""
使用示例
展示如何使用重构后的代码结构
"""

import torch
import numpy as np

# ============================================================================
# 示例1: 基础使用流程
# ============================================================================

def example_basic_usage():
    """基础使用流程示例"""
    print("\n" + "="*60)
    print("示例1: 基础使用流程")
    print("="*60)
    
    # 1. 导入必要模块
    from config import device, update_augmentation_params
    from utils import set_random_seed, load_dataset, construct_adjacency
    from models import ModelFactory
    
    # 2. 设置随机种子
    set_random_seed(42)
    print("✓ 随机种子设置完成")
    
    # 3. 加载数据（这里使用模拟数据）
    print("✓ 数据加载流程（实际使用时替换为真实数据路径）")
    # X_full, y_full, total_features, num_omics, train_data, test_data, num_classes = load_dataset('data/')
    
    # 4. 创建模型
    print("✓ 创建模型...")
    model = ModelFactory.create_model(
        model_type='gcn',
        input_dim=100,  # 示例值
        hidden_dim=64,
        num_omics=3,
        num_classes=5,
        gcn_layers=2
    )
    print(f"  模型类型: {type(model).__name__}")
    print(f"  设备: {device}")


# ============================================================================
# 示例2: 数据增强使用
# ============================================================================

def example_augmentation_usage():
    """数据增强使用示例"""
    print("\n" + "="*60)
    print("示例2: 数据增强使用")
    print("="*60)
    
    from augmentation import DataAugmentor, print_augmentation_effects
    from config import update_augmentation_params
    
    # 1. 设置增强参数
    aug_params = {
        'minority_drop_rate': 0.15,
        'majority_drop_rate': 0.35,
        'edge_add_prob': 0.3,
        'max_new_edges': 50
    }
    update_augmentation_params(aug_params)
    print("✓ 增强参数设置完成")
    
    # 2. 创建增强器
    augmentor = DataAugmentor()
    print("✓ 数据增强器创建完成")
    
    # 3. 应用增强（实际使用时替换为真实数据）
    print("✓ 数据增强流程（实际使用时传入真实数据）")
    # X_aug, edge_indices_aug, aug_stats = augmentor.augment(
    #     X_full, y_full, edge_indices, train_mask, aug_params
    # )
    # print_augmentation_effects(aug_stats, aug_params)


# ============================================================================
# 示例3: 对比学习训练
# ============================================================================

def example_contrastive_training():
    """对比学习训练示例"""
    print("\n" + "="*60)
    print("示例3: 对比学习训练")
    print("="*60)
    
    from models import ModelFactory
    from trainer import train_with_contrastive_learning
    import torch.nn as nn
    
    # 1. 创建对比学习模型
    model = ModelFactory.create_model(
        model_type='gcn_contrastive',
        input_dim=100,
        hidden_dim=64,
        num_omics=3,
        num_classes=5,
        gcn_layers=2,
        embedding_dim=128,
        temperature=0.5
    )
    print("✓ 对比学习模型创建完成")
    
    # 2. 设置优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    print("✓ 优化器和损失函数设置完成")
    
    # 3. 训练（实际使用时传入真实数据）
    print("✓ 训练流程（实际使用时传入真实数据）")
    # train_results = train_with_contrastive_learning(
    #     model, X_full, X_aug, edge_indices, edge_indices_aug,
    #     labels, train_idx, optimizer, criterion,
    #     cb_weight=0.6, cl_weight=0.4
    # )


# ============================================================================
# 示例4: 评估和结果保存
# ============================================================================

def example_evaluation_and_save():
    """评估和结果保存示例"""
    print("\n" + "="*60)
    print("示例4: 评估和结果保存")
    print("="*60)
    
    from utils import calculate_metrics, save_results, create_result_directory
    from trainer import evaluate_model
    
    # 1. 评估模型（实际使用时传入真实数据和模型）
    print("✓ 模型评估流程（实际使用时传入真实数据）")
    # predictions, probabilities = evaluate_model(model, X_full, edge_indices, test_idx)
    
    # 2. 计算指标（示例数据）
    y_true = np.array([0, 1, 2, 0, 1])
    y_pred = np.array([0, 1, 2, 0, 2])
    y_prob = np.array([
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.85, 0.1, 0.05],
        [0.2, 0.3, 0.5]
    ])
    
    metrics = calculate_metrics(y_true, y_pred, y_prob, num_classes=3)
    print("✓ 指标计算完成:")
    print(f"  - Accuracy: {metrics['acc']:.4f}")
    print(f"  - F1-Score (macro): {metrics['f1_macro']:.4f}")
    
    # 3. 保存结果
    result_dir = create_result_directory('example_experiment')
    print(f"✓ 结果目录创建: {result_dir}")
    
    results = {
        'test_metrics': metrics,
        'best_val_acc': 0.85,
        'config': {'lr': 0.001, 'epochs': 100}
    }
    # save_results(results, f"{result_dir}/results.json")
    print("✓ 结果保存流程示例")


# ============================================================================
# 示例5: 配置管理
# ============================================================================

def example_config_management():
    """配置管理示例"""
    print("\n" + "="*60)
    print("示例5: 配置管理")
    print("="*60)
    
    from utils import merge_configs, validate_config, print_config_summary
    
    # 1. 合并配置
    base_config = {
        'data_dir': 'data/',
        'experiment_name': 'test_experiment',
        'hidden_dim': 64,
        'lr': 0.001,
        'main_epochs': 100
    }
    
    aug_config = {
        'use_augmentation': True,
        'minority_drop_rate': 0.15,
        'majority_drop_rate': 0.35
    }
    
    full_config = merge_configs(base_config, aug_config)
    print("✓ 配置合并完成")
    
    # 2. 验证配置
    is_valid = validate_config(full_config)
    print(f"✓ 配置验证: {'通过' if is_valid else '失败'}")
    
    # 3. 打印配置摘要
    print_config_summary(full_config)


# ============================================================================
# 主函数
# ============================================================================

def main():
    """运行所有示例"""
    print("\n" + "="*80)
    print("重构代码使用示例")
    print("="*80)
    
    example_basic_usage()
    example_augmentation_usage()
    example_contrastive_training()
    example_evaluation_and_save()
    example_config_management()
    
    print("\n" + "="*80)
    print("所有示例运行完成！")
    print("="*80)


if __name__ == "__main__":
    main()
