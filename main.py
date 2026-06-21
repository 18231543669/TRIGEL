"""
主训练脚本（集成错误分析和样本检查）
多组学GCN模型训练，支持数据增强和对比学习

修复内容：
1. update_augmentation_params 添加缺失的类别分层参数
2. 调参后更新参数也添加新参数
"""

import time
import sys

# 导入参数解析
from parser1 import parse_arguments

# 导入配置管理
from config import update_augmentation_params

# 导入实验运行器
from models.experiment_runner import ExperimentRunner

# ✓ 导入样本检查器
from utils.sample_checker import check_error_samples


BRCA_20260118_BEST_PRESET = {
    'data_dir': './datasets/BRCA',
    'graph_source': 'knn',
    'seed': 97043,
    'num_runs': 20,
    'experiment_top_k': 10,
    'cosine_k': 10,
    'hidden_dim': 64,
    'lr': 0.001,
    'weight_decay': 1e-4,
    'main_epochs': 400,
    'min_training_epochs': 300,
    'early_stopping_patience': 20,
    'early_stopping_min_delta': 0.001,
    'use_augmentation': True,
    'use_contrastive': True,
    'cb_weight': 0.8,
    'cl_weight': 0.2,
    'temperature': 0.05,
    'embedding_dim': 64,
    'minority_drop_rate': 0.0,
    'majority_drop_rate': 0.3,
    'edge_add_prob': 0.4,
    'max_new_edges': 100,
    'use_edge_weights': True,
    'minority_edge_weight': 2.0,
    'quality_weight_factor': 0.3,
    'fusion_num_heads': 2,
    'fusion_dropout': 0.2,
    'class_tier_severe_threshold': 0.05,
    'class_tier_moderate_threshold': 0.15,
    'class_tier_intermediate_threshold': 0.30,
    'effective_num_beta': 0.99,
}


def apply_brca_20260118_best_preset(config: dict) -> dict:
    """将配置覆盖为 BRCA 2026-01-18 调参最佳组合，确保复现实验口径一致。"""
    config.update(BRCA_20260118_BEST_PRESET)
    if not config.get('experiment_name'):
        config['experiment_name'] = 'BRCA_reproduce_20260118_best'
    return config


def get_augmentation_params(config: dict) -> dict:
    """从配置中提取数据增强参数（包含新增的类别分层参数）
    
    Args:
        config: 配置字典
        
    Returns:
        数据增强参数字典
    """
    return {
        # 原有参数
        'minority_drop_rate': config.get('minority_drop_rate', 0.0),
        'majority_drop_rate': config.get('majority_drop_rate', 0.3),
        'edge_add_prob': config.get('edge_add_prob', 0.5),
        'max_new_edges': config.get('max_new_edges', 120),
        
        # ✨ 新增：类别分层参数
        'class_tier_severe_threshold': config.get('class_tier_severe_threshold', 0.05),
        'class_tier_moderate_threshold': config.get('class_tier_moderate_threshold', 0.15),
        'class_tier_intermediate_threshold': config.get('class_tier_intermediate_threshold', 0.30),
        'effective_num_beta': config.get('effective_num_beta', 0.9999),
    }


def run_auto_hyperparameter_tuning(config: dict) -> dict:
    """执行自动超参数调优

    Args:
        config: 基础配置字典

    Returns:
        最佳参数字典，失败时返回None
    """
    print("\n" + "=" * 60)
    print("启动自动超参数调优模式")
    print("=" * 60)

    try:
        from tuning import auto_tune_hyperparameters
    except ImportError:
        print("错误: 无法导入tuning模块")
        print("请确保tuning模块已正确安装")
        return None

    print("开始超参数搜索...")

    start_time = time.time()
    tune_num_seeds = int(config.get('tune_num_seeds', 5))
    print(f"调参种子数/组合: {tune_num_seeds}")
    tune_results = auto_tune_hyperparameters(base_config=config, num_seeds=tune_num_seeds)
    tune_elapsed = time.time() - start_time

    print(f"\n自动调参完成，耗时: {tune_elapsed / 60:.1f} 分钟")

    if tune_results and 'best_result' in tune_results:
        br = tune_results['best_result']
        best_params = br['params']
        obj = config.get('hyperparam_objective', 'test_acc')
        if obj == 'test_f1_macro':
            print(f"最佳 F1-macro(选优指标): {br['test_f1_macro_mean']:.4f}")
        elif obj == 'test_acc_with_f1_floor':
            f1_floor = float(config.get('hyperparam_f1_floor', 0.37))
            print(
                f"最佳测试准确率(选优指标, F1约束>={f1_floor:.4f}): "
                f"{br['test_acc_mean']:.4f}"
            )
        else:
            print(f"最佳测试准确率(选优指标): {br['test_acc_mean']:.4f}")
        print(
            f"  同组聚合: ACC={br['test_acc_mean']:.4f}±{br['test_acc_std']:.4f}, "
            f"F1-macro={br['test_f1_macro_mean']:.4f}±{br['test_f1_macro_std']:.4f}"
        )
        print("最佳参数组合:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")

        return best_params
    else:
        print("自动调参失败，将使用原始参数")
        return None


def main():
    """主函数 - 实验入口"""
    # 解析命令行参数
    config = parse_arguments()

    # 一键复现历史最佳组合（允许后续命令行参数继续覆盖）
    if config.get('reproduce_brca_20260118_best', False):
        config = apply_brca_20260118_best_preset(config)
        print("已启用复现预设: BRCA_hyperparameter_search_20260118_202538")
        print("复现口径: seed=97043, num_runs=20, top_k=10, best params 全量覆盖")

    # ✨ 使用统一函数更新数据增强参数
    if config.get('use_augmentation', False):
        aug_params = get_augmentation_params(config)
        update_augmentation_params(aug_params)
        print(f"数据增强参数已更新:")
        for key, value in aug_params.items():
            print(f"  {key}: {value}")

    # 打印实验配置概览
    print(f"\n{'=' * 60}")
    print(f"实验配置概览")
    print(f"{'=' * 60}")
    print(f"实验名称: {config['experiment_name']}")
    print(f"数据目录: {config['data_dir']}")
    if config.get('graph_source', 'precomputed') == 'knn':
        print(f"图来源: 在线KNN构图 (cosine_k={config.get('cosine_k', 5)})")
    else:
        print(f"图来源: 预计算adj文件 (adj_dir={config.get('adj_dir') or config.get('data_dir')})")
    print(f"数据增强: {'启用' if config['use_augmentation'] else '禁用'}")
    print(f"对比学习: {'启用' if config['use_contrastive'] else '禁用'}")
    print(f"训练早停: 启用 (最少{config['min_training_epochs']}轮, "
          f"耐心值={config['early_stopping_patience']})")
    print(f"自动调参: {'启用' if config['use_auto_tune'] else '禁用'}")
    
    # ✨ 打印新增参数
    print(f"\n融合模块参数:")
    print(f"  fusion_num_heads: {config.get('fusion_num_heads', 4)}")
    print(f"  fusion_dropout: {config.get('fusion_dropout', 0.3)}")
    print(f"\n类别分层参数:")
    print(f"  effective_num_beta: {config.get('effective_num_beta', 0.9999)}")
    print(f"  严重少数类阈值: {config.get('class_tier_severe_threshold', 0.05)}")
    print(f"  中度少数类阈值: {config.get('class_tier_moderate_threshold', 0.15)}")
    print(f"  中间类阈值: {config.get('class_tier_intermediate_threshold', 0.30)}")

    # ✓ 新增功能标识
    print(f"\n附加功能:")
    print(f"  错误分析: 启用")
    print(f"  样本检查: 启用")
    print(f"{'=' * 60}")

    # 如果启用自动调参，只执行调参并退出
    if config['use_auto_tune']:
        best_params = run_auto_hyperparameter_tuning(config)
        if best_params:
            print("\n自动调参完成，已保存最佳参数与报告。按需使用最佳参数单独启动正式实验。")
            return 0
        print("\n自动调参失败。")
        return 1

    # 运行完整实验
    print(f"\n开始 {config['num_runs']} 次完整实验...")
    runner = ExperimentRunner(config)

    start_time = time.time()
    results = runner.run_multiple_experiments(config['num_runs'])
    experiment_elapsed = time.time() - start_time

    print(f"\n实验耗时: {experiment_elapsed:.2f} 秒")

    # ✓ 运行样本特征检查（如果有错误样本）
    if len(results) > 0 and hasattr(runner, 'main_result_dir'):
        print(f"\n{'=' * 60}")
        print("开始样本特征检查...")
        print(f"{'=' * 60}")

        try:
            check_start = time.time()
            check_error_samples(runner.main_result_dir)
            check_elapsed = time.time() - check_start

            print(f"\n样本检查耗时: {check_elapsed:.2f} 秒")
            print(f"总耗时: {experiment_elapsed + check_elapsed:.2f} 秒")
        except Exception as e:
            print(f"\n⚠️  样本检查失败: {e}")
            print("实验结果已保存，但样本检查未完成")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n总耗时: {experiment_elapsed:.2f} 秒")

    # 返回状态码
    if len(results) == 0:
        print("\n所有实验都失败了")
        return 1
    elif len(results) < config['num_runs']:
        print(f"\n部分实验失败 ({len(results)}/{config['num_runs']} 成功)")
        return 2
    else:
        print("\n所有实验成功完成")
        return 0


if __name__ == "__main__":
    sys.exit(main())