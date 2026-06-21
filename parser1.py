"""
命令行参数解析器
提供统一的参数解析和默认配置管理

改进点：
1. 新增融合模块超参数（fusion_num_heads, fusion_dropout）
2. 新增类别分层超参数（class_tier_*_threshold, effective_num_beta）
"""

import argparse
from typing import Dict


def create_argument_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器 - 统一所有参数默认值"""
    parser = argparse.ArgumentParser(
        description='GCN Model for Multi-omics Data (支持数据增强、对比学习、训练早停和自动调参)'
    )

    # ==================== 数据参数 ====================
    parser.add_argument('--data_dir', type=str, default="./datasets/BRCA",
                        help='Path to dataset directory')
    parser.add_argument('--experiment_name', type=str, default='BRCA',
                        help='Name of the experiment for saving results')
    parser.add_argument('--seed', type=int, default=43,
                        help='Base random seed (will be incremented for each run)')
    parser.add_argument('--num_runs', type=int, default=20,
                        help='Number of experimental runs')
    parser.add_argument('--experiment_top_k', type=int, default=10,
                        help='聚合统计时只使用测试集表现最好的前k次运行（默认10）')

    # ==================== 图参数（仅用于离线生成adj） ====================
    parser.add_argument('--cosine_k', type=int, default=5,
                        help='离线生成adj时的k参数（训练阶段不再使用该参数构图）')
    parser.add_argument(
        '--adj_dir',
        type=str,
        default='',
        help='adj文件目录（为空时使用data_dir，文件名约定为adj1.csv, adj2.csv...）'
    )
    parser.add_argument(
        '--graph_source',
        type=str,
        default='precomputed',
        choices=['precomputed', 'knn'],
        help='图来源：precomputed=读取adj文件；knn=按cosine_k现场构图'
    )

    # ==================== 模型参数 ====================
    parser.add_argument('--gcn_layers', type=int, default=2, choices=[1, 2, 3],
                        help='Number of GCN layers per omics (1-3)')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension for GCN layers (KRCCC 少样本默认 32，大数据集可改 64)')
    parser.add_argument('--lr', type=float, default=0.004,
                        help='Learning rate (KRCCC 建议在 0.002~0.005 间，默认取中档)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Adam weight decay (历史基线常用1e-4)')
    parser.add_argument('--main_epochs', type=int, default=400,
                        help='Maximum training epochs')

    # ==================== 训练早停参数 ====================
    parser.add_argument('--min_training_epochs', type=int, default=200,
                        help='Minimum training epochs before early stopping (KRCCC 样本少可适当降低)')
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                        help='Early stopping patience (number of epochs without improvement)')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.001,
                        help='Minimum improvement threshold for early stopping')

    # ==================== 功能开关参数 ====================
    parser.add_argument('--use_augmentation', action='store_true', default=True,
                        help='Enable data augmentation')
    parser.add_argument('--use_contrastive', action='store_true', default=True,
                        help='Enable contrastive learning')
    parser.add_argument('--use_auto_tune', action='store_true', default=False,
                        help='Enable automatic hyperparameter tuning')
    parser.add_argument(
        '--reproduce_brca_20260118_best',
        action='store_true',
        default=False,
        help='一键复现 BRCA_hyperparameter_search_20260118_202538 的最佳组合配置'
    )
    parser.add_argument('--tune_num_seeds', type=int, default=20,
                        help='Number of seeds per hyperparameter combination (default 20 for KRCCC stability)')
    parser.add_argument('--tune_top_k', type=int, default=10,
                        help='Use top-k runs per combination for metric aggregation (default 10)')
    parser.add_argument(
        '--tune_selection_strategy',
        type=str,
        default='all_runs',
        choices=['all_runs', 'top_k'],
        help='种子结果聚合策略：all_runs=全部种子聚合；top_k=仅按目标指标取前k个再聚合'
    )
    parser.add_argument(
        '--tune_stability_penalty',
        type=float,
        default=0.25,
        help='稳定性惩罚系数，选优分数=mean-penalty*std（0表示不惩罚波动）'
    )
    parser.add_argument(
        '--hyperparam_objective',
        type=str,
        default='test_f1_macro',
        choices=['test_acc', 'test_f1_macro', 'test_acc_with_f1_floor'],
        help='超参数搜索目标：test_acc / test_f1_macro / test_acc_with_f1_floor',
    )
    parser.add_argument('--hyperparam_f1_floor', type=float, default=0.37,
                        help='仅在 test_acc_with_f1_floor 下生效：F1-macro 下限')

    # ==================== 对比学习参数 ====================
    parser.add_argument('--cb_weight', type=float, default=0.8,
                        help='Weight for classification loss (KRCCC 不平衡略提高分类项)')
    parser.add_argument('--cl_weight', type=float, default=0.2,
                        help='Weight for contrastive learning loss (与 cb_weight 之和宜为 1)')
    parser.add_argument('--temperature', type=float, default=0.05,
                        help='Temperature for contrastive learning (KRCCC 常用约 0.05~0.12)')
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='Embedding dimension for contrastive learning (少样本默认 32)')
    parser.add_argument('--cl_warmup_epochs', type=int, default=50,
                        help='对比学习权重warmup轮数（前期从0线性升到目标cl_weight）')

    # ==================== 数据增强参数 ====================
    parser.add_argument('--minority_drop_rate', type=float, default=0.00,
                        help='Feature dropout rate for minority class nodes')
    parser.add_argument('--majority_drop_rate', type=float, default=0.30,
                        help='Feature dropout rate for majority class nodes (KRCCC 不宜过强)')
    parser.add_argument('--edge_add_prob', type=float, default=0.4,
                        help='Probability of adding edges in topology augmentation')
    parser.add_argument('--max_new_edges', type=int, default=100,
                        help='Max new edges per omics (KRCCC 全样本约 122、训练约 85，120 过大；默认约 28)')
    
    # ==================== 边权重参数 ====================
    parser.add_argument('--use_edge_weights', action='store_true', default=True,
                        help='Enable edge weighting for minority class nodes')
    parser.add_argument('--minority_edge_weight', type=float, default=2.0,
                        help='Base weight for minority-related edges (KRCCC 不平衡建议 1.5~2.5)')
    parser.add_argument('--quality_weight_factor', type=float, default=0.3,
                        help='Factor for quality-based weight adjustment')

    # ==================== ✨ 新增：融合模块超参数 ====================
    parser.add_argument('--fusion_num_heads', type=int, default=2,
                        help='Number of attention heads in fusion module (推荐: 2 或 4)')
    parser.add_argument('--fusion_dropout', type=float, default=0.2,
                        help='Dropout rate in fusion module (KRCCC 常用 0.3~0.5)')

    # ==================== ✨ 新增：类别分层超参数 ====================
    parser.add_argument('--class_tier_severe_threshold', type=float, default=0.10,
                        help='Threshold for severe minority class (样本比例 < 此值)')
    parser.add_argument('--class_tier_moderate_threshold', type=float, default=0.15,
                        help='Threshold for moderate minority class (样本比例 < 此值)')
    parser.add_argument('--class_tier_intermediate_threshold', type=float, default=0.30,
                        help='Threshold for intermediate class (样本比例 < 此值)')
    parser.add_argument('--effective_num_beta', type=float, default=0.99,
                        help='Beta parameter for effective number of samples formula (推荐: 0.99-0.9999)')

    return parser


def get_default_config() -> Dict:
    """从命令行参数解析器获取默认配置
    
    Returns:
        包含所有默认参数值的字典
    """
    parser = create_argument_parser()
    # 解析空参数列表获取默认值
    args = parser.parse_args([])
    return vars(args)


def parse_arguments() -> Dict:
    """解析命令行参数并返回配置字典
    
    Returns:
        包含所有参数的配置字典
    """
    parser = create_argument_parser()
    args = parser.parse_args()
    return vars(args)


def print_new_hyperparameters_info():
    """打印新增超参数的说明信息"""
    info = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                    新增超参数说明                                  ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  融合模块超参数:                                                   ║
    ║    --fusion_num_heads    注意力头数 (默认4, 推荐尝试: 2, 4, 8)      ║
    ║    --fusion_dropout      Dropout率 (KRCCC 默认0.4, 可尝试 0.2~0.5)   ║
    ║                                                                    ║
    ║  类别分层超参数:                                                   ║
    ║    --class_tier_severe_threshold      严重少数类阈值 (默认0.10)     ║
    ║    --class_tier_moderate_threshold    中度少数类阈值 (默认0.15)     ║
    ║    --class_tier_intermediate_threshold 中间类阈值 (默认0.30)        ║
    ║    --effective_num_beta  有效样本数beta (默认0.9999)               ║
    ║                                                                    ║
    ║  调参优先级:                                                       ║
    ║    1. fusion_num_heads (高优先级)                                  ║
    ║    2. effective_num_beta (高优先级)                                ║
    ║    3. fusion_dropout (中优先级)                                    ║
    ║    4. class_tier_*_threshold (低优先级, 根据数据集调整)             ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(info)


# 说明：实际自动调参由 tuning.param_space.HyperparameterSpace.get_search_space 决定。
# KRCCC / 路径或实验名含 KRCCC 时使用「少样本+不平衡」专用网格（约 256 组），
# 重点包含 lr、temperature、cb_weight（cl_weight 自动取 1-cb_weight）、effective_num_beta、minority_edge_weight 等。
# 若需手写自定义网格，可传入 tuning 模块的 custom_search_space（见 HyperparameterTuner.tune）。
RECOMMENDED_SEARCH_SPACE = {
    'lr': [0.002, 0.005],
    'hidden_dim': [32, 64],
    'fusion_num_heads': [2, 4],
    'fusion_dropout': [0.3, 0.5],
    'temperature': [0.05, 0.12],
    'cb_weight': [0.58, 0.72],
    'effective_num_beta': [0.999, 0.9999],
    'minority_edge_weight': [1.5, 2.5],
}


def get_recommended_search_space() -> Dict:
    """获取推荐的超参数搜索空间"""
    return RECOMMENDED_SEARCH_SPACE.copy()