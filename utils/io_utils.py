"""
IO和配置管理工具模块
包含文件操作、结果保存和配置验证等功能
"""

import os
import json
from datetime import datetime
from typing import Dict


def create_result_directory(experiment_name: str) -> str:
    """创建结果保存目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"results/{experiment_name}_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def save_results(results: Dict, filepath: str) -> None:
    """保存实验结果到JSON文件"""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=4, default=str)


def load_results(filepath: str) -> Dict:
    """从JSON文件加载实验结果"""
    with open(filepath, 'r') as f:
        return json.load(f)


def merge_configs(*configs: Dict) -> Dict:
    """合并多个配置字典"""
    merged = {}
    for config in configs:
        if config:
            merged.update(config)
    return merged


def validate_config(config: Dict) -> bool:
    """验证配置参数的合法性"""
    required_keys = ['data_dir', 'experiment_name', 'hidden_dim', 'lr', 'main_epochs']

    for key in required_keys:
        if key not in config:
            print(f"配置中缺少必需参数: {key}")
            return False

    # 验证数据类型和范围
    if config.get('lr', 0) <= 0 or config.get('lr', 0) > 1:
        print("学习率应在 (0, 1] 范围内")
        return False

    if config.get('main_epochs', 0) <= 0:
        print("训练轮数应为正整数")
        return False

    return True


def print_config_summary(config: Dict) -> None:
    """打印配置摘要"""
    print(f"\n{'=' * 60}")
    print("实验配置摘要")
    print(f"{'=' * 60}")

    print(f"实验名称: {config.get('experiment_name', 'N/A')}")
    print(f"数据目录: {config.get('data_dir', 'N/A')}")
    print(f"随机种子: {config.get('seed', 'N/A')}")
    print(f"数据增强: {'启用' if config.get('use_augmentation', False) else '禁用'}")
    print(f"对比学习: {'启用' if config.get('use_contrastive', False) else '禁用'}")

    print(f"\n模型参数:")
    print(f"  GCN层数: {config.get('gcn_layers', 'N/A')}")
    print(f"  隐藏维度: {config.get('hidden_dim', 'N/A')}")
    print(f"  学习率: {config.get('lr', 'N/A')}")
    print(f"  训练轮数: {config.get('main_epochs', 'N/A')}")

    if config.get('use_contrastive', False):
        print(f"\n对比学习参数:")
        print(f"  分类损失权重: {config.get('cb_weight', 'N/A')}")
        print(f"  对比学习权重: {config.get('cl_weight', 'N/A')}")
        print(f"  温度参数: {config.get('temperature', 'N/A')}")
        print(f"  嵌入维度: {config.get('embedding_dim', 'N/A')}")

    if config.get('use_augmentation', False):
        print(f"\n数据增强参数:")
        print(f"  少数类丢弃率: {config.get('minority_drop_rate', 'N/A')}")
        print(f"  多数类丢弃率: {config.get('majority_drop_rate', 'N/A')}")
        print(f"  边添加概率: {config.get('edge_add_prob', 'N/A')}")
        print(f"  最大新增边数: {config.get('max_new_edges', 'N/A')}")

    print(f"{'=' * 60}")
