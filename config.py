"""
全局配置文件
包含设备配置、环境变量和数据增强参数管理
"""

import os
import torch
import warnings
from typing import Dict
from utils.data_utils import set_random_seed

# =============================================================================
# 设备配置
# =============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# 环境配置
# =============================================================================

# 解决joblib警告
os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count() or 4)
warnings.filterwarnings("ignore", category=UserWarning, module="joblib.externals.loky")

# =============================================================================
# 数据增强参数管理
# =============================================================================

# 用于动态更新的全局变量 - 将通过命令行参数初始化
CURRENT_AUGMENTATION_PARAMS = {}


def update_augmentation_params(new_params: Dict) -> None:
    """动态更新数据增强参数"""
    global CURRENT_AUGMENTATION_PARAMS
    CURRENT_AUGMENTATION_PARAMS.update(new_params)


def get_current_augmentation_params() -> Dict:
    """获取当前数据增强参数"""
    return CURRENT_AUGMENTATION_PARAMS.copy()
