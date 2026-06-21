"""
训练工具模块
包含训练过程中使用的工具类和函数
"""

import torch
from typing import Optional


class TrainingEarlyStopping:
    """训练过程早停机制（最少运行指定轮数）
    
    该类实现了一个早停策略，只有在达到最少训练轮数后才会触发早停。
    在达到最少轮数之前，只记录最佳模型但不触发早停。
    
    Attributes:
        patience: 没有改善的最大容忍轮数
        min_delta: 认定为改善的最小变化量
        restore_best_weights: 是否在早停时恢复最佳权重
        min_epochs: 触发早停前的最少训练轮数
        best_score: 目前为止的最佳验证分数
        best_epoch: 最佳分数对应的轮数
        best_weights: 最佳模型的权重（如果启用保存）
        counter: 没有改善的连续轮数计数器
        early_stop: 是否应该早停的标志
    """

    def __init__(self, 
                 patience: int = 20, 
                 min_delta: float = 0.001,
                 restore_best_weights: bool = True, 
                 min_epochs: int = 300):
        """初始化早停机制
        
        Args:
            patience: 没有改善的最大容忍轮数
            min_delta: 认定为改善的最小变化量
            restore_best_weights: 是否在早停时恢复最佳权重
            min_epochs: 触发早停前的最少训练轮数
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.min_epochs = min_epochs

        self.best_score = -float('inf')
        self.best_epoch = 0
        self.best_weights = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_score: float, model: torch.nn.Module, epoch: int) -> bool:
        """检查是否应该早停
        
        Args:
            val_score: 当前验证分数
            model: 当前模型
            epoch: 当前轮数（从0开始）
            
        Returns:
            是否应该早停
        """
        # 如果还没达到最少轮数，只更新最佳模型，不触发早停
        if epoch < self.min_epochs:
            if val_score > self.best_score:
                self.best_score = val_score
                self.best_epoch = epoch
                self.counter = 0
                if self.restore_best_weights:
                    self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False

        # 达到最少轮数后，正常执行早停逻辑
        if val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def restore_best_model(self, model: torch.nn.Module) -> None:
        """恢复最佳模型权重
        
        Args:
            model: 要恢复权重的模型
        """
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)

    def get_status_message(self, epoch: int) -> str:
        """获取当前早停状态的描述信息
        
        Args:
            epoch: 当前轮数
            
        Returns:
            状态描述字符串
        """
        if epoch < self.min_epochs:
            return "warming up"
        else:
            return "ready for early stop"

    def get_patience_info(self, epoch: int) -> str:
        """获取耐心值信息
        
        Args:
            epoch: 当前轮数
            
        Returns:
            耐心值信息字符串
        """
        if epoch < self.min_epochs:
            return "N/A"
        else:
            return f"{self.counter}/{self.patience}"


def get_training_summary(early_stopping: TrainingEarlyStopping, final_epoch: int) -> dict:
    """获取训练总结信息
    
    Args:
        early_stopping: 早停对象
        final_epoch: 最终训练轮数
        
    Returns:
        包含训练总结信息的字典
    """
    return {
        'best_val_score': early_stopping.best_score,
        'best_epoch': early_stopping.best_epoch,
        'final_epoch': final_epoch,
        'early_stopped': early_stopping.early_stop,
        'min_epochs_reached': final_epoch >= early_stopping.min_epochs,
        'total_patience_count': early_stopping.counter
    }


def print_training_summary(summary: dict, run_id: Optional[int] = None) -> None:
    """打印训练总结信息
    
    Args:
        summary: 训练总结字典
        run_id: 运行ID（可选）
    """
    prefix = f"训练状态 (运行 {run_id}): " if run_id else "训练状态: "
    status_msg = "早停" if summary['early_stopped'] else "完整训练"
    min_epochs_msg = "✓" if summary['min_epochs_reached'] else "✗"
    
    print(f"{prefix}{status_msg} | 最少轮数达标: {min_epochs_msg}")
    print(f"最佳验证准确率: {summary['best_val_score']:.4f} 在第 {summary['best_epoch'] + 1} 轮")
    print(f"实际训练轮数: {summary['final_epoch']}")
