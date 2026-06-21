"""
可视化工具模块
包含训练过程可视化等功能
"""

import matplotlib.pyplot as plt
from typing import Dict


def visualize_training(history: Dict, save_dir: str) -> None:
    """可视化训练过程"""
    plt.figure(figsize=(15, 10))

    # 损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # 准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # 精确率曲线
    plt.subplot(2, 2, 3)
    plt.plot(history['val_precision'], label='Validation Precision')
    plt.title('Validation Precision (macro)')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()

    # F1分数曲线
    plt.subplot(2, 2, 4)
    plt.plot(history['val_f1_macro'], label='Validation F1 Score')
    plt.title('Validation F1 Score (macro)')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curves.png")
    plt.close()
