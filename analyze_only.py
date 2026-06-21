"""
简化版：仅分析数据集，不修改文件（用于预览）
"""

import os
import pandas as pd
import numpy as np


def analyze_dataset(data_dir: str):
    """
    分析数据集并预览将要进行的操作
    """
    print("="*60)
    print("数据集分析（预览模式）")
    print("="*60)
    
    # 加载标签
    print("\n加载标签文件...")
    labels_tr = pd.read_csv(os.path.join(data_dir, 'labels_tr.csv'), header=None).values.flatten()
    labels_te = pd.read_csv(os.path.join(data_dir, 'labels_te.csv'), header=None).values.flatten()
    
    # 合并所有标签
    y_full = np.concatenate([labels_tr, labels_te])
    
    # 分析类别分布
    unique_classes, class_counts = np.unique(y_full, return_counts=True)
    
    print(f"\n当前数据集信息：")
    print(f"  训练集样本数: {len(labels_tr)}")
    print(f"  测试集样本数: {len(labels_te)}")
    print(f"  总样本数: {len(y_full)}")
    print(f"  类别数: {len(unique_classes)}")
    
    print(f"\n类别分布：")
    for cls, count in zip(unique_classes, class_counts):
        percentage = count/len(y_full)*100
        print(f"  类别 {int(cls)}: {count:4d} 个样本 ({percentage:5.2f}%)")
    
    # 找出样本数最少的两个类别
    sorted_indices = np.argsort(class_counts)
    classes_to_remove = unique_classes[sorted_indices[:2]]
    classes_to_keep = unique_classes[sorted_indices[2:]]
    
    print(f"\n" + "-"*60)
    print("预计操作：")
    print("-"*60)
    
    print(f"\n将要删除的类别（样本数最少的两个）：")
    for cls in classes_to_remove:
        idx = np.where(unique_classes == cls)[0][0]
        print(f"  ❌ 类别 {int(cls)}: {class_counts[idx]} 个样本")
    
    print(f"\n将要保留的类别：")
    for cls in classes_to_keep:
        idx = np.where(unique_classes == cls)[0][0]
        print(f"  ✅ 类别 {int(cls)}: {class_counts[idx]} 个样本")
    
    # 计算保留的样本数
    mask_tr = np.isin(labels_tr, classes_to_keep)
    mask_te = np.isin(labels_te, classes_to_keep)
    
    # 重新映射标签
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(classes_to_keep))}
    
    print(f"\n标签重映射：")
    for old_label, new_label in label_mapping.items():
        old_count = class_counts[unique_classes == old_label][0]
        print(f"  原标签 {int(old_label)} ({old_count}样本) -> 新标签 {new_label}")
    
    print(f"\n处理后的数据集预览：")
    print(f"  训练集样本数: {len(labels_tr)} -> {mask_tr.sum()} (删除 {len(labels_tr)-mask_tr.sum()})")
    print(f"  测试集样本数: {len(labels_te)} -> {mask_te.sum()} (删除 {len(labels_te)-mask_te.sum()})")
    print(f"  总样本数: {len(y_full)} -> {mask_tr.sum()+mask_te.sum()} (删除 {len(y_full)-(mask_tr.sum()+mask_te.sum())})")
    print(f"  类别数: {len(unique_classes)} -> 3")
    
    # 显示新的类别分布
    new_labels_tr = np.array([label_mapping[label] for label in labels_tr[mask_tr]])
    new_labels_te = np.array([label_mapping[label] for label in labels_te[mask_te]])
    new_y_full = np.concatenate([new_labels_tr, new_labels_te])
    new_unique, new_counts = np.unique(new_y_full, return_counts=True)
    
    print(f"\n新的类别分布：")
    for cls, count in zip(new_unique, new_counts):
        percentage = count/len(new_y_full)*100
        print(f"  类别 {int(cls)}: {count:4d} 个样本 ({percentage:5.2f}%)")
    
    print("\n" + "="*60)
    print("分析完成！这只是预览，没有修改任何文件。")
    print("如需执行实际操作，请运行 reduce_classes.py")
    print("="*60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "./datasets/Lung"
    
    print(f"数据目录: {data_dir}\n")
    
    try:
        analyze_dataset(data_dir)
    except FileNotFoundError as e:
        print(f"\n错误: 找不到文件 - {e}")
        print("请确保数据目录路径正确。")
    except Exception as e:
        print(f"\n错误: {e}")
