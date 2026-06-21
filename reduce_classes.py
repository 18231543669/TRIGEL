"""
删除样本数最少的两个类别，并重新保存数据集
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path


def reduce_dataset_classes(data_dir: str, output_dir: str = None):
    """
    删除样本数最少的两个类别，并重新保存数据集
    
    Args:
        data_dir: 原始数据集目录
        output_dir: 输出目录，如果为None则覆盖原文件
    """
    if output_dir is None:
        output_dir = data_dir
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("删除少数类别并重建数据集")
    print("="*60)
    
    # 1. 加载标签
    print("\n步骤 1: 加载标签文件...")
    labels_tr = pd.read_csv(os.path.join(data_dir, 'labels_tr.csv'), header=None).values.flatten()
    labels_te = pd.read_csv(os.path.join(data_dir, 'labels_te.csv'), header=None).values.flatten()
    
    # 合并所有标签
    y_full = np.concatenate([labels_tr, labels_te])
    n_train = len(labels_tr)
    n_test = len(labels_te)
    
    # 2. 分析类别分布
    print("\n步骤 2: 分析类别分布...")
    unique_classes, class_counts = np.unique(y_full, return_counts=True)
    
    print(f"\n原始数据集信息：")
    print(f"  训练集样本数: {n_train}")
    print(f"  测试集样本数: {n_test}")
    print(f"  总样本数: {len(y_full)}")
    print(f"  类别数: {len(unique_classes)}")
    print(f"\n原始类别分布：")
    for cls, count in zip(unique_classes, class_counts):
        print(f"  类别 {int(cls)}: {count} 个样本 ({count/len(y_full)*100:.2f}%)")
    
    # 3. 找出样本数最少的两个类别
    print("\n步骤 3: 识别要删除的类别...")
    sorted_indices = np.argsort(class_counts)
    classes_to_remove = unique_classes[sorted_indices[:2]]
    classes_to_keep = unique_classes[sorted_indices[2:]]
    
    print(f"\n将要删除的类别（样本数最少的两个）：")
    for cls in classes_to_remove:
        idx = np.where(unique_classes == cls)[0][0]
        print(f"  类别 {int(cls)}: {class_counts[idx]} 个样本")
    
    print(f"\n保留的类别：")
    for cls in classes_to_keep:
        idx = np.where(unique_classes == cls)[0][0]
        print(f"  类别 {int(cls)}: {class_counts[idx]} 个样本")
    
    # 4. 创建掩码
    print("\n步骤 4: 创建样本掩码...")
    # 训练集掩码
    mask_tr = np.isin(labels_tr, classes_to_keep)
    # 测试集掩码
    mask_te = np.isin(labels_te, classes_to_keep)
    
    print(f"  训练集: 保留 {mask_tr.sum()}/{len(mask_tr)} 个样本")
    print(f"  测试集: 保留 {mask_te.sum()}/{len(mask_te)} 个样本")
    
    # 5. 重新映射标签到0, 1, 2
    print("\n步骤 5: 重新映射标签...")
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(classes_to_keep))}
    
    print(f"  标签映射关系：")
    for old_label, new_label in label_mapping.items():
        print(f"    原标签 {int(old_label)} -> 新标签 {new_label}")
    
    # 应用映射
    new_labels_tr = np.array([label_mapping[label] for label in labels_tr[mask_tr]])
    new_labels_te = np.array([label_mapping[label] for label in labels_te[mask_te]])
    
    # 6. 处理组学数据文件
    print("\n步骤 6: 处理组学数据文件...")
    
    # 查找组学文件
    import glob
    omics_files = glob.glob(os.path.join(data_dir, '*_featname.csv'))
    num_omics = len(omics_files)
    print(f"  发现 {num_omics} 个组学数据集")
    
    for i in range(1, num_omics + 1):
        print(f"\n  处理组学 {i}...")
        
        # 加载训练和测试数据
        tr_data = pd.read_csv(os.path.join(data_dir, f'{i}_tr.csv'), header=None)
        te_data = pd.read_csv(os.path.join(data_dir, f'{i}_te.csv'), header=None)
        
        # 应用掩码
        tr_data_filtered = tr_data.iloc[mask_tr]
        te_data_filtered = te_data.iloc[mask_te]
        
        print(f"    训练数据: {tr_data.shape} -> {tr_data_filtered.shape}")
        print(f"    测试数据: {te_data.shape} -> {te_data_filtered.shape}")
        
        # 保存过滤后的数据
        tr_data_filtered.to_csv(os.path.join(output_dir, f'{i}_tr.csv'), 
                               header=False, index=False)
        te_data_filtered.to_csv(os.path.join(output_dir, f'{i}_te.csv'), 
                               header=False, index=False)
        
        # 复制特征名文件（不需要修改）
        featname = pd.read_csv(os.path.join(data_dir, f'{i}_featname.csv'), header=None)
        featname.to_csv(os.path.join(output_dir, f'{i}_featname.csv'), 
                       header=False, index=False)
        print(f"    特征名文件已复制")
    
    # 7. 保存新的标签文件
    print("\n步骤 7: 保存新的标签文件...")
    pd.DataFrame(new_labels_tr).to_csv(os.path.join(output_dir, 'labels_tr.csv'), 
                                       header=False, index=False)
    pd.DataFrame(new_labels_te).to_csv(os.path.join(output_dir, 'labels_te.csv'), 
                                       header=False, index=False)
    
    # 8. 处理其他文件
    print("\n步骤 8: 处理其他辅助文件...")
    
    # 复制npy文件（如果存在）
    for i in range(1, num_omics + 1):
        npy_file = os.path.join(data_dir, f'{i}.npy')
        if os.path.exists(npy_file):
            npy_data = np.load(npy_file)
            # 合并训练和测试掩码
            full_mask = np.concatenate([mask_tr, mask_te])
            npy_filtered = npy_data[full_mask]
            np.save(os.path.join(output_dir, f'{i}.npy'), npy_filtered)
            print(f"  {i}.npy: {npy_data.shape} -> {npy_filtered.shape}")
    
    # 处理labels.npy（如果存在）
    labels_npy_file = os.path.join(data_dir, 'labels.npy')
    if os.path.exists(labels_npy_file):
        labels_npy = np.load(labels_npy_file)
        full_mask = np.concatenate([mask_tr, mask_te])
        labels_filtered = labels_npy[full_mask]
        # 重新映射标签
        labels_filtered_mapped = np.array([label_mapping[label] for label in labels_filtered])
        np.save(os.path.join(output_dir, 'labels.npy'), labels_filtered_mapped)
        print(f"  labels.npy: {labels_npy.shape} -> {labels_filtered_mapped.shape}")
    
    # 处理sample_ids.csv（如果存在）
    sample_ids_file = os.path.join(data_dir, 'sample_ids.csv')
    if os.path.exists(sample_ids_file):
        sample_ids = pd.read_csv(sample_ids_file, header=None)
        full_mask = np.concatenate([mask_tr, mask_te])
        sample_ids_filtered = sample_ids.iloc[full_mask]
        sample_ids_filtered.to_csv(os.path.join(output_dir, 'sample_ids.csv'), 
                                   header=False, index=False)
        print(f"  sample_ids.csv: {sample_ids.shape} -> {sample_ids_filtered.shape}")
    
    # 处理class_weights.csv（如果存在）
    class_weights_file = os.path.join(data_dir, 'class_weights.csv')
    if os.path.exists(class_weights_file):
        # 重新计算类别权重
        y_new = np.concatenate([new_labels_tr, new_labels_te])
        unique_new, counts_new = np.unique(y_new, return_counts=True)
        total_samples = len(y_new)
        n_classes = len(unique_new)
        
        # 计算新的类别权重
        class_weights = total_samples / (n_classes * counts_new)
        
        pd.DataFrame(class_weights).to_csv(os.path.join(output_dir, 'class_weights.csv'), 
                                          header=False, index=False)
        print(f"  class_weights.csv: 已重新计算")
        for cls, weight in zip(unique_new, class_weights):
            print(f"    类别 {cls}: 权重 = {weight:.4f}")
    
    # 处理label_mapping.csv（如果存在）
    label_mapping_file = os.path.join(data_dir, 'label_mapping.csv')
    if os.path.exists(label_mapping_file):
        # 创建新的标签映射
        new_mapping_df = pd.DataFrame([
            {'old_label': int(old), 'new_label': new} 
            for old, new in label_mapping.items()
        ])
        new_mapping_df.to_csv(os.path.join(output_dir, 'label_mapping.csv'), index=False)
        print(f"  label_mapping.csv: 已更新")
    
    # 9. 显示最终结果
    print("\n" + "="*60)
    print("处理完成！")
    print("="*60)
    
    # 验证新数据集
    new_y_full = np.concatenate([new_labels_tr, new_labels_te])
    new_unique, new_counts = np.unique(new_y_full, return_counts=True)
    
    print(f"\n新数据集信息：")
    print(f"  训练集样本数: {len(new_labels_tr)}")
    print(f"  测试集样本数: {len(new_labels_te)}")
    print(f"  总样本数: {len(new_y_full)}")
    print(f"  类别数: {len(new_unique)}")
    print(f"\n新类别分布：")
    for cls, count in zip(new_unique, new_counts):
        print(f"  类别 {int(cls)}: {count} 个样本 ({count/len(new_y_full)*100:.2f}%)")
    
    print(f"\n所有文件已保存到: {output_dir}")
    
    return {
        'classes_removed': classes_to_remove,
        'classes_kept': classes_to_keep,
        'label_mapping': label_mapping,
        'original_counts': dict(zip(unique_classes, class_counts)),
        'new_counts': dict(zip(new_unique, new_counts)),
        'n_train_original': n_train,
        'n_test_original': n_test,
        'n_train_new': len(new_labels_tr),
        'n_test_new': len(new_labels_te)
    }


if __name__ == "__main__":
    import sys
    
    # 使用示例
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "./datasets/Lung"  # 默认路径
    
    # 如果想保存到新目录，取消注释下面这行
    output_dir = "./datasets/Lung_3classes"
    # output_dir = None  # 覆盖原文件
    
    print(f"数据目录: {data_dir}")
    if output_dir:
        print(f"输出目录: {output_dir}")
    else:
        print(f"将覆盖原文件")
    
    # 执行处理
    result = reduce_dataset_classes(data_dir, output_dir)
    
    print("\n处理摘要：")
    print(f"删除的类别: {result['classes_removed']}")
    print(f"保留的类别: {result['classes_kept']}")
    print(f"样本数变化: {result['n_train_original']+result['n_test_original']} -> {result['n_train_new']+result['n_test_new']}")
