import numpy as np
import pandas as pd
import os
import glob
from typing import Tuple, List


def prepare_multi_omics_dataset(
    omics_files: List[str],
    labels_file: str,
    output_dir: str,
    train_size: int = 70,
    auto_transpose: bool = True,  # 自动检测是否需要转置
):
    """
    准备多组学数据集，使用固定划分（与BRCA模型代码一致）
    
    参数说明:
    - omics_files: 组学特征矩阵文件路径列表
    - labels_file: 标签文件路径
    - output_dir: 输出目录
    - train_size: 训练集样本数（默认70，前train_size个样本为训练集）
    - auto_transpose: 是否自动检测并转置数据（默认True）
    
    数据处理流程（与BRCA模型完全一致）:
    1. 加载数据并自动检测格式
    2. 每个组学数据单独标准化
    3. 合并所有组学后再次整体标准化
    4. 固定划分：前train_size个样本为训练集，其余为测试集
    """
    print("="*50)
    print("开始准备多组学数据集（固定划分模式）...")
    print("="*50)
    
    # 1. 先加载标签以确定样本数
    print("\n[1/6] 加载标签以确定样本数...")
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"文件不存在: {labels_file}")
    
    print(f"   标签文件: {os.path.basename(labels_file)}")
    try:
        labels = np.loadtxt(labels_file, dtype=np.str_)
        labels = np.array(labels).astype(np.float32)
        num_samples = len(labels)
        print(f"   样本数: {num_samples}")
    except Exception as e:
        raise ValueError(f"加载标签文件失败 {labels_file}: {str(e)}") from e
    
    # 2. 加载并预处理组学数据
    print("\n[2/6] 加载并预处理组学特征矩阵...")
    omics_data = []
    
    for i, file_path in enumerate(omics_files, 1):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        print(f"\n   组学 {i}: {os.path.basename(file_path)}")
        try:
            # 加载数据
            data = np.loadtxt(file_path, dtype=np.str_)
            data = np.array(data).astype(np.float32)
            
            print(f"      原始形状: {data.shape}")
            
            # 自动检测是否需要转置
            if auto_transpose:
                # 如果第一个维度不等于样本数，则需要转置
                if data.shape[0] != num_samples:
                    print(f"      ⚠️ 检测到维度不匹配，进行转置...")
                    data = data.T
                    print(f"      转置后形状: {data.shape}")
                else:
                    print(f"      ✓ 维度匹配，无需转置")
            
            # 验证样本数
            if data.shape[0] != num_samples:
                raise ValueError(
                    f"组学 {i} 的样本数 ({data.shape[0]}) 与标签样本数 ({num_samples}) 不匹配！\n"
                    f"请检查数据格式或设置 auto_transpose=False 手动处理"
                )
            
            # 单独标准化每个组学（与BRCA模型一致）
            data = data - data.mean(axis=0, keepdims=True)
            data = data / np.sqrt(data.var(axis=0, keepdims=True) + 1e-8)  # 添加小值避免除零
            
            omics_data.append(data)
            print(f"      ✓ 已标准化 - 特征数: {data.shape[1]}")
            
        except Exception as e:
            raise ValueError(f"加载文件失败 {file_path}: {str(e)}\n"
                           f"请确保文件是数值矩阵，使用空格或制表符分隔") from e
    
    # 3. 合并所有组学特征
    print("\n[3/6] 合并组学特征...")
    X_combined = np.concatenate(omics_data, axis=1)
    print(f"   合并后形状: {X_combined.shape}")
    print(f"   总特征数: {X_combined.shape[1]}")
    
    # 整体标准化（与BRCA模型一致）
    print("   对合并后的数据进行整体标准化...")
    X_combined = X_combined - X_combined.mean(axis=0, keepdims=True)
    X_combined = X_combined / np.sqrt(X_combined.var(axis=0, keepdims=True) + 1e-8)
    print("   ✓ 整体标准化完成")
    
    # 4. 处理标签
    print("\n[4/6] 处理标签数据...")
    
    # 检查是否需要转换为整数标签
    unique_labels = np.unique(labels)
    print(f"   原始标签值: {sorted(unique_labels)}")
    
    # 如果标签是浮点数但实际是整数，转换为整数
    if all(label == int(label) for label in unique_labels):
        # 自动检测起始标签并转换为从0开始
        if labels.min() == 1:
            labels = labels.astype(int) - 1
            print("   已将标签转换为从0开始（例如：1→0, 2→1, ...）")
        else:
            labels = labels.astype(int)
            if labels.min() == 0:
                print("   标签已是从0开始，无需转换")
            else:
                print(f"   ⚠️ 警告: 标签起始值为 {labels.min()}")
    
    # 验证数据完整性
    if X_combined.shape[0] != len(labels):
        raise ValueError(f"样本数量不匹配: 特征矩阵有 {X_combined.shape[0]} 行，"
                        f"但标签有 {len(labels)} 个")
    
    num_classes = len(np.unique(labels))
    print(f"   类别数: {num_classes}")
    print(f"   类别分布: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    # 5. 固定划分训练集和测试集（与BRCA模型一致）
    print(f"\n[5/6] 进行固定划分...")
    print(f"   ⚠️ 注意：使用固定划分模式（非随机）")
    print(f"   训练集：前 {train_size} 个样本 (索引 0-{train_size-1})")
    print(f"   测试集：第 {train_size+1} 到 {X_combined.shape[0]} 个样本 (索引 {train_size}-{X_combined.shape[0]-1})")
    
    # 验证训练集大小
    if train_size >= X_combined.shape[0]:
        raise ValueError(f"训练集大小 ({train_size}) 必须小于总样本数 ({X_combined.shape[0]})")
    
    # 固定划分
    X_train = X_combined[:train_size]
    X_test = X_combined[train_size:]
    y_train = labels[:train_size]
    y_test = labels[train_size:]
    
    print(f"\n   训练集: {X_train.shape[0]} 样本")
    print(f"   测试集: {X_test.shape[0]} 样本")
    print(f"   训练集类别分布: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"   测试集类别分布: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    
    # 6. 按原始组学维度拆分特征
    print(f"\n[6/6] 拆分训练集和测试集的特征...")
    train_data = []
    test_data = []
    
    start_idx = 0
    for i, omic_data in enumerate(omics_data, 1):
        end_idx = start_idx + omic_data.shape[1]
        train_data.append(X_train[:, start_idx:end_idx])
        test_data.append(X_test[:, start_idx:end_idx])
        print(f"   组学 {i}: 训练{train_data[-1].shape}, 测试{test_data[-1].shape}")
        start_idx = end_idx
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存文件
    print("\n" + "="*50)
    print("保存文件...")
    print("="*50)
    
    # 保存标签（无行列名）
    pd.DataFrame(y_train).to_csv(
        os.path.join(output_dir, 'labels_tr.csv'), 
        header=False, index=False
    )
    pd.DataFrame(y_test).to_csv(
        os.path.join(output_dir, 'labels_te.csv'), 
        header=False, index=False
    )
    print("✓ 已保存: labels_tr.csv, labels_te.csv")
    
    # 保存每个组学的数据（无行列名）
    for i, (tr_data, te_data) in enumerate(zip(train_data, test_data), 1):
        pd.DataFrame(tr_data).to_csv(
            os.path.join(output_dir, f'{i}_tr.csv'), 
            header=False, index=False
        )
        pd.DataFrame(te_data).to_csv(
            os.path.join(output_dir, f'{i}_te.csv'), 
            header=False, index=False
        )
        print(f"✓ 已保存: {i}_tr.csv, {i}_te.csv")
    
    # 创建特征名称文件
    for i, omic_data in enumerate(omics_data, 1):
        feat_names = [f'omics{i}_feature_{j}' for j in range(omic_data.shape[1])]
        pd.DataFrame(feat_names).to_csv(
            os.path.join(output_dir, f'{i}_featname.csv'), 
            header=False, index=False
        )
    print("✓ 已创建特征名称文件: *_featname.csv")
    
    print(f"\n{'='*50}")
    print(f"所有文件已保存到: {output_dir}")
    print(f"数据处理方式: 与BRCA模型完全一致")
    print(f"  - 自动检测并调整数据格式")
    print(f"  - 每个组学单独标准化")
    print(f"  - 合并后整体标准化")
    print(f"  - 固定划分（前{train_size}个为训练集）")
    print(f"{'='*50}")


def load_dataset(data_dir: str) -> Tuple[np.ndarray, np.ndarray, int, int, List, List, int]:
    """
    加载多组学数据集（与原代码保持一致）
    """
    print("\n加载数据集...")
    # 获取组学数量
    omics_files = glob.glob(os.path.join(data_dir, '*_featname.csv'))
    num_omics = len(omics_files)
    print(f"发现 {num_omics} 个组学数据集")

    # 加载标签
    labels_tr = pd.read_csv(os.path.join(data_dir, 'labels_tr.csv'), header=None).values.flatten()
    labels_te = pd.read_csv(os.path.join(data_dir, 'labels_te.csv'), header=None).values.flatten()

    # 初始化数据结构
    train_data, test_data = [], []
    total_features = 0

    # 加载每个组学数据
    for i in range(1, num_omics + 1):
        # 加载训练和测试数据
        tr_data = pd.read_csv(os.path.join(data_dir, f'{i}_tr.csv'), header=None).values
        te_data = pd.read_csv(os.path.join(data_dir, f'{i}_te.csv'), header=None).values

        train_data.append(tr_data)
        test_data.append(te_data)
        total_features += tr_data.shape[1]
        print(f"组学 {i}: {tr_data.shape[1]} 特征, {tr_data.shape[0]} 训练样本, {te_data.shape[0]} 测试样本")

    # 合并所有组学的特征
    X_train = np.concatenate(train_data, axis=1)
    X_test = np.concatenate(test_data, axis=1)

    # 合并所有样本
    X_full = np.vstack([X_train, X_test])
    y_full = np.concatenate([labels_tr, labels_te])

    # 分析数据集
    unique_classes, class_counts = np.unique(y_full, return_counts=True)
    num_classes = len(unique_classes)

    print(f"\n完整数据集: {X_full.shape[0]} 样本, {total_features} 特征")
    print(f"训练标签: {len(labels_tr)}, 测试标签: {len(labels_te)}")
    print(f"类别数量: {num_classes}")
    print("类别分布:")
    for cls, count in zip(unique_classes, class_counts):
        print(f"  类别 {cls}: {count} 样本")

    return X_full, y_full, total_features, num_omics, train_data, test_data, num_classes


# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 请根据实际文件路径修改这些变量
    RAW_OMICS_FILES = [
        "./datasets/BRCA104/feature_1.txt",
        "./datasets/BRCA104/feature_2.txt",
        "./datasets/BRCA104/feature_3.txt"
    ]  # 三个组学特征矩阵文件路径
    
    RAW_LABELS_FILE = "./datasets/BRCA104/label.txt"  # 标签文件路径
    
    OUTPUT_DIR = "./datasets/BRCA104"  # 输出目录
    
    # 1. 准备数据集（只需运行一次）
    print("\n准备数据集中...")
    prepare_multi_omics_dataset(
        omics_files=RAW_OMICS_FILES,
        labels_file=RAW_LABELS_FILE,
        output_dir=OUTPUT_DIR,
        train_size=70,  # 固定前70个样本为训练集（与BRCA模型一致）
        auto_transpose=True  # 自动检测是否需要转置
    )
    
    # 2. 验证load_dataset函数是否能正确加载
    print("\n\n验证数据加载...")
    try:
        X_full, y_full, total_features, num_omics, train_data, test_data, num_classes = load_dataset(OUTPUT_DIR)
        
        print("\n" + "="*50)
        print("数据加载成功！")
        print(f"完整特征矩阵形状: {X_full.shape}")
        print(f"完整标签形状: {y_full.shape}")
        print(f"组学数量: {num_omics}")
        print(f"总特征数: {total_features}")
        print(f"类别数: {num_classes}")
        print("="*50)
        
        # 数据一致性检查
        print(f"\n数据一致性检查:")
        print(f"  总样本数: {X_full.shape[0]}")
        print(f"  训练集样本数: {train_data[0].shape[0]}")
        print(f"  测试集样本数: {test_data[0].shape[0]}")
        print(f"  组学数量: {len(train_data)}")
        
        # 检查所有组学样本数是否一致
        train_samples_consistent = len(set(d.shape[0] for d in train_data)) == 1
        test_samples_consistent = len(set(d.shape[0] for d in test_data)) == 1
        total_match = X_full.shape[0] == train_data[0].shape[0] + test_data[0].shape[0]
        
        print(f"  检查1 - 所有组学训练集样本数一致: {'✓' if train_samples_consistent else '✗'}")
        print(f"  检查2 - 所有组学测试集样本数一致: {'✓' if test_samples_consistent else '✗'}")
        print(f"  检查3 - 总样本数 = 训练集+测试集: {'✓' if total_match else '✗'}")
        
        if train_samples_consistent and test_samples_consistent and total_match:
            print("\n✓ 所有检查通过！数据划分正确。")
            
            # 验证标签范围是否正确
            print(f"\n标签范围检查:")
            print(f"  标签最小值: {y_full.min()}")
            print(f"  标签最大值: {y_full.max()}")
            print(f"  期望范围: [0, {num_classes-1}]")
            label_range_ok = y_full.min() >= 0 and y_full.max() < num_classes
            print(f"  是否正常: {'✓' if label_range_ok else '✗'}")
            
            # 验证数据是否已标准化
            print(f"\n数据标准化验证:")
            print(f"  完整数据集均值: {np.mean(X_full):.6f} (应接近0)")
            print(f"  完整数据集标准差: {np.std(X_full):.6f} (应接近1)")
            
            # 详细的标签分布分析
            print(f"\n{'='*50}")
            print("详细标签分布分析")
            print(f"{'='*50}")
            
            # 训练集标签分布
            train_unique, train_counts = np.unique(labels_tr, return_counts=True)
            train_dist = dict(zip(train_unique, train_counts))
            
            print(f"\n训练集标签分布 (总计 {len(labels_tr)} 样本):")
            print(f"{'类别':<8} {'样本数':<10} {'比例':<10} {'可视化'}")
            print("-" * 50)
            for label in sorted(train_dist.keys()):
                count = train_dist[label]
                ratio = count / len(labels_tr) * 100
                bar = "█" * int(ratio / 2)  # 每2%一个方块
                print(f"{int(label):<8} {count:<10} {ratio:>6.2f}%    {bar}")
            
            # 测试集标签分布
            test_unique, test_counts = np.unique(labels_te, return_counts=True)
            test_dist = dict(zip(test_unique, test_counts))
            
            print(f"\n测试集标签分布 (总计 {len(labels_te)} 样本):")
            print(f"{'类别':<8} {'样本数':<10} {'比例':<10} {'可视化'}")
            print("-" * 50)
            for label in sorted(test_dist.keys()):
                count = test_dist[label]
                ratio = count / len(labels_te) * 100
                bar = "█" * int(ratio / 2)
                print(f"{int(label):<8} {count:<10} {ratio:>6.2f}%    {bar}")
            
            # 对比分析
            print(f"\n训练集 vs 测试集 对比:")
            print(f"{'类别':<8} {'训练集':<12} {'测试集':<12} {'差异'}")
            print("-" * 50)
            all_labels = sorted(set(list(train_dist.keys()) + list(test_dist.keys())))
            for label in all_labels:
                train_count = train_dist.get(label, 0)
                test_count = test_dist.get(label, 0)
                train_ratio = train_count / len(labels_tr) * 100 if len(labels_tr) > 0 else 0
                test_ratio = test_count / len(labels_te) * 100 if len(labels_te) > 0 else 0
                diff = test_ratio - train_ratio
                diff_str = f"{diff:+.2f}%" if diff != 0 else "0.00%"
                print(f"{int(label):<8} {train_count:>3} ({train_ratio:>5.2f}%)  {test_count:>3} ({test_ratio:>5.2f}%)  {diff_str}")
            
            # 类别缺失检查
            train_labels_set = set(train_unique)
            test_labels_set = set(test_unique)
            missing_in_train = test_labels_set - train_labels_set
            missing_in_test = train_labels_set - test_labels_set
            
            if missing_in_train or missing_in_test:
                print(f"\n⚠️ 警告：类别分布不平衡")
                if missing_in_train:
                    print(f"  训练集缺失类别: {sorted([int(x) for x in missing_in_train])}")
                if missing_in_test:
                    print(f"  测试集缺失类别: {sorted([int(x) for x in missing_in_test])}")
            else:
                print(f"\n✓ 训练集和测试集都包含所有类别")
            
            print(f"\n{'='*50}")
            
        else:
            print("\n✗ 检查发现潜在问题，请检查数据！")
        
    except Exception as e:
        print(f"\n数据加载失败: {str(e)}")
        import traceback
        traceback.print_exc()