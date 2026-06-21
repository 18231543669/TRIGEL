# Multi-Omics GCN 项目架构（✓ 集成PSN+SNF）

## 📁 目录结构

```
multi-omics-gcn/
│
├── main.py                          # 主入口
├── parser.py                        # 命令行参数解析
├── config.py                        # 全局配置
├── trainer.py                       # 训练评估模块
│   ├── train_with_contrastive_learning()
│   └── evaluate_model()
│
├── models/                          # 模型包
│   ├── __init__.py
│   ├── losses.py                   # 损失函数
│   │   └── ContrastiveLoss
│   ├── gcn_models.py               # GCN模型
│   │   ├── GCNModel
│   │   ├── GCNContrastiveModel
│   │   └── ModelFactory
│   ├── graph_fusion.py             # 图融合（注意力机制）
│   │   └── AttentionFusion
│   └── experiment_runner.py        # 实验运行器
│       └── ExperimentRunner
│
├── utils/                           # 工具函数包
│   ├── __init__.py
│   ├── data_utils.py               # ✨ 修改：数据处理（集成PSN+SNF）
│   │   ├── set_random_seed()
│   │   ├── load_dataset()          # ✓ 支持LASSO+PSN+SNF
│   │   └── construct_graphs_with_psn_snf()  # ✓ 新增
│   ├── metrics.py                  # 评估指标
│   │   ├── calculate_metrics()
│   │   ├── calculate_average_results()
│   │   └── print_average_results()
│   ├── visualization.py            # 可视化
│   │   └── visualize_training()
│   ├── io_utils.py                 # IO和配置管理
│   │   ├── create_result_directory()
│   │   ├── save_results()
│   │   └── load_results()
│   ├── training_utils.py           # 训练工具
│   │   └── EarlyStopping
│   └── error_analysis.py           # 错误分析
│       └── ErrorAnalyzer
│
├── augmentation/                    # 数据增强包
│   ├── __init__.py
│   ├── augmentors.py               # 增强器
│   │   ├── PageRankCalculator
│   │   ├── MinorityClassIdentifier
│   │   ├── FeatureAugmentor
│   │   ├── TopologyAugmentor
│   │   └── DataAugmentor
│   └── reporter.py                 # 报告模块
│       └── AugmentationEffectsReporter
│
├── tuning/                          # 超参数调优包
│   ├── __init__.py
│   ├── param_space.py              # 参数空间
│   ├── result_tracker.py           # 结果跟踪
│   ├── grid_search.py              # 网格搜索
│   ├── report_generator.py         # 报告生成
│   └── auto_tuner.py               # 主调优器
│
└── README.md                        # 项目文档
```

---


## 🟢 优先级7：Q2 - 执行顺序和合理性

### 完整执行流程图
```
main.py
  │
  ├─→ parse_arguments()
  │
  └─→ ExperimentRunner(config)
        │
        ├─→ run_multiple_experiments(num_runs=5)
        │     │
        │     ├─→ _load_and_build_once() ←────────────────┐
        │     │     ├─→ _load_experiment_data()           │ 只执行一次
        │     │     └─→ _build_graph_structures()  ←──────┘
        │     │
        │     └─→ for run_id in 1..5:
        │           │
        │           └─→ run_single_experiment(cached_data)
        │                 │
        │                 ├─→ _apply_data_augmentation()
        │                 │     ├─→ MinorityClassIdentifier.identify()
        │                 │     ├─→ PageRankCalculator.compute()
        │                 │     ├─→ FeatureAugmentor.augment()
        │                 │     ├─→ TopologyAugmentor.augment()
        │                 │     └─→ apply_edge_weighting() [可选]
        │                 │
        │                 ├─→ _split_dataset()
        │                 │
        │                 ├─→ _initialize_model()
        │                 │     ├─→ ModelFactory.create_model()
        │                 │     └─→ compute_class_weight()
        │                 │
        │                 ├─→ _train_model()
        │                 │     │
        │                 │     └─→ for epoch in 1..400:
        │                 │           ├─→ model(x_aug, edge_indices_aug)
        │                 │           ├─→ classification_loss = CrossEntropy
        │                 │           ├─→ contrastive_loss = model.compute_contrastive_loss()
        │                 │           ├─→ total_loss = 0.5*cls + 0.5*cl
        │                 │           └─→ early_stopping check
        │                 │
        │                 └─→ _evaluate_model()
        │
        └─→ _finalize_experiments()
              ├─→ calculate_average_results()
              ├─→ MisclassificationAnalyzer.analyze()
              └─→ run_enhanced_analysis()


## 🎯 方案B核心改进

### 新增模块
1. **integration/patient_similarity_network.py** (220行)
   - 在patient空间(875×875)而非特征空间(2503维)构建图
   - 避免维度灾难

2. **integration/snf_fusion.py** (150行)
   - 融合多个PSN，充分利用组学间互补信息
   - 基于Nature Methods 2014经典算法

3. **integration/lasso_feature_selection.py** (180行)
   - 特征降维：2503 → 1250维
   - 降噪：L1正则化自动移除噪声特征

4. **integration/sparse_denoising.py** (100行)
   - 可选：适用于极度稀疏(>80%)数据
   - 包含诊断工具

### 修改模块
5. **utils/data_utils.py** (修改)
   - 集成LASSO特征选择
   - 集成PSN+SNF图构建
   - 保持向后兼容

---

## 🔄 数据流程（方案B）

```
原始数据 (3组学)
    ↓
┌───────────────────────────────────────┐
│ 1. LASSO特征选择                      │
│    omics1: 1000 → 500维               │
│    omics2: 1000 → 500维               │
│    omics3: 500 → 250维                │
│    总计: 2500 → 1250维 (50%保留)      │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│ 2. PSN构建（patient空间）             │
│    PSN1: 875×875 (omics1)             │
│    PSN2: 875×875 (omics2)             │
│    PSN3: 875×875 (omics3)             │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│ 3. SNF融合                            │
│    fused_PSN: 875×875                 │
│    （综合3个组学信息）                │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│ 4. 图构建                             │
│    edge_indices + edge_weights        │
│    （基于融合PSN）                    │
└───────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│ 5. GCN训练                            │
│    GCN(features, fused_graph)         │
│    预期准确率: 87-89%                 │
└───────────────────────────────────────┘
```

---

## 📊 性能对比

### 当前方法
```python
X = concat([omics1, omics2, omics3])  # 2503维特征空间
graph = cosine_KNN(X, k=20)           # ❌ 高维空间失效
GCN(X, graph)
→ 准确率: 84.5%
```

### 方案B
```python
# LASSO降维
omics1 = LASSO(omics1) → 500维

# PSN构建（patient空间）
PSN1 = patient_similarity(omics1)  # ✓ 875×875低维空间

# SNF融合
fused_PSN = SNF([PSN1, PSN2, PSN3])  # ✓ 综合信息

# GCN训练
GCN(features, fused_PSN)
→ 预期准确率: 87-89% (+3-5%)
```

---

## 🔧 配置参数

### integration/相关参数
```python
# LASSO特征选择
lasso_top_k_percent = 50.0        # 保留特征百分比
lasso_method = 'logistic_lasso'   # 或 'f_test'

# PSN构建
psn_k_neighbors = 20              # K近邻数

# SNF融合
snf_iterations = 20               # 迭代次数
snf_alpha = 0.8                   # 平衡参数

# 边权重
use_edge_weights = True           # 使用边权重
```

---

## 📈 预期改进

| 指标 | 当前 | 方案B | 改进 |
|------|------|-------|------|
| 测试准确率 | 84.5% | 87-89% | +3-5% |
| 100%错误样本 | 38个 | <15个 | -60% |
| 特征数 | 2503 | 1250 | -50% |
| 训练时间 | 基准 | ~1.2x | +20% |

---

## 🎓 文献依据

1. **Alharbi et al. (2024)** - LASSO-MOGAT
   - arXiv 2410.05325
   - 95.9%准确率
   - 核心：LASSO + PSN + GAT

2. **Li & Nabavi (2024)** - Multimodal GNN
   - BMC Bioinformatics
   - 核心发现：GAT适合小图，GCN适合大图

3. **Wang et al. (2014)** - SNF原文
   - Nature Methods
   - 被引3000+次

**共同结论**：图构建方法 >> GNN架构

---

## 🚀 使用方式

### 方法1：自动（推荐）
```bash
python main.py --num_runs 5
# 默认启用LASSO+PSN+SNF
```

### 方法2：自定义
```bash
python main.py \
    --lasso_top_k 30 \
    --psn_k_neighbors 30 \
    --snf_iterations 30 \
    --num_runs 5
```

### 方法3：关闭PSN+SNF（使用传统方法）
```bash
python main.py --use_psn_snf False
```

---

## ✅ 验证检查清单

### 部署前
- [ ] `integration/`文件夹在根目录
- [ ] `utils/data_utils.py`已替换（备份旧版本）
- [ ] Python环境有sklearn, numpy, torch

### 运行时
- [ ] 看到"LASSO特征选择"输出
- [ ] 看到"构建Patient Similarity Networks"输出
- [ ] 看到"Similarity Network Fusion"输出

### 结果验证
- [ ] 测试准确率 ≥ 87%
- [ ] 100%错误样本 ≤ 15个
- [ ] 各类别错误率更均衡

---

## 💡 下一步

### 如果达到87-89%
✅ 成功！可以：
1. 运行20次完整实验
2. 在其他数据集验证
3. 准备论文投稿
4. 可选：尝试GAT（+1-2%）

### 如果效果不够
可以尝试：
1. 调整LASSO保留率（30-70%）
2. 调整K值（15-30）
3. 使用稀疏降噪
4. 考虑加入AutoEncoder（方案C）

---

## 📞 技术支持

- 详细部署：见`DEPLOYMENT_GUIDE.md`
- 技术原理：见`PSN_vs_GAT_ANALYSIS.md`（如需要）
- 代码注释：所有模块都有详细注释

---

**更新日期**: 2025-12-16
**版本**: 方案B v1.0
**核心改进**: LASSO + PSN + SNF