# TRIGEL 第四章实验设计框架

## 1. 总体判断

参考论文采用“实验设置—总体对比—增强方法比较—消融—参数敏感性”的结构，基本完整，也适合一般机器学习论文。但其消融和参数实验主要集中在单一数据集，评价指标较少，且缺少生物医学任务所需要的类别级结果和下游解释。

TRIGEL 的第四章不宜完全照搬。建议围绕以下证据链展开：

> **总体性能是否更好 → 少数类是否真正改善 → 哪些模块产生增益 → 参数和不平衡程度变化时是否稳定 → 学到的表示是否具有可解释性 → 方法在哪些情况下仍会失败。**

建议用研究问题组织全文：

- **RQ1：有效性。** TRIGEL 是否优于主流多组学分类方法和不平衡学习方法？
- **RQ2：不平衡改善。** 性能提升是否来自少数类识别改善，而不是多数类准确率进一步提高？
- **RQ3：模块贡献。** TGS、MGE 和 CFC 中的关键设计分别贡献了什么，是否存在协同作用？
- **RQ4：稳健性。** 模型对关键超参数、类别不平衡程度和随机划分是否稳定？
- **RQ5：解释性与应用价值。** 融合表示、组学注意力和错误样本能否揭示有意义的患者异质性？

---

## 2. 推荐章节结构

## 4. Experimental Results and Analysis

### 4.1 Experimental Setup

这一节只交代复现实验所需的信息，不讨论结果。

#### 4.1.1 Datasets and Preprocessing

每个数据集报告：

- 患者数量、训练/验证/测试数量；
- 癌症亚型数量及各类别样本数；
- 组学类型和各组学特征维度；
- 不平衡率 $\mathrm{IR}=n_{\max}/n_{\min}$；
- 图构建方式、特征标准化与数据划分方式。

**建议选择：**

- BRCA：主数据集，用于完整对比、消融、参数和下游分析；
- KRCCC 或 BRCA104：小样本且不平衡明显，用于检验低样本稳定性；
- GBM：补充不同癌种或较低不平衡程度下的泛化；
- 其余数据集根据来源独立性选择。

BRCA、BRCA1031、BRCAqx 等若只是同一队列的不同预处理版本，不应作为多个独立数据集共同宣称跨队列泛化。

**表 1：Dataset statistics。**

#### 4.1.2 Compared Methods

现有方法很少同时深入处理多组学融合、癌症亚型分类和类别不平衡，因此基线分为两个互补类别：

1. **多组学癌症分类方法**：MOGONET、Dynamic、MLCLNet、DPNET、HyperTMO 和 MOHGCN。该组用于比较异质组学表示学习和融合能力。
2. **图类别不平衡方法**：GraphSHA 和 IceBerg。该组用于比较少数类增强、偏置校正和不平衡节点分类能力。

所有方法必须使用相同的数据划分、预处理和调参预算，不能直接混用其他论文在不同划分下报告的数字。GraphSHA 和 IceBerg 并非原生多组学融合模型，正式稿必须说明其多组学输入适配方式，并保证其获得的信息量与 TRIGEL 一致。

#### 4.1.3 Evaluation Metrics and Statistical Protocol

由于任务存在类别不平衡，建议：

- **主指标：Macro-F1**；
- **核心辅助指标：Balanced Accuracy 或 Macro-Recall**；
- Accuracy、Macro-Precision、Weighted-F1、Macro-AUC 作为补充；
- 报告每个类别的 Precision、Recall 和 F1；
- 单独报告 minority-class recall/F1。

所有正式结果应：

- 使用固定且公开的多组随机种子；
- 报告全部运行的 `mean ± std`；
- 超参数和早停仅依据验证集 Macro-F1；
- 测试集只用于最终评价；
- 对 TRIGEL 与最强基线进行配对 Wilcoxon 检验或配对 t 检验，并报告效应量。

当前代码中的 `top_test_acc` 和按测试 F1 选择 top-k 运行不适合作为论文正式协议，需要改为 `all_runs`，或者由验证集确定模型后汇总全部测试结果。

#### 4.1.4 Implementation Details

集中报告 GCN 层数、隐藏维度、优化器、学习率、权重衰减、对比温度、损失权重、注意力头数、训练轮数、早停设置、KNN 的 $k$ 值、关键增强参数和软硬件环境。

---

### 4.2 Comparison with State-of-the-Art Methods (RQ1)

#### 4.2.1 Overall Classification Performance

使用一张主表报告所有数据集上的 Macro-F1、Balanced Accuracy 和 Accuracy。正文依次分析总体排名、相对最强基线的提升、高低不平衡数据集上的差异，以及未取得最优的情况。

**表 2：Main comparison results。**

#### 4.2.2 Class-wise and Minority-class Performance

单独报告各亚型 F1/Recall，minority、intermediate、majority 三组的平均 F1，以及归一化混淆矩阵。该小节用于证明性能提升确实来自少数类识别改善。

**图 3：不同类别或三类层级的 Recall/F1 柱状图。**  
**图 4：TRIGEL 与最强基线的归一化混淆矩阵。**

---

### 4.3 Ablation and Mechanism Analysis (RQ2–RQ3)

#### 4.3.1 Component Ablation

不建议只做逐步累加式消融。主流写法是以完整模型为中心，采用 `w/o` 去除实验：

| Variant | Removed design | Purpose |
|---|---|---|
| Full TRIGEL | None | 完整模型 |
| w/o TGS | 三类分层及其差异化策略 | 验证三类指导 |
| w/o Feature Enhancement | 类别引导特征增强 | 验证特征路径 |
| w/o Topology Augmentation | 边增删 | 验证拓扑路径 |
| w/o Edge Weighting | 类别感知边权 | 验证传播调节 |
| w/o Contrastive Learning | 对比损失 | 验证跨视图一致性 |
| w/o Attention Fusion | 改为平均融合或简单拼接 | 验证动态组学融合 |
| Backbone | 去除全部新增模块 | 衡量总体增益 |

主表报告 Macro-F1、Balanced Accuracy、minority-F1 和标准差。

**表 3：Component ablation results。**

#### 4.3.2 Interaction Analysis

若算力允许，增加 Feature only、Topology only、Feature + Topology、Feature + Topology + Edge Weighting 和 Full TRIGEL，证明“特征—结构—传播”三个层面的协同关系。

#### 4.3.3 Structural and Representation Evidence

避免仅用分类分数解释机制。建议补充：

- 各层级节点的新增边数、删除边数和平均边权；
- 增强前后类内距离与类间距离；
- 原始视图与增强视图的同患者余弦相似度；
- Full 与关键消融组的表示可视化。

**图 5：增强前后图结构统计或类内/类间距离。**  
**图 6：Full 与关键消融组的 UMAP/t-SNE。**

---

### 4.4 Parameter Sensitivity and Robustness (RQ4)

#### 4.4.1 Sensitivity to Key Parameters

只选择与创新直接相关的参数：

- 有效样本数参数 $\beta$；
- majority feature drop rate；
- `max_new_edges` 或 edge-add probability；
- minority edge weight；
- 对比损失权重 $\lambda_{\mathrm{con}}$ 或温度 $\tau$。

每个参数采用单变量分析，报告 Macro-F1 与 minority-F1 的 `mean ± std`。建议组织为一张多子图。

**图 7：关键参数敏感性曲线。**

#### 4.4.2 Robustness to Imbalance Severity

通过训练集下采样构造原始、中度、高度和极端不平衡设置，比较 TRIGEL、Backbone 与最强不平衡基线随 IR 增大时的 Macro-F1/minority-F1 下降幅度。

**图 8：性能随不平衡率变化的曲线。**

#### 4.4.3 Stability and Efficiency

简要报告多随机种子标准差、参数量、单次训练时间、峰值显存和相对 Backbone 的额外成本。可放正文末尾或补充材料。

---

### 4.5 Downstream Analysis and Model Interpretation (RQ5)

#### 4.5.1 Representation Visualization

使用最终融合表示展示 TRIGEL 与 Backbone/最强基线的 UMAP 或 t-SNE，并补充 silhouette score 或 Davies–Bouldin index。

现有工作区中的 t-SNE 是各组学输入特征可视化，不能直接作为“模型学到更好表示”的证据，需要导出最终融合嵌入重新绘制。

#### 4.5.2 Omics Attention Analysis

分析不同癌症亚型的平均组学权重、minority 与 majority 患者的权重分布，以及典型患者的个体组学贡献。

**图 9：亚型 × 组学注意力热图或箱线图。**

注意力权重只能解释模型依赖，不能直接称为生物因果重要性。

#### 4.5.3 Error and Hard-case Analysis

将现有错误分析压缩为：

- 经常误判的类别对；
- 多次随机运行中持续误判的患者；
- 错误患者到真实类/预测类中心的距离；
- 离群点与错误样本的重合程度。

正文选择 2–3 个典型错误模式，不列出全部患者 ID。类别空间重叠可作为困难样本证据，但没有临床注释时不能直接归因于生物异质性。

#### 4.5.4 Optional Biological Validation

仅在具备基因名称、临床信息或生存数据时开展差异特征、GO/KEGG 富集、Kaplan–Meier 生存分析或标志物一致性验证。若当前数据只有特征矩阵和标签，则删除该小节。

---

### 4.6 Failure Modes and Limitations

结合实际实验讨论：

- 极少数类样本仍可能被相邻大类吸收；
- 图构建质量依赖 KNN 和输入特征；
- 增强强度过大会引入结构噪声；
- 注意力权重不等于生物因果贡献；
- 小队列上的方差和外部泛化仍受限制。

---

## 3. 推荐图表总表

| 编号 | 内容 | 优先级 |
|---|---|---|
| Table 1 | 数据集统计与不平衡率 | 必须 |
| Table 2 | 多数据集主对比实验 | 必须 |
| Table 3 | 核心模块消融 | 必须 |
| Figure 3 | 分层或逐类 F1/Recall | 必须 |
| Figure 4 | 混淆矩阵 | 推荐 |
| Figure 5 | 图结构/表示机制证据 | 推荐 |
| Figure 6 | 最终融合表示可视化 | 推荐 |
| Figure 7 | 参数敏感性 | 必须 |
| Figure 8 | 不平衡程度压力测试 | 强烈推荐 |
| Figure 9 | 组学注意力分析 | 强烈推荐 |
| Table 4 | 参数量、时间和显存 | 可选 |

---

## 4. 最小可发表版本

1. 3 个具有不同样本规模或不平衡程度的数据集；
2. 一张公平协议下的主对比表；
3. 每个核心设计对应的 `w/o` 消融；
4. Macro-F1、Balanced Accuracy 和 minority-F1；
5. 4–5 个关键参数的敏感性分析；
6. 一项不平衡程度压力测试；
7. 一项融合表示或注意力下游分析；
8. 全部运行的 `mean ± std`，不使用测试集选择 top-k。

---

## 5. 当前需要优先补齐的证据

1. 现有正式结果主要来自 BRCA，尚不足以支持跨数据集泛化结论。
2. 当前 `top_test_acc`/`test_f1_macro` 调参协议存在测试集泄漏风险，需改为验证集选参。
3. 现有消融设计尚未完整覆盖 CFC 中的对比学习和注意力融合。
4. 现有 t-SNE 主要基于各组学输入特征，需要补充最终融合嵌入。
5. 现有错误分析可以保留，但应压缩为类别边界与典型困难样本分析。
6. 生存、通路或标志物分析是否可做，取决于数据中是否包含临床信息和可映射的特征名称。
