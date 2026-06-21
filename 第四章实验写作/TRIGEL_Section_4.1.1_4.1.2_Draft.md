# 4.1.1 and 4.1.2 Draft

## English Manuscript Text

### 4.1.1 Datasets and Preprocessing

We evaluated TRIGEL on four multi-omics cancer subtype classification datasets with different cohort sizes and class-imbalance levels: BRCA, BRCA104, GBM, and KRCCC. The datasets were obtained from [data source and accession number], and each patient was assigned a single cancer subtype label. Each cohort contained [omics type 1], [omics type 2], and [omics type 3], except that the complete modality files and feature dimensions of KRCCC remain to be verified. Table 1 summarizes the dataset statistics derived from the current local data files.

**Table 1. Statistics of the multi-omics datasets used in the experiments.**

| Dataset | Patients | Classes | Class distribution | Feature dimensions | Train/validation/test | IR |
|---|---:|---:|---|---|---|---:|
| BRCA | 875 | 5 | 115/131/46/436/147 | 1000/1000/503 | [to be specified] | 9.48 |
| BRCA104 | 104 | 4 | 18/51/12/23 | 17814/354/23094 | [to be specified] | 4.25 |
| GBM | 263 | 4 | 68/80/46/69 | 3010/1250/534 | [to be specified] | 1.74 |
| KRCCC | 122 | 4 | 24/24/62/12 | 329/[to be verified]/[to be verified] | [to be specified] | 5.17 |

The imbalance ratio of a dataset was calculated as

$$
\mathrm{IR}=\frac{\max_{c\in\{1,\ldots,C\}} n_c}
{\min_{c\in\{1,\ldots,C\}} n_c},
\tag{19}
$$

where $C$ is the number of cancer subtypes and $n_c$ is the number of patients belonging to subtype $c$. A larger $\mathrm{IR}$ indicates a more uneven class distribution.

For each dataset, preprocessing was performed independently for each omics modality. Missing values were handled using [missing-value strategy], and features were [standardized/normalized] using statistics estimated from the training set only. The data were divided into mutually exclusive training, validation, and test sets using stratified sampling with a ratio of [training ratio]:[validation ratio]:[test ratio]. The validation set was used for hyperparameter selection and early stopping, whereas the test set was used only for final evaluation. Unless otherwise stated, all methods used the same data partitions across repeated runs.

For each omics modality, patients were represented as graph nodes, and an undirected patient-similarity graph was constructed from cosine similarity. Each patient was connected to its $k$ nearest neighbours, where $k$ was selected from [candidate values] on the validation set. The same initial graphs, input features, and data partitions were provided to TRIGEL and all graph-based comparison methods whenever their model definitions allowed this setting.

### 4.1.2 Compared Methods

Few existing methods jointly address multi-omics integration, cancer subtype classification, and severe class imbalance within a unified framework. We therefore selected comparison methods from two complementary categories. The first category contains six representative biomedical multi-omics classification methods and evaluates the ability to learn and integrate heterogeneous molecular information. The second category contains two graph-based imbalance learning methods and evaluates the ability to reduce majority-class bias under skewed subtype distributions.

The multi-omics classification methods include:

1. **MOGONET**, which constructs an independent patient graph for each omics modality and models cross-omics correlations through view correlation discovery [Ref.].
2. **Dynamic**, which dynamically estimates feature- and modality-level contributions for trustworthy multimodal fusion [Ref.].
3. **MLCLNet**, which combines feature confidence learning, cross-modal label fusion, and label confidence learning for multi-omics classification [Ref.].
4. **DPNET**, which learns complementary multi-level representations through a dual-path multimodal network [Ref.].
5. **HyperTMO**, which represents sample relationships using modality-specific hypergraphs and performs evidence-level multi-omics integration [Ref.].
6. **MOHGCN**, which captures higher-order sample correlations through graph-based multi-omics representation learning and fusion [Ref.].

The graph imbalance learning methods include:

7. **GraphSHA**, which synthesizes hard minority-class nodes near class boundaries to improve the discrimination of under-represented classes [Ref.].
8. **IceBerg**, which uses debiased self-training and balanced pseudo-label selection to exploit unlabeled nodes under class-imbalanced and few-shot settings [Ref.].

For fair comparison, all methods were evaluated using identical training, validation, and test partitions. Hyperparameters were selected according to validation-set Macro-F1 using the search spaces recommended by the original studies or their official implementations. GraphSHA and IceBerg were originally developed for general class-imbalanced node classification rather than multi-omics integration. They were therefore adapted to the same patient-graph input used in this study through [specific multi-omics adaptation strategy], without changing their core imbalance learning mechanisms. Each method was run with [number of random seeds] random seeds, and the mean and standard deviation over all runs were reported. Results from the original papers were not directly reused because their data partitions, preprocessing procedures, and task settings may differ from those adopted here.

---

## 中文对照

### 4.1.1 数据集与预处理

我们在四个具有不同队列规模和类别不平衡程度的多组学癌症亚型分类数据集上评估 TRIGEL，包括 BRCA、BRCA104、GBM 和 KRCCC。数据集来源于[数据来源及登录号]，每名患者对应一个癌症亚型标签。每个队列包含[组学类型1]、[组学类型2]和[组学类型3]，但 KRCCC 的完整组学文件及特征维度仍需进一步核实。表1列出了根据当前本地数据文件统计得到的数据集信息。

**表1. 实验所用多组学数据集的统计信息。**

| 数据集 | 患者数 | 类别数 | 各类别样本数 | 各组学特征维度 | 训练/验证/测试划分 | IR |
|---|---:|---:|---|---|---|---:|
| BRCA | 875 | 5 | 115/131/46/436/147 | 1000/1000/503 | [待填写] | 9.48 |
| BRCA104 | 104 | 4 | 18/51/12/23 | 17814/354/23094 | [待填写] | 4.25 |
| GBM | 263 | 4 | 68/80/46/69 | 3010/1250/534 | [待填写] | 1.74 |
| KRCCC | 122 | 4 | 24/24/62/12 | 329/[待核实]/[待核实] | [待填写] | 5.17 |

数据集的不平衡率按照式（19）计算，其中，$C$ 表示癌症亚型数量，$n_c$ 表示第 $c$ 个亚型的患者数量。$\mathrm{IR}$ 越大，说明类别分布越不均衡。

对于每个数据集，各组学分别进行预处理。缺失值采用[缺失值处理方法]处理，并仅使用训练集统计量对特征进行[标准化/归一化]。采用分层抽样将数据划分为互不重叠的训练集、验证集和测试集，比例为[训练集比例]:[验证集比例]:[测试集比例]。验证集用于超参数选择和早停，测试集仅用于最终评估。除非另有说明，所有方法在重复实验中使用相同的数据划分。

对于每种组学，将患者视为图节点，并根据余弦相似度构建无向患者相似图。每名患者与其 $k$ 个最近邻相连，$k$ 从[候选取值]中根据验证集性能确定。在模型定义允许的情况下，TRIGEL 与所有图方法使用相同的初始图、输入特征和数据划分。

### 4.1.2 对比方法

现有方法很少在同一框架中同时深入处理多组学信息融合、癌症亚型分类和严重类别不平衡问题。因此，本文从两个互补的类别中选取对比方法。第一类包含六种具有代表性的生物医学多组学分类方法，主要检验模型对异质分子信息的学习与融合能力；第二类包含两种图类别不平衡学习方法，主要检验模型在亚型分布偏斜条件下减轻多数类偏置的能力。

多组学分类方法包括：

1. **MOGONET**：为每种组学构建独立的患者图，并通过视图相关性发现模块建模跨组学关系 [Ref.]。
2. **Dynamic**：动态估计特征层和模态层的贡献，实现可信的多模态信息融合 [Ref.]。
3. **MLCLNet**：结合特征置信度学习、跨模态标签融合和标签置信度学习完成多组学分类 [Ref.]。
4. **DPNET**：通过双路径多模态网络学习互补的多层次表示 [Ref.]。
5. **HyperTMO**：使用各组学特异的超图表示样本关系，并在证据层完成多组学信息融合 [Ref.]。
6. **MOHGCN**：通过图表示学习与融合捕获多组学样本之间的高阶相关性 [Ref.]。

图类别不平衡学习方法包括：

7. **GraphSHA**：在类别边界附近合成较难分类的少数类节点，以提高模型对代表不足类别的判别能力 [Ref.]。
8. **IceBerg**：利用去偏自训练和平衡伪标签选择，在类别不平衡和少样本条件下挖掘未标记节点中的监督信息 [Ref.]。

为保证比较公平，所有方法采用相同的训练集、验证集和测试集划分，并根据验证集 Macro-F1 选择超参数，搜索范围参考原论文或官方实现。GraphSHA 和 IceBerg 最初面向通用的类别不平衡节点分类，而不是多组学融合任务。因此，本文通过[具体的多组学适配方式]使其接收与本文方法一致的患者图输入，同时不改变其核心不平衡学习机制。每种方法使用[随机种子数量]个随机种子重复运行，并报告全部运行结果的均值和标准差。由于数据划分、预处理流程和任务设置可能不同，本文不直接采用原论文报告的结果。

---

## 写作说明与待补充信息

- **数据集选择**：当前正文暂定 BRCA、BRCA104、GBM 和 KRCCC。BRCA、BRCA1031 与 BRCAqx 可能是同一来源队列的不同预处理或划分版本，正式论文中不宜将其直接视为三个独立外部数据集。
- **KRCCC 文件**：当前目录中可以确认3种组学的特征名称文件，但只发现第1种组学的训练和测试特征矩阵，因此后两种组学的维度暂不写死。
- **数据划分**：当前代码将验证集与测试集设为同一批样本。正式实验应改为互不重叠的训练集、验证集和测试集，否则存在测试信息参与早停和调参的问题。
- **数据预处理**：当前加载代码未明确执行缺失值填补或特征标准化。应根据数据的实际来源和生成流程补充，不应直接保留正文中的占位描述而不修改代码。
- **图构建**：本地 BRCA 复现实验使用余弦相似度 KNN 图，当前预设为 $k=10$。若其他数据集采用预计算邻接矩阵或不同的 $k$，需要在最终稿中分别说明。
- **对比方法构成**：最终主表采用六种多组学分类方法 MOGONET、Dynamic、MLCLNet、DPNET、HyperTMO 和 MOHGCN，以及两种图类别不平衡方法 GraphSHA 和 IceBerg。
- **不平衡方法适配**：GraphSHA 和 IceBerg 本身不负责多组学融合，必须补充它们接收多组学数据的具体方式，例如使用拼接特征构建统一患者图，或接入相同的多组学图编码与融合主干。该信息不能省略，否则不同方法的输入信息和模型容量可能不公平。
- **参考文献**：所有 `[Ref.]` 需在确定全文参考文献顺序后替换为正式编号。
- **公式编号**：式（19）暂按第3章以式（18）结束进行顺延；合并全文后需再次检查。

## 符号与术语一致性

| 符号或术语 | 本节含义 | 与前文的关系 |
|---|---|---|
| $C$ | 癌症亚型数量 | 与分类任务中的类别数一致 |
| $c$ | 类别索引 | 不与前文的患者索引 $n$、组学索引 $v$ 冲突 |
| $n_c$ | 第 $c$ 类患者数量 | 仅用于数据统计 |
| $\mathrm{IR}$ | 最大类样本数与最小类样本数之比 | 本节首次定义 |
| patient/node | 患者在图中对应一个节点 | 与第3章一致 |
| omics modality | 一种组学模态 | 与第3章的 $V$ 和 $v$ 一致 |
| Macro-F1 | 对各类别 F1 等权平均 | 适合作为不平衡分类的主要选参指标 |

## Claim-Evidence Check

| Claim | Evidence | Status |
|---|---|---|
| 四个数据集具有不同样本规模和不平衡程度 | 本地标签文件及表1统计 | Supported |
| 所有方法使用独立验证集且测试集仅用于最终评估 | 尚未与当前代码一致 | Needs implementation |
| 所有基线使用相同划分和调参预算 | 尚未完成基线实验 | Needs evidence |
| KRCCC 包含完整的三组学特征矩阵 | 当前目录文件不完整 | Needs verification |
