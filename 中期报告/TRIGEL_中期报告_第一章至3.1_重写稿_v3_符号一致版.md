# 河北工业大学研究生学位论文中期报告（第一章至 3.1 重写稿）

论文题目：基于三分类引导多维图增强对比学习的非平衡癌症分型研究  
类别：硕士  
所属学院：人工智能与数据科学学院  
学科专业：[待补充]  
研究方向：生物信息学  
导师姓名：[待补充]  
学号：[待补充]  
姓名：[待补充]  
日期：[待补充]

## 一、问题的提出

癌症分型在精准医学中具有重要作用，能够为个体化治疗方案制定和患者预后评估提供支持[1]。然而，癌症数据普遍存在数据不平衡问题[2]。此外，不同分子层面的组学数据在特征维度、稀疏程度和信息含量上存在显著差异，阻碍了多组学数据的有效整合[3]。这些问题严重限制了癌症亚型分类的准确性。因此，同时解决数据不平衡和多组学融合问题，仍然是癌症分型研究中的重要挑战。

数据不平衡表现为多个层面，包括癌症亚型之间的类别不平衡、不同组学模态之间的特征不平衡，以及图结构表示中的拓扑不平衡[2,13]。现有类别不平衡处理方法主要从数据层面、算法层面和集成策略三个方面展开。例如，SMOTE 在数据层面通过特征空间插值生成合成样本，从而缓解数据不平衡问题[4]。然而，当少数类样本数量极少时，该方法可能难以充分刻画真实数据分布。GAN 和 VAE 等生成模型则从算法层面学习潜在数据分布，以生成更接近真实分布的合成样本[5-6]。与直接改变数据的方法不同，Focal Loss 采用代价敏感策略处理不平衡问题，通过降低易分类样本的权重，使模型在学习过程中动态关注难分类样本和少数类样本[7]。集成策略则通过聚合多个在重平衡子集上训练的学习器，进一步提高模型的鲁棒性[8-9]。尽管这些方法能够缓解类别不平衡问题，但多数方法只是简单地将样本划分为少数类和多数类。这种二分方式难以描述数据内部的分布结构，也难以适应多组学数据中常见的梯度式不平衡特征。

在多组学数据中，不同组学模态在维度和稀疏程度上存在明显差异，导致分子层面之间出现特征不平衡，并对特征提取造成直接影响。针对这一问题，MOVE 使用多组学变分自编码器对不同模态进行联合降维，从而建立统一的低维表示[10]。MoGCN 进一步引入图卷积网络，先使用自编码器提取紧凑特征，再结合图结构进行分类[11]。然而，这类方法可能忽略少数类特征在高维空间中的分布偏斜。AE-CTGAN 通过引入自编码器框架，在降维的同时对少数类特征进行增强，从而缓解这一问题[12]。但是，当多组学数据存在极高稀疏性和严重偏斜的模态分布时，如何在不丢失跨组学交互信息的前提下实现平衡的特征提取，仍然是一个尚未充分解决的问题。

图学习的引入为不平衡数据建模带来了新的挑战，其中图结构中的拓扑不平衡已成为一个关键问题[13]。少数类节点通常处于结构不利位置，并在消息传递过程中接收较弱的监督信号。为了解决这一问题，GraphSMOTE 在嵌入空间中生成少数类节点，并使用可学习的生成器构造边[14]。TAM 采用另一种策略，通过引入拓扑感知间隔，根据节点连接模式自适应调整决策边界[15]。此外，ReNode 为少数类节点分配拓扑感知权重，重新平衡其在传播过程中的影响，从而缓解少数类节点在图结构中的边缘化问题[16]。尽管现有方法能够在一定程度上缓解图结构数据中的拓扑不平衡，但它们在处理多组学数据中更加复杂的场景时仍然存在局限。

有效融合多组学数据也是提高癌症亚型分类准确性的关键。MoGCN 将自编码器与图卷积网络结合，用于多组学数据整合[11]。MOGONET 通过多视图图神经网络联合探索模态特异学习和跨组学关联[17]。基于异质多层图的多组学图神经网络框架进一步将多组学数据建模为异质多层图，以嵌入组学内部和组学之间的连接关系[18]。MMGN 将多重网络与泛癌多组学数据结合，采用图神经网络和负样本推断识别癌症驱动基因，并通过互信息和一致性正则项增强基因特征学习[19]。DrugCellGNN 将多组学数据拼接为经过 PCA 降维的特征矩阵，通过加权融合相似性度量构建相似网络，并使用图卷积网络实现多组学数据融合[20]。尽管这些方法在特征提取和图融合方面取得了一定进展，但在融合过程中，仍然缺少一种能够同时处理类别不平衡、特征不平衡和拓扑不平衡的整体解决方案。

综上所述，非平衡多组学癌症分型仍面临以下问题：现有类别不平衡方法多采用少数类和多数类二分策略，难以刻画癌症亚型样本规模的连续变化；现有多组学特征学习方法虽然能够降低高维噪声影响，但对不同模态之间的维度差异、稀疏差异以及少数类特征偏斜考虑不足；现有图不平衡方法主要面向单一图结构，难以适应多组学数据中类别不平衡、特征不平衡和拓扑不平衡相互耦合的复杂情况；现有多组学融合方法多关注如何提高融合表示质量，而缺少在融合过程中同步校正多层不平衡的统一建模方式。因此，研究一种能够同时考虑类别、特征和拓扑不平衡的多组学癌症分型方法，对于提高癌症亚型识别的准确性和稳定性具有重要意义。

## 二、课题研究内容

### 2.1 主要研究内容

针对非平衡多组学癌症分型中同时存在的数据不平衡和多组学融合问题，本课题研究一种基于三分类引导多维图增强对比学习的癌症分型方法（Tri-class Guided Multi-dimensional Graph Enhanced Contrastive Learning，TRIGEL）。该方法主要包括三分类引导分层、多维图增强和对比融合分类三个部分。首先，根据不同癌症亚型的样本比例，将类别划分为少数类、中间类和多数类，以更细致地刻画类别规模差异；其次，在三分类引导下，从特征维度和拓扑维度对多组学患者图进行增强，以缓解特征不平衡和拓扑不平衡；最后，通过图对比学习对原始图视图和增强图视图进行表示对齐，并利用节点级注意力机制融合不同组学模态的患者表示，从而完成癌症亚型分类。

（1）三分类引导分层。针对传统不平衡处理方法通常只区分少数类和多数类，难以描述癌症亚型样本规模连续变化的问题，本课题根据训练集中各癌症亚型的样本比例进行类别分层。对于每一种组学模态，首先构建患者相似图，并根据癌症亚型样本数量将类别划分为少数类、中间类和多数类。与传统二分类划分方式相比，三分类引导分层能够显式保留中等规模类别，使类别规模差异从简单的少数类/多数类划分转变为更平滑的类别梯度，为后续特征增强、拓扑增强和类别平衡训练提供指导。

（2）多维图增强。针对多组学数据中的特征不平衡和图结构中的拓扑不平衡问题，本课题设计多维图增强机制。在特征维度上，根据特征重要性和类别分层结果设置差异化掩蔽强度。对于少数类节点，保留更多特征信息，以避免关键判别特征被过度削弱；对于中间类节点，采用适度的特征扰动；对于多数类节点，则对低重要性特征进行更强的掩蔽，以降低多数类特征对训练过程的主导作用。在拓扑维度上，根据节点所属类别调整图中的边连接关系和边权。对于少数类节点，增加连接并提高边权，以扩大其邻域信息来源；对于多数类节点，删除部分弱连接并降低边权，以减少冗余传播；对于中间类节点，则进行适度调整。通过特征增强和拓扑增强的联合设计，模型能够在保留关键判别结构和分布信息的同时，缓解多组学数据中的多层不平衡问题。

（3）对比融合分类。针对非平衡条件下多组学数据融合困难的问题，本课题设计对比学习融合分类框架。对于每一种组学模态，将原始患者图和增强患者图作为两种视图输入图卷积编码器，学习患者节点表示[24]。随后，将同一患者在原始视图和增强视图中的表示作为正样本对，将不同患者之间的表示作为负样本对，通过对比学习约束增强前后表示的一致性[25]，使模型能够学习对特征扰动和拓扑扰动更加稳定的患者表示。在多组学融合阶段，采用节点级注意力机制为不同组学模态分配自适应权重[26]，并将各模态表示融合为最终患者表示。分类阶段结合类别平衡损失，减弱多数类样本对模型训练的影响[22]，最终实现非平衡多组学癌症亚型分类。

后续实验将在公开多组学癌症数据集上验证模型性能，并与现有不平衡学习方法、图学习方法和多组学融合方法进行对比。实验评价指标包括 Accuracy、F1-score、Macro-F1、Precision 和 Recall 等，其中重点关注 Macro-F1 和 Recall 等对非平衡分类更敏感的指标。同时，通过消融实验验证三分类引导分层、多维图增强和对比融合分类三个模块的有效性，并进一步开展参数敏感性分析和表示可视化分析，以验证模型在非平衡癌症分型任务中的稳定性和有效性。

### 2.2 研究的创新性

（1）针对类别不平衡问题，提出三分类引导分层策略。现有不平衡处理方法通常只将样本划分为少数类和多数类，这种二分方式难以描述癌症亚型样本规模的内部差异。本课题根据样本比例将癌症亚型划分为少数类、中间类和多数类，更细致地刻画数据内部的分布结构和类别之间渐进的密度差异，为后续多维图增强提供类别引导信息。

（2）针对特征不平衡和拓扑不平衡问题，提出多维图增强机制。在特征维度上，TRIGEL 根据特征重要性权重和类别差异化强度进行特征掩蔽，以缓解不同组学模态之间的特征偏斜。在拓扑维度上，TRIGEL 为少数类节点增加边并提高边权，为多数类节点删除边并降低边权，同时对中间类节点进行适度调整。该多维增强策略能够在缓解多组学数据不平衡的同时，保留关键判别结构和分布信息。

（3）针对非平衡多组学数据融合问题，设计对比学习融合分类框架。TRIGEL 在类别平衡目标下对增强后的多组学视图进行表示对齐，并通过节点级注意力机制融合不同组学模态的患者表示。该框架将跨组学整合和不平衡缓解放在统一流程中，使二者能够相互促进，从而提升非平衡癌症分型的表示学习能力和分类性能。

## 三、研究进展及阶段性成果

### 3.1 课题研究进展

本课题目前已完成 TRIGEL 方法的总体框架设计。TRIGEL 面向非平衡多组学癌症分型任务，主要由三分类引导分层模块（Tri-class Guided Stratification，TGS）、多维图增强模块（Multi-dimensional Graph Enhancement，MGE）和对比融合分类模块（Contrastive Fusion Classification，CFC）组成。模型总体流程如图 3.1 所示。

图 3.1 TRIGEL 模型总体框架图：[待插入]

TRIGEL 首先针对每一种组学模态构建患者相似图，并根据癌症亚型样本数量将类别划分为少数类、中间类和多数类；随后，在三分类分层结果的引导下，对每个组学图分别进行类别感知的特征增强和拓扑增强，得到增强图；最后，将原始图和增强图输入图卷积编码器，通过对比学习对齐增强前后的患者表示，并利用节点级自注意力机制融合不同组学模态的表示，完成癌症亚型分类。

#### 3.1.1 三分类引导分层模块

三分类引导分层模块将每一种组学模态的数据转换为患者相似图，并根据癌症亚型样本数量进行类别分层。得到的分层结果为多维图增强模块提供类别感知的指导信息。

**（1）患者相似图构建。** 对于 $V$ 种组学模态中的第 $v$ 种组学，使用 K 近邻方法构建患者相似图。设 $X_v \in \mathbb{R}^{N\times D_v}$ 表示第 $v$ 种组学模态的特征矩阵，其中 $N$ 表示患者数量，$D_v$ 表示该组学模态的特征维度。对于患者 $p$ 和患者 $q$，二者在第 $v$ 种组学模态下的余弦相似度定义为：
\[
sim\left(x_p^{(v)},x_q^{(v)}\right)=
\frac{\left(x_p^{(v)}\right)^T x_q^{(v)}}{\left\|x_p^{(v)}\right\|_2\left\|x_q^{(v)}\right\|_2}.
\tag{3.1}
\]
其中，$x_p^{(v)}$ 和 $x_q^{(v)}$ 分别表示患者 $p$ 和患者 $q$ 在第 $v$ 种组学模态下的特征向量，$\|\cdot\|$ 表示 $L_2$ 范数。根据计算得到的相似性，为每个患者选择 $k$ 个最近邻节点，并构建邻接矩阵 $A_v \in \{0,1\}^{N\times N}$。同时，在邻接矩阵中加入自环，使节点在图卷积过程中能够保留自身特征信息。由此得到第 $v$ 种组学模态对应的患者相似图 $G_v=(X_v,A_v)$。

**（2）三分类引导分层。** 为了对不同样本规模的癌症亚型进行差异化处理，TRIGEL 根据训练集中各亚型的样本比例进行三分类引导分层。设癌症分型任务中共有 $C$ 个亚型，第 $c$ 个亚型的训练样本数量为 $n_c$，训练样本总数为 $N_{train}$，则亚型 $c$ 的样本比例定义为：
\[
p_c=\frac{n_c}{N_{train}}.
\tag{3.2}
\]
根据 $p_c$ 与预设阈值 $\eta_1,\eta_2$ 的关系，将亚型 $c$ 划分为不同类别。当 $p_c\leq \eta_1$ 时，将其划分为少数类；当 $\eta_1<p_c<\eta_2$ 时，将其划分为中间类；当 $p_c\geq \eta_2$ 时，将其划分为多数类。与传统少数类和多数类二分方式相比，该策略显式保留中等规模类别，为后续图增强提供更细粒度的类别指导。

进一步地，TRIGEL 使用有效样本数刻画不同类别样本数量增加时的边际收益变化。亚型 $c$ 的有效样本数定义为：
\[
E_c=\frac{1-\beta^{n_c}}{1-\beta}.
\tag{3.3}
\]
其中，$\beta\in[0,1)$ 控制样本数量增加时的边际收益。当 $\beta=0$ 时，所有亚型的 $E_c=1$；当 $\beta$ 越接近 1 时，$E_c$ 越接近真实样本数量 $n_c$。在有效样本数的基础上，TRIGEL 将三分类分层结果进一步引入亚型权重计算：
\[
w_c=\frac{\tau_{t(c)}}{E_c\cdot \bar{w}}.
\tag{3.4}
\]
其中，$t(c)\in\{minority,intermediate,majority\}$ 表示亚型 $c$ 的分层结果，$\tau_{t(c)}$ 表示对应分层的组特异权重系数。归一化因子 $\bar{w}=\frac{1}{C}\sum_{j=1}^{C}\frac{\tau_{t(j)}}{E_j}$ 用于保证所有亚型权重的均值为 1。该权重同时结合了有效样本数捕获的连续频率信息和 TGS 得到的离散分层信息，后续用于类别平衡分类损失。

#### 3.1.2 多维图增强模块

多维图增强模块在 TGS 产生的类别分层结果指导下，从特征维度和拓扑维度增强每一个原始患者图 $G_v=(X_v,A_v)$，以缓解特征不平衡和拓扑不平衡。该模块生成增强图 $\hat{G}_v=(\hat{X}_v,\hat{A}_v,W_v)$，其中 $\hat{X}_v$ 表示增强后的特征矩阵，$\hat{A}_v$ 表示修改后的邻接矩阵，$W_v$ 表示类别感知边权矩阵。增强操作仅作用于训练节点，验证节点和测试节点保留原始特征和连接关系。

**（1）类别引导的特征增强。** 类别引导的特征增强对少数类节点保留更多信息，对中间类节点采用适度掩蔽，对多数类节点更强地抑制低显著性特征。特征显著性由原始患者图中的特征幅值和节点中心性共同估计。

首先，在原始患者相似图上使用 PageRank 计算节点中心性：
\[
PR(i)=\alpha\sum_{j\in\mathcal{N}(i)}\frac{A_{ji}}{d_j^{out}}PR(j)+\frac{1-\alpha}{N}.
\tag{3.5}
\]
其中，$\alpha\in(0,1)$ 为阻尼系数，$\mathcal{N}(i)$ 表示与节点 $i$ 相连的节点集合，$A_{ji}$ 表示节点 $j$ 到节点 $i$ 是否存在边，$d_j^{out}=\sum_{k=1}^{N}A_{jk}$ 表示节点 $j$ 的出度，$N$ 表示节点总数。由于患者图为无向图，因此 $A_{ji}=A_{ij}$。

基于 PageRank 得分，第 $f$ 个特征维度的结构显著性得分定义为：
\[
s_f=\sum_{i=1}^{N}\left|x_{if}\right|\cdot PR(i).
\tag{3.6}
\]
其中，$x_{if}$ 表示节点 $i$ 在特征维度 $f$ 上的取值。$s_f$ 越大，表示特征 $f$ 在结构重要节点上具有更大取值，因此在特征掩蔽过程中具有更高重要性。

为了将 $s_f$ 转换为掩蔽概率，先对其进行对数变换，变换后的结果仍记为 $s_f=\log(s_f+\epsilon)$，其中 $\epsilon$ 为防止数值不稳定的极小正数。随后采用反向最小-最大归一化计算特征 $f$ 的相对掩蔽得分：
\[
\rho_f=\frac{s_{f,max}-s_f}{s_{f,max}-s_{f,min}+\epsilon}.
\tag{3.7}
\]
其中，$s_{f,max}$ 和 $s_{f,min}$ 分别表示所有特征维度中变换后结构显著性得分的最大值和最小值。$\rho_f$ 越大，表示该特征结构显著性越低，被掩蔽的倾向越强。

对于训练节点 $n$，其特征 $f$ 的掩蔽概率定义为：
\[
p_{drop}^{(n)}(f)=min\left(\sqrt{\rho_f\cdot r_{t(y_n)}},0.9\right).
\tag{3.8}
\]
其中，$y_n$ 表示节点 $n$ 的亚型标签，$t(y_n)$ 表示其所属的少数类、中间类或多数类分层，$r_{t(y_n)}\in[0,1]$ 表示对应分层的组特异特征丢弃率。根据 $p_{drop}^{(n)}(f)$ 采样伯努利掩码，得到增强后的特征矩阵 $\hat{X}_v$。上界 0.9 用于避免所有特征信息被完全删除。

**（2）类别感知的拓扑增强。** 类别感知的拓扑增强根据 TGS 提供的分层结果调整图连接关系和传播强度。具体来说，对少数类节点引入新边以扩展其邻域，对中间类和多数类节点删除弱连接以抑制冗余传播，并为得到的边分配组特异权重。

对于每个少数类训练节点 $n$，候选节点从高中心性节点集合 $\mathcal{H}=\{j:PR(j)\geq PR_{75}\}$ 中选取，同时排除已有邻居和节点自身。每个候选节点根据特征相似性和结构中心性共同评分：
\[
score(n,j)=\lambda\cdot sim(x_n,x_j)+(1-\lambda)\cdot \widetilde{PR}(j).
\tag{3.9}
\]
其中，$\widetilde{PR}(j)=PR(j)/PR_{max}$，$\lambda\in[0,1]$ 用于平衡两个部分，$PR_{max}$ 表示最大的 PageRank 得分。随后更新邻接矩阵：
\[
\hat{A}_{nj}=\begin{cases}
1, & A_{nj}^{(v)}=0,\ t(y_n)=minority,\ score(n,j)>\theta,\ u<p_{add},\\
0, & A_{nj}^{(v)}=1,\ t(y_n)\in\{intermediate,majority\},\ sim(x_n,x_j)<\delta,\ u<p_{rem},\\
A_{nj}, & otherwise.
\end{cases}
\tag{3.10}
\]
其中，$u\sim U(0,1)$，$\theta$ 和 $\delta$ 分别表示边添加和边删除阈值，$p_{add}$ 和 $p_{rem}$ 分别控制对应的增强概率。由于患者图为无向图，每次边添加或边删除会同时作用于 $\hat{A}_{nj}^{(v)}$ 和 $\hat{A}_{jn}^{(v)}$。

为了进一步调节信息传播，每条边被赋予其两端节点中优先级更高的类别，优先级顺序为 minority $>$ intermediate $>$ majority。边 $(n,j)$ 的结构质量定义为：
\[
q(n,j)=\frac{PR(n)+PR(j)}{2PR_{max}}.
\tag{3.11}
\]
最终边权计算为：
\[
w(e_{nj})=\tau_e(t(e_{nj}))\kappa(e_{nj})\left[1+\gamma q(n,j)\right].
\tag{3.12}
\]
其中，$\tau_e$ 表示组特异基础权重，并满足 $\tau_e(minority)>\tau_e(intermediate)>\tau_e(majority)=1$。当 $y_n=y_j$ 时，亚型一致性因子设为 $\kappa(e_{nj})=\kappa>1$；否则 $\kappa(e_{nj})=1$。参数 $\gamma$ 用于控制结构质量的影响。所有边权被限制在 $[0.5,5.0]$ 范围内。结合特征增强，该过程为每一种组学模态得到增强图 $\hat{G}_v=(\hat{X}_v,\hat{A}_v,W_v)$。

#### 3.1.3 对比融合分类模块

对比融合分类模块以原始图集合 $\mathcal{G}=\{G_1,\ldots,G_V\}$ 和增强图集合 $\hat{\mathcal{G}}=\{\hat{G}_1,\ldots,\hat{G}_V\}$ 为输入。该模块首先从两种图视图中提取模态特异表示，再通过对比学习对齐原始视图和增强视图下的患者表示，最后通过节点级自注意力机制整合互补的多组学信息并进行亚型分类。

**（1）对比表示学习。** 对于第 $v$ 种组学模态，原始图和增强图由同一个模态特异 GCN 编码器 $f_v(\cdot)$ 处理：
\[
H_v=f_v(G_v),\quad \hat{H}_v=f_v(\hat{G}_v).
\tag{3.13}
\]
其中，$H_v$ 和 $\hat{H}_v$ 分别表示两种视图下得到的节点表示矩阵。原始图 $G_v$ 使用单位边权，增强图 $\hat{G}_v$ 使用式（3.12）中定义的边权 $W_v$。不同组学模态使用独立编码器以保留模态特异图模式，同一组学模态下的原始视图和增强视图共享编码器参数。

对于每一种视图，先按照多组学融合方式得到患者级表示，再通过两层投影头映射为 $L_2$ 归一化嵌入 $z_n$ 和 $\hat{z}_n$。同一患者在两种视图中的表示构成正样本对，不同患者之间的表示构成负样本对。对于训练患者，跨视图相似性矩阵 $S$ 定义为：
\[
S_{nj}=\frac{z_n^T\hat{z}_j}{\tau}.
\tag{3.14}
\]
其中，$\tau>0$ 表示对比学习温度系数。双向对比损失定义为：
\[
\mathcal{L}_{con}=\frac{1}{2}\left[CE(S,\mathbf{y}^{id})+CE(S^T,\mathbf{y}^{id})\right].
\tag{3.15}
\]
其中，$CE$ 表示行方向交叉熵，$\mathbf{y}^{id}$ 用于标识相似性矩阵 $S$ 的对角线元素为正样本对。该目标用于保持同一患者在类别感知特征扰动和拓扑扰动下的身份一致性。

**（2）多组学融合与分类。** 由于不同组学模态对不同患者的贡献不同，CFC 使用多头自注意力估计患者特异的模态权重。设 $h_{n,v}$ 表示患者 $n$ 在第 $v$ 种组学模态下的表示，线性变换后得到：
\[
\tilde{h}_{n,v}=\Phi_v h_{n,v},\quad \bar{h}_n=V^{-1}\sum_{v=1}^{V}\tilde{h}_{n,v}.
\]
对于注意力头 $k$，查询由 $\bar{h}_n$ 构造，键由 $\tilde{h}_{n,v}$ 构造。第 $v$ 种组学模态的权重为：
\[
\alpha_{n,v}=\frac{1}{K}\sum_{k=1}^{K}softmax_v\left(\frac{\left(q_n^{(k)}\right)^T k_{n,v}^{(k)}}{\xi\sqrt{d_k}}\right).
\tag{3.16}
\]
其中，$K$ 表示注意力头数量，$d_k$ 表示每个注意力头的维度，$\xi>0$ 控制注意力分布的尖锐程度。

患者 $n$ 的融合表示和最终多组学表示分别为：
\[
h_n^F=LN\left[W_o\left(\sum_{v=1}^{V}\alpha_{n,v}\tilde{h}_{n,v}\right)+\bar{h}_n\right],\quad
r_n=Concat(h_{n,1},h_{n,2},\ldots,h_{n,V},h_n^F).
\tag{3.17}
\]
其中，$W_o$ 表示输出投影矩阵，$LN$ 表示层归一化。增强视图采用相同操作得到 $\hat{r}_n$。随后，将增强视图表示 $\hat{r}_n$ 输入分类器，并将式（3.4）中定义的亚型权重 $w_{y_n}$ 引入交叉熵损失 $\mathcal{L}_{cls}$。最终训练目标为：
\[
\mathcal{L}=\lambda_{cls}\mathcal{L}_{cls}+\lambda_{con}\mathcal{L}_{con}.
\tag{3.18}
\]
其中，$\lambda_{cls}$ 和 $\lambda_{con}$ 分别控制亚型分类和跨视图对齐的损失贡献。通过上述方式，CFC 在统一目标中结合类别平衡监督、增强不变表示学习和患者特异多组学融合，从而完成非平衡多组学癌症分型。

## 参考文献

[1] Rappoport N, Shamir R. Multi-omic and multi-view clustering algorithms: review and cancer benchmark[J]. Nucleic Acids Research, 2018, 46(20): 10546-10562.

[2] He H, Garcia E A. Learning from imbalanced data[J]. IEEE Transactions on Knowledge and Data Engineering, 2009, 21(9): 1263-1284.

[3] Wörheide M A, Krumsiek J, Kastenmüller G, et al. Multi-omics integration in biomedical research: a metabolomics-centric review[J]. Analytica Chimica Acta, 2021, 1141: 144-162.

[4] Chawla N V, Bowyer K W, Hall L O, et al. SMOTE: Synthetic Minority Over-sampling Technique[J]. Journal of Artificial Intelligence Research, 2002, 16: 321-357.

[5] Goodfellow I, Pouget-Abadie J, Mirza M, et al. Generative adversarial nets[C]. Advances in Neural Information Processing Systems, 2014: 2672-2680.

[6] Kingma D P, Welling M. Auto-Encoding Variational Bayes[C]. International Conference on Learning Representations, 2014.

[7] Lin T Y, Goyal P, Girshick R, et al. Focal Loss for Dense Object Detection[C]. IEEE International Conference on Computer Vision, 2017: 2980-2988.

[8] Liu X Y, Wu J, Zhou Z H. Exploratory undersampling for class-imbalance learning[J]. IEEE Transactions on Systems, Man, and Cybernetics, Part B, 2009, 39(2): 539-550.

[9] Galar M, Fernández A, Barrenechea E, et al. A review on ensembles for the class imbalance problem: bagging-, boosting-, and hybrid-based approaches[J]. IEEE Transactions on Systems, Man, and Cybernetics, Part C, 2012, 42(4): 463-484.

[10] Allesøe R L, Lundgaard A T, Møller N, et al. Discovery of drug-omics associations in type 2 diabetes with generative deep-learning models[J]. Nature Biotechnology, 2023, 41(3): 399-408.

[11] Li X, Ma J, Leng L, et al. MoGCN: A Multi-Omics Integration Method Based on Graph Convolutional Network for Cancer Subtype Analysis[J]. Frontiers in Genetics, 2022, 13: 806842.

[12] Al-Hurani I, Alkhawaldeh R S, Al-Habashneh S. AE-CTGAN: Autoencoder–Conditional Tabular GAN for Class Imbalance Learning[J]. Algorithms, 2026, 19(2): 95.

[13] Chen D, Lin Y, Li W, et al. Topology-Imbalance Learning for Semi-Supervised Node Classification[C]. Advances in Neural Information Processing Systems, 2021: 29885-29897.

[14] Zhao T, Zhang X, Wang S. GraphSMOTE: Imbalanced Node Classification on Graphs with Graph Neural Networks[C]. ACM International Conference on Web Search and Data Mining, 2021: 833-841.

[15] Song J, Park J, Yang E. TAM: Topology-Aware Margin Loss for Class-Imbalanced Node Classification[C]. International Conference on Machine Learning, 2022: 20369-20383.

[16] Chen D, Lin Y, Li W, et al. ReNode: Topology-Imbalance Learning for Semi-Supervised Node Classification[C]. Advances in Neural Information Processing Systems, 2021: 29885-29897.

[17] Wang T, Shao W, Huang Z, et al. MOGONET integrates multi-omics data using graph convolutional networks allowing patient classification and biomarker identification[J]. Nature Communications, 2021, 12: 3445.

[18] Li B, Nabavi S. A Multimodal Graph Neural Network Framework of Cancer Molecular Subtype Classification[EB/OL]. arXiv:2302.12838, 2023.

[19] Li X, Lin B, Yang H, et al. Multiplex Networks and Pan-Cancer Multiomics-Based Driver Gene Identification Using Graph Neural Networks[J]. Big Data Mining and Analytics, 2024, 7(4): 1062-1077.

[20] Aly G A, Seoud R A A, Salem D A. DrugCellGNN: Graph Convolutional Networks for Integrating Omics and Drug Similarities in Cancer Therapy Prediction[J]. International Journal of Advanced Computer Science and Applications, 2025, 16(12): 1-10.

[21] Cover T, Hart P. Nearest neighbor pattern classification[J]. IEEE Transactions on Information Theory, 1967, 13(1): 21-27.

[22] Cui Y, Jia M, Lin T Y, et al. Class-Balanced Loss Based on Effective Number of Samples[C]. IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019: 9268-9277.

[23] Page L, Brin S, Motwani R, et al. The PageRank citation ranking: Bringing order to the web[R]. Stanford InfoLab, 1999.

[24] Kipf T N, Welling M. Semi-Supervised Classification with Graph Convolutional Networks[C]. International Conference on Learning Representations, 2017.

[25] Chen T, Kornblith S, Norouzi M, et al. A Simple Framework for Contrastive Learning of Visual Representations[C]. International Conference on Machine Learning, 2020: 1597-1607.

[26] Vaswani A, Shazeer N, Parmar N, et al. Attention Is All You Need[C]. Advances in Neural Information Processing Systems, 2017: 5998-6008.
