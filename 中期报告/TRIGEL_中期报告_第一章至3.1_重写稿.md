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

本课题目前已初步完成 TRIGEL 方法的总体框架设计，并围绕三分类引导分层、多维图增强和对比融合分类三个核心模块开展模型构建。基于三分类引导多维图增强对比学习的非平衡癌症分型模型图如图 3.1 所示。

图 3.1 TRIGEL 模型总体框架图：[待插入]

TRIGEL 主要包括三分类引导分层模块、多维图增强模块和对比融合分类模块。首先，三分类引导分层模块针对每一种组学模态构建患者相似图，并根据癌症亚型样本比例将类别划分为少数类、中间类和多数类；其次，多维图增强模块在类别分层结果的引导下，从特征维度和拓扑维度对患者图进行增强，得到类别感知的增强图；最后，对比融合分类模块将原始图和增强图输入图卷积编码器，通过对比学习对齐增强前后的患者表示，并利用节点级注意力机制融合不同组学模态的表示，完成癌症亚型分类。

#### 3.1.1 三分类引导分层模块

三分类引导分层模块首先针对每一种组学模态构建患者相似图。设第 \(v\) 种组学模态的特征矩阵为 \(X_v \in \mathbb{R}^{N \times D_v}\)，其中 \(N\) 表示患者数量，\(D_v\) 表示该组学模态的特征维度。对于患者 \(p\) 和患者 \(q\)，采用余弦相似度计算二者在第 \(v\) 种组学模态下的相似性：
\[
sim(x_p^{(v)},x_q^{(v)})=
\frac{(x_p^{(v)})^T x_q^{(v)}}{\|x_p^{(v)}\|_2\|x_q^{(v)}\|_2}.
\tag{3.1}
\]
根据患者之间的相似性，为每个患者选择 \(k\) 个最近邻节点，构建邻接矩阵 \(A_v\)，并加入自环以保留节点自身特征，由此得到第 \(v\) 种组学模态下的患者相似图 \(G_v=(X_v,A_v)\)[21]。

在完成患者相似图构建后，模块根据训练集中各癌症亚型的样本数量进行三分类引导分层。设第 \(c\) 个亚型的训练样本数量为 \(n_c\)，训练样本总数为 \(N_{train}\)，则该亚型的样本比例为：
\[
p_c=\frac{n_c}{N_{train}}.
\tag{3.2}
\]
根据样本比例 \(p_c\) 和预设阈值 \(\tau_1,\tau_2\)，当 \(p_c \leq \tau_1\) 时，将该亚型划分为少数类；当 \(\tau_1 < p_c < \tau_2\) 时，将其划分为中间类；当 \(p_c \geq \tau_2\) 时，将其划分为多数类。与传统少数类和多数类二分方式相比，三分类引导分层能够显式保留中间规模类别，使类别划分更加符合癌症亚型样本规模连续变化的特点。

为了进一步减弱多数类样本对模型训练的主导作用，本课题结合有效样本数思想计算类别权重[22]。第 \(c\) 个亚型的有效样本数表示为：
\[
E_c=\frac{1-\beta^{n_c}}{1-\beta}.
\tag{3.3}
\]
其中，\(\beta \in [0,1)\) 用于控制样本数量增加时的边际收益变化。在此基础上，模型将三分类分层结果引入类别权重计算，使分类损失同时考虑样本数量差异和类别分层信息，为后续类别平衡训练提供权重依据。

#### 3.1.2 多维图增强模块

多维图增强模块根据三分类引导分层结果，对每一种组学模态下的原始患者图 \(G_v=(X_v,A_v)\) 进行特征维度和拓扑维度的联合增强，得到增强图 \(\widetilde{G}_v=(\widetilde{X}_v,\widetilde{A}_v,W_v)\)。其中，\(\widetilde{X}_v\) 表示增强后的特征矩阵，\(\widetilde{A}_v\) 表示调整后的邻接矩阵，\(W_v\) 表示类别感知边权矩阵。增强操作主要作用于训练节点，验证节点和测试节点保留原始特征和连接关系，以避免引入数据泄露。

**（1）类别引导的特征增强。**  
类别引导的特征增强根据节点所属类别设置不同强度的特征掩蔽。对于少数类节点，模型保留更多特征信息，以避免有限样本中的关键判别特征被削弱；对于中间类节点，模型采用适度的特征扰动；对于多数类节点，模型更强地抑制低重要性特征，以降低多数类特征对训练过程的主导作用。

具体来说，模型首先在原始患者相似图上计算节点的 PageRank 中心性[23]，并结合节点中心性和特征取值计算第 \(r\) 个特征维度的结构显著性得分：
\[
s_r=\sum_{i=1}^{N}|x_{ir}|\cdot P(i).
\tag{3.4}
\]
其中，\(x_{ir}\) 表示节点 \(i\) 在第 \(r\) 个特征维度上的取值，\(P(i)\) 表示节点 \(i\) 的 PageRank 中心性。若某一特征在结构重要节点上具有较高取值，则其结构显著性得分较高，在增强过程中应被更多保留。随后，模型通过对数变换和反向归一化得到特征掩蔽得分 \(m_r\)，并结合节点所属类别对应的掩蔽强度计算特征掩蔽概率：
\[
p_{drop}^{(i)}(r)=\min(m_r\cdot \rho_{g(y_i)},0.9).
\tag{3.5}
\]
其中，\(y_i\) 表示节点 \(i\) 的类别标签，\(g(y_i)\) 表示其所属的少数类、中间类或多数类分层，\(\rho_{g(y_i)}\) 表示对应类别分层的特征掩蔽强度。最后，根据该概率采样伯努利掩码，得到增强后的特征矩阵 \(\widetilde{X}_v\)。

**（2）类别感知的拓扑增强。**  
类别感知的拓扑增强根据节点所属类别调整图中的连接关系和传播强度。对于少数类训练节点，模型从高中心性候选节点中选择新的邻居，以扩大少数类节点的信息接收范围；对于中间类和多数类训练节点，模型删除部分相似度较低的弱连接，以减少冗余传播和多数类过度平滑问题。

对于少数类训练节点 \(i\)，候选节点从高中心性节点集合中选取，同时排除节点自身和已有邻居。候选边 \((i,j)\) 的得分由特征相似性和结构中心性共同决定：
\[
score(i,j)=\gamma \cdot sim(x_i,x_j)+(1-\gamma)\cdot \overline{P}(j).
\tag{3.6}
\]
其中，\(\gamma\) 用于平衡特征相似性和结构中心性，\(\overline{P}(j)\) 表示归一化后的 PageRank 得分。根据候选边得分和边添加概率，对少数类节点增加新的连接；同时，根据边相似性和边删除概率，对中间类和多数类节点删除部分弱连接。

在完成边连接调整后，模块进一步为边分配类别感知权重。对于边 \((i,j)\)，先根据两端节点所属类别确定该边的类别优先级，优先级顺序为少数类高于中间类，中间类高于多数类。随后结合边的结构质量和类别一致性信息计算最终边权：
\[
w_{ij}=b_{g(i,j)}\cdot \eta_{ij}\cdot (1+\delta q(i,j)).
\tag{3.7}
\]
其中，\(b_{g(i,j)}\) 表示边所属类别优先级对应的基础权重，\(\eta_{ij}\) 表示类别一致性系数，\(q(i,j)\) 表示边的结构质量，\(\delta\) 控制结构质量对边权的影响。通过上述方式，多维图增强模块能够同时从特征和拓扑两个维度缓解非平衡问题，并得到每一种组学模态下的增强图。

#### 3.1.3 对比融合分类模块

对比融合分类模块以原始图集合 \(\mathcal{G}=\{G_1,G_2,\ldots,G_V\}\) 和增强图集合 \(\widetilde{\mathcal{G}}=\{\widetilde{G}_1,\widetilde{G}_2,\ldots,\widetilde{G}_V\}\) 为输入，首先通过图卷积编码器提取不同视图下的患者表示，再通过对比学习对齐原始视图和增强视图中的患者表示，最后利用节点级注意力机制融合不同组学模态的信息，并完成癌症亚型分类。

对于第 \(v\) 种组学模态，原始图和增强图输入同一个组学特异的 GCN 编码器 \(f_v(\cdot)\)，得到两种视图下的节点表示[24]：
\[
H_v=f_v(G_v), \quad \widetilde{H}_v=f_v(\widetilde{G}_v).
\tag{3.8}
\]
其中，原始图使用单位边权，增强图使用多维图增强模块得到的类别感知边权。不同组学模态使用不同的编码器，以保留各组学模态特有的图结构和特征分布；同一组学模态下的原始视图和增强视图共享编码器参数，以保证两种视图的表示处于一致空间。

为了提高患者表示在特征扰动和拓扑扰动下的稳定性，模块将同一患者在原始视图和增强视图下的表示作为正样本对，将不同患者之间的表示作为负样本对，构建双向对比学习损失[25]。设原始视图和增强视图的患者表示经过投影头和归一化后分别为 \(z_i\) 和 \(\widetilde{z}_i\)，则跨视图相似性矩阵为：
\[
S_{ij}=\frac{z_i^T \widetilde{z}_j}{\tau}.
\tag{3.9}
\]
其中，\(\tau\) 为温度系数。对比学习约束同一患者在增强前后的表示保持一致，同时保持不同患者表示之间的区分性，从而提升模型在非平衡扰动下的表示稳定性。

在多组学融合阶段，模块采用节点级注意力机制为不同组学模态分配自适应权重[26]。设患者 \(i\) 在第 \(v\) 种组学模态下的表示为 \(h_{i,v}\)，根据患者整体表示和各模态表示之间的相关性，计算不同组学模态的注意力权重 \(\alpha_{i,v}\)，并得到融合表示：
\[
h_i^f=\sum_{v=1}^{V}\alpha_{i,v}h_{i,v}.
\tag{3.10}
\]
随后，将融合表示输入分类器进行癌症亚型预测，并结合三分类引导分层得到的类别权重计算类别平衡分类损失。模型总损失由分类损失和对比学习损失共同组成：
\[
\mathcal{L}=\lambda_{cls}\mathcal{L}_{cls}+\lambda_{con}\mathcal{L}_{con}.
\tag{3.11}
\]
其中，\(\lambda_{cls}\) 和 \(\lambda_{con}\) 分别控制分类损失和对比学习损失的贡献。通过上述设计，对比融合分类模块能够在类别平衡目标下对齐原始视图和增强视图，并融合不同组学模态的患者表示，从而实现非平衡多组学癌症分型。

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
