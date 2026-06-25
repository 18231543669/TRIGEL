# 4.1 Experimental Setup Draft

## English Manuscript Text

### 4.1 Experimental Setting

This section introduces the datasets, comparison methods, evaluation metrics, and implementation settings used in the experiments. Following the objective of imbalanced multi-omics cancer subtype classification, the experimental protocol evaluates both overall predictive performance and class-balanced performance under skewed subtype distributions.

#### 4.1.1 Datasets and Preprocessing

We evaluate TRIGEL on four public multi-omics cancer subtype classification datasets, including BRCA, BRCA104, GBM, and KRCCC. Each dataset contains multiple omics modalities measured from the same set of patients, and each patient is assigned a single cancer subtype label. Table 1 lists the number of patients, the number of subtypes, the subtype distribution, the feature dimensions of different omics modalities, and the imbalance ratio of each dataset. BRCA is used as the main dataset for full comparison, mechanism analysis, and downstream interpretation because it contains more samples and exhibits a pronounced subtype imbalance. BRCA104, GBM, and KRCCC are used to examine model behavior under different cohort sizes and imbalance levels.

**Table 1. Statistics of the multi-omics datasets used in the experiments.**

| Dataset | Patients | Classes | Class distribution | Feature dimensions | Train/validation/test | IR |
|---|---:|---:|---|---|---|---:|
| BRCA | 875 | 5 | 115/131/46/436/147 | 1000/1000/503 | [to be specified] | 9.48 |
| BRCA104 | 104 | 4 | 18/51/12/23 | 17814/354/23094 | [to be specified] | 4.25 |
| GBM | 263 | 4 | 68/80/46/69 | 3010/1250/534 | [to be specified] | 1.74 |
| KRCCC | 122 | 4 | 24/24/62/12 | 329/[to be verified]/[to be verified] | [to be specified] | 5.17 |

The imbalance ratio of each dataset was calculated as

$$
\mathrm{IR}=\frac{\max_{c\in\{1,\ldots,C\}} n_c}
{\min_{c\in\{1,\ldots,C\}} n_c},
\tag{19}
$$

where $C$ is the number of cancer subtypes and $n_c$ is the number of patients belonging to subtype $c$. A larger $\mathrm{IR}$ indicates a more uneven class distribution. In BRCA, the largest subtype contained 436 patients, whereas the smallest subtype contained 46 patients, resulting in an imbalance ratio of 9.48. This setting is consistent with the main problem addressed by TRIGEL, namely learning discriminative patient representations when some subtypes are substantially under-represented.

For each dataset, preprocessing is performed independently for each omics modality. Missing values are handled using [missing-value strategy], and features are [standardized/normalized] using statistics estimated from the training set only. The samples are divided into mutually exclusive training, validation, and test sets using stratified sampling with a ratio of [training ratio]:[validation ratio]:[test ratio]. The validation set is used for hyperparameter selection and early stopping, whereas the test set is used only for final evaluation. Unless otherwise stated, all methods use identical data partitions across repeated runs.

Following the patient graph construction described in Section 3.1.1, each patient is treated as a graph node for every omics modality. For each modality, an undirected patient-similarity graph is constructed using cosine similarity and k-nearest neighbors (KNN). Each patient is connected to its $k$ nearest neighbors, where $k$ is set to [to be specified] or selected from [candidate values] using validation-set performance. The same initial patient-similarity graphs, input features, subtype labels, and data partitions are used for TRIGEL and graph-based comparison methods whenever their model definitions allow this setting.

#### 4.1.2 Comparison Methods

We compare TRIGEL with two groups of methods. The first group includes representative biomedical multi-omics classification methods, which are used to evaluate the ability to learn and integrate heterogeneous molecular information. The second group includes graph-based imbalance learning methods, which are used to evaluate the ability to reduce majority-class bias under skewed subtype distributions.

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

For fair comparison, all methods are evaluated using the same training, validation, and test partitions. Hyperparameters are selected according to validation-set Macro-F1 using the search spaces recommended by the original studies or their official implementations. GraphSHA and IceBerg are originally developed for general class-imbalanced node classification rather than multi-omics integration. Therefore, they are adapted to the same patient-graph input used in this study through [specific multi-omics adaptation strategy], without changing their core imbalance learning mechanisms. Results reported in the original papers are not directly reused because their data partitions, preprocessing procedures, and task settings may differ from those adopted here.

#### 4.1.3 Evaluation Metrics

We use Macro-F1 as the primary evaluation metric because the subtype classification tasks are class-imbalanced. Macro-F1 assigns equal weight to each subtype and is therefore more informative than Accuracy when minority subtypes are under-represented. We also report Accuracy, Macro-Precision, Macro-Recall, Weighted-F1, and Macro-AUC to provide complementary views of model performance. When class-wise results are available, Precision, Recall, and F1 are reported for each subtype to examine whether the performance improvement comes from better recognition of under-represented subtypes.

Each method is repeated with [number of random seeds] random seeds, and the results are reported as mean $\pm$ standard deviation. Hyperparameters and early stopping are determined only by validation-set Macro-F1. The test set is kept fixed for final evaluation and is not used for model selection.

When comparing TRIGEL with the strongest baseline, statistical significance is assessed using [paired t-test/Wilcoxon signed-rank test] across repeated runs, and the corresponding effect size is reported when available.

#### 4.1.4 Experiment Design

TRIGEL is implemented according to the framework described in Chapter 3. For each omics modality, a patient-similarity graph is first constructed using KNN with cosine similarity. The original graph and the class-aware enhanced graph are encoded by modality-specific GCN encoders. The original and enhanced views of the same modality share encoder parameters, and their patient representations are aligned by contrastive learning. Finally, modality-specific representations are integrated using node-level self-attention, and the fused patient representation is used for cancer subtype classification.

The main implementation settings are as follows. The number of GCN layers is [to be specified], the hidden dimension is [to be specified], and the embedding dimension is [to be specified]. The model is optimized using [optimizer] with a learning rate of [to be specified] and weight decay of [to be specified]. Training is performed for at most [maximum epochs] epochs, with early stopping based on validation-set Macro-F1 and a patience of [to be specified]. The contrastive temperature is set to [to be specified], and the classification and contrastive loss weights are set to [to be specified] and [to be specified], respectively.

For Tri-class Guided Stratification (TGS), cancer subtypes are assigned to minority, intermediate, and majority groups according to the subtype proportions in the training set, using thresholds [to be specified]. For Multi-dimensional Graph Enhancement (MGE), class-guided feature enhancement uses group-specific masking rates of [to be specified], and class-aware topology augmentation uses an edge addition probability of [to be specified], a maximum number of new edges of [to be specified], a minority edge weight of [to be specified], and a structural-quality weighting factor of [to be specified]. For Contrastive Fusion Classification (CFC), the number of attention heads is [to be specified], and the attention dropout rate is [to be specified].

All baselines are implemented using their official code when available or reproduced according to the settings described in the original papers. To ensure comparable input information, all methods use the same preprocessed omics matrices and the same training, validation, and test splits. For methods that require graph inputs, the same patient-similarity graphs are used whenever possible. For non-graph methods, the multi-omics inputs are adapted following [specific adaptation strategy].

---

## Terminology Ledger

| Canonical term | Use in Chapter 4 |
|---|---|
| TRIGEL | Full method name; avoid Trigel or TriGEL |
| Tri-class Guided Stratification (TGS) | The subtype grouping strategy into minority, intermediate, and majority groups |
| Multi-dimensional Graph Enhancement (MGE) | The feature and topology enhancement module guided by TGS |
| Contrastive Fusion Classification (CFC) | The contrastive learning, node-level self-attention, and classification component |
| multi-omics data | Use instead of multi-group data |
| omics modality | Main term for each molecular data type |
| patient-similarity graph | Main term for KNN graph over patients |
| k-nearest neighbors (KNN) | Use American spelling: neighbors |
| class imbalance, feature imbalance, topological imbalance | The three imbalance levels introduced in the paper |
| minority, intermediate, and majority groups | Use for TGS groups; reserve subtype/class for cancer labels |
| node-level self-attention | Use instead of attention fusion when referring to CFC |
| class-balanced classification loss | Use instead of vague weighted loss |

---

## Information Still Needed

1. Data source and accession numbers for BRCA, BRCA104, GBM, and KRCCC.
2. Exact omics modality names for each dataset.
3. Missing-value strategy and feature scaling strategy.
4. Train/validation/test split ratio and number of random seeds.
5. Whether KRCCC contains complete files for all omics modalities.
6. Baseline adaptation details for GraphSHA, IceBerg, and non-graph methods.
7. Final implementation hyperparameters used in formal experiments.

---

## Claim-Evidence Check

| Claim | Evidence | Status |
|---|---|---|
| The datasets have different cohort sizes and imbalance levels | Local label statistics and Table 1 | Supported |
| BRCA is the main dataset for downstream interpretation | Current downstream analysis and exported TRIGEL embeddings are BRCA-based | Supported |
| All methods use identical data partitions | Requires final experiment protocol confirmation | Needs confirmation |
| Hyperparameters are selected by validation-set Macro-F1 | Required for a clean formal protocol | Needs confirmation |
| KRCCC contains complete multi-omics matrices | Current local files remain incomplete or unverified | Needs verification |
