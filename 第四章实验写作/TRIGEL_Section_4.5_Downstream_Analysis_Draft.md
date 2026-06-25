# 4.5 Downstream Analysis and Model Interpretation Draft

## English Manuscript Text

### 4.5 Downstream Analysis and Model Interpretation

To further examine whether TRIGEL learns subtype-consistent patient representations, we conduct downstream analyses on the BRCA cohort. BRCA is selected because it contains five PAM50 subtypes and shows a clear class-imbalance pattern. The analysis focuses on three aspects: molecular subtype patterns in the original omics data, subtype separation in the learned TRIGEL embedding, and subtype-consistent topology in patient-similarity graphs.

It should be noted that these analyses are not used for model training, hyperparameter selection, or early stopping. The raw omics heatmap is used to validate the subtype signal in the original data. The embedding similarity analysis directly evaluates the model output. The graph connectivity analysis examines whether the patient-similarity graphs used by TRIGEL provide a reasonable neighborhood structure for graph-based learning.

#### 4.5.1 Molecular subtype structure in the original omics space

We first examine whether the subtype annotations in BRCA are reflected in the original omics measurements. For each omics modality, features are ranked by an ANOVA-like ratio between subtype-level variation and within-subtype variation. The top-ranked features are then visualized after row-wise standardization.

As shown in Fig. X, samples sorted by PAM50 subtype show block-like variation among the top subtype-discriminative features in Omics 1. Several highly ranked features, including ESR1, FOXC1, MLPH, ERBB4, and TFF3, exhibit subtype-dependent patterns. This result indicates that the BRCA cohort contains molecular signals aligned with the PAM50 annotations. Therefore, the dataset provides a meaningful basis for subtype classification and multi-omics representation learning. Since this heatmap is generated from the original data, it is used as data-level validation rather than direct evidence of TRIGEL's learned representation.

**Figure X. PAM50-discriminative molecular profiles in Omics 1.** Samples were sorted by PAM50 subtype, and the top 15 subtype-discriminative features in Omics 1 were shown after row-wise z-score normalization. Red and blue indicate relatively high and low standardized feature values, respectively. The block-like patterns indicate that the original omics measurements contain subtype-associated molecular variation.

#### 4.5.2 TRIGEL strengthens subtype separation in the learned representation space

We then evaluate the final representation learned by TRIGEL. Pairwise cosine similarities are computed for patients represented in each individual omics modality and in the final TRIGEL embedding. For each representation, we calculate the mean similarity between patients from the same PAM50 subtype and the mean similarity between patients from different PAM50 subtypes.

As shown in Fig. X, the individual omics modalities contain subtype information but show limited separation. Omics 1 achieves a within-subtype mean similarity of 0.213 and a between-subtype mean similarity of -0.084, corresponding to a separation score of 0.297. Omics 2 and Omics 3 show smaller separation scores of 0.214 and 0.108, respectively. In contrast, the TRIGEL embedding increases the within-subtype mean similarity to 0.526 and reduces the between-subtype mean similarity to -0.172, yielding a separation score of 0.699. The separation score of TRIGEL is more than twice that of Omics 1 and is substantially higher than those of Omics 2 and Omics 3.

This result indicates that TRIGEL learns a more subtype-discriminative patient representation than any single omics modality. Compared with the original input spaces, the final embedding makes patients from the same subtype more compact and patients from different subtypes more separated. This provides representation-level evidence for the learned integrated embedding.

**Figure X. Within-subtype and between-subtype similarity across representations.** Pairwise cosine similarities were computed for each single-omics representation and for the learned TRIGEL embedding. Within-subtype similarity denotes the mean similarity between patients with the same PAM50 subtype, whereas between-subtype similarity denotes the mean similarity between patients with different PAM50 subtypes. TRIGEL produced the largest gap between within-subtype and between-subtype similarity.

The sample-level similarity heatmap further confirms this trend. As shown in Fig. X, after patients are ordered by PAM50 subtype, the TRIGEL embedding forms high-similarity blocks along the diagonal and lower-similarity regions across several subtype pairs. The bar plot summarizes the global within- and between-subtype similarity, whereas the heatmap shows the patient-level organization of the learned embedding. Together, these results show that the final TRIGEL embedding is more aligned with known PAM50 subtype structure than the single-omics representations.

**Figure X. Sample similarity heatmap of the learned TRIGEL embedding.** Rows and columns correspond to BRCA patients sorted by PAM50 subtype. Each entry represents the cosine similarity between two patients in the TRIGEL embedding space. Brighter diagonal blocks indicate stronger within-subtype similarity.

#### 4.5.3 Subtype-consistent topology in patient-similarity graphs

Because TRIGEL uses patient-similarity graphs as graph encoder inputs, we further examine whether the initial graph topology is consistent with PAM50 subtype organization. For each omics-specific graph, we calculate the proportion of edge weight connecting patients from the same PAM50 subtype.

As shown in Fig. X, all three patient-similarity graphs show enrichment of intra-subtype connections. Graph 1 contains 6,095 edges, among which 65.8% of the edge weight connects patients from the same PAM50 subtype. Graph 2 and Graph 3 contain 6,068 and 5,975 edges, with intra-subtype edge ratios of 57.1% and 56.3%, respectively. In comparison, the expected proportion of same-subtype patient pairs under random pairing is approximately 31.8%. Therefore, the observed intra-subtype edge ratios are clearly higher than the random-pairing baseline.

This result indicates that the patient-similarity graphs are not arbitrary connectivity structures. Instead, they preferentially connect patients sharing the same molecular subtype, providing a reasonable topological basis for graph-based message passing. This analysis does not independently prove the effectiveness of MGE or contrastive learning; rather, it shows that the graph inputs preserve subtype-relevant information before representation learning.

**Figure X. Intra-subtype edge ratios in patient-similarity graphs.** For each omics-specific patient-similarity graph, the ratio of edge weight connecting patients with the same PAM50 subtype was calculated. All graphs showed intra-subtype edge ratios above the random-pairing baseline estimated from the cohort subtype distribution.

#### 4.5.4 Summary of downstream evidence

In summary, the downstream analyses provide multi-level evidence for interpreting TRIGEL. The molecular profile heatmap confirms that PAM50 subtypes in BRCA are associated with observable molecular patterns. The representation-level similarity analysis shows that the final TRIGEL embedding achieves stronger within-subtype compactness and between-subtype separation than individual omics modalities. The graph connectivity analysis further shows that the patient-similarity graphs contain subtype-consistent topology. These findings support that TRIGEL learns an integrated patient representation aligned with known molecular subtype structure. However, this evidence should be interpreted as representation-level and topology-level support, rather than causal biological validation.

---

## Figure Placement

| Figure | File | Role in the section |
|---|---|---|
| Fig. X | `E:/TRIGEL/下游实验分析/results/01_molecular_profile_heatmap/20260622_154746/omics1_profile_heatmap.png` | Data-level molecular separability of PAM50 subtypes |
| Fig. X | `E:/TRIGEL/下游实验分析/results/03_multiomics_similarity_structure/20260623_084213/within_between_similarity.png` | Core model-output evidence: TRIGEL embedding improves subtype compactness and separation |
| Fig. X | `E:/TRIGEL/下游实验分析/results/03_multiomics_similarity_structure/20260623_084213/trigel_similarity_heatmap.png` | Qualitative visualization of the TRIGEL embedding geometry |
| Fig. X | `E:/TRIGEL/下游实验分析/results/04_graph_subtype_connectivity/20260622_162111/intra_subtype_edge_ratio.png` | Input-graph topology evidence: same-subtype edge enrichment |

---

## Claim-Evidence Check

| Claim | Evidence | Boundary |
|---|---|---|
| PAM50 labels correspond to molecular differences in the raw data | Omics 1 subtype-discriminative feature heatmap; top features include ESR1, FOXC1, MLPH, ERBB4, and TFF3 | This is a data-level validation, not model-output evidence |
| TRIGEL learns a representation with stronger subtype structure than single omics modalities | Within/between cosine similarity: TRIGEL separation 0.699 vs Omics 1/2/3 separations 0.297/0.214/0.108 | Based on BRCA-875 and cosine similarity of exported embedding |
| Patient-similarity graphs contain subtype-consistent topology | Intra-subtype edge ratios: 0.658, 0.571, and 0.563 vs random same-subtype pair ratio about 0.318 | This validates graph input structure, not the effectiveness of graph augmentation by itself |
| Downstream evidence supports representation-level interpretation | Agreement across raw molecular patterns, embedding geometry, and graph topology | Does not yet establish causal biomarkers, pathway mechanisms, or clinical associations |

---

## Notes for Integration

1. This section should remain under `4.5 Downstream Analysis and Model Interpretation`.
2. The marker heatmap was intentionally not included in the main section to avoid redundancy with the molecular profile heatmap. It can be moved to supplementary material if a stronger biomarker-consistency argument is needed.
3. The wording distinguishes raw-data analyses, model-output embedding analyses, and graph-input analyses. This distinction should be preserved in the final thesis to avoid overstating what each figure proves.
4. Once Figure numbers for Sections 4.2-4.4 are fixed, replace every `Fig. X` in this section with the final numbering.
