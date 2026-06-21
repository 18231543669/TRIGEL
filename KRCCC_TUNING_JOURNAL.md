# KRCCC Tuning Journal

该文档由搜索流程自动维护。每完成一轮搜索会追加一节。

## Run 2026-04-17 14:54:54 (BRCA_hyperparameter_search_20260416_162756)

- Objective: `test_f1_macro`
- Total combinations: `192`
- Completed combinations: `192`
- Time (min): `1347.0`
- Round best (test_f1_macro): `0.8385 +- 0.0032`
- Round best ACC: `0.8551 +- 0.0040`
- Round best F1-macro: `0.8385 +- 0.0032`
- Round best params:
  - `hidden_dim=64`
  - `fusion_num_heads=4`
  - `fusion_dropout=0.3`
  - `temperature=0.2`
  - `cb_weight=0.6`
  - `cl_weight=0.4`
  - `embedding_dim=32`
  - `minority_drop_rate=0.0`
  - `majority_drop_rate=0.2`
  - `edge_add_prob=0.4`
  - `max_new_edges=15`
  - `effective_num_beta=0.9999`
  - `minority_edge_weight=1.5`
  - `quality_weight_factor=0.3`
- Average objective across all combinations: `0.8089`
- Trials summary file: `results/BRCA_hyperparameter_search_20260416_162756\trials_summary.csv`

### Top-5 Combinations in This Run

1. objective=0.8385, acc=0.8551, f1_macro=0.8385
   params: `{"cb_weight": 0.6, "cl_weight": 0.4, "edge_add_prob": 0.4, "effective_num_beta": 0.9999, "embedding_dim": 32, "fusion_dropout": 0.3, "fusion_num_heads": 4, "hidden_dim": 64, "majority_drop_rate": 0.2, "max_new_edges": 15, "minority_drop_rate": 0.0, "minority_edge_weight": 1.5, "quality_weight_factor": 0.3, "temperature": 0.2}`
2. objective=0.8382, acc=0.8570, f1_macro=0.8382
   params: `{"cb_weight": 0.6, "cl_weight": 0.4, "edge_add_prob": 0.6, "effective_num_beta": 0.9999, "embedding_dim": 32, "fusion_dropout": 0.2, "fusion_num_heads": 4, "hidden_dim": 64, "majority_drop_rate": 0.1, "max_new_edges": 15, "minority_drop_rate": 0.0, "minority_edge_weight": 1.5, "quality_weight_factor": 0.3, "temperature": 0.2}`
3. objective=0.8378, acc=0.8536, f1_macro=0.8378
   params: `{"cb_weight": 0.6, "cl_weight": 0.4, "edge_add_prob": 0.4, "effective_num_beta": 0.9999, "embedding_dim": 32, "fusion_dropout": 0.3, "fusion_num_heads": 4, "hidden_dim": 32, "majority_drop_rate": 0.1, "max_new_edges": 15, "minority_drop_rate": 0.0, "minority_edge_weight": 1.5, "quality_weight_factor": 0.3, "temperature": 0.2}`
4. objective=0.8368, acc=0.8570, f1_macro=0.8368
   params: `{"cb_weight": 0.6, "cl_weight": 0.4, "edge_add_prob": 0.6, "effective_num_beta": 0.9999, "embedding_dim": 32, "fusion_dropout": 0.3, "fusion_num_heads": 2, "hidden_dim": 64, "majority_drop_rate": 0.1, "max_new_edges": 15, "minority_drop_rate": 0.0, "minority_edge_weight": 1.5, "quality_weight_factor": 0.3, "temperature": 0.2}`
5. objective=0.8366, acc=0.8532, f1_macro=0.8366
   params: `{"cb_weight": 0.6, "cl_weight": 0.4, "edge_add_prob": 0.6, "effective_num_beta": 0.9999, "embedding_dim": 32, "fusion_dropout": 0.3, "fusion_num_heads": 4, "hidden_dim": 64, "majority_drop_rate": 0.1, "max_new_edges": 15, "minority_drop_rate": 0.0, "minority_edge_weight": 1.5, "quality_weight_factor": 0.3, "temperature": 0.2}`

