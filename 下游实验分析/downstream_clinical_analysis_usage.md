# Downstream clinical subtype analysis usage

## 1. Representation-space comparison inputs

The representation-space comparison should compare TRIGEL with other models,
not with only raw input data. To draw that figure, provide one embedding table
or one `.npy` embedding file per method.

Minimum required columns for a table:

```text
sample_id, subtype, dim1, dim2
```

Preferred raw embedding format before UMAP:

```text
sample_id, subtype, emb_1, emb_2, ..., emb_n
```

Needed methods can include:

```text
Early fusion, MOGONET, MOHGCN, MMHN, GCN/GAT, TRIGEL
```

The subtype column should use the same label mapping across all methods.

## 2. Survival analysis input

Run `downstream_clinical_subtype_analysis.py` after preparing:

- A subtype assignment table.
- A clinical table.

The subtype assignment table must contain:

```text
sample_id, pred_subtype
```

The current default file
`results/BRCA_hyperparameter_search_20260118_202538/best_result/brca875_umap/BRCA875_umap_source_data.csv`
contains `class_id`, which is the true class label used for UMAP coloring. For
downstream validation of model predictions, export a table with predicted labels
and pass `--subtype-col pred_subtype`.

The clinical table should contain at least:

```text
sample_id, OS_time, OS_event
```

Accepted aliases include `patient_id`, `barcode`, `bcr_patient_barcode` for the
sample column; `OS.time`, `survival_time`, `days_to_death`, `time` for survival
time; and `OS`, `status`, `vital_status`, `event` for event status.

Example:

```bash
python downstream_clinical_subtype_analysis.py \
  --subtypes results/predicted_subtypes.csv \
  --clinical datasets/BRCA_clinical.csv \
  --subtype-col pred_subtype \
  --subtype-id-col sample_id \
  --clinical-id-col sample_id \
  --time-col OS_time \
  --event-col OS_event \
  --time-scale days
```

Outputs:

```text
results/downstream_clinical_analysis/kaplan_meier_by_subtype.png
results/downstream_clinical_analysis/survival_summary.csv
results/downstream_clinical_analysis/survival_merged_data.csv
```

## 3. Clinicopathological association input

The same clinical table can include clinical variables such as:

```text
age, stage, grade, T_stage, N_stage, M_stage, ER_status, PR_status, HER2_status
```

If `--clinical-vars` is omitted, the script infers usable categorical and numeric
variables. To control the paper figures, pass variables explicitly:

```bash
python downstream_clinical_subtype_analysis.py \
  --subtypes results/predicted_subtypes.csv \
  --clinical datasets/BRCA_clinical.csv \
  --subtype-col pred_subtype \
  --clinical-vars age stage grade ER_status PR_status HER2_status
```

Outputs:

```text
results/downstream_clinical_analysis/clinical_association_summary.csv
results/downstream_clinical_analysis/clinical_categorical_<variable>.png
results/downstream_clinical_analysis/clinical_numeric_<variable>.png
```
