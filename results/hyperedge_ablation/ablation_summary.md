# 4-Hyperedge Ablation 결과 (fold 0, seed 42)


Total time: 44s (0.7min)


## Scenario: early


### TR (binary classification)

| Condition | TR AUROC | TR F1_pos | TR MCC | Bal Acc |
|---|---|---|---|---|
| − E_trajectory | 0.585 | 0.222 | 0.160 | 0.548 |
| − E_co-trajectory | 0.476 | 0.118 | -0.258 | 0.357 |
| − E_intervention-response | 0.748 | 0.400 | 0.054 | 0.524 |
| − E_clinical_prior | 0.476 | 0.438 | 0.200 | 0.571 |

### Diagnosis (4-class)

| Condition | Diag macro F1 | Diag Accuracy |
|---|---|---|
| − E_trajectory | 0.855 | 0.857 |
| − E_co-trajectory | 0.888 | 0.893 |
| − E_intervention-response | 1.000 | 1.000 |
| − E_clinical_prior | 0.825 | 0.821 |

## Scenario: full


### TR (binary classification)

| Condition | TR AUROC | TR F1_pos | TR MCC | Bal Acc |
|---|---|---|---|---|
| − E_trajectory | 0.721 | 0.222 | 0.160 | 0.548 |
| − E_co-trajectory | 0.381 | 0.000 | 0.000 | 0.500 |
| − E_intervention-response | 0.707 | 0.250 | 0.333 | 0.571 |
| − E_clinical_prior | 0.367 | 0.000 | 0.000 | 0.500 |

### Diagnosis (4-class)

| Condition | Diag macro F1 | Diag Accuracy |
|---|---|---|
| − E_trajectory | 0.961 | 0.964 |
| − E_co-trajectory | 0.713 | 0.714 |
| − E_intervention-response | 0.894 | 0.893 |
| − E_clinical_prior | 0.579 | 0.607 |