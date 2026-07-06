# 4-Hyperedge Ablation 결과 (5-fold × 5-seed, 각 condition n=25 runs)


총 학습 시간: 15.6분 (250 runs)


## Scenario: early


### TR (binary classification)

| Condition | AUROC | F1_pos | MCC | Bal Acc | Δ AUROC vs baseline |
|---|---|---|---|---|---|
| **full B6 (baseline)** | 0.557 ± 0.040 | 0.261 ± 0.066 | 0.001 ± 0.055 | 0.497 ± 0.024 | — |
| − E_trajectory | 0.555 ± 0.048 | 0.256 ± 0.063 | -0.000 ± 0.044 | 0.500 ± 0.022 | -0.002 |
| − E_co-trajectory | 0.543 ± 0.039 | 0.225 ± 0.070 | -0.026 ± 0.036 | 0.490 ± 0.018 | -0.015 |
| − E_intervention-response | 0.558 ± 0.051 | 0.252 ± 0.065 | -0.027 ± 0.038 | 0.488 ± 0.018 | +0.001 |
| − E_clinical_prior | 0.541 ± 0.044 | 0.292 ± 0.069 | 0.025 ± 0.050 | 0.513 ± 0.025 | -0.016 |

### Diagnosis (4-class)

| Condition | Macro F1 | Accuracy | Δ F1 vs baseline |
|---|---|---|---|
| **full B6 (baseline)** | 0.656 ± 0.117 | 0.713 ± 0.095 | — |
| − E_trajectory | 0.671 ± 0.121 | 0.720 ± 0.099 | +0.015 |
| − E_co-trajectory | 0.627 ± 0.110 | 0.684 ± 0.090 | -0.029 |
| − E_intervention-response | 0.644 ± 0.124 | 0.691 ± 0.104 | -0.011 |
| − E_clinical_prior | 0.632 ± 0.123 | 0.689 ± 0.102 | -0.024 |

### 진단군별 TR AUROC

| Condition | Depression | Anxiety | Addiction |
|---|---|---|---|
| **full B6 (baseline)** | 0.521 ± 0.092 | 0.645 ± 0.072 | 0.619 ± 0.095 |
| − E_trajectory | 0.494 ± 0.089 | 0.662 ± 0.076 | 0.654 ± 0.104 |
| − E_co-trajectory | 0.489 ± 0.091 | 0.622 ± 0.066 | 0.649 ± 0.101 |
| − E_intervention-response | 0.611 ± 0.099 | 0.629 ± 0.085 | 0.630 ± 0.099 |
| − E_clinical_prior | 0.510 ± 0.089 | 0.647 ± 0.077 | 0.641 ± 0.103 |

## Scenario: full


### TR (binary classification)

| Condition | AUROC | F1_pos | MCC | Bal Acc | Δ AUROC vs baseline |
|---|---|---|---|---|---|
| **full B6 (baseline)** | 0.528 ± 0.041 | 0.279 ± 0.069 | 0.013 ± 0.042 | 0.507 ± 0.019 | — |
| − E_trajectory | 0.555 ± 0.042 | 0.295 ± 0.066 | 0.044 ± 0.054 | 0.520 ± 0.023 | +0.026 |
| − E_co-trajectory | 0.518 ± 0.047 | 0.262 ± 0.065 | 0.000 ± 0.044 | 0.500 ± 0.019 | -0.011 |
| − E_intervention-response | 0.536 ± 0.043 | 0.262 ± 0.066 | 0.007 ± 0.054 | 0.502 ± 0.024 | +0.008 |
| − E_clinical_prior | 0.550 ± 0.044 | 0.268 ± 0.072 | 0.046 ± 0.051 | 0.521 ± 0.022 | +0.021 |

### Diagnosis (4-class)

| Condition | Macro F1 | Accuracy | Δ F1 vs baseline |
|---|---|---|---|
| **full B6 (baseline)** | 0.663 ± 0.112 | 0.707 ± 0.092 | — |
| − E_trajectory | 0.704 ± 0.106 | 0.736 ± 0.091 | +0.041 |
| − E_co-trajectory | 0.698 ± 0.097 | 0.734 ± 0.083 | +0.036 |
| − E_intervention-response | 0.635 ± 0.110 | 0.679 ± 0.091 | -0.027 |
| − E_clinical_prior | 0.675 ± 0.107 | 0.714 ± 0.091 | +0.013 |

### 진단군별 TR AUROC

| Condition | Depression | Anxiety | Addiction |
|---|---|---|---|
| **full B6 (baseline)** | 0.507 ± 0.094 | 0.569 ± 0.085 | 0.578 ± 0.108 |
| − E_trajectory | 0.466 ± 0.097 | 0.609 ± 0.090 | 0.616 ± 0.105 |
| − E_co-trajectory | 0.501 ± 0.091 | 0.554 ± 0.090 | 0.617 ± 0.094 |
| − E_intervention-response | 0.466 ± 0.095 | 0.576 ± 0.071 | 0.584 ± 0.098 |
| − E_clinical_prior | 0.494 ± 0.105 | 0.586 ± 0.090 | 0.623 ± 0.091 |