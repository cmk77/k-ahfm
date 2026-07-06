# Youden's J 임계값 보정 분석 (B6 fold 0 / seed 42 / Early)


Total patients: 140 (TR-valid)

- pos: 35 (25.0%)

- neg: 105 (75.0%)

- Overall AUROC: 0.6180

- Optimal threshold (Youden's J): 0.4960

- Optimal Youden's J: 0.2667


## 예측 확률 분포

- min: 0.4308, max: 0.5052

- mean: 0.4730 ± 0.0194

- 5%: 0.4400, 50%: 0.4783, 95%: 0.4985


## 임계값별 메트릭 비교 (140 환자)

| Threshold | TP | TN | FP | FN | Recall_pos | Precision_pos | F1_pos | MCC | Bal Acc |
|---|---|---|---|---|---|---|---|---|---|
| **0.5 (baseline)** | 4 | 104 | 1 | 31 | 0.114 | 0.800 | 0.200 | 0.244 | 0.552 |
| **0.49** | 12 | 89 | 16 | 23 | 0.343 | 0.429 | 0.381 | 0.206 | 0.595 |
| **0.48** | 20 | 63 | 42 | 15 | 0.571 | 0.323 | 0.412 | 0.149 | 0.586 |
| **0.4960 (Youden\'s J)** | 11 | 100 | 5 | 24 | 0.314 | 0.688 | 0.431 | 0.363 | 0.633 |
| **0.45** | 29 | 24 | 81 | 6 | 0.829 | 0.264 | 0.400 | 0.060 | 0.529 |


## 진단군별 비교 (Youden's optimal vs baseline 0.5)

| 진단군 | n | n_pos | Baseline TP/FN | Baseline Recall_pos | Youden TP/FN | Youden Recall_pos |
|---|---|---|---|---|---|---|
| depression | 49 | 10 | 0/10 | 0.000 | 0/10 | 0.000 |
| anxiety | 47 | 15 | 1/14 | 0.067 | 3/12 | 0.200 |
| addiction | 44 | 10 | 3/7 | 0.300 | 8/2 | 0.800 |