# Cross-Modal Alignment 정량 검증 결과

평가 sample: 5,000개  |  f_T-f_V 공동 임베딩 공간 (512-dim)


## 1. Paired vs Random Cosine Similarity

| 조건 | Cosine Similarity |
|---|---|
| **Paired (같은 instance)** | **0.2048 ± 0.1022** |
| Random (다른 instance) | 0.0399 ± 0.0210 |
| **Alignment Gap** | **0.1648** |


## 2. Within-clip vs Across-clip

| 조건 | Cosine Similarity |
|---|---|
| Within-clip | 0.1485 |
| Across-clip | 0.0398 |
| **Gap** | **0.1087** |


## 3. Same Emotion vs Different Emotion

| 조건 | Cosine Similarity |
|---|---|
| Same emotion | 0.0516 |
| Different emotion | 0.0368 |
| **Gap** | **0.0148** |


## 4. Cross-modal Retrieval

| 방향 | R@1 | R@5 | R@10 | Median Rank |
|---|---|---|---|---|
| Image → Text | 0.0410 | 0.1170 | 0.1610 | 173 |
| Text → Image | 0.0346 | 0.1016 | 0.1490 | 187 |

## 5. Per-Emotion Paired Similarity

| 감정 | N | Paired Sim |
|---|---|---|
| angry | 529 | 0.2188 ± 0.1175 |
| contempt | 172 | 0.1918 ± 0.1007 |
| dislike | 1,835 | 0.2014 ± 0.1016 |
| fear | 173 | 0.1850 ± 0.1049 |
| happy | 971 | 0.2033 ± 0.0974 |
| neutral | 372 | 0.2016 ± 0.0930 |
| sad | 404 | 0.2183 ± 0.1084 |
| surprise | 544 | 0.2074 ± 0.0939 |

## 해석

✓ **강한 alignment**: Paired similarity가 random보다 명확히 높음.