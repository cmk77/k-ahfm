# Cross-Modal Alignment 정량 검증 결과

평가 sample: 10,000개  |  f_T-f_V 공동 임베딩 공간 (512-dim)


## 1. Paired vs Random Cosine Similarity

| 조건 | Cosine Similarity |
|---|---|
| **Paired (같은 instance)** | **0.2051 ± 0.1028** |
| Random (다른 instance) | 0.0398 ± 0.0209 |
| **Alignment Gap** | **0.1653** |


## 2. Within-clip vs Across-clip

| 조건 | Cosine Similarity |
|---|---|
| Within-clip | 0.1489 |
| Across-clip | 0.0397 |
| **Gap** | **0.1093** |


## 3. Same Emotion vs Different Emotion

| 조건 | Cosine Similarity |
|---|---|
| Same emotion | 0.0515 |
| Different emotion | 0.0366 |
| **Gap** | **0.0149** |


## 4. Cross-modal Retrieval

| 방향 | R@1 | R@5 | R@10 | Median Rank |
|---|---|---|---|---|
| Image → Text | 0.0236 | 0.0833 | 0.1187 | 350 |
| Text → Image | 0.0199 | 0.0716 | 0.1056 | 386 |

## 5. Per-Emotion Paired Similarity

| 감정 | N | Paired Sim |
|---|---|---|
| angry | 1,066 | 0.2199 ± 0.1169 |
| contempt | 330 | 0.1969 ± 0.0974 |
| dislike | 3,693 | 0.2036 ± 0.1015 |
| fear | 336 | 0.1943 ± 0.1084 |
| happy | 1,951 | 0.2048 ± 0.0996 |
| neutral | 720 | 0.1953 ± 0.0932 |
| sad | 804 | 0.2108 ± 0.1104 |
| surprise | 1,100 | 0.2044 ± 0.0960 |

## 해석

✓ **강한 alignment**: Paired similarity가 random보다 명확히 높음.