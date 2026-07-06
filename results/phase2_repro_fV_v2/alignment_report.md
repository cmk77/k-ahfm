# Cross-Modal Alignment 정량 검증 결과

평가 sample: 10,000개  |  f_T-f_V 공동 임베딩 공간 (512-dim)


## 1. Paired vs Random Cosine Similarity

| 조건 | Cosine Similarity |
|---|---|
| **Paired (같은 instance)** | **0.1981 ± 0.1357** |
| Random (다른 instance) | 0.0347 ± 0.0255 |
| **Alignment Gap** | **0.1635** |


## 2. Within-clip vs Across-clip

| 조건 | Cosine Similarity |
|---|---|
| Within-clip | 0.1448 |
| Across-clip | 0.0346 |
| **Gap** | **0.1103** |


## 3. Same Emotion vs Different Emotion

| 조건 | Cosine Similarity |
|---|---|
| Same emotion | 0.0525 |
| Different emotion | 0.0299 |
| **Gap** | **0.0226** |


## 4. Cross-modal Retrieval

| 방향 | R@1 | R@5 | R@10 | Median Rank |
|---|---|---|---|---|
| Image → Text | 0.0047 | 0.0202 | 0.0298 | 1147 |
| Text → Image | 0.0032 | 0.0165 | 0.0266 | 1176 |

## 5. Per-Emotion Paired Similarity

| 감정 | N | Paired Sim |
|---|---|---|
| angry | 1,066 | 0.2146 ± 0.1576 |
| contempt | 330 | 0.1849 ± 0.1237 |
| dislike | 3,693 | 0.1957 ± 0.1367 |
| fear | 336 | 0.1777 ± 0.1275 |
| happy | 1,951 | 0.2029 ± 0.1330 |
| neutral | 720 | 0.1930 ± 0.1238 |
| sad | 804 | 0.2067 ± 0.1351 |
| surprise | 1,100 | 0.1892 ± 0.1251 |

## 해석

✓ **강한 alignment**: Paired similarity가 random보다 명확히 높음.