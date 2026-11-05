# MariusGNN Mélange Splitwise TetriServe의 비용 비교와 멀티모달 연구 적용

정리일: 2026-10-15. 네 논문의 비용 비교는 작업 완료 비용, 수요를 만족하는 시간당 자원 비용, 비용당 SLO 충족 처리능력, 고정 자원의 SLO 충족률로 나뉜다. 우리 연구는 고정 수요의 최소 비용과 고정 비용의 최대 SLO 충족 수요를 함께 평가할 수 있다. 고정 GPU 풀에서도 비용당 처리능력은 개선될 수 있으며, 이를 실제 총비용 절감과 구분해 보고해야 한다.

## 읽은 원문과 범위

| 논문 | 로컬 원문과 버전 | 전체 읽기 범위 | 비용 근거를 그림으로 재확인한 PDF 쪽 |
|---|---|---|---|
| MariusGNN | [mariusgnn.pdf](../../papers/refs/mariusgnn.pdf), EuroSys 2023, 18쪽 | 초록, §1–9, 참고문헌 [1]–[54], Artifact Appendix A | 10–11쪽, Table 2–5와 Fig. 7 |
| Mélange | [melange-2404.14527.pdf](../../papers/refs/melange-2404.14527.pdf), arXiv v4, 2024-07-22, 20쪽 | 초록, §1–7, 참고문헌 [1]–[60], Appendix A–C, 모든 할당 표 | 9쪽 Table 1, 17쪽 Table 3 |
| Splitwise | [splitwise-2311.18677.pdf](../../papers/refs/splitwise-2311.18677.pdf), arXiv v2, 2024-05-20, 15쪽 | 초록, §I–IX, 참고문헌 [1]–[85], Artifact Appendix A–E | 8쪽 Table V, 11쪽 Fig. 18–19 |
| TetriServe | [tetriserve-arxiv-v4.pdf](../../papers/refs/tetriserve-arxiv-v4.pdf), 2026-06-18, 16쪽 | 초록, §1–8, 참고문헌 [1]–[47], Appendix A–B | 6쪽 GPU-hour 목적식, 10쪽 Fig. 7–8 |

TetriServe의 [ASPLOS 2026 PDF](../../papers/refs/tetriserve-asplos26.pdf)도 대조했다. 공백을 정규화한 페이지별 비교에서 본문 2–14쪽은 동일하고, 차이는 첫 쪽 arXiv 표기 및 15–16쪽 참고문헌·부록·표 배치다. 모든 그림을 이미지로 개별 확대 판독했다는 의미는 아니며, 전체 추출 텍스트를 읽고 위의 핵심 가격·결과 표와 그림을 렌더링하여 확인했다. 인용된 외부 참고문헌 전체를 추가로 읽은 것은 아니다.

공식 출처: [MariusGNN](https://arxiv.org/abs/2202.02365), [Mélange v4](https://arxiv.org/html/2404.14527v4), [Splitwise ISCA 논문](https://www.microsoft.com/en-us/research/wp-content/uploads/2023/12/Splitwise_ISCA24.pdf), [TetriServe ASPLOS 논문](https://symbioticlab.org/publications/files/tetriserve%3Aasplos26/tetriserve-asplos26.pdf). 아래 가격은 논문이 사용한 과거 값이며 현재 판매 가격이 아니다.

## 네 논문을 같은 기준으로 비교

| 항목 | MariusGNN | Mélange | Splitwise | TetriServe |
|---|---|---|---|---|
| 대상 | 유한 GNN 학습 작업 | 지속 LLM 서비스의 GPU 구성 | P/D 분리 LLM 클러스터 | 고정 GPU 풀의 혼합 해상도 DiT 서비스 |
| 주 목적 | 유사한 모델 품질에서 학습 시간·비용 감소 | 주어진 수요와 SLO를 만족하는 최소 시간당 비용 | SLO 아래에서 비용·처리능력·전력의 여러 최적화 | 주어진 자원에서 deadline을 지키는 요청 수 최대화 |
| 비용 단위 | USD/epoch, 전체 학습 USD | USD/hour, 보조로 input+output tokens/USD | 클러스터 비용, RPS/비용, provisioned power | GPU-time 자원 대용량; 주요 평가는 SAR |
| 가격 근거 | AWS P3 인스턴스 요금 | 여러 공급자의 공개 요금 + H100 정규화 | CoreWeave 인용 단가 및 별도 정규화 설계 비용 | 달러 단가·임대료 비교 없음 |
| 고정하는 서비스 조건 | 그래프·모델·학습 epoch; 품질을 함께 보고 | 모델·엔진·길이 분포·요청률·TPOT SLO | 모델·길이 분포·아홉 지연 SLO | 모델·해상도 mix·도착률·deadline scale |
| 바꾸는 것 | CPU/GPU/디스크 사용법과 인스턴스 크기, 실행시간 | GPU 종류·수·요청 분배 | prompt/token 서버 수·종류·전력 제한 | 요청별·step별 SP degree, 배치·packing |
| 주 결과 근거 | 실측 학습 시간 × 인스턴스 요금 | 실측 프로파일 기반 ILP 비용 + 선택된 조건의 배포 검증 | 통신 실측·성능 보정 + 클러스터 시뮬레이션 | 실제 GPU 클러스터의 SAR·지연 실험 |

서로 다른 정의의 최대 향상률을 같은 열에 놓고 우열을 판단할 수는 없다. 같은 수요와 품질을 만족했는지, 어떤 SLO를 적용했는지, 얼마의 자원을 얼마 동안 과금했는지가 공통 비교 기준이다.

## MariusGNN의 가격 비교

### 논문의 기법과 비용 절감 경로

DENSE는 여러 GNN layer의 one-hop 이웃 샘플을 재사용하고 dense GPU 연산을 사용하여 CPU 샘플링과 GPU 계산의 병목을 줄인다. COMET은 물리·논리 partition과 학습 예제 순서를 분리하여 디스크 IO와 예제 상관관계 사이의 절충을 개선한다. 즉 GPU를 추가하는 대신 CPU·GPU·디스크 계층의 활용을 개선하는 연구다.

§7.1–7.2, PDF 9–11쪽에서 측정한 epoch 실행시간에 AWS 인스턴스 가격을 적용한다. Table 2의 단가는 P3.2xLarge 3.06 USD/h, P3.8xLarge 12.24 USD/h, P3.16xLarge 24.48 USD/h다. 비용의 근사식은 `Cost_epoch = VM_price_hour * epoch_minutes / 60`이다.

핵심은 GPU를 실제로 몇 장 사용했는지와 인스턴스를 몇 장짜리로 임대했는지가 다르다는 점이다. Graph 데이터를 CPU RAM에 올리기 위해 큰 P3 인스턴스가 필요하면, GPU 한 장만 계산에 쓰더라도 큰 인스턴스 전체 가격을 부담한다. Table 4의 link prediction에서는 모든 시스템이 GPU 한 장으로 학습하지만, 디스크를 사용하는 MariusGNN은 더 작은 인스턴스를 사용할 수 있다.

### 표의 수치를 재계산한 예

| 비교 | 원문 수치 | 해석 |
|---|---|---|
| Mag240M-Cites node classification, DGL vs M-GNN Disk, Table 3 | epoch 7.83 vs 0.94분; 비용 3.19 vs 0.05 USD; 정확도 63.73 vs 62.53 | 실행시간과 인스턴스 단가가 함께 감소. 표의 반올림 비용 기준 약 64배 차이지만 정확도는 1.20%p 다름 |
| Freebase86M link prediction, PyG vs M-GNN Disk, Table 4 | epoch 108 vs 34.2분; 비용 22.0 vs 1.74 USD; MRR .7267 vs .7216 | 약 12.6배 비용 차이와 품질 차이를 함께 제시할 수 있는 사례 |
| Freebase86M, M-GNN Mem vs Disk, Table 4 | epoch 17.5 vs 34.2분; 비용 3.57 vs 1.74 USD; MRR .7285 vs .7216 | Disk가 약 두 배 느려도 작은 인스턴스 덕분에 비용은 낮음. 가장 빠른 실행이 가장 싼 실행이라는 뜻은 아님 |

직접 계산하면 `7.83/60*24.48 = 3.19464`, `0.94/60*3.06 = 0.04794`로 Table 3 비용을 재현한다. Table 4의 Wiki disk 비용도 `69.9/60*3.06 = 3.5649 USD/epoch`로 재현된다.

### 주장 범위와 제한

논문은 같은 epoch 수에서 시간·비용·품질을 함께 보고하고 Fig. 7에서 일부 데이터셋의 time-to-accuracy도 제시한다. 모든 조건에서 정확도를 수치상 완전히 같게 맞춘 time-to-target-quality 비용 비교는 아니다. 특히 WikiKG90Mv2의 disk MRR .4156은 PyG .4683과 차이가 있고, DGL은 완료 제한시간을 넘겼다. 서론의 약 48배 사례만으로 동일 품질 우위를 일반화하면 안 된다.

서론의 Wiki 사례인 약 6일·1,720 USD 대비 8시간·36 USD는 시간과 비용을 같은 구성으로 재현할 수 없다. Table 4와 열 epoch를 기준으로 Mem은 `46.6*10/60 = 7.77시간`, `9.38*10 = 93.8 USD`이고, Disk는 `69.9*10/60 = 11.65시간`, `3.56*10 = 35.6 USD`다. 8시간은 Mem 시간, 36 USD는 Disk 비용에 가깝다. 표의 반올림 값 기준으로 비용 약 48배라는 비율은 재현되지만, 그때의 Disk 시간·MRR을 함께 제시해야 한다. 원 실행 로그나 별도 미기재 구성을 확인한 것은 아니다.

PyG의 일부 batch size와 DGL의 negative sample 수는 OOM 때문에 다르다. 이는 논문이 명시한 조건이다. EBS 1 GB/s·10,000 IOPS를 사용하지만 표의 비용 계산에서 별도의 EBS 요금 항목을 확인할 수 없다. 따라서 이를 저장소·전송·준비 비용까지 포괄한 총 소유비용 또는 실제 청구서 비교로 확대하지 않는다.

우리 연구에 적용하면, 단계 분리나 offloading으로 GPU 수뿐 아니라 필요한 CPU RAM·GPU 메모리·네트워크 때문에 선택하는 전체 인스턴스가 달라지는지를 비용에 반영해야 한다. 유한 작업에서는 같은 작업·품질·완료 조건 아래에 실제 점유시간이 짧아지고 자원을 반환할 수 있을 때 시간 감소가 비용 감소로 이어진다.

## Mélange의 가격 비교

### 용량 대비 비용과 고정 수요 비용을 구분

§4.1은 SLO를 만족하는 최대 포화 상태의 input+output token 처리량을 GPU 임대 비용으로 나눈 T/USD를 사용한다. §4.4와 §5는 같은 길이 분포와 요청률을 처리하므로 분자가 고정되어, GPU 구성의 시간당 비용 최소화로 바뀐다. Eq. (1)의 목적은 `sum_j B_j * c_j` 최소화다.

§5.4는 요청 크기 bucket을 작은 수요 slice로 나누고, 단독 profiling에서 얻은 최대 SLO 충족 처리량으로 정규화한 load를 합산하는 ILP다. 엔진·모델을 바꾸는 대신 GPU 종류와 개수 및 요청 배분을 선택한다. 같은 공식을 GPU 한 종류로 제한한 baseline과 비교하는 점이 우리 연구의 공정한 후보 공간 비교에 유용하다.

### 가격 출처와 최대 절감률의 기준선

§6.1, PDF 9쪽의 시간당 가격은 L4 0.70, A10G 1.01, A100 3.67, H100 7.516 USD/h다. 앞의 세 GPU는 GCP·AWS·Azure에서 선택한 공개 on-demand 요금이며, 단일 공급자의 한 구매 견적은 아니다. H100은 RunPod의 H100/A100 가격 비율을 Azure A100 단가에 적용한 값이다.

`H100_normalized = (4.69/2.29)*3.67 = 7.516288... USD/h`

Appendix C, Table 3, PDF 17쪽의 Arena 1 req/s·120 ms SLO에서 Mélange는 L4 한 장 + A10G 한 장으로 1.71 USD/h다. H100-only 7.516 USD/h 대비 77.25% 절감이지만, 같은 셀의 가장 싼 단일 종류 baseline인 A10G-only 두 장 2.02 USD/h 대비 절감은 15.35%다. 따라서 최대 77%를 모든 조건의 최선 homogeneous baseline 대비 절감으로 바꾸어 말하지 않는다.

### SLO와 검증 범위

§4.1의 TPOT는 요청 전체 latency를 output token 수로 나눈 값이다. 우리 코드의 `(end-first)/(tokens-1)` 또는 token/chunk 간 간격과 동일하지 않다. §6.3은 요청별 평균 TPOT의 CDF를 보고하며, 40 ms에서 99.5%, 120 ms에서 99.95% 이상의 충족률을 보고한다. 이 배포 검증은 각 데이터셋·SLO에서 4 req/s, 2,000개 요청으로 수행한다. 모든 cost-allocation 셀을 동일 규모의 실제 배포로 검증했다는 뜻은 아니다.

평가 workload·요청률·SLO가 달라지면 선택 구성이 달라진다. burst와 큰 요청의 연속 도착으로 SLO 위반도 발생한다고 명시한다. 우리에게는 고정 수요의 최소 비용 실험과 SLO·가격 민감도 분석의 직접적인 선례다. 실제 기업 SLO가 없어도 가정을 밝힌 연구 평가가 가능하나, 우리가 사용한 SLO가 해당 논문과 같은 통계량인지는 별도로 확인해야 한다.

## Splitwise의 가격 비교

### 같은 비용과 같은 처리능력 양쪽을 평가

Splitwise는 prompt와 token generation 단계의 계산·메모리·전력 특성 차이를 이용해 풀을 분리하고, KV 전송을 layer 단위로 겹치며, 혼합 풀로 부하에 따른 분할 손실을 줄인다. §IV-D의 구성 선택은 prompt/token 서버 수·종류·전력 제한을 함께 탐색한다.

§VI와 Fig. 18–19, PDF 10–11쪽에서 다음을 구분한다.

- Iso-cost: 같은 클러스터 비용에서 SLO를 만족하는 최대 RPS 비교.
- Iso-throughput: 같은 SLO 충족 RPS를 제공하는 데 필요한 클러스터 비용 비교.
- Iso-power: 같은 provisioned power에서 처리능력 비교. 이는 실측 전력 적분 Wh 또는 전기요금 비교와 다르다.

§VI-C는 iso-cost에서 Splitwise-AA가 Baseline-H100보다 1.4배 throughput이지만 power는 25% 증가하고 공간도 더 필요하다고 설명한다. Fig. 19b 설명은 같은 throughput에서 Splitwise-AA가 Baseline-H100 대비 비용을 25% 줄이는 사례다. 초록의 최대 수치를 서로 다른 그림·기준선의 모든 조건에 일괄 적용하지 않는다. §VI-E의 batch 고부하에서는 혼합 풀로 수렴하면서 AA와 A100 baseline의 throughput/비용이 같아지는 조건도 보고한다.

### SLO와 가격 모델

Table VI, PDF 9쪽은 DGX-A100에서 경합 없이 실행한 해당 요청 대비 slowdown으로 SLO를 정한다. TTFT p50/p90/p99는 2/3/6배, TBT와 E2E는 각각 1.25/1.5/5배이며 아홉 제약을 모두 만족해야 최대 처리능력으로 인정한다. 실제 한 기업의 SLA를 확보한 것과는 다른, 명시된 연구용 상대 기준이다.

Table I, PDF 1쪽은 CoreWeave를 인용하며 A100/H100 machine의 17.6/38 USD/h를 제시한다. Table V, PDF 8쪽은 별도의 DGX-A100 대비 정규화 비용을 사용하며 H100 prompt 2.35배, token 2.5배로 기재한다. 서로 다른 표의 가격·비율이 완전히 같은 산식으로 재현되는지는 본문만으로 확정하지 않고, 우리 모델에서는 하나의 일관된 가격표로 계산한다.

§V–VI에서 통신과 성능은 실측으로 보정하지만 대규모 cluster cost/throughput 결과는 discrete-event 시뮬레이터를 이용한다. 이를 수십 대 실제 클러스터에서 측정한 청구 비용 절감이라고 부르지 않는다. 또한 H100과 A100 사이 InfiniBand 연결은 기술적으로 가능하더라도 실제 cloud 제공 여부에 제약이 있다고 §VII에서 설명한다.

우리에게 중요한 결론은 고정 비용·고정 GPU 풀에서도 SLO를 지키며 더 많은 수요를 처리하면 비용당 처리능력이 개선된다는 것이다. 총비용이 줄었다는 뜻과는 구분된다. 더 싼 GPU가 없는 환경이라는 이유로 이 평가를 배제할 필요는 없다.

## TetriServe의 비용 관점

### Cost model은 달러 가격 모델이 아니다

§4.1의 전역 문제는 고정 GPU 수에서 deadline을 만족하는 요청 수 최대화다. §4.2.1, PDF 6쪽은 요청별로 deadline을 지키는 step allocation 중 `sum_j A_ij*T_ij(A_ij)`를 줄이는 GPU-time 목표를 사용한다. 시간 단위가 초라면 이 값은 GPU-seconds이며 GPU-hours로 표기하려면 3,600으로 나누어야 한다.

이는 profiling한 step 실행시간을 이용하는 자원 사용 모델이다. GPU 단가가 없으며, step별 할당 GPU-time이 줄었다고 서버의 임대 청구액이 자동으로 줄어들지는 않는다. 같은 GPU 풀을 계속 예약한다면 빈 자원으로 다른 요청을 처리할 여지가 늘어난 것이다. 전역 최적해를 보장하는 최소 비용 solver가 아니라 round-based heuristic과 DP packing을 사용한다.

### 실제 보여준 결과와 조건

§6은 FLUX.1-dev의 8×H100, SD3의 4×A40에서 고정 GPU 풀을 사용한다. 주요 지표는 SAR이며 latency CDF를 보조로 사용한다. 해상도 mix·도착률·SLO scale을 바꾸고 fixed SP 및 resolution-specific SP와 비교한다. 300개 DiffusionDB prompt와 기본 12 req/min의 Poisson 도착을 사용한다.

§6.1의 기본 deadline은 해상도 256/512/1024/2048에 대해 1.5/2/3/5초이며 scale 1.0–1.5를 평가한다. 원문이 보고한 최대 32% SAR 향상을 동일 비율의 달러 절감 또는 GPU-hour 감소로 환산하지 않는다. 상대 %와 절대 %p 역시 원시 값을 확인하지 않고 혼용하지 않는다.

§6.2의 latency CDF는 완료 요청만 포함하고 drop/timeout은 제외하므로 CDF와 전체 SAR를 함께 읽어야 한다. 기본 step-level SP 조정과 Nirvana caching 결합 실험도 구분해야 한다. 후자는 initial step을 생략하므로, 우리 비교에서는 품질과 계산량 조건을 따로 맞춰야 한다.

우리에게는 같은 자원과 같은 도착 수요에서 서비스별 SLO 충족률과 자원 점유를 개선하는 평가의 선례다. 이 결과만으로 더 높은 최대 수요를 감당한다고 단정하지 말고, 고정 SLO에서 요청률을 올리는 별도 실험으로 확인한다.

## 우리 연구에서 비용 효율을 정의하는 방법

### 총비용과 시간당 비용의 구분

자원 구성의 시간당 비용을 `C_rate(sigma) = sum_m n_m*p_m`로 정의한다. 여기서 m은 실제 과금하는 GPU 또는 VM 단위이며 RAM·네트워크가 묶인 VM이면 VM 전체를 과금한다. 유한 작업의 총비용은 `C_total = sum_m p_m*h_m + 별도 저장소/전송 등`이다. h_m은 GPU kernel이 일한 시간만이 아니라 과금·점유한 시간이다.

보유 장비의 상각, 전기요금, 새 구매 비용, cloud 임대료는 다른 비용 모델이다. 특정 모델을 명시하고 바꾸어 평가할 수 있지만 서로 혼합하지 않는다. GPU 종류별 점유시간과 Wh를 보존하면 가격 가정 변경에 대응할 수 있다.

### 평가 1 고정 수요에서 SLO 충족 최소 비용

LLM·VLM·DiT의 수요 벡터 lambda, 모델·품질, 입출력 분포, SLO와 목표 충족률을 고정한다. 공유 실행을 고려해 실제로 목표를 만족하는 전략·단계 배치·GPU/VM 구성 중 `C_rate`를 최소화한다. 이는 Mélange와 LLM-PILOT의 목표에 직접 대응한다.

단독 성능 기반 선택, 같은 후보 공간의 혼합 성능 반영 선택, 강한 고정 배치 및 실측 최선 구성을 비교한다. 예측하는 것은 성능과 가용성이고 가격표가 정해졌을 때 후보 비용은 계산값이다. SLO 위반 후보의 낮은 비용을 성공한 절감으로 세지 않는다.

### 평가 2 고정 비용에서 최대 SLO 충족 수요

기준 mix `lambda0 = (lambda_LLM, lambda_VLM, lambda_DiT)`의 비율과 요청 크기·품질을 고정하고, 모든 서비스의 수요를 공통 배율 alpha로 늘린다. 서비스별 SLO와 충족률, 관측한 안정성 조건을 만족하는 최대 alpha를 비교한다.

`alpha_max(sigma) = sup {alpha : 모든 서비스가 alpha*lambda0에서 각자 SLO를 만족}`

비교 지표는 `alpha_max / C_rate`다. 이는 정규화된 수요배율/시간당 비용이며 token/s와 image/s를 더한 물리적 처리량이 아니다. 같은 mix와 같은 가격 조건에서 비율로 비교한다. 서로 다른 mix의 alpha 숫자를 그대로 합치거나 우열 비교하지 않는다.

예를 들어 같은 4090+3090 비용에서 한 배치가 기준 mix의 1.0배까지, 다른 배치가 1.4배까지 모든 서비스의 SLO를 만족한다면, 비용당 SLO 충족 처리능력이 1.4배라고 할 수 있다. 이는 설명용 가상 예이며 기존 실험의 결과가 아니다. GPU 두 장의 청구 총액이 40% 감소했다는 뜻도 아니다.

현재 장비에서는 이 평가를 주요 실측 결과로 삼을 수 있다. 평가 1의 가변 자원 후보를 확보하면 같은 frontier를 고정 수요 기준으로도 비교한다. 정수 자원·메모리·가격·SLO 때문에 두 관점의 수치가 단순히 역수가 된다고 가정하지 않는다.

### 평가 3 같은 자원과 같은 수요에서 유효 서비스 개선

서비스별 공동 SLO 충족률, 전체 오류·미완료, 지연 분포를 함께 보고한다. 이때 같은 비용으로 deadline 안에 완료한 유효 요청이 늘었다면 SLO 유효 처리 효율 개선이라고 할 수 있다. 다만 한 서비스의 개선으로 다른 서비스의 실패를 가리지 않도록 서비스별 벡터와 최소 목표를 유지한다. 실제 최대 수요나 더 작은 자원 구성을 확인하지 않았다면 각각의 capacity 향상 또는 비용 절감으로 확대하지 않는다.

### 평가 4 유한 작업의 완료 비용

같은 서비스별 작업 수·모델·품질·완료 목표를 고정하고, 완료까지 점유한 자원시간과 비용을 비교한다. MariusGNN의 논리에 대응한다. 한 요청의 p90 감소를 전체 작업 완료시간 감소로 대신할 수 없으며, 예약을 계속 유지하는 운영이면 짧아진 kernel 시간만큼 청구액이 줄어든다고 볼 수 없다.

## 현재 결과에 적용할 판단

현재 [요약 그림](../../archive/legacy/early-2026-10/results/mixed_summary/mixed_split_summary.png)은 두 배치의 일부 성공 요청 지연 p90을 비교한다. LLM+DiT에서는 trade-off, LLM+VLM에서는 한 배치의 주요 지표 우세를 관측했다. 고정 SLO에서 감당하는 최대 수요 또는 고정 수요의 최소 비용을 측정한 결과는 아니다.

따라서 지금은 간섭·단계 배치가 SLO 충족 성능에 영향을 줄 수 있다는 근거로 사용하고, 다음 결과로 비용 효율을 연결한다.

1. SLO·충족률·품질과 workload mix를 사전에 정하고 전체 민감도 범위를 공개한다. SLO를 결과에 맞춰 바꾼 조건은 탐색으로 표시한다.
2. 같은 두 GPU에서 모든 서비스 수요를 공통 배율로 늘려 SLO 충족 한계를 측정한다. threshold 인근은 새 seed와 충분한 관측시간으로 재검증하고, 시험한 최대값을 실제 최대값으로 단정하지 않는다.
3. 가능하면 1 GPU 공유, 2 GPU 서비스 분리, 검증된 단계 분리와 배치를 같은 후보 공간에 둔다. 메모리·기능상 불가능한 후보는 제외 근거를 남긴다.
4. 선택기 정확도와 실측 최선 대비 선택 손실을 비용 효율 결과와 분리한다. 혼합 실측 lookup으로 사후 우위를 확인한 것만으로 새로운 예측 solver의 효과를 입증하지 않는다.
5. 가격 비율과 SLO scale에 따른 Pareto frontier를 제시한다. 가격이 달라질 때도 성립하는 개선과 특정 가정에서만 성립하는 개선을 구분한다.

기존 `splitwise_score.py`와 `map_score.py`는 요청 평균 TPOT를 쓰고 오류 요청을 제외하는 경로가 있다. 전자의 원 논문과 동일하다는 주석은 그대로 재현성을 보장하지 않는다. 새 평가에서는 실제 token/chunk 지표와 요청별 평균의 차이, 도착 기준 지연, 전체 계획 요청을 분모로 한 오류·미완료 처리를 명시해야 한다. 이번 검토에서는 채점 코드를 수정하거나 GPU 실험을 실행하지 않았다.

## 이전 논의의 정정

기존 인계에서 고정 GPU 두 장의 비용이 같다는 이유로 공통 수요배율 최대화를 보조 목표로만 제한한 것은 연구 선택지를 불필요하게 좁혔다. 같은 비용에서 같은 서비스별 SLO·품질을 만족하는 처리능력이 늘면 비용 효율 개선으로 정당하게 평가할 수 있다. 다만 SLO 충족 최소 비용이라는 LLM-PILOT의 목적함수와 고정 비용 최대 용량이라는 목적함수는 각각 명시해야 한다.

최종 연구 주장은 여러 모달리티의 전략·단계 배치와 공유 실행 성능을 함께 고려하여, 서비스별 SLO를 만족하는 비용과 처리능력의 경계를 개선하는 것이다. 기존 원시 결과가 그 경계를 개선했다고 이미 입증한 것은 아니다.

## 공동 검토 기록

Claude에게도 네 논문의 전체 텍스트·부록·참고문헌을 독립 정독하도록 맡겼다. Codex가 표를 재계산하고 핵심 페이지 이미지를 확인한 뒤, 고정 비용의 capacity 개선, 실제 점유시간과 할당시간의 차이, MariusGNN의 시간·비용 구성, Mélange의 기준선과 TPOT 정의를 재논의했다. 위 결론은 두 차례 검토를 반영한다.

- [Claude 첫 검토](research_discussion/20261015_claude_cost_papers_review.txt)는 검토 전 해석도 포함한 원문 기록이다.
- [Codex 수정 의견](research_discussion/20261015_codex_cost_papers_feedback.txt).
- [Claude 재검토](research_discussion/20261015_claude_cost_papers_revised.txt).
- [읽은 PDF의 SHA256과 페이지 범위](research_discussion/20261015_cost_paper_reading_manifest.json).

원 실행 코드나 로그로 원 논문의 모든 표 숫자를 재현한 작업은 아니다. 숫자 불일치는 해당 PDF의 표·본문 기준에서 확인한 범위로 한정한다. 새 GPU 실험은 실행하지 않았다.
