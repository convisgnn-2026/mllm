# LLMServingSim 2.0 — 네이티브 빌드와 우리 패치

우리가 쓰는 LLM/VLM 시뮬레이터. 이 문서 하나로 새 머신에서 재현된다.

- 상류: https://github.com/casys-kaist/LLMServingSim (1.0 IISWC'24, 2.0 arXiv 2602.23036)
- 우리 패치: `results/servingsim/llmservingsim-hetpilot.patch` (7 files, +289 −239)
- 로컬 클론 위치(이 서버): `~/papers/refs/code/LLMServingSim`, 브랜치 `hetpilot-patches`
- **상류 저장소는 우리 저장소에 넣지 않았다.** 클론 + patch 적용으로 복원한다.

---

## 1. 왜 이걸 골랐나

| | Vidur (MLSys'24) | Frontier (arXiv) | Splitwise-sim (ISCA'24) | **LLMServingSim 2.0** |
|---|---|---|---|---|
| PD 분리 | ❌ | ✅ | ✅ | ✅ |
| KV 오프로딩 | ❌ | ❌ | ❌ | ✅ (`prefix-storage CPU`, `kv_evict_loc`) |
| 이종 클러스터 | ❌ | 용례 1개 | ✅ | ✅ (인스턴스마다 `hardware`, `npu_mem`) |
| **소비자 GPU 프로파일** | ❌ | A100/H100만 | ❌ | **RTX 4090 · RTX PRO 6000** |
| 검증 | 잘 됨 | 분리 2.6% | — | vLLM 대비 TTFT/TPOT **0.2~4%** |

결정적인 건 마지막 두 줄. 우리 실제 카드(4090, PRO 6000)가 vLLM 실측 대비 검증된 채로 들어 있다.
AFD(attention-FFN 분리)는 Frontier에만 있으나 우리는 AO를 뺐으므로 문제 없음.

**동봉 검증 재현 결과** (우리 네이티브 빌드, `bench/examples/RTX4090/Llama-3.1-8B`):
TTFT 평균 +0.6%, TPOT 평균 +0.2%, 지연 P99 +0.4%. 저장소 `validation/summary.txt`와 소수점까지 일치.

---

## 2. 네이티브 빌드 (docker·conda 없이)

`scripts/setup_llmservingsim.sh` 한 방. 15분쯤 걸린다(protobuf 소스 빌드가 대부분).

```bash
bash scripts/setup_llmservingsim.sh
source ~/venv/servingsim/env.sh     # 이후 셸마다
```

docker 이미지가 실제로 주는 것을 분해해 보면 이렇다.

| 이미지가 까는 것 | 진짜 필요? | 우리 해법 |
|---|---|---|
| gcc/g++/make | 필요 | 시스템에 있음 (gcc 11) |
| cmake, ninja | 필요 | **pip wheel** |
| boost, openmpi | analytical 백엔드 코드에서 **0회 사용** | 안 깖 |
| Python 3.11 + numpy·pandas·sympy·graphviz | 필요 | venv `~/venv/servingsim` (3.12) |
| abseil 20240722.0 + protobuf 29.0 | 필요 | **이미지도 소스 빌드함** → 같은 레시피로 `~/local`에 |
| fmt, spdlog | 필요 | 저장소 서브모듈, 같이 컴파일 |

즉 docker는 "protobuf를 소스로 빌드하는 우분투 껍데기"였다. 폐쇄망이라 `dnf`가 저장소 조회에서 멈추는데
어차피 시스템 패키지가 필요 없다.

주의:
- **중첩 서브모듈**을 재귀로 받아야 한다(`chakra`, `fmt`, `spdlog`가 비어 있으면 빌드 실패). 스크립트가 처리.
- venv를 연구용(`~/venv/jinwoo`)과 **분리**했다. chakra가 `protobuf==5.29`를 고정해 torch·diffusers와 충돌할 수 있다.
- 시뮬레이터는 경로를 **저장소 루트 기준 상대경로**로 받는다. 그래서 `LLMServingSim/ext` → `results/servingsim`
  심볼릭 링크를 만들고 `ext/...`로 넘긴다.

---

## 3. 우리 패치 3개

전부 `results/servingsim/llmservingsim-hetpilot.patch`에 들어 있다. 복원:

```bash
git clone --recurse-submodules https://github.com/casys-kaist/LLMServingSim.git
cd LLMServingSim && git checkout -b hetpilot-patches
git apply /path/to/results/servingsim/llmservingsim-hetpilot.patch
```

### P1. PD 디코드 풀 초과 시 예외 → 대기

`serving/core/router.py`, `serving/__main__.py`

**문제.** 프리필이 끝난 요청을 디코드 인스턴스에 넘길 때 KV 블록이 모자라면 `RuntimeError`로 **실행 전체가 죽는다**
(`scheduler.add_decode`). 24 GB 카드에서 8B는 가중치 16 GB를 빼면 KV가 41k 토큰뿐이라, PD 구성이 4 req/s에서도 죽었다.

**패치.** 라우터가 `_pd_pending` 큐를 두고, 자리가 없으면 다른 디코드 인스턴스를 순회하고, 그래도 없으면 대기시킨다.
디코드 인스턴스에서 요청이 끝날 때마다 `retry_pending_decode()`로 재시도. DistServe가 KV를 스테이징하는 것과 같은 의미.

**영향.** 이게 없으면 PD 실험 자체가 불가능. 우리 P:D 스윕 전부가 이 패치 위에 있다.

### P2. `kv_loc`이 CXL이면 KV 풀 용량을 그 계층에서 + 비율 `kv_host_frac`

`serving/core/memory_model.py`, `scheduler.py`, `__main__.py`, `config_builder.py`

**문제.** 상류는 `kv_loc`을 배치 정보로만 쓰고 **KV 풀 크기는 언제나 NPU 메모리에서** 계산한다.
호스트에 KV를 두어도 풀이 안 커져서 오프로딩의 이득(= 더 큰 배치)이 표현되지 않았다.

**패치.**
- `kv_loc`이 `CXL:*`이면 풀 용량을 `cxl_mem`에서 잡는다.
- `kv_host_frac` f (0~1)를 placement `default`에 추가. 시퀀스마다 (1−f)는 NPU, f는 호스트에 둔다는 의미로
  `kv_bytes = min(npu_kv/(1−f), host_share/f)`. f=1이면 전부 호스트.
- 디코드 인스턴스가 여럿이면 `cxl_mem`을 나눠 갖는다(`kv_share`).

**쓰는 법.** 인스턴스 placement에 `{"weights":"npu","kv_loc":"cxl:0","kv_evict_loc":"cpu","kv_host_frac":0.25}`,
클러스터에 `cxl_mem: {mem_size, mem_bw, mem_latency, num_devices}`.
**디코드 GPU마다 PCIe 링크가 따로 있으므로 `num_devices`를 디코드 수만큼 주고 `cxl:0,1,2…`로 하나씩 할당해야 한다.**
하나를 공유시키면 로드가 직렬화되어 결과가 틀린다(처음에 이 실수를 했다).

### P3. attention이 호스트 KV를 레이어마다 그 계층 대역폭으로 읽는다

`serving/core/trace_generator.py`

**문제.** 호스트에 둔 KV를 디코드 스텝마다 GPU로 가져오는 시간이 어디에도 안 잡혔다.
처음엔 attention의 **입력 위치**를 CXL로 바꿨는데 효과가 없었다. 변환기(`llm_converter.py`)가 입력 로드 노드를
**배치당 한 번**만 만들고 레이어별로는 **가중치 필드만** 로드 노드로 만들기 때문.

**패치.** attention 레이어에서 `kv_loc`이 CXL이면 `wt_loc`을 그 계층으로 바꾸고 `wt`에 KV 바이트를 싣는다.

```python
wt = f × 2 × kv_head × head_dim × kv_len × kv_fp // tp_size
```

attention 가중치는 이미 프로파일된 지연 안에 있으므로 그 필드를 빌려 써도 이중 계산이 아니다.
ASTRA-Sim이 `runtime = mem_latency + tensor_size / mem_bw`로 과금한다(`AnalyticalMemory.cc:272`).

**검증.** `mem_bw`를 25 → 2 GB/s로 낮추면 TPOT가 비례해서 늘어난다(f=0.5, 279 ms → 2.2 s급).
생성된 트레이스에 `attention_N ... CXL:0 2422784 ...` 줄이 레이어마다(32개) 찍히는 것도 확인.

---

## 4. 시뮬레이터가 못 하는 것

| | 대안 |
|---|---|
| FlexGen식 zigzag 블록 스케줄 | 없음. vLLM continuous batching만 |
| AO를 다른 **GPU**로 | `--enable-attn-offloading`은 **PIM 대상**. 우리는 AO를 뺐으므로 무관 |
| VLM / 인코더 단계 | `pd_type`이 prefill/decode/None뿐. **우리는 이미지 토큰 + 인코더 고정비로 우회** (§15.7) |
| DiT | 우리 `solver/sim.py` |
| 3090, 8B 외 모델 프로파일 | `profiler/` 모듈로 데스크탑에서 직접 뜬다 |

---

## 5. 재현 절차

```bash
# 0. 빌드
bash scripts/setup_llmservingsim.sh
source ~/venv/servingsim/env.sh

# 1. 동봉 예제로 빌드 검증 (vLLM 대비 1% 이내여야 함)
cd ~/papers/refs/code/LLMServingSim && PYTHON=$(which python) bash bench/examples/run.sh RTX4090/Llama-3.1-8B

# 2. 스윕
cd ~/vidbench
python scripts/sweep_pd_kvo.py        # P:D × 호스트 KV 비율 × 도착률
python scripts/sweep_llm_pool.py      # colo 1~4장 (co-solver 재료)
python scripts/sweep_vlm_pool.py      # VLM (이미지 토큰 +576)
python scripts/sweep_pool_parking.py  # 위 둘을 파킹 on으로

# 3. co-solver
python solver/cosolve.py --dit-rates 0,0.01,0.02,0.04 --llm-rates 0,1,2,3,4,6,8 --vlm-rates 0,1,2
```

**새 하드웨어 추가**는 `profiler/`를 그 GPU에서 한 번 돌려 `profiler/perf/<GPU>/<model>/` 표를 만들고,
클러스터 JSON의 `hardware`와 `npu_mem`을 바꾸면 끝. 4090 예제 설정을 복사해서 쓰면 된다.
