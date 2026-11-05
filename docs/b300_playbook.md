# B300 / RTX PRO 6000 포팅 플레이북 (2026-11-05)

> 동료가 같은 환경에서 그대로 재현할 수 있게 쓴 실행 안내서다. "무엇을 왜"는 `docs/handoff_b300.md`(배경)와
> `docs/solver_need_experiments.md`(실험 기록, "B300 포팅" 절부터)에 있고, 여기에는 **환경·명령·결과 위치**만 적는다.

---

## 0. 한눈에 보기

| 질문 | 답 | 어디서 |
|---|---|---|
| 한 GPU를 두 프로세스가 나누면 왜 느려지나 | CUDA가 컨텍스트를 ~2.3 ms 단위로 번갈아 실행(시간 분할). 커널은 안 느려지고 상대 차례만큼 기다림 | `nsys_s4_b300/`, `nsys_c_b300/` |
| B300에서 모양이 4090과 다른 점 | decode 스텝(1.84 ms)이 슬라이스보다 짧아 스텝은 그대로, 스텝 사이 간격이 0.2 → 1.7 ms | 같은 곳, `fig_dit_kernel_lengths.png` |
| MPS를 켜면 | 대기가 사라지고 경합으로 바뀜. 토큰당 1.72× → 1.23× | `nsys_s4_b300_mps/` |
| MIG로 나누면 | 간섭 0 (k = 1.00). 대신 반쪽 SM이라 토큰당 1.36× 느림 | `nsys_s4_b300_mig/` |
| 같은 노드 KV 전송 | NVLink(cuda_ipc) 20 ms / 235 MB. UCX를 TCP로 강제하면 160 ms | `pd_xfer_b300/` |
| 노드 간 KV 전송 (25 GbE) | 튜닝 TCP 1.45 GB/s. 설정 없으면 Calico VXLAN으로 0.36 GB/s | `pd_xfer_mn/` |

---

## 1. 환경

### 노드
| | B300 노드 (`vanta-gpu-worker-study01`, 10.201.4.151) | PRO 6000 노드 (`vanta-dev-gpu-worker-infer01`, 10.201.4.101) |
|---|---|---|
| GPU | B300 SXM6 × 8 (sm_103, 275 GB). **우리가 쓰는 것: GPU1, GPU3** (UUID `GPU-1b5a091d…`, `GPU-55867e89…`) | RTX PRO 6000 Blackwell × 4 (sm_120, 96 GB). **우리가 쓰는 것: GPU1** (`GPU-d8d3cf1e…`). GPU0은 MIG Enabled(인스턴스 없음)라 CUDA 불가 |
| 다른 사용자 | GPU 0·2·5·7 사용 중 → 절대 건드리지 않음 | GPU 2·3 사용 중 |
| 공유 방식 기본값 | compute mode Default, MPS 없음, MIG off | 동일 |
| 노드 간 링크 | bond0 (25 GbE, active-backup, MTU 1500). RoCE는 꺼져 있고 범위 밖 | 동일 |
| sudo | 비밀번호 없이 가능 (MIG 설정에 사용) | 비밀번호 sudo |

`CUDA_VISIBLE_DEVICES`는 항상 UUID로 지정한다(인덱스는 재부팅·MIG에 따라 바뀜).

### 소프트웨어 (양쪽 동일)
```
~/venv/mllm          Python 3.12 (uv), vLLM 0.28.0+cu129, torch 2.13+cu129, diffusers 0.40, transformers 5.14.1, NIXL 1.5.2
~/mllm               이 저장소, 브랜치 research/latest-multimodal-2026-10-15
~/.cache/huggingface meta-llama/Llama-3.2-3B-Instruct, stabilityai/stable-diffusion-3.5-medium (HF_HUB_OFFLINE=1 로 실행)
~/tools/nsys/...     Nsight Systems CLI 2026.5.1 (B300 노드만; 설치법은 §2)
```
venv 만들기(처음 한 번):
```
uv python install 3.12 && uv venv ~/venv/mllm --python 3.12
uv pip install --python ~/venv/mllm/bin/python --index-strategy unsafe-best-match -r env/requirements-b300-freeze.txt
```
항상 `export PATH=$HOME/venv/mllm/bin:$PATH` 를 먼저 한다. FlashInfer JIT가 `ninja`를 PATH에서 찾는다(없으면 vLLM 기동 실패).

### 어텐션 백엔드
Blackwell에서 vLLM은 B300 → FLASHINFER, PRO 6000 → FLASH_ATTN 을 자동 선택한다. **PD 분할(NIXL)은 양쪽 백엔드가 같아야 한다**
(호환 해시에 포함). 노드 간 PD는 `--attention-backend FLASHINFER` 를 양쪽에 준다. 환경변수 `VLLM_ATTENTION_BACKEND`는 0.28에 없다.

---

## 2. Nsight Systems

시스템 `nsys`(/usr/local/cuda)는 깨져 있고, `/opt/nvidia/nsight-systems/2025.6.3`은 vLLM과 조합이 나쁘다
(graph 모드에서 EngineCore 교착, node 모드에서 `--trace-fork-before-exec=true` 없이는 EngineCore 커널 누락). **2026.5.1을 홈에 푼다:**
```
mkdir -p ~/tools/nsys_dl ~/tools/nsys && cd ~/tools/nsys_dl && \
curl -L -o nsys-cli-2026.5.1.rpm https://developer.download.nvidia.com/devtools/repos/rhel9/x86_64/NsightSystems-linux-cli-public-2026.5.1.161-3889610.rpm && \
cd ~/tools/nsys && rpm2cpio ~/tools/nsys_dl/nsys-cli-2026.5.1.rpm | cpio -idm && \
find ~/tools/nsys -name nsys -type f -exec {} --version \;
```
(6 KB짜리 `nsight-systems-cli-…rpm`은 메타 패키지라 받지 말 것.) 캡처 스크립트의 B300 기본 경로가 이 위치다.
MPS 모드에서는 nsys가 소프트웨어 계측 트레이스로 떨어져 절대값이 조금 다르다 → MPS 결과는 MPS 단독과만 비교한다.

---

## 3. 실행 스크립트

모두 `~/mllm`에서 실행. 각 실행은 서버를 띄우고 → 플랜을 재생하고 → 정리까지 스스로 한다. 끝나면 `pgrep -u $USER -af "vllm serve|dit_server|nsys launch"` 로 남은 것이 없는지 확인.

### 3.1 간섭 캡처 `bench/hetero/profile_mixed_nsys.sh`
```
PLACE=<배치> [옵션] bench/hetero/profile_mixed_nsys.sh mixed|llm|dit
```
한 배치당 세 번(혼합 / LLM만 / DiT만), 각 약 5분(기동 1분 + 워밍업 90 s + 캡처 90 s). 출력 `results/hetero/<배치 폴더>/<모드>/`.

| PLACE | 뜻 | GPU |
|---|---|---|
| `S4_B300` | 한 장 공유: vLLM + dit_server 가 GPU1 공유 | GPU1 |
| `C_B300` | PD 분할 + decode 카드 공유: prefill GPU3, decode + dit_server GPU1 | GPU1, GPU3 |
| `S4_B300_MIG` | GPU1을 3g.135gb 두 조각으로 나눠 vLLM / dit_server 하나씩 (§4.2 먼저) | GPU1 (MIG) |
| `S4_B300` + `MPS_PIPE=...` | 위 S4를 MPS 클라이언트로 (§4.1 먼저). 출력 폴더에 `_mps` 가 붙음 | GPU1 |

옵션(환경변수): `MPS_PIPE`, `MPS_PCT_LLM`, `MPS_PCT_DIT`(SM 비율 상한 %), `MPS_TAG`(출력 폴더 접미사), `GRAPH_TRACE_B300`(기본 graph), `NSYS_B300`(nsys 경로).

예: 세 모드를 연달아
```
for m in mixed llm dit; do PLACE=S4_B300 bench/hetero/profile_mixed_nsys.sh $m; done
```
**주의** 실행 중에 이 스크립트 파일을 편집하면 돌고 있는 인스턴스가 깨진다(bash는 스크립트를 읽으면서 실행). 하네스는 8100/8192/8300/5600 포트를 쓰므로 같은 노드에서 다른 실험을 동시에 돌리면 포트가 겹치지 않게 한다.

### 3.2 분석
```
python3 bench/hetero/analyze_nsys.py       <dir>/mixed <dir>/llm <dir>/dit          # 저장소 표준 (스텝 = graph ≥ 2 ms, 4090 기준)
STEP_MIN_MS=1 python3 bench/hetero/analyze_nsys_steps.py <dir>/mixed <dir>/llm <dir>/dit   # B300용 (스텝 ≥ 1 ms)
~/venv/mllm/bin/python bench/hetero/plot_nsys_timeline.py <dir>/mixed --t0 <s> --span 0.03   # 타임라인 그림
```
둘 다 `<dir>/<mode>/report.sqlite`가 필요하다. 없으면 먼저 내보낸다:
```
~/tools/nsys/opt/nvidia/nsight-systems-cli/2026.5.1/target-linux-x64/nsys export --type sqlite --output <dir>/<mode>/report.sqlite <dir>/<mode>/report.nsys-rep
```
`.nsys-rep`/`.sqlite`는 100 MB급이라 git에 없다(`nsys_summary*.json`, 그림, 로그만 커밋). 원본이 필요하면 B300 노드의 `~/mllm/results/hetero/...`에 있다.
B300 캡처는 반드시 1 ms 문턱 값을 본다. 표준 분석기는 B300 스텝의 95%를 버린다(기록 참조).

### 3.3 KV 전송 프로브
```
# 같은 노드 (GPU3 → GPU1). UCX 환경변수 없이 돌리면 NVLink(cuda_ipc) 경로.
bench/hetero/pd_xfer_probe_b300.sh default
# TCP 경로 비교용
UCX_TLS=cuda_copy,tcp,self UCX_TCP_TX_SEG_SIZE=4m UCX_TCP_RX_SEG_SIZE=4m UCX_TCP_SNDBUF=64m UCX_TCP_RCVBUF=64m UCX_MAX_RNDV_LANES=4 UCX_RNDV_THRESH=64k UCX_NET_DEVICES=lo \
  bench/hetero/pd_xfer_probe_b300.sh tcp_tuned_lo

# 노드 간 (B300 GPU3 prefill → PRO 6000 GPU1 decode). PRO 6000 쪽 서버는 ssh 로 스크립트가 직접 띄운다 (비밀번호 없는 ssh 필요).
UCX_TLS=cuda_copy,tcp,self UCX_TCP_TX_SEG_SIZE=4m UCX_TCP_RX_SEG_SIZE=4m UCX_TCP_SNDBUF=64m UCX_TCP_RCVBUF=64m UCX_MAX_RNDV_LANES=4 UCX_RNDV_THRESH=64k UCX_NET_DEVICES=bond0 \
  bench/hetero/pd_xfer_probe_mn.sh tcp_tuned_bond0
```
결과는 콘솔의 `seq : KV Transfer metrics ...`(순차 8건) / `conc :`(동시 16건) 줄과 `results/hetero/pd_xfer_{b300,mn}/<tag>/`. 2048 토큰 프롬프트 = KV 235 MB(3B 기준)이므로 실효 GB/s = 0.235 / (평균 ms / 1000).
노드 간은 `UCX_NET_DEVICES=bond0` 필수(없으면 Calico VXLAN을 타서 4배 느림). 프로브는 8110/8193/5610 포트를 써서 3.1과 동시에 돌 수 있다.

### 3.4 단일 카드 스모크
`results/hetero/stage0_b300/`의 `smoke2`가 Stage 0-1 기록이다(vLLM 3B + dit_server 동거, KV 37,440 토큰 = 4090과 동일).

---

## 4. 공유 방식 바꾸기 (GPU1 한 장만, 다른 사용자 영향 없음)

### 4.1 MPS (root 불필요)
```
export CUDA_MPS_PIPE_DIRECTORY=$HOME/mps_gpu1/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/mps_gpu1/log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
CUDA_VISIBLE_DEVICES=GPU-1b5a091d-563d-e5ed-8201-2f3a95e09483 nvidia-cuda-mps-control -d      # 데몬 (GPU1 전용)
MPS_PIPE=$HOME/mps_gpu1/pipe PLACE=S4_B300 bench/hetero/profile_mixed_nsys.sh mixed            # 클라이언트로 실행
echo quit | nvidia-cuda-mps-control                                                             # 끝나면 내리기
```
SM 비율을 주려면 `MPS_PCT_LLM=70 MPS_PCT_DIT=30 MPS_TAG=_70_30` 을 추가. MPS 서버는 사용자당 하나이고 클라이언트 간 격리가 없다.

### 4.2 MIG (sudo 필요, GPU1이 비어 있어야 함)
```
sudo nvidia-smi -i 1 -mig 1                      # 켜기
sudo nvidia-smi mig -i 1 -cgi 9,9 -C             # 3g.135gb × 2 (+ compute instance)
nvidia-smi -L | grep -A2 "^GPU 1:"               # MIG-… UUID 두 개 → profile_mixed_nsys.sh 의 MIG_LLM / MIG_DIT 와 같은지 확인
PLACE=S4_B300_MIG bench/hetero/profile_mixed_nsys.sh mixed
# 되돌리기 (통째 GPU1이 필요할 때)
sudo nvidia-smi mig -i 1 -dci && sudo nvidia-smi mig -i 1 -dgi && sudo nvidia-smi -i 1 -mig 0
```
프로파일 목록은 `sudo nvidia-smi mig -i 1 -lgip`. 2026-11-05 현재 GPU1은 **MIG가 켜진 상태**로 남아 있다.

---

## 5. 결과 폴더

```
results/hetero/
  stage0_b300/            Stage 0-1 스모크, nsys 옵션 진단(diag_*), 설정 분리 실험
  nsys_s4_b300/           한 장 공유, 시간 분할      (mixed/llm/dit, nsys_summary.json, nsys_summary_thr1ms.json, fig_*.png)
  nsys_c_b300/            PD 분할 + decode 카드 공유
  nsys_s4_b300_mps/       한 장 공유, MPS            (dit 1차는 _dit_contaminated_by_probe, 재실행본이 dit/)
  nsys_s4_b300_mig/       MIG 반반
  nsys_s4_b300_node/      (참고) 2025.6.3 node 모드: EngineCore 커널 누락, 판정에 안 씀
  nsys_s4_b300_graphtrace_stalled/  (참고) 2025.6.3 graph 모드: EngineCore 교착
  pd_xfer_b300/           같은 노드 KV 전송 4설정
  pd_xfer_mn/             노드 간 KV 전송 2설정 (_로 시작하는 폴더는 실패한 시도: ssh 세션, 포트 충돌, 해시 불일치)
```
각 캡처 폴더의 `replay/requests.live.jsonl`이 요청 기록(TTFT/TPOT/DiT 단계 시간), `serve_*.log`가 vLLM 로그, `dmon.txt`가 nvidia-smi 1 s 샘플이다.

---

## 6. 알아두면 시간이 절약되는 것

- **실행 중 스크립트 편집 금지.** 체인이 끝난 뒤 고친다.
- **pkill -f 패턴에 자기 명령이 걸린다.** 셸에서 `pkill -f "vllm serve"` 를 치면 그 명령줄 자체가 매치돼 자신이 죽을 수 있다. PID로 죽이거나 `pgrep -f "[v]llm serve"` 꼴을 쓴다.
- **원격 서버는 `ssh -n -f host "cmd"`** 로 띄운다. `ssh host "cmd &"` 는 서버가 세션을 잡아 돌아오지 않는다.
- **같은 노드 PD는 UCX 환경변수를 주지 않는다**(NVLink). 노드 간만 튜닝 TCP + `UCX_NET_DEVICES=bond0`.
- **DiT 512²는 B300에서 CPU launch에 절반이 묶인다**(요청당 커널 67k개, GPU 몫 0.54). 요청 수준 DiT 지연은 CPU 부하·nsys 오버헤드에 흔들리므로 커널 수준 지표로 판정한다. 큰 해상도/배치로 가면 해결된다.
- 다른 사용자의 프로세스는 절대 종료하지 않는다. `pgrep -u $USER` 로 자기 것만 본다.
