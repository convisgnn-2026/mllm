#!/usr/bin/env bash
# loadgen.py 파이썬 프로세스가 LIMIT 초를 넘기면 끊는다 (병적인 PD 셀 상한).
# 패턴은 python 실행 파일로 시작하는 명령줄만 잡는다 — bash -c 래퍼의 명령줄 문자열에 걸리지 않도록.
#   bench/hetero/watchdog.sh <stop-file> [LIMIT]   ; stop-file 이 생기면 종료
STOP=${1:?stop file}; LIMIT=${2:-600}
while [ ! -e "$STOP" ]; do
  for p in $(pgrep -f '^[^ ]*/python [^ ]*loadgen\.py '); do
    e=$(ps -o etimes= -p "$p" 2>/dev/null | tr -d ' ') || continue
    if [ "${e:-0}" -gt "$LIMIT" ]; then echo "$(date +%T) capped pid $p after ${e}s"; kill "$p"; fi
  done
  sleep 15
done
# 참고: vllm bench/serve 를 kill 하면 EngineCore 자식이 고아로 남아 GPU 메모리를 잡을 수 있다.
#       정리는  pkill -9 -f "VLLM::EngineCore"  또는  nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs kill -9
