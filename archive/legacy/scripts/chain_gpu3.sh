#!/bin/bash
cd "$(dirname "${BASH_SOURCE[0]}")/.."
while pgrep -f "run_gpu3_mixed.sh" > /dev/null; do sleep 30; done
echo "[chain] mig_mixed done -> reconfiguring GPU3 to 7 x 1g.34gb"
while pgrep -f "serve_bench.py --workload.*mig_mixed" > /dev/null; do sleep 10; done
sleep 10
sudo nvidia-smi mig -i 3 -dci  2>&1 | tail -1
sudo nvidia-smi mig -i 3 -dgi  2>&1 | tail -1
sudo nvidia-smi mig -i 3 -cgi 19,19,19,19,19,19,19 -C 2>&1 | grep -c Successfully
nvidia-smi -L | grep -c "MIG 1g.34gb"
./scripts/run_gpu3_uniform.sh
