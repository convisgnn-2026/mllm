#!/bin/bash

MODELS=("sd3" "flux")

for model in ${MODELS[@]}; do
    # export LOG_DIR="results/scaling_efficiency/$model"
    export LOG_DIR="/workspaces/FluidServe/FluidServe_logs/benchmark/scaling_efficiency/$model"
    mkdir -p $LOG_DIR

    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 benchmark/scaling_efficiency/test_e2e.py \
        -m $model -g 4 --repeat 20 --logging &
    CUDA_VISIBLE_DEVICES=4 python3 benchmark/scaling_efficiency/test_e2e.py \
        -m $model -g 1 --repeat 20 --logging &
    CUDA_VISIBLE_DEVICES=6,7 python3 benchmark/scaling_efficiency/test_e2e.py \
        -m $model -g 2 --repeat 20 --logging

    pkill test_e2e.py

    python3 benchmark/scaling_efficiency/test_e2e.py \
        -m $model -g 8 --repeat 20 --logging

done

# only for debugging
# CUDA_VISIBLE_DEVICES=0,1 python3 benchmark/scaling_efficiency/test_e2e.py \
    # -g 2 --repeat 3 --batch-size 1 --height 1024 --width 1024 2>&1 | tee log1.log
# CUDA_VISIBLE_DEVICES=0 python3 benchmark/scaling_efficiency/test_e2e.py \
    # -g 1 --repeat 3 --batch-size 1 --height 1024 --width 1024 2>&1 | tee log1.log

# export SIZE=2048
# CUDA_VISIBLE_DEVICES=0,1 python3 benchmark/scaling_efficiency/test_e2e.py \
#     -g 2 --repeat 8 --batch-size 1 --height $SIZE --width $SIZE 2>&1 | tee log2.log

# export SIZE=2048
# CUDA_VISIBLE_DEVICES=2,3 python3 benchmark/scaling_efficiency/test_e2e.py \
#     -g 2 --repeat 8 --batch-size 1 --height $SIZE --width $SIZE 2>&1 | tee log2.log
