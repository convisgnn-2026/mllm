#!/bin/bash

# Run detailed communication/computation timing benchmark with 100 iterations each
# This will collect individual timing measurements and calculate statistics

CUDA_VISIBLE_DEVICES=0 python3 benchmark/comm_comp/run_comm_comp.py \
    -g 1 --model flux --repeat 100 --logging --output-dir results/comm_comp/ &

CUDA_VISIBLE_DEVICES=2,3 python3 benchmark/comm_comp/run_comm_comp.py \
    -g 2 --model flux --repeat 100 --logging --output-dir results/comm_comp/ &

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 benchmark/comm_comp/run_comm_comp.py \
    -g 4 --model flux --repeat 100 --logging --output-dir results/comm_comp/

pkill run_comm_comp.py
