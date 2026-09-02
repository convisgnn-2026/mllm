#!/bin/bash

# Run detailed communication/computation timing benchmark with 100 iterations each
# This will collect individual timing measurements and calculate statistics

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 benchmark/comm_comp/run_comm_comp.py \
    -g 8 --model flux --repeat 100 --logging --output-dir results/comm_comp/
