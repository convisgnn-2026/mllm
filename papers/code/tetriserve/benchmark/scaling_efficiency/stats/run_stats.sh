#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 benchmark/scaling_efficiency/stats/test_e2e_stats.py \
    -g 1 --logging &
CUDA_VISIBLE_DEVICES=2,3 python3 benchmark/scaling_efficiency/stats/test_e2e_stats.py \
    -g 2 --logging

pkill -f test_e2e_stats.py

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 benchmark/scaling_efficiency/stats/test_e2e_stats.py \
    -g 4 --logging
