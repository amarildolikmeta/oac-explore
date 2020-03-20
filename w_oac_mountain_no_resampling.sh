#!/usr/bin/env bash
python3 run_mountain_modified_replay_buffer  --delta=0.95 --no_gpu --domain mountain --num_layers 1 --layer_size 16 --n_estimators 10  --alg p-oac --ensemble --no_resampling