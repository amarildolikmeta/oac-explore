#!/usr/bin/env bash

# RUN OAC
for ((i=0;i<5;i+=1))
do
    python main.py --seed=$i --domain riverswim --alg p-oac --n_estimators 10 --delta 0.95 --max_path_length 100 --share_layers --num_layers 1 --layer_size 16 &
done

for ((i=10;i<15;i+=1))
do
    python main.py --seed=$i --domain riverswim --alg p-oac --n_estimators 10 --delta 0.95 --max_path_length 100 --share_layers --num_layers 1 --layer_size 16 --counts  &
done
