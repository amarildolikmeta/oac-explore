#!/usr/bin/env bash
for ((i=0;i<5;i+=1))
do
  python3 main.py --seed $i --delta=0.95 --no_gpu --domain riverswim --num_layers 1 --layer_size 16 --n_estimators 10  --alg g-oac
done