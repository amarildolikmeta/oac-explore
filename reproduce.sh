#!/usr/bin/env bash

# RUN OAC
for ((i=0;i<5;i+=1))
do
    python3 main.py --alg p-oac --seed $i --domain=lqg --num_layers 1 --layer_size 8 --n_estimators 10  &

done

# RUN SAC
for ((i=0;i<5;i+=1))
do
  python3 main.py --alg oac --seed $i --domain=lqg --num_layers 1 --layer_size 8 --beta_UB=4.66 --delta=23.53 &
done

for ((i=0;i<5;i+=1))
do
    python3 main.py --alg sac --seed $i --domain=lqg --num_layers 1 --layer_size 8 &
done
