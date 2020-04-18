#!/usr/bin/env bash

# RUN P-OAC counts
for ((i=0;i<5;i+=1))
do
    python3 main.py --seed=$i --domain riverswim --alg p-oac --n_estimators 10 --delta 0.95 --max_path_length 100 --share_layers --counts --mean_update --global_opt &
done


# RUN G-OAC counts
for ((i=0;i<5;i+=1))
do
    python3 main.py --seed=$i --domain riverswim --alg g-oac  --delta 0.95 --max_path_length 100 --share_layers --counts --mean_update --global_opt &
done

