#!/usr/bin/env bash

# RUN OAC
for ((i=0;i<5;i+=1))
do
    python3 main.py --seed=$i --domain=humanoid --max_path_length 1000 --min_num_steps_before_training 10000 --epochs 9000 --num_eval_steps_per_epoch 5000 --num_expl_steps_per_train_loop 1000 --num_trains_per_train_loop 1000 --layer_size 256 --num_layers 2 --batch_size 256 --r_max 5 --replay_buffer_size 1e6 --alg p-oac --delta=0.95 --n_estimators 10 --share_layers  &
done
