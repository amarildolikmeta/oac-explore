#!/usr/bin/env bash

## RUN P-OAC counts
#for ((i=0;i<5;i+=1))
#do
#    python3 main.py --seed=$i --domain=point --max_path_length 300 --min_num_steps_before_training 10000 --epochs 400 --num_eval_steps_per_epoch 3000 --num_expl_steps_per_train_loop 3000 --num_trains_per_train_loop 1000 --layer_size 64 --num_layers 2 --batch_size 64 --r_max 0 --r_min -20 --replay_buffer_size 1e5 --alg p-oac --delta=0.95 --n_estimators 10 --share_layers --counts --mean_update --global_opt &
#done
#
#
## RUN G-OAC counts
#for ((i=0;i<5;i+=1))
#do
#    python3 main.py --seed=$i --domain=point --max_path_length 300 --min_num_steps_before_training 10000 --epochs 400 --num_eval_steps_per_epoch 3000 --num_expl_steps_per_train_loop 3000 --num_trains_per_train_loop 1000 --layer_size 64 --num_layers 2 --batch_size 64 --r_max 0 --r_min -20 --replay_buffer_size 1e5 --alg g-oac --delta=0.95 --share_layers --counts --mean_update --global_opt &
#done
#
#
### RUN OAC
##for ((i=0;i<5;i+=1))
##do
##    python3 main.py --seed=$i --domain=point --max_path_length 300 --min_num_steps_before_training 10000 --epochs 400 --num_eval_steps_per_epoch 3000 --num_expl_steps_per_train_loop 3000 --num_trains_per_train_loop 1000 --layer_size 64 --num_layers 2 --batch_size 64 --r_max 0 --r_min -20 --replay_buffer_size 1e5 --alg oac  --beta_UB=4.66 --delta=23.53 &
##done
##
##
### RUN SAC
##for ((i=0;i<5;i+=1))
##do
##    python3 main.py --seed=$i --domain=point --max_path_length 300 --min_num_steps_before_training 10000 --epochs 400 --num_eval_steps_per_epoch 3000 --num_expl_steps_per_train_loop 3000 --num_trains_per_train_loop 1000 --layer_size 64 --num_layers 2 --batch_size 64 --r_max 0 --r_min -20 --replay_buffer_size 1e5 --alg sac  &
##done



for ((i=0;i<5;i+=1))
do
    python3 main.py  --domain point --difficulty hard --alg g-oac --n_estimators 2 --delta 0.95 --max_path_length 300   --num_layers 2 --layer_size 32 --batch_size 32 --no_gpu  --num_eval_steps_per_epoch 3000 --num_expl_steps_per_train_loop 3000 --replay_buffer_size 1e5 --r_max -1 --r_min -1.1 --share_layers  --epochs 2000 --clip_state --max_state 25 --terminal --no_train_bias --suffix range_10/ &
done

for ((i=0;i<5;i+=1))
do
    python3 main.py  --domain point --difficulty hard --alg g-oac --n_estimators 2 --delta 0.95 --max_path_length 300   --num_layers 2 --layer_size 32 --batch_size 32 --no_gpu  --num_eval_steps_per_epoch 3000 --num_expl_steps_per_train_loop 3000 --replay_buffer_size 1e5 --r_max -1 --r_min -1.05 --share_layers  --epochs 2000 --clip_state --max_state 25 --terminal --no_train_bias --suffix range_5/ &
done
for ((i=0;i<5;i+=1))
do
    python3 main.py  --domain point --difficulty hard --alg g-oac --n_estimators 2 --delta 0.95 --max_path_length 300   --num_layers 2 --layer_size 32 --batch_size 32 --no_gpu  --num_eval_steps_per_epoch 3000 --num_expl_steps_per_train_loop 3000 --replay_buffer_size 1e5 --r_max -1 --r_min -1.2 --share_layers  --epochs 2000 --clip_state --max_state 25 --terminal --no_train_bias --suffix range_20/ &
done
for ((i=0;i<5;i+=1))
do
    python3 main.py  --domain point --difficulty hard --alg g-oac --n_estimators 2 --delta 0.95 --max_path_length 300   --num_layers 2 --layer_size 32 --batch_size 32 --no_gpu  --num_eval_steps_per_epoch 3000 --num_expl_steps_per_train_loop 3000 --replay_buffer_size 1e5 --r_max -1 --r_min -1.5 --share_layers  --epochs 2000 --clip_state --max_state 25 --terminal --no_train_bias --suffix range_50/ &
done
for ((i=0;i<5;i+=1))
do
    python3 main.py  --domain point --difficulty hard --alg g-oac --n_estimators 2 --delta 0.95 --max_path_length 300   --num_layers 2 --layer_size 32 --batch_size 32 --no_gpu  --num_eval_steps_per_epoch 3000 --num_expl_steps_per_train_loop 3000 --replay_buffer_size 1e5 --r_max -1 --r_min -2 --share_layers  --epochs 2000 --clip_state --max_state 25 --terminal --no_train_bias --suffix range_100/ &
done
