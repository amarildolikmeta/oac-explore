#!/usr/bin/env bash
betas=(3 4 5 6)
delta=(10 20 30 40)

for beta in "${betas[@]}"
do
  for delta in "${delta[@]}"
  do
      for ((i=0;i<3;i+=1))
      do
          python3 main.py --domain cliff_mono --alg oac  --beta_UB=$beta --delta=$delta --num_layers 2 --layer_size 8 --batch_size 32 --no_gpu  --num_eval_steps_per_epoch 1000 --num_expl_steps_per_train_loop 1000 --replay_buffer_size 1e5  --epochs 500 --suffix delta_"${delta}"_beta_"${beta}"/ &
      done
  done
done

