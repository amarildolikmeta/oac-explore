#!/usr/bin/env bash

# RUN P-OAC global
for ((i=0;i<5;i+=1))
do
    python3 main.py --seed=$i --domain riverswim --alg p-oac --n_estimators 10 --delta 0.95 --max_path_length 100 --share_layers  --global_opt &
done


# RUN G-OAC global
for ((i=0;i<5;i+=1))
do
    python3 main.py --seed=$i --domain riverswim --alg g-oac  --delta 0.95 --max_path_length 100 --share_layers  --global_opt  &
done

# RUN P-OAC global mean update
for ((i=0;i<5;i+=1))
do
    python3 main.py --seed=$i --domain riverswim --alg p-oac --n_estimators 10  --delta 0.95 --max_path_length 100 --share_layers  --global_opt --mean_update &
done

# RUN G-OAC global mean update
for ((i=0;i<5;i+=1))
do
    python3 main.py --seed=$i --domain riverswim --alg g-oac  --delta 0.95 --max_path_length 100 --share_layers  --global_opt --mean_update &
done

# RUN P-OAC global mean update counts
for ((i=0;i<5;i+=1))
do
    python3 main.py --seed=$i --domain riverswim --alg p-oac --n_estimators 10  --delta 0.95 --max_path_length 100 --share_layers  --global_opt --mean_update --counts &
done

# RUN G-OAC global mean update counts
for ((i=0;i<5;i+=1))
do
    python3 main.py --seed=$i --domain riverswim --alg g-oac  --delta 0.95 --max_path_length 100 --share_layers  --global_opt --mean_update --counts &
done

