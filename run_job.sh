#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
for i in $(seq 0 4); do
    nohup python dist_rl.py --ind1 0 --ind2 $i >| gpu.txt &
done

export CUDA_VISIBLE_DEVICES=1
for i in $(seq 0 4); do
    nohup python dist_rl.py --ind1 1 --ind2 $i >| gpu.txt &
done
