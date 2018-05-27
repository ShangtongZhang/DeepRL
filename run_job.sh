#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
for i in $(seq 0 5); do
    nohup python dist_rl.py --ind1 $i --ind2 0 >| gpu.txt &
done

export CUDA_VISIBLE_DEVICES=1
for i in $(seq 0 5); do
    nohup python dist_rl.py --ind1 $i --ind2 1 >| gpu.txt &
done
