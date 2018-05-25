#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=2
for i in $(seq 0 0); do
    for j in $(seq 0 2); do
        nohup python dist_rl.py --ind1 $i --ind2 $j >| gpu.txt &
    done
done

export CUDA_VISIBLE_DEVICES=3
for i in $(seq 1 1); do
    for j in $(seq 0 2); do
        nohup python dist_rl.py --ind1 $i --ind2 $j >| gpu.txt &
    done
done
