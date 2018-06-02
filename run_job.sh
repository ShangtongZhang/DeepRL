#!/usr/bin/env bash
#export CUDA_VISIBLE_DEVICES=0
GPUs=(0 1 2 3 0)
for i in $(seq 0 4); do
    export CUDA_VISIBLE_DEVICES=${GPUs[$i]}
#    nohup python dist_rl.py --ind1 0 --ind2 $i >| gpu.txt &
    sleep 4s
    nohup py dist_rl.py --ind1 4 --ind2 $i >| debug3.txt &
done

#export CUDA_VISIBLE_DEVICES=1
#for i in $(seq 0 4); do
#    nohup python dist_rl.py --ind1 1 --ind2 $i >| gpu.txt &
#done
