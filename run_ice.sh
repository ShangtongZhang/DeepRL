#!/usr/bin/env bash
GPUs=(0 1 2 3)
for i in $(seq 0 3); do
    export CUDA_VISIBLE_DEVICES=${GPUs[$i]}
    sleep 4s
    id=$(($i + $1))
    echo $id
    nohup py dist_rl.py $id >| ice_cliff_$id.txt &
done
