#!/usr/bin/env bash
GPUs=(0 1 2 3)
for i in $(seq 0 2); do
    export CUDA_VISIBLE_DEVICES=${GPUs[$i]}
    sleep 4s
    nohup py dist_rl.py --ind1 $1 --ind2 $(($i + $2)) >| atari_$1_program_$i.out &
done
