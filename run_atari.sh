#!/usr/bin/env bash
GPUs=(0 1 2 3 0)
for i in $(seq 0 4); do
    export CUDA_VISIBLE_DEVICES=${GPUs[$i]}
    sleep 4s
#    nohup py dist_rl.py --ind1 $1 --ind2 $(($i + $2)) >| atari_$1_program_$i.out &
#    nohup py dist_rl.py --ind1 $1 --ind2 $i >| atari_$1_program_$i.out &
    nohup py dist_rl.py --ind1 $i >| extra_qr_$i.out &
done
