#!/usr/bin/env bash
#export CUDA_VISIBLE_DEVICES=0
GPUs=(0 1 2 3 0 1)
for i in $(seq 0 2); do
    export CUDA_VISIBLE_DEVICES=${GPUs[$i]}
    sleep 4s
#    echo $(($i + $1))
    nohup py dist_rl.py --ind1 0 --ind2 $(($i + $1)) >| ice_cliff_$1_$i.txt &
#    echo atari_$1_program_$i
#    nohup py dist_rl.py --ind1 $1 --ind2 $i >| atari_$1_program_$i.out &
done

#export CUDA_VISIBLE_DEVICES=1
#for i in $(seq 0 4); do
#    nohup python dist_rl.py --ind1 1 --ind2 $i >| gpu.txt &
#done
