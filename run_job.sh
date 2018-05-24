#!/usr/bin/env bash
for i in $(seq 0 4); do
    nohup python ensemble-ddpg.py --ind1 $i --ind2 0&
#    for j in $(seq 0 1); do
#        nohup python ensemble-ddpg.py $i $j&
#    done
done