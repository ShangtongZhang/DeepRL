#!/usr/bin/env bash
for i in $(seq 0 3); do
    for j in $(seq 0 1); do
        nohup python plan-ddpg.py --ind1 $i --ind2 $j >| nohup.out &
    done
done