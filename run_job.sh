#!/usr/bin/env bash
for i in $(seq 0 5); do
    for j in $(seq 0 0); do
        nohup python plan-ddpg.py --ind1 $i --ind2 $j >| nohup.out &
    done
done