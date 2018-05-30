#!/usr/bin/env bash
for i in $(seq 0 6); do
    for j in $(seq 0 0); do
        nohup python ensemble-ddpg.py --ind1 $i --ind2 $j &
    done
done