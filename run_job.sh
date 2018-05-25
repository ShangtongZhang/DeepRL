#!/usr/bin/env bash
for i in $(seq 0 1); do
    for j in $(seq 0 2); do
        nohup python dist_rl.py --ind1 $i --ind2 $j &
    done
done
