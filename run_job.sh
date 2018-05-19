#!/usr/bin/env bash
for i in $(seq 0 2); do
    for j in $(seq 0 1); do
        nohup python ensemble-ddpg.py $i $j &
    done
done