#!/usr/bin/env bash
for i in $(seq 0 5); do
#    for j in $(seq 0 1); do
    nohup python ensemble-ddpg.py $i &
#    done
done