#!/usr/bin/env bash
#GPUs=(0 0 1 2 3 4 5 6)
#GPUs=(0 1 2 3 4 5)
GPUs=(0 1 2 3 4 5 6 7)
#for i in $(seq 0 7); do
#    for j in $(seq 0 4); do
#        nohup bash docker_python.sh ${GPUs[$i]} "job.py --i1 $i --i2 $j" >| job_${i}_${j}.out &
#    done
#done

#for i in $(seq 0 7); do
#    for j in $(seq 0 4); do
#        nohup bash docker_python.sh $i "job.py --i1 $i --i2 $j" >| job_${i}_${j}.out &
#        nohup bash docker_python.sh $i "MDP.py --i1 $i --i2 $j" >| job_${i}_${j}.out &
#    done
#done

rm -f jobs.txt
touch jobs.txt
for i in $(seq 0 305); do
    echo "$i" >> jobs.txt
done
cat jobs.txt | xargs -n 1 -P 50 sh -c 'bash docker_python.sh 0 "job.py --i1 $0"'



