#!/usr/bin/env bash
GPUs=(0 1 2 3 4 5 6 7)

#for i in $(seq 0 7); do
#    for j in $(seq 0 0); do
#        nohup bash docker_python.sh ${GPUs[$i]} "template_jobs.py --i $i --j $j" >| job_${i}_${j}.out &
#    done
#done


rm -f jobs.txt
touch jobs.txt
#for i in $(seq 0 3888); do
for i in $(seq 0 2592); do
    echo "$i" >> jobs.txt
done
cat jobs.txt | xargs -n 1 -P 40 sh -c 'bash docker_python.sh 0 "template_jobs.py --i $0"'
rm -f jobs.txt


#rm -f jobs.txt
#touch jobs.txt
#for i in $(seq 0 6); do
#    echo "$i ${GPUs[$(($i % 8))]}" >> jobs.txt
#done
#cat jobs.txt | xargs -n 2 -P 40 sh -c 'bash docker_python.sh $1 "template_jobs.py --i $0"'
#rm -f jobs.txt
