#!/bin/bash
sleep 1
if [[ $i -lt 10 ]]
then
    echo $1 $2 $3 $4
    #sbatch single_run_gpu.sh $1 $2 $3 $4;
else
    echo $1 $2 $3 $4
    #sbatch single_run_cpu.sh $1 $2 $3 $4;
fi












