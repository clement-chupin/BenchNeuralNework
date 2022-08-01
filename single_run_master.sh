#!/bin/bash

if [$i -lt 10 ]
then
    sbatch -o "./log_meso/log_propre_gpu" single_run_gpu.sh $1 $2 $3 $4;
else
    sbatch -o "./log_meso/log_propre_cpu" single_run_cpu.sh $1 $2 $3 $4;
fi












