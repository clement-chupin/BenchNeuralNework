#!/bin/bash

if [$i -lt 10 ]
then
    sbatch single_run_gpu.sh $1 $2 $3 $4;
else
    sbatch single_run_cpu.sh $1 $2 $3 $4;
fi












