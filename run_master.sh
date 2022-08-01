#!/bin/bash
source activate py_conda
source ~/IA_chupin/py_env/bin/activate

for i in {6..16} 
do 
    for j in {0..5} 
    do 
        for k in {0..51}
        do 
            if [$i -lt 10 ]
            then
                sbatch -o "./log_meso/log_propre_gpu" single_run_gpu.sh $i $j $k 12;
            else
                sbatch -o "./log_meso/log_propre_cpu" single_run_cpu.sh $i $j $k 12;
            fi
            
            #./single_run.sh $i $j $k 12
            #echo "./single_run.sh $i $j $k 12"
        done
    done
done







