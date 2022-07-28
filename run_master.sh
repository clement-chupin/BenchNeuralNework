#!/bin/bash
source activate py_conda
source ~/IA_chupin/py_env/bin/activate

for i in {0..16}#env 
do 
    for j in {0..5}#policy 
    do 
        for k in {0..51}#feature 
        do
            #sbatch single_run.sh $i $j $k 12
            ./single_run.sh $i $j $k 12
        done
    done
done







