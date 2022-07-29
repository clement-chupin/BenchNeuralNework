#!/bin/bash
source activate py_conda
source ~/IA_chupin/py_env/bin/activate

for i in {0..5} 
do 
    for j in {1..5} 
    do 
        for k in {0..51}
        do
            sbatch -o ("./log_meso/lo_"+$1+"_"+$2+"_"+$3) single_run.sh $i $j $k 12
            #./single_run.sh $i $j $k 12
            #echo "./single_run.sh $i $j $k 12"
        done
    done
done







