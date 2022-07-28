#!/bin/bash
source activate py_conda
source ~/IA_chupin/py_env/bin/activate

for i in {0..0} 
do 
    for j in {0..0} 
    do 
        for k in {0..5}
        do
            log_output = "./log_meso/log" + $i + "-" + $j + "-" + $k
            sbatch --output=$log_output single_run.sh $i $j $k 12
            #./single_run.sh $i $j $k 12
            #echo "./single_run.sh $i $j $k 12"
        done
    done
done







