#!/bin/bash
source activate py_conda
source ~/IA_chupin/py_env/bin/activate




for i in {0..16} 
do 
    for j in {0..5} 
    do 
        for k in 9 10 11
        do 
            for l in {0..2}
            do 
                sbatch single_run_cpu_experiment.sh $i $j $k $l 2000;
                sleep 0.1
            done
        done
    done
done



