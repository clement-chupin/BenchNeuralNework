#!/bin/bash
source activate py_conda
source ~/IA_chupin/py_env/bin/activate











for i in {0..16} 
do 
    for j in {0..5}
    do 
        for k in 42
        do 
            for l in {0..4}
            do 
                sbatch single_run_cpu_experiment.sh $i $j $k $l 11011;
                # sleep 0.1
            done
        done
    done
done










# for i in {0..16} 
# do 
#     for j in {0..5}
#     do 
#         for k in 39 40
#         do 
#             for l in {0..3}
#             do 
#                 sbatch single_run_cpu_experiment.sh $i $j $k $l 11011;
#                 # sleep 0.1
#             done
#         done
#     done
# done




