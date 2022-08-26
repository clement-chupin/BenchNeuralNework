#!/bin/bash
source activate py_conda
source ~/IA_chupin/py_env/bin/activate




for i in {0..16} 
do 
    for j in {0..5} 
    do 
        for k in 6 7 11 12
        do 
            for l in {0..3}
            do 
                sbatch single_run_cpu_experiment.sh $i $j $k $l 1088;
                sleep 0.1
            done
        done
    done
done



# for i in {9..9} 
# do 
#     for j in {0..5} 
#     do 
#         for k in {0..51}
#         do 
#             sbatch single_run_cpu.sh $i $j $k 99;
#             sleep 1
#         done
#     done
# done
# for i in {10..16} 
# do 
#     for j in {0..5} 
#     do 
#         for k in {0..51}
#         do 
#             if [$i -lt 10 ]
#             then
#                 sbatch single_run_gpu.sh $i $j $k 12;
#             else
#                 sbatch single_run_cpu.sh $i $j $k 12;
#             fi
#             sleep 1
#             #./single_run.sh $i $j $k 12
#             #echo "./single_run.sh $i $j $k 12"
#         done
#     done
# done

# for i in {0..9} 
# do 
#     for j in {0..5} 
#     do 
#         for k in {0..51}
#         do 
#             if [$i -lt 10 ]
#             then
#                 sbatch single_run_gpu.sh $i $j $k 12;
#             else
#                 sbatch single_run_cpu.sh $i $j $k 12;
#             fi
#             sleep 1
#             #./single_run.sh $i $j $k 12
#             #echo "./single_run.sh $i $j $k 12"
#         done
#     done
# done


