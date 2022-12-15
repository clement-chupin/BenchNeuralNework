#!/bin/bash
source activate py_conda
source ~/IA_chupin/py_env/bin/activate







for i in 0 1 6 7 8 9
do 
    for j in {0..5}
    do 
        for k in 112 113 114
        do 
            for l in 0 1 2 
            do 
                sbatch cpu_run.sh $i $j $k $l 11011;
                # sleep 0.1
            done
        done
    done
done





# # for i in 0 1 6 7 8 9


# for i in 0 1 6 7 8 9
# do 
#     for j in {0..5}
#     do 
#         for k in 115 116 #117 118 119
#         do 
#             for l in 0 1 2 
#             do 
#                 sbatch cpu_run.sh $i $j $k $l 11011;
#                 # sleep 0.1
#             done
#         done
#     done
# done

# for i in 0 1 6 7 8 9
# do 
#     for j in {0..5}
#     do 
#         for k in 120 121 #122 123 124 125 126
#         do 
#             for l in 0 1 2 
#             do 
#                 sbatch cpu_run.sh $i $j $k $l 11011;
#                 # sleep 0.1
#             done
#         done
#     done
# done


# for i in 0 1 6 7 8 9
# do 
#     for j in {0..5}
#     do 
#         for k in 127 128 #129 130 131
#         do 
#             for l in 0 1 2 
#             do 
#                 sbatch cpu_run.sh $i $j $k $l 11011;
#                 # sleep 0.1
#             done
#         done
#     done
# done


# for i in 0 1 6 7 8 9
# do 
#     for j in {0..5}
#     do 
#         for k in 132 133 #134 135 136 137 138
#         do 
#             for l in 0 1 2 
#             do 
#                 sbatch cpu_run.sh $i $j $k $l 11011;
#                 # sleep 0.1
#             done
#         done
#     done
# done




