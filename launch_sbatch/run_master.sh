#!/bin/bash
source activate py_conda
source ~/IA_chupin/py_env/bin/activate







# for i in 0 1 6 7 8 9
# do 
#     for j in {0..5}
#     do 
#         for k in 112 113 114
#         do 
#             for l in 0 1 2 
#             do 
#                 sbatch cpu_run.sh $i $j $k $l 11011;
#                 # sleep 0.1
#             done
#         done
#     done
# done





# # for i in 0 1 6 7 8 9


for i in 0 1 6 7 8 9
do 
    for j in {0..5}
    do 
        for k in 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90
        do 
            for l in 0 1 2 
            do 
                sbatch cpu_run.sh $i $j $k $l 11011;
                # sleep 0.1
            done
        done
    done
done


for i in 0 1 6 7 8 9
do 
    for j in {0..5}
    do 
        for k in 106 107 108 
        do 
            for l in 0 1 2 
            do 
                sbatch cpu_run.sh $i $j $k $l 11011;
                # sleep 0.1
            done
        done
    done
done


for i in 0 1 6 7 8 9
do 
    for j in {0..5}
    do 
        for k in 117 118 119
        do 
            for l in 0 1 2 
            do 
                sbatch cpu_run.sh $i $j $k $l 11011;
                # sleep 0.1
            done
        done
    done
done

for i in 0 1 6 7 8 9
do 
    for j in {0..5}
    do 
        for k in 122 123 124 125 126
        do 
            for l in 0 1 2 
            do 
                sbatch cpu_run.sh $i $j $k $l 11011;
                # sleep 0.1
            done
        done
    done
done


for i in 0 1 6 7 8 9
do 
    for j in {0..5}
    do 
        for k in 129 130 131
        do 
            for l in 0 1 2 
            do 
                sbatch cpu_run.sh $i $j $k $l 11011;
                # sleep 0.1
            done
        done
    done
done


for i in 0 1 6 7 8 9
do 
    for j in {0..5}
    do 
        for k in 134 135 136 137 138
        do 
            for l in 0 1 2 
            do 
                sbatch cpu_run.sh $i $j $k $l 11011;
                # sleep 0.1
            done
        done
    done
done




