#!/bin/bash
source activate py_conda
source ~/IA_chupin/py_env/bin/activate





# for i in {0..16}
# do 
#     for j in {0..5}
#     do 
#         for k in 0
#         do 
#             for l in 0
#             do 
#                 sbatch cpu_run.sh $i $j $k $l 11011;
#                 # sleep 0.1
#             done
#         done
#     done
# done



#p3 e8 f0 v0
#p3 e8 f2 v1
#p3 e8 f172 v3
#p3 e8 f168 v0

#p2 e16 f0 v0
#p2 e16 f1 v2
#p2 e16 f167 v0

#p3 e15 f0 v0
#p3 e15 f2 v1
#p3 e15 f168 v0



for i in 0 1 7 8 15
do 
    for j in 0 3 4 5
    do 
        for k in 173 174
        do 
            for l in 0
            do 
                sbatch cpu_run.sh $i $j $k $l 11011;
            done
        done
    done
done


for i in 10 11 16
do 
    for j in 0 2 3
    do 
        for k in 173 174
        do 
            for l in 0
            do 
                sbatch cpu_run.sh $i $j $k $l 11011;
            done
        done
    done
done



# for index in 11011 11012 11013 11014 11015
# do 

#     for i in 1 7 8 15
#     do 
#         for j in 0 1 3 4 5
#         do 
#             for k in 2
#             do 
#                 for l in 3 4
#                 do 
#                         sbatch cpu_run.sh $i $j $k $l $index;
#                         # sleep 0.1
#                 done
#             done
#         done
#     done
#     for i in 10 11 16
#     do 
#         for j in 0 2 3 
#         do 
#             for k in 2
#             do 
#                 for l in 3 4
#                 do 
#                         sbatch cpu_run.sh $i $j $k $l $index;
#                         # sleep 0.1
#                 done
#             done
#         done
#     done

    # for i in 10 11 16
    # do 
    #     for j in 0 2 3 
    #     do 
    #         for k in 169
    #         do 
    #             for l in 0
    #             do 
    #                     sbatch cpu_run.sh $i $j $k $l $index;
    #                     # sleep 0.1
    #             done
    #         done
    #     done
    # done

    # sbatch cpu_run.sh 8 3 0 0 $index;
    # sbatch cpu_run.sh 8 3 2 1 $index;
    # sbatch cpu_run.sh 8 3 172 3 $index;
    # sbatch cpu_run.sh 8 3 168 0 $index;

    # sbatch cpu_run.sh 15 3 0 0 $index;
    # sbatch cpu_run.sh 15 3 2 1 $index;
    # sbatch cpu_run.sh 15 3 168 0 $index;

    # sbatch cpu_run.sh 16 2 0 0 $index;
    #sbatch cpu_run.sh 16 2 1 1 $index;
    # sbatch cpu_run.sh 16 2 167 0 $index;

# done
# sbatch cpu_run.sh 16 3 0 0 11011;
# sbatch cpu_run.sh 16 0 0 0 11011;

# sbatch cpu_run.sh 11 3 0 0 11011;
# sbatch cpu_run.sh 11 0 0 0 11011;