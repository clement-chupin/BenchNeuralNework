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


for $ind in 11011 #11012 11013 11014 11015
do 
    #sin vs trian 0 plot
    for i in 10 11 13 16
    do 
        for j in 2
        do 
            for k in 0
            do 
                for l in 0
                do 
                    sbatch cpu_run.sh $i $j $k $l $ind;
                    # sleep 0.1
                done
            done
        done
    done
    #trian robot 0 plot
    for i in 0 1 2 5 6 7 8 9 15
    do 
        for j in 0 3 4
        do 
            for k in 0
            do 
                for l in 0
                do 
                    sbatch cpu_run.sh $i $j $k $l $ind;
                    # sleep 0.1
                done
            done
        done
    done

    #trian robot method plot
    for i in 0 1 2 5 6 7 8 9 15
    do 
        for j in 0 3 4
        do 
            for k in 167 168 169
            do 
                for l in 0
                do 
                    sbatch cpu_run.sh $i $j $k $l $ind;
                    # sleep 0.1
                done
            done
        done
    done

    #trian sinus robot method plot
    for i in 0 1 2 5 6 7 8 9 15
    do 
        for j in 0 3 4
        do 
            for k in 171
            do 
                for l in 0 1 2
                do 
                    sbatch cpu_run.sh $i $j $k $l $ind;
                    # sleep 0.1
                done
            done
        done
    done
    for i in 0 1 2 5 6 7 8 9 15
    do 
        for j in 0 3 4
        do 
            for k in 172
            do 
                for l in 0 1 2 3 4
                do 
                    sbatch cpu_run.sh $i $j $k $l $ind;
                    # sleep 0.1
                done
            done
        done
    done


    for i in 10 11 13 16
    do 
        for j in 2
        do 
            for k in 0
            do 
                for l in 0
                do 
                    sbatch cpu_run.sh $i $j $k $l $ind;
                    # sleep 0.1
                done
            done
        done
    done


    sbatch cpu_run.sh 10 2 1 1 $ind;
    sbatch cpu_run.sh 10 2 171 4 $ind;
    sbatch cpu_run.sh 10 2 2 1 $ind;
    sbatch cpu_run.sh 10 2 172 4 $ind;

    sbatch cpu_run.sh 11 2 1 1 $ind;
    sbatch cpu_run.sh 11 2 171 2 $ind;
    sbatch cpu_run.sh 11 2 2 1 $ind;
    sbatch cpu_run.sh 11 2 172 2 $ind;

    sbatch cpu_run.sh 13 2 1 2 $ind;
    sbatch cpu_run.sh 13 2 171 3 $ind;
    sbatch cpu_run.sh 13 2 2 2 $ind;
    sbatch cpu_run.sh 13 2 172 3 $ind;

    sbatch cpu_run.sh 16 2 1 0 $ind;
    sbatch cpu_run.sh 16 2 171 3 $ind;
    sbatch cpu_run.sh 16 2 2 0 $ind;
    sbatch cpu_run.sh 16 2 172 3 $ind;





done