
for i in 0 
do 
    for j in 2 3 4 5 
    do 
        for k in 7
        do 
            for l in {0..2}
            do 
                echo $i $j $k $l 
                python3.8 ../main.py --mode manual_all --env $i --policy $j --feature $k --feature_var $l --index 222 --compute auto
                sleep 1
            done
        done
    done
done

for i in 0 
do 
    for j in 0 2 3 4 5
    do 
        for k in 8
        do 
            for l in {0..2}
            do 
                echo $i $j $k $l 
                python3.8 ../main.py --mode manual_all --env $i --policy $j --feature $k --feature_var $l --index 222 --compute auto
                sleep 1
            done
        done
    done
done

for i in 1 2
do 
    for j in 2 3 4 5
    do 
        for k in 7 8
        do 
            for l in {0..2}
            do 
                echo $i $j $k $l 
                python3.8 ../main.py --mode manual_all --env $i --policy $j --feature $k --feature_var $l --index 222 --compute cpu
                sleep 1
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


