for i in 0 1 6 7 8 9
do 
    for j in {0..5}
    do 
        for k in 0
        do 
            for l in 0 1 2 
            do 
                sbatch cpu_run.sh $i $j $k $l 11011;
                # sleep 0.1
            done
        done
    done
done