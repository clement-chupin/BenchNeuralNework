source activate py_conda
source ~/IA_chupin/py_env/bin/activate


for i in 16
do 
    for j in 2
    do 
        for k in 167 168 169
        do 
            for l in 0
            do 
                sbatch cpu_run.sh $i $j $k $l 11011;
                # sleep 0.1
            done
        done
    done
done