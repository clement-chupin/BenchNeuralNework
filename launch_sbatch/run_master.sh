#!/bin/bash
source activate py_conda
source ~/IA_chupin/py_env/bin/activate







for i in 0 1 6 7 8 9
do 
    for j in {0..5}
    do 
        for k in 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182
        do 
            for l in 0 1 2 
            do 
                sbatch cpu_run.sh $i $j $k $l 11011;
                # sleep 0.1
            done
        done
    done
done



