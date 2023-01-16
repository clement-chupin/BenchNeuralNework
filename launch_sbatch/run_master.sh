#!/bin/bash
source activate py_conda
source ~/IA_chupin/py_env/bin/activate







for i in 0 1 6 7 8 9
do 
    for j in {0..5}
    do 
        for k in 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 162 166 164 165 166
        do 
            for l in 0
            do 
                sbatch cpu_run.sh $i $j $k $l 11011;
                # sleep 0.1
            done
        done
    done
done



