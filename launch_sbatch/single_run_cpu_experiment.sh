#!/bin/bash
#SBATCH --partition=normal
#SBATCH --job-name=ia_chupin
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --output=../log_meso/log_propre_cpu
#SBATCH --mem-per-cpu=8G



echo start 
echo $0
python3.8 ../main.py --mode manual_all --env $1 --policy $2 --feature $3 --feature_var $4 --index $5 --compute cpu

echo end






