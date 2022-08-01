#!/bin/bash
#SBATCH --partition=normal
#SBATCH --job-name=ia_chupin
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
# --output=./log_meso/log_output.txt
#SBATCH --mem-per-cpu=4G



echo start 
echo $0
python3.8 main.py --mode manual --env $1 --policy $2 --feature $3 --index $4

echo end







