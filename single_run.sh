#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=ia_chupin
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --output=log_output.txt
# --mem-per-cpu=1G


echo start 
echo $0
python3.8 main.py --mode manual --env $1 --policy $2 --feature $3 --index $4

echo end







