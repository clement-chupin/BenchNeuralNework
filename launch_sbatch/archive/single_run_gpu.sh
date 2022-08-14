#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=ia_chupin
#SBATCH --time=1-00:00:00
#SBATCH --output=../log_meso/log_propre_gpu
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G

echo start 
echo $0
python3.8 ../main.py --mode manual --env $1 --policy $2 --feature $3 --index $4 --compute auto

echo end







