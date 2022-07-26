#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2G

#SBATCH --job-name=ia_chupin
#SBATCH --time=23:00:00

#SBATCH --output=log_output.txt



hostname
sleep 10
echo Début du job.
python test_gpu.py
echo Fin du job.
