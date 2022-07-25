#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=ia_chupin_test
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --output=log_output.txt
#SBATCH --mem-per-cpu=2G


hostname
sleep 100
echo Fin du job.
