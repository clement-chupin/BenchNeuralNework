#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=2G

#SBATCH --job-name=ia_chupin
#SBATCH --time=23:00:00

#SBATCH --output=log_output.txt



hostname
sleep 10
echo Lancement de la tache :
python3.8 test_gym_ppo.py
echo Fin de la tache
