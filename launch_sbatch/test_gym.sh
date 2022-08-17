#!/bin/bash



#SBATCH --partition=normal
#SBATCH --job-name=ia_chupin
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --output=../log_meso/log_propre_test_gym
#SBATCH --mem-per-cpu=4G

source activate py_conda
source ~/IA_chupin/py_env/bin/activate



python3.8 test_gym.sh









