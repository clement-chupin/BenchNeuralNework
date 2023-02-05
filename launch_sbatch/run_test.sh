source activate py_conda
source ~/IA_chupin/py_env/bin/activate


sbatch cpu_run.sh 8 3 0 0 11;
sbatch cpu_run.sh 15 3 0 0 11;
sbatch cpu_run.sh 16 2 0 0 11;