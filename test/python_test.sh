
#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=ia_chupin
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --output=log_output.txt
#SBATCH --mem-per-cpu=2G


echo start 
python python_test.py
python3.8 python_test.py
echo end