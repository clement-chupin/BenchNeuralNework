$ srun --partition=gpu --pty bash
$ module load gcc/8.1.0 
$ module load python/3.7.1
$ virtualenv --python=python3.7 pytorch_env
$ cd pytorch_env
$ source bin/activate

pip3 install --user torch torchvision torchaudio




export PATH="$HOME/apps/conda/condabin:$PATH"


source activate py_conda
source activate py_conda