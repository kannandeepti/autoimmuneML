#!/bin/bash

#SBATCH -o train_nn.log-%A-%a
#SBATCH -a 1-1
#SBATCH -c 20

source /etc/profile
module load anaconda/Python-ML-2024b

eval "$(conda shell.bash hook)"
source activate autoimmuneML_env

python -u nn_predictions.py