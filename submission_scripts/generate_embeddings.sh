#!/bin/bash

#SBATCH -o generate_embeddings_518.log-%A-%a
#SBATCH -a 1-1
#SBATCH -c 20
#SBATCH --gres=gpu:volta:1

source /etc/profile
module load anaconda/Python-ML-2024b
module load cuda/12.6
module load nccl/2.23.4-cuda12.6

eval "$(conda shell.bash hook)"
source activate autoimmuneML_env

python -u generate_embeddings.py 518