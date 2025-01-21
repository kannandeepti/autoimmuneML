#!/bin/bash

#SBATCH -p download
#SBATCH -o output/data_download.out-%j

source /etc/profile
module load anaconda/Python-ML-2024b

eval "$(conda shell.bash hook)"
source activate autoimmuneML_env

crunch setup --size default broad-1 aecv1 --token FAXL5rNprhxMkvi92MuEW5raFWSWit9SxJ6SYeiGPi4w3qvnvtJw5ZsFbVldMK9H