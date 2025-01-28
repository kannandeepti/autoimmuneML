#!/bin/bash

#SBATCH -p download
#SBATCH -o output/data_download.out-%j

source /etc/profile
module load anaconda/Python-ML-2024b

eval "$(conda shell.bash hook)"
source activate autoimmuneML_env
export PATH=/home/gridsan/dkannan/.conda/envs/autoimmuneML_env/bin:$PATH 
echo $PATH 

crunch setup --size default broad-1 test --token ZMlbw4FswTxIf3pEpbA8YhsOKSjFLDhTo5TbDoOIY1LjWR5MYmWK52OqUSAaQdge