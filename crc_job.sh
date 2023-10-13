#!/usr/bin/env bash

# Slurm script for running finetune.py on the CRC cluster
# Alejandro Ciuba, alc307@pitt.edu

#SBATCH --job-name=cs3730
#SBATCH --output=crc_output/%x.out
#SBATCH --mail-type=FAIL
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --time=01:00:00
#SBATCH --qos=short

echo "RUN:" `date`

# Load necessary modules
module load gcc/8.2.0 python/anaconda3.10-2022.10

# Activate the conda environment
source activate cs3730

unset PYTHONHOME
unset PYTHONPATH

version=`python finetune.py --version`

echo "RUNNING $version SCRIPT"

python finetune.py -m google/mt5-small \
		   -f 1 \
		   -e 5

echo "DONE"

# Run CRC job stats script if it exists
command -v crc-job-stats &> /dev/null && command crc-job-stats
