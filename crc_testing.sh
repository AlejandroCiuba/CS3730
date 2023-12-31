#!/usr/bin/env bash

# Slurm script for running scores.py on the CRC cluster
# Alejandro Ciuba, alc307@pitt.edu

############## SBATCH HEADER BEGIN ##############
#SBATCH --job-name=cs3730-testing
#SBATCH --output=output/%x-%A.out
#SBATCH --mail-user=alc307@pitt.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --constraint=amd
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0-02:00:00
#SBATCH --qos=short
############## SBATCH HEADER END ##############

echo "RUN:" `date`

# Load necessary modules
module load gcc/8.2.0 python/anaconda3.10-2022.10

# Activate the conda environment
source activate cs3730

unset PYTHONHOME
unset PYTHONPATH

echo "RUN: `date`"

# version=`python scores.py --version`

echo "RUNNING testing.py SCRIPT"

python testing.py

echo "DONE"

# Run CRC job stats script if it exists
command -v crc-job-stats &> /dev/null && command crc-job-stats
