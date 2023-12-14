#!/usr/bin/env bash

# Slurm script for running dataset.py on the CRC cluster
# Alejandro Ciuba, alc307@pitt.edu

############## SBATCH HEADER BEGIN ##############
#SBATCH --job-name=cs3730-dataset
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
#SBATCH --time=5-00:00:00
#SBATCH --qos=short
############## SBATCH HEADER END ##############

# Load necessary modules
module load gcc/8.2.0 python/anaconda3.10-2022.10

# Activate the conda environment
source activate cs3730

unset PYTHONHOME
unset PYTHONPATH

echo "RUN: `date`"

version=`python dataset.py --version`

echo "RUNNING $version SCRIPT"

python dataset.py -d opus_books opus_wikipedia \
                  -s train \
                  -sl en \
                  -tl es \
                  -op 1 \
                  -b 128 \
                  -o datasets/opus \
                  -lo logs

echo "DONE"

# Run CRC job stats script if it exists
command -v crc-job-stats &> /dev/null && command crc-job-stats
