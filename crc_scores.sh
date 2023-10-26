#!/usr/bin/env bash

# Slurm script for running scores.py on the CRC cluster
# Alejandro Ciuba, alc307@pitt.edu

############## SBATCH HEADER BEGIN ##############
#SBATCH --job-name=cs3730
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
#SBATCH --time=10:00:00
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

version=`python scores.py --version`

echo "RUNNING $version SCRIPT"

python scores.py -m facebook/nllb-200-distilled-600M \
                 -tc spa_Latn \
                 -d opus_books opus_wikipedia \
                 -s train \
                 -sl en \
                 -tl es \
                 -op 1 \
                 -tb 128 \
                 -b 32 \
                 -me sacrebleu \
                 -mk score \
                 -o datasets/opus_test \
                 -lo logs

echo "DONE"
