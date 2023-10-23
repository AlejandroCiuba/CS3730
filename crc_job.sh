#!/usr/bin/env bash

# Slurm script for running finetune.py on the CRC cluster
# Alejandro Ciuba, alc307@pitt.edu

############## SBATCH HEADER BEGIN ##############
#SBATCH --job-name=cs3730
#SBATCH --output=output/%x-%A.out
#SBATCH --mail-user=alc307@pitt.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4-00:00:00
#SBATCH --qos=short
############## SBATCH HEADER END ##############

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
				   -d opus_books opus_wikipedia \
				   -s train \
				   -sl en \
				   -tl es \
				   -ts 0.3 \
				   -t "English to Spanish" \
				   -me sacrebleu \
				   -mk score \
				   -f 1 \
				   -l 4e-5 \
				   -e 1 \
				   -b 8 \
				   -sa 0.25 \
				   -x 25 \
				   -o models/baseline \
                   -lo logs

echo "DONE"

# Run CRC job stats script if it exists
command -v crc-job-stats &> /dev/null && command crc-job-stats
