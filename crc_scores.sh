#!/usr/bin/env bash

# Slurm script for running scores.py on the CRC cluster
# Alejandro Ciuba, alc307@pitt.edu

############## SBATCH HEADER BEGIN ##############
#SBATCH --job-name=cs3730-scores
#SBATCH --output=output/%x-%a-%A.out
#SBATCH --mail-user=alc307@pitt.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --constraint=amd
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --array=0-1
#SBATCH --time=4-10:00:00
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

echo "RUNNING $version SCRIPT ON $SLURM_ARRAY_TASK_ID"

if [ "${SLURM_ARRAY_TASK_ID}" = "0" ]; then

    python scores.py -m facebook/nllb-200-distilled-600M \
                     -f 1 \
                     -tc spa_Latn \
                     -ts "Translate from English to Spanish" \
                     -d datasets/ix_datasets/opus \
                     -lc 1 \
                     -s train \
                     -sl en \
                     -tl es \
                     -op 0 \
                     -tb 128 \
                     -b 32 \
                     -me sacrebleu \
                     -mk score \
                     -o datasets/ix_datasets/opus_nllb \
                     -lo logs \
                     -ln $SLURM_ARRAY_TASK_ID

elif [ "${SLURM_ARRAY_TASK_ID}" = "1" ]; then

        python scores.py -m google/flan-t5-large \
                         -f 1 \
                         -tc spa_Latn \
                         -ts "Translate from English to Spanish" \
                         -d datasets/ix_datasets/opus \
                         -lc 1 \
                         -s train \
                         -sl en \
                         -tl es \
                         -op 0 \
                         -tb 128 \
                         -b 32 \
                         -me sacrebleu \
                         -mk score \
                         -o datasets/ix_datasets/opus_flan \
                         -lo logs \
                         -ln $SLURM_ARRAY_TASK_ID

fi

echo "DONE"

# Run CRC job stats script if it exists
command -v crc-job-stats &> /dev/null && command crc-job-stats
