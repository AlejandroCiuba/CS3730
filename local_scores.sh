#!/usr/bin/env bash

# Run scores.py on a local (LINUX) machine

echo "RUN: `date`"

version=`python3 scores.py --version`

echo "RUNNING $version SCRIPT"

python3 scores.py -m facebook/nllb-200-distilled-600M \
                  -tc spa_Latn \
                  -d datasets/opus_test \
                  -lc 1 \
                  -s train \
                  -sl en \
                  -tl es \
                  -op 1 \
                  -tb 128 \
                  -b 16 \
                  -me sacrebleu \
                  -mk score \
                  -o datasets/opus_nllb_test \
                  -lo logs

echo "DONE"
