#!/usr/bin/env bash

# Run scores.py on a local (LINUX) machine

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
