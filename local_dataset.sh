#!/usr/bin/env bash

# Run scores.py on a local (LINUX) machine

echo "RUN: `date`"

version=`python3 dataset.py --version`

echo "RUNNING $version SCRIPT"

python3 dataset.py -d opus_books opus_wikipedia \
                   -s train \
                   -sl en \
                   -tl es \
                   -op 1 \
                   -b 128 \
                   -o datasets/opus \
                   -lo logs

echo "DONE"
