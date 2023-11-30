#!/usr/bin/env bash

# Run scores.py on a local (LINUX) machine

echo "RUN: `date`"

version=`python3 merge.py --version`

echo "RUNNING $version SCRIPT"

python3 merge.py -d datasets/opus_flan_test datasets/opus_nllb_test \
                 -s train test valid \
                 -o datasets \
                 -l logs

echo "DONE"
