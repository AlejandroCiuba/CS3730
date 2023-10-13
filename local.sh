#!/usr/bin/env bash

# Run finetune.py on a local (LINUX) machine

echo "RUN: `date`"

version=`python finetune.py --version`

echo "RUNNING $version SCRIPT"

python finetune.py -m google/mt5-small \
                   -f 1 \
                   -e 5

echo "DONE"
