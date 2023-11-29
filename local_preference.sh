#!/usr/bin/env bash

# Run finetune.py on a local (LINUX) machine

echo "RUN: `date`"

version=`python finetune_preference.py --version`

echo "RUNNING $version SCRIPT"

python finetune_preference.py -m google/mt5-small \
							  -d datasets/opus_test \
							  -s train \
							  -sl en \
							  -lc 1 \
							  -tl es \
							  -ts 0.3 \
							  -op 0 \
							  -tb 128 \
							  -t "English to Spanish" \
							  -me sacrebleu \
							  -mk score \
							  -sk 1 \
							  -f 1 \
							  -l 4e-5 \
							  -e 1 \
							  -b 1 \
							  -sa 0.5 \
							  -x 5 \
							  -o models/test_pref1 \
							  -lo logs

echo "DONE"
