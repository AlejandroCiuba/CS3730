#!/usr/bin/env bash

# Run finetune.py on a local (LINUX) machine

echo "RUN: `date`"

version=`python finetune.py --version`

echo "RUNNING $version SCRIPT"

python finetune.py -m google/mt5-small \
				   -d opus_books opus_wikipedia \
				   -s train \
				   -sl en \
				   -tl es \
                   -ts 0.3 \
                   -op 1 \
				   -tb 128 \
				   -t "English to Spanish" \
				   -me sacrebleu \
				   -mk score \
				   -f 1 \
				   -l 4e-5 \
				   -e 1 \
				   -b 2 \
				   -sa 0.5 \
				   -x 5 \
				   -o models/test_run2 \
                   -lo logs

echo "DONE"
