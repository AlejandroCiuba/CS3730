#!/usr/bin/env bash

# Run finetune.py on a local (LINUX) machine

echo "RUN: `date`"

version=`python finetune_preference.py --version`

echo "RUNNING $version SCRIPT"

python finetune_preference.py -m google/mt5-small \
							  -dmt datasets/opus_test \
							  -s train \
							  -lc 1 \
							  -sl en \
							  -tl es \
							  -op 1 \
							  -dpt datasets/opus_flan_test_opus_nllb_test \
							  -mts flan-t5-large nllb-200-distilled-600M \
							  -mtk flan-t5-large_score nllb-200-distilled-600M_score \
							  -ts 0.3 \
							  -tb 128 \
							  -g 1.0 \
							  -t "English to Spanish" \
							  -me sacrebleu \
							  -mk score \
							  -sk 1 \
							  -f 1 \
							  -l 4e-5 \
							  -e 2 \
							  -b 2 \
							  -tm 2 \
							  -sa 1 \
							  -x 100 \
							  -o models/preference-pretest \
							  -lo logs

echo "DONE"
