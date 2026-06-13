#!/bin/bash

python benchmark.py \
	--image images/broadway_tower_1428x968.jpg \
	--seams     10 \
	--runs      2  \
	--warmup    0  \
	--progress     \
	--skip-cpu 
