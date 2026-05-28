#!/bin/bash

python benchmark.py \
	--image Broadway_tower_edit.jpg \
	--seams     10 \
	--runs      2  \
	--warmup    0  \
	--progress     \
	--skip-cpu 
