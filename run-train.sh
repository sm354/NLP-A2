#!/bin/bash
data_dir="$1"
model_dir="$2"

python runtrain.py --data_dir "$data_dir" --model_dir "$model_dir"