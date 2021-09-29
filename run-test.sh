#!/bin/bash
model_dir="$1"
input_file="$2"
output_file="$3"

python runtest.py --model_dir "$model_dir" --input_file "$input_file" --output_file "$output_file"