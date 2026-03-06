#!/bin/bash

set -euo pipefail

data_root=../../data/realworld_data/memetracker

# folder to save results
output_dir=results/greedy

python evaluate_greedy.py \
    -data_root $data_root \
    -output_dir $output_dir
