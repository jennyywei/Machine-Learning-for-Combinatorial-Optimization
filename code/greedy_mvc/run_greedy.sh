#!/bin/bash

g_type=barabasi_albert

data_test=../../data/mvc/gtype-$g_type-nrange-15-20-n_graph-100-p-0.00-m-4.pkl

num_graphs=100 # TODO: change this

# folder to save results
output_dir=results/greedy

python evaluate_greedy.py \
    -data_test $data_test \
    -num_graphs $num_graphs \
    -output_dir $output_dir \
    -save_csv 1
