#!/bin/bash

set -euo pipefail

result_root=results/dqn-meme

# max belief propagation iteration
max_bp_iter=1

# embedding size
embed_dim=64

# gpu card id
dev_id=${DEV_ID:-2}

# max batch size for training/testing
batch_size=64

net_type=QNet

# set reg_hidden=0 to make a linear regression
reg_hidden=64

# learning rate
learning_rate=0.0001

# init weights with rand normal(0, w_scale)
w_scale=0.01

# nstep
n_step=5

min_n=5
max_n=300

num_env=10
mem_size=500000
prob_q=7
max_iter=${REALWORLD_MVC_MAX_ITER:-100000}

# folder to save the trained model
save_dir=$result_root/embed-$embed_dim-nbp-$max_bp_iter-rh-$reg_hidden-prob_q-$prob_q

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

lib_so=mvc_lib/build/dll/libmvc.so
if [ ! -f "$lib_so" ]; then
    echo "building real-world mvc shared library"
    if [ ! -f mvc_lib/Makefile ] && [ -f mvc_lib/Makefile.example ]; then
        cp mvc_lib/Makefile.example mvc_lib/Makefile
    fi
    (cd mvc_lib && make -j4)
fi

last_checkpoint=$(( ((max_iter - 1) / 300) * 300 ))
complete_model="$save_dir/iter_${last_checkpoint}.model"
if [ -f "$complete_model" ]; then
    echo "skipping completed real-world training"
    exit 0
fi

resume_args=()
log_mode=
latest_model=$(find "$save_dir" -maxdepth 1 -name "iter_*.model" | sort -V | tail -n 1)
if [ -n "$latest_model" ]; then
    echo "resuming from $latest_model"
    resume_args=(-load_model "$latest_model")
    log_mode=-a
fi

python main.py \
    -prob_q $prob_q \
    -n_step $n_step \
    -data_root ../../data/realworld_data/memetracker \
    -min_n $min_n \
    -max_n $max_n \
    -num_env $num_env \
    -dev_id $dev_id \
    -max_iter $max_iter \
    -mem_size $mem_size \
    -learning_rate $learning_rate \
    -max_bp_iter $max_bp_iter \
    -net_type $net_type \
    -max_iter $max_iter \
    -save_dir $save_dir \
    -embed_dim $embed_dim \
    -batch_size $batch_size \
    -reg_hidden $reg_hidden \
    -momentum 0.9 \
    -l2 0.00 \
    -w_scale $w_scale \
    "${resume_args[@]}" \
    2>&1 | tee $log_mode $save_dir/log-$min_n-${max_n}.txt
#    -load_model $save_dir/iter_5.model \
