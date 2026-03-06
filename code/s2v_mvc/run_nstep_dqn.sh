#!/bin/bash

set -euo pipefail

g_type=${G_TYPE:-barabasi_albert}

result_root=${SYNTHETIC_MVC_RESULT_ROOT:-results/dqn-$g_type}

# max belief propagation iteration
max_bp_iter=5

# embedding size
embed_dim=64

# gpu card id
dev_id=${DEV_ID:-0}

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
n_step=2

num_env=1
mem_size=500000

max_iter=${SYNTHETIC_MVC_MAX_ITER:-100000}

data_dir=${SYNTHETIC_MVC_TEST_DATA_DIR:-../../data/test_mvc_maxcut}
valid_data_dir=${SYNTHETIC_MVC_VALID_DATA_DIR:-}

# folder to save the trained model
save_dir=$result_root/embed-$embed_dim-nbp-$max_bp_iter-rh-$reg_hidden

if [ ! -e $save_dir ];
then
    mkdir -p $save_dir
fi

mapfile -t ranges < <(
    find "$data_dir" -maxdepth 1 -type f -name "gtype-${g_type}-nrange-*-n_graph-1000-*.pkl" \
        | sed "s#^.*/gtype-${g_type}-nrange-##" \
        | sed 's/-n_graph-1000-.*$//' \
        | sort -V
)

if [ -n "${TRAIN_RANGE:-}" ]; then
    ranges=("$TRAIN_RANGE")
fi

for range in "${ranges[@]}"; do
    min_n=${range%-*}
    max_n=${range#*-}
    done_file="$save_dir/nrange_${min_n}_${max_n}.done"
    last_checkpoint=$(( ((max_iter - 1) / 300) * 300 ))
    complete_model="$save_dir/nrange_${min_n}_${max_n}_iter_${last_checkpoint}.model"

    if [ -f "$complete_model" ]; then
        echo "skipping completed nrange=$range"
        continue
    fi

    if [ -f "$done_file" ]; then
        echo "ignoring stale completion marker for nrange=$range"
    fi

    echo "=== training $g_type nrange=$range ==="

    resume_args=()
    valid_args=()
    log_mode=
    latest_model=$(find "$save_dir" -maxdepth 1 -name "nrange_${min_n}_${max_n}_iter_*.model" | sort -V | tail -n 1)
    if [ -n "$latest_model" ]; then
        echo "resuming from $latest_model"
        resume_args=(-load_model "$latest_model")
        log_mode=-a
    fi

    if [ -n "$valid_data_dir" ]; then
        data_valid=$(find "$valid_data_dir" -maxdepth 1 -type f -name "gtype-${g_type}-nrange-${range}-n_graph-100-*.pkl" | sort -V | head -n 1)
        if [ -n "$data_valid" ]; then
            valid_args=(-data_valid "$data_valid")
        else
            echo "warning: no validation file found for $g_type nrange=$range in $valid_data_dir"
        fi
    fi

    python main.py \
        -n_step $n_step \
        -dev_id $dev_id \
        -min_n $min_n \
        -max_n $max_n \
        -num_env $num_env \
        -max_iter $max_iter \
        -mem_size $mem_size \
        -g_type $g_type \
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
        "${valid_args[@]}" \
        "${resume_args[@]}" \
        2>&1 | tee $log_mode $save_dir/log-$min_n-${max_n}.txt

    touch "$done_file"
done
