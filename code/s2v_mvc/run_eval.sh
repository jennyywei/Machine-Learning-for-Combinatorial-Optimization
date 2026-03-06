#!/bin/bash

set -euo pipefail

g_type=${G_TYPE:-barabasi_albert}

result_root=${SYNTHETIC_MVC_RESULT_ROOT:-results/dqn-$g_type}

# max belief propagation iteration
max_bp_iter=5

# embedding size
embed_dim=64

# gpu card id
dev_id=${DEV_ID:-3}

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

num_env=10
mem_size=500000

max_iter=100000

# folder to save the trained model
save_dir=$result_root/embed-$embed_dim-nbp-$max_bp_iter-rh-$reg_hidden

data_dir=${SYNTHETIC_MVC_TEST_DATA_DIR:-../../data/test_mvc_maxcut}

test_ranges=${TEST_RANGES:-}
test_ranges=${test_ranges//,/ }

model_args=()
if [ -n "${MODEL_RANGE:-}" ]; then
    model_min_n=${MODEL_RANGE%-*}
    model_max_n=${MODEL_RANGE#*-}
    model_args=(-model_min_n "$model_min_n" -model_max_n "$model_max_n")
fi

for data_test in "$data_dir"/gtype-${g_type}-nrange-*-n_graph-1000-*.pkl; do
    [ -f "$data_test" ] || continue

    # extract nrange from filename (e.g. "15-20")
    range=$(echo "$data_test" | sed 's/.*nrange-\([0-9]*-[0-9]*\)-.*/\1/')
    min_n=${range%-*}
    max_n=${range#*-}

    if [ -n "$test_ranges" ]; then
        case " $test_ranges " in
            *" $range "*) ;;
            *) continue ;;
        esac
    fi

    test_name=$(basename "$data_test")
    if [ -n "${MODEL_RANGE:-}" ]; then
        result_file="$save_dir/test-${test_name}-gnn-train-${model_min_n}-${model_max_n}-test-${min_n}-${max_n}.csv"
    else
        result_file="$save_dir/test-${test_name}-gnn-${min_n}-${max_n}.csv"
    fi
    if [ -f "$result_file" ]; then
        echo "skipping completed eval for ${test_name}"
        continue
    fi

    if [ -n "${MODEL_RANGE:-}" ]; then
        echo "=== $g_type train=$MODEL_RANGE test=$range ==="
    else
        echo "=== $g_type nrange=$range ==="
    fi
    python evaluate.py \
        -n_step $n_step \
        -dev_id $dev_id \
        -data_test "$data_test" \
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
        "${model_args[@]}"
done
