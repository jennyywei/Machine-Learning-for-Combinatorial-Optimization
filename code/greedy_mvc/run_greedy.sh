#!/bin/bash

set -euo pipefail

# folder to save results
output_dir=${GREEDY_OUTPUT_DIR:-results/greedy}

data_dir=${SYNTHETIC_MVC_TEST_DATA_DIR:-../../data/test_mvc_maxcut}
opt_dir=${SYNTHETIC_MVC_OPT_DIR:-../../data/test_mvc}
test_ranges=${TEST_RANGES:-}
test_ranges=${test_ranges//,/ }

if [ -n "${G_TYPES:-}" ]; then
    graph_families=$G_TYPES
elif [ -n "${G_TYPE:-}" ]; then
    graph_families=$G_TYPE
else
    graph_families="barabasi_albert erdos_renyi"
fi

for g_type in $graph_families; do
    # find all test data files for this graph type
    for data_test in "$data_dir"/gtype-${g_type}-nrange-*-n_graph-1000-*.pkl; do
        [ -f "$data_test" ] || continue

        test_basename=$(basename "$data_test" .pkl)
        csv_out="${output_dir}/${test_basename}-greedy.csv"
        if [ -f "$csv_out" ]; then
            echo "skipping completed greedy eval for ${test_basename}"
            continue
        fi

        # extract nrange from filename (e.g. "15-20")
        range=$(echo "$data_test" | sed 's/.*nrange-\([0-9]*-[0-9]*\)-.*/\1/')

        if [ -n "$test_ranges" ]; then
            case " $test_ranges " in
                *" $range "*) ;;
                *) continue ;;
            esac
        fi

        opt_sol=${opt_dir}/gtype-${g_type}-nrange-${range}-n_graph-1000-p-0-tlim-3600-opt.pkl

        echo "=== $g_type nrange=$range ==="
        python evaluate_greedy.py \
            -data_test "$data_test" \
            -num_graphs 1000 \
            -output_dir $output_dir \
            -opt_sol "$opt_sol" \
            -save_csv 1
    done
done
