#!/bin/bash
#SBATCH --job-name=mvc-pipeline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-mvc-%j.out
#SBATCH --error=slurm-mvc-%j.err

set -euo pipefail

module load intel-oneapi-mkl/2024.2.2
module load intel-oneapi-tbb/2022.2.0
module load cuda
source ~/venvs/math195/bin/activate

cd $HOME/graph_comb_opt
export DEV_ID="${DEV_ID:-0}"
export SYNTHETIC_MVC_MAX_ITER="${SYNTHETIC_MVC_MAX_ITER:-100000}"
export REALWORLD_MVC_MAX_ITER="${REALWORLD_MVC_MAX_ITER:-100000}"
export MVC_PIPELINE_MODE="${MVC_PIPELINE_MODE:-full}"
export PAPER_G_TYPE="${PAPER_G_TYPE:-barabasi_albert}"
export PAPER_GRAPH_FAMILIES="${PAPER_GRAPH_FAMILIES:-barabasi_albert erdos_renyi}"
export PAPER_BUCKETS="${PAPER_BUCKETS:-15-20 40-50 50-100 100-200 400-500}"
export PAPER_SYNTHETIC_DATA_ROOT="${PAPER_SYNTHETIC_DATA_ROOT:-$HOME/graph_comb_opt/data/paper_synthetic_data}"
export PAPER_SYNTHETIC_OPT_ROOT="${PAPER_SYNTHETIC_OPT_ROOT:-$HOME/graph_comb_opt/data/paper_synthetic_opt}"
export GENERALIZATION_TRAIN_RANGES="${GENERALIZATION_TRAIN_RANGES:-50-100}"
export GENERALIZATION_TEST_RANGES="${GENERALIZATION_TEST_RANGES:-}"

paper_test_data_dir="$PAPER_SYNTHETIC_DATA_ROOT/test_mvc_maxcut"
paper_valid_data_dir="$PAPER_SYNTHETIC_DATA_ROOT/validation_mvc_maxcut"
paper_opt_dir="$PAPER_SYNTHETIC_OPT_ROOT/test_mvc"
paper_result_root="results/paper-dqn-$PAPER_G_TYPE"
paper_greedy_output_dir="results/paper_greedy"

if [ "$MVC_PIPELINE_MODE" = "paper_full" ]; then
    echo "============================================"
    echo "  full paper-style mvc reproduction"
    echo "============================================"
    echo "graph families: $PAPER_GRAPH_FAMILIES"
    echo "same-bucket buckets: $PAPER_BUCKETS"
    echo "generalization train ranges: $GENERALIZATION_TRAIN_RANGES"
    if [ -n "$GENERALIZATION_TEST_RANGES" ]; then
        echo "generalization test ranges: $GENERALIZATION_TEST_RANGES"
    else
        echo "generalization test ranges: all discovered paper test buckets"
    fi

    for g_type in $PAPER_GRAPH_FAMILIES; do
        family_result_root="results/paper-dqn-$g_type"

        echo ""
        echo "============================================"
        echo "  same-bucket synthetic mvc for $g_type"
        echo "============================================"

        for bucket in $PAPER_BUCKETS; do
            echo ""
            echo "============================================"
            echo "  train/eval synthetic mvc for $g_type bucket $bucket"
            echo "============================================"

            (
                cd code/s2v_mvc
                G_TYPE="$g_type" \
                TRAIN_RANGE="$bucket" \
                SYNTHETIC_MVC_TEST_DATA_DIR="$paper_test_data_dir" \
                SYNTHETIC_MVC_VALID_DATA_DIR="$paper_valid_data_dir" \
                SYNTHETIC_MVC_RESULT_ROOT="$family_result_root" \
                ./run_nstep_dqn.sh
                G_TYPE="$g_type" \
                MODEL_RANGE="$bucket" \
                TEST_RANGES="$bucket" \
                SYNTHETIC_MVC_TEST_DATA_DIR="$paper_test_data_dir" \
                SYNTHETIC_MVC_RESULT_ROOT="$family_result_root" \
                ./run_eval.sh
            )
        done

        (
            cd code/greedy_mvc
            G_TYPE="$g_type" \
            TEST_RANGES="$PAPER_BUCKETS" \
            SYNTHETIC_MVC_TEST_DATA_DIR="$paper_test_data_dir" \
            SYNTHETIC_MVC_OPT_DIR="$paper_opt_dir" \
            GREEDY_OUTPUT_DIR="$paper_greedy_output_dir" \
            ./run_greedy.sh
        )

        echo ""
        echo "============================================"
        echo "  synthetic mvc generalization for $g_type"
        echo "============================================"

        for train_bucket in $GENERALIZATION_TRAIN_RANGES; do
            echo ""
            echo "============================================"
            echo "  train on $g_type $train_bucket, evaluate generalization"
            echo "============================================"

            (
                cd code/s2v_mvc
                G_TYPE="$g_type" \
                TRAIN_RANGE="$train_bucket" \
                SYNTHETIC_MVC_TEST_DATA_DIR="$paper_test_data_dir" \
                SYNTHETIC_MVC_VALID_DATA_DIR="$paper_valid_data_dir" \
                SYNTHETIC_MVC_RESULT_ROOT="$family_result_root" \
                ./run_nstep_dqn.sh
                G_TYPE="$g_type" \
                MODEL_RANGE="$train_bucket" \
                TEST_RANGES="$GENERALIZATION_TEST_RANGES" \
                SYNTHETIC_MVC_TEST_DATA_DIR="$paper_test_data_dir" \
                SYNTHETIC_MVC_RESULT_ROOT="$family_result_root" \
                ./run_eval.sh
            )
        done

        (
            cd code/greedy_mvc
            G_TYPE="$g_type" \
            TEST_RANGES="$GENERALIZATION_TEST_RANGES" \
            SYNTHETIC_MVC_TEST_DATA_DIR="$paper_test_data_dir" \
            SYNTHETIC_MVC_OPT_DIR="$paper_opt_dir" \
            GREEDY_OUTPUT_DIR="$paper_greedy_output_dir" \
            ./run_greedy.sh
        )
    done

    echo ""
    echo "============================================"
    echo "  real-world mvc from paper data"
    echo "============================================"
    cd code/realworld_s2v_mvc
    ./run_nstep_dqn.sh

    echo ""
    echo "============================================"
    echo "  evaluate real-world s2v-dqn"
    echo "============================================"
    ./eval.sh

    echo ""
    echo "============================================"
    echo "  evaluate real-world greedy"
    echo "============================================"
    cd ../realworld_greedy_mvc
    ./run_greedy.sh

    echo ""
    echo "============================================"
    echo "  full paper-style mvc reproduction complete"
    echo "============================================"
    exit 0
fi

if [ "$MVC_PIPELINE_MODE" = "paper_synthetic" ]; then
    echo "============================================"
    echo "  paper-style synthetic mvc reproduction"
    echo "============================================"
    echo "graph family: $PAPER_G_TYPE"
    echo "buckets: $PAPER_BUCKETS"

    for bucket in $PAPER_BUCKETS; do
        echo ""
        echo "============================================"
        echo "  train/eval synthetic mvc for bucket $bucket"
        echo "============================================"

        (
            cd code/s2v_mvc
            G_TYPE="$PAPER_G_TYPE" \
            TRAIN_RANGE="$bucket" \
            SYNTHETIC_MVC_TEST_DATA_DIR="$paper_test_data_dir" \
            SYNTHETIC_MVC_VALID_DATA_DIR="$paper_valid_data_dir" \
            SYNTHETIC_MVC_RESULT_ROOT="$paper_result_root" \
            ./run_nstep_dqn.sh
            G_TYPE="$PAPER_G_TYPE" \
            MODEL_RANGE="$bucket" \
            TEST_RANGES="$bucket" \
            SYNTHETIC_MVC_TEST_DATA_DIR="$paper_test_data_dir" \
            SYNTHETIC_MVC_RESULT_ROOT="$paper_result_root" \
            ./run_eval.sh
        )

        (
            cd code/greedy_mvc
            G_TYPE="$PAPER_G_TYPE" \
            TEST_RANGES="$bucket" \
            SYNTHETIC_MVC_TEST_DATA_DIR="$paper_test_data_dir" \
            SYNTHETIC_MVC_OPT_DIR="$paper_opt_dir" \
            GREEDY_OUTPUT_DIR="$paper_greedy_output_dir" \
            ./run_greedy.sh
        )
    done

    echo ""
    echo "============================================"
    echo "  paper-style synthetic mvc complete"
    echo "============================================"
    exit 0
fi

if [ "$MVC_PIPELINE_MODE" = "paper_generalization" ]; then
    echo "============================================"
    echo "  paper-style synthetic mvc generalization"
    echo "============================================"
    echo "graph family: $PAPER_G_TYPE"
    echo "train ranges: $GENERALIZATION_TRAIN_RANGES"
    if [ -n "$GENERALIZATION_TEST_RANGES" ]; then
        echo "test ranges: $GENERALIZATION_TEST_RANGES"
    else
        echo "test ranges: all discovered paper test buckets"
    fi

    for train_bucket in $GENERALIZATION_TRAIN_RANGES; do
        echo ""
        echo "============================================"
        echo "  train on $train_bucket, evaluate generalization"
        echo "============================================"

        (
            cd code/s2v_mvc
            G_TYPE="$PAPER_G_TYPE" \
            TRAIN_RANGE="$train_bucket" \
            SYNTHETIC_MVC_TEST_DATA_DIR="$paper_test_data_dir" \
            SYNTHETIC_MVC_VALID_DATA_DIR="$paper_valid_data_dir" \
            SYNTHETIC_MVC_RESULT_ROOT="$paper_result_root" \
            ./run_nstep_dqn.sh
            G_TYPE="$PAPER_G_TYPE" \
            MODEL_RANGE="$train_bucket" \
            TEST_RANGES="$GENERALIZATION_TEST_RANGES" \
            SYNTHETIC_MVC_TEST_DATA_DIR="$paper_test_data_dir" \
            SYNTHETIC_MVC_RESULT_ROOT="$paper_result_root" \
            ./run_eval.sh
        )
    done

    (
        cd code/greedy_mvc
        G_TYPE="$PAPER_G_TYPE" \
        TEST_RANGES="$GENERALIZATION_TEST_RANGES" \
        SYNTHETIC_MVC_TEST_DATA_DIR="$paper_test_data_dir" \
        SYNTHETIC_MVC_OPT_DIR="$paper_opt_dir" \
        GREEDY_OUTPUT_DIR="$paper_greedy_output_dir" \
        ./run_greedy.sh
    )

    echo ""
    echo "============================================"
    echo "  paper-style synthetic mvc generalization complete"
    echo "============================================"
    exit 0
fi

# --- synthetic data ---

echo "============================================"
echo "  1. train s2v-dqn on synthetic data"
echo "============================================"
cd code/s2v_mvc
./run_nstep_dqn.sh

echo ""
echo "============================================"
echo "  2. evaluate s2v-dqn on synthetic data"
echo "============================================"
./run_eval.sh

echo ""
echo "============================================"
echo "  3. evaluate greedy on synthetic data"
echo "============================================"
cd ../greedy_mvc
./run_greedy.sh

# --- real-world data ---

echo ""
echo "============================================"
echo "  4. train s2v-dqn on real-world data"
echo "============================================"
cd ../realworld_s2v_mvc
./run_nstep_dqn.sh

echo ""
echo "============================================"
echo "  5. evaluate s2v-dqn on real-world data"
echo "============================================"
./eval.sh

echo ""
echo "============================================"
echo "  6. evaluate greedy on real-world data"
echo "============================================"
cd ../realworld_greedy_mvc
./run_greedy.sh

echo ""
echo "============================================"
echo "  all mvc tests complete"
echo "============================================"
