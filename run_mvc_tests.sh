#!/bin/bash
# run full mvc pipeline: training and evaluation
# uses author's pre-generated test data in data/test_mvc_maxcut/
#
# before running, check these parameters in the scripts:
#
# s2v training (code/s2v_mvc/run_nstep_dqn.sh):
#   max_iter      training iterations (default 1000000)
#   dev_id        gpu card id
#
# s2v eval (code/s2v_mvc/run_eval.sh):
#   dev_id        gpu card id
#
# realworld s2v training (code/realworld_s2v_mvc/run_nstep_dqn.sh):
#   max_iter      training iterations (default 1000000)
#   dev_id        gpu card id
#
# realworld s2v eval (code/realworld_s2v_mvc/eval.sh):
#   dev_id        gpu card id

set -e

source ~/venvs/math195/bin/activate

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
# requires data/realworld_data/memetracker/InfoNet5000Q1000NEXP.txt

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
