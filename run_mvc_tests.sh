#!/bin/bash
# run full mvc pipeline: data generation, training, and evaluation
#
# before running, check these parameters in the scripts:
#
# data generation (code/data_generator/mvc/run_generate.sh):
#   -num_graph    number of test graphs (default 100, paper uses 1000)
#   -min_n/-max_n graph size range (default 15-20)
#
# s2v training (code/s2v_mvc/run_nstep_dqn.sh):
#   max_iter      training iterations (default 1000, paper uses 1000000)
#   dev_id        gpu card id
#
# s2v eval (code/s2v_mvc/run_eval.sh):
#   n_test        in evaluate.py (default 100, paper uses 1000)
#   dev_id        gpu card id
#
# realworld s2v training (code/realworld_s2v_mvc/run_nstep_dqn.sh):
#   max_iter      training iterations (default 1000000)
#   dev_id        gpu card id
#
# realworld s2v eval (code/realworld_s2v_mvc/eval.sh):
#   dev_id        gpu card id
#
# greedy scripts have no gpu or iteration params, just:
#   num_graphs    in code/greedy_mvc/run_greedy.sh
#   data_root     in code/realworld_greedy_mvc/run_greedy.sh

set -e

source ~/venvs/math195/bin/activate

# --- synthetic data ---

echo "============================================"
echo "  1. generate synthetic test graphs"
echo "============================================"
cd code/data_generator/mvc
./run_generate.sh

echo ""
echo "============================================"
echo "  2. train s2v-dqn on synthetic data"
echo "============================================"
cd ../../s2v_mvc
./run_nstep_dqn.sh

echo ""
echo "============================================"
echo "  3. evaluate s2v-dqn on synthetic data"
echo "============================================"
./run_eval.sh

echo ""
echo "============================================"
echo "  4. evaluate greedy on synthetic data"
echo "============================================"
cd ../greedy_mvc
./run_greedy.sh

# --- real-world data ---
# requires data/memetracker/InfoNet5000Q1000NEXP.txt
# download from the authors' dropbox (see README)

echo ""
echo "============================================"
echo "  5. train s2v-dqn on real-world data"
echo "============================================"
cd ../realworld_s2v_mvc
./run_nstep_dqn.sh

echo ""
echo "============================================"
echo "  6. evaluate s2v-dqn on real-world data"
echo "============================================"
./eval.sh

echo ""
echo "============================================"
echo "  7. evaluate greedy on real-world data"
echo "============================================"
cd ../realworld_greedy_mvc
./run_greedy.sh

echo ""
echo "============================================"
echo "  all mvc tests complete"
echo "============================================"
