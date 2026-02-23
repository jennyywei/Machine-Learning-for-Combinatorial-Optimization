#!/bin/bash
# Run: source setup_env.sh

module load cuda
module load python
module load intel-oneapi-mkl/2024.2.2
module load intel-oneapi-tbb/2022.2.0

export LD_LIBRARY_PATH=/software/spack/opt/spack/linux-x86_64_v3/cuda-12.9.0-jmleofbt4f2ctfltnzfkgly2quee5d6y/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

source ~/venvs/math195/bin/activate
