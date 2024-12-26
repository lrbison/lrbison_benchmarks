#!/bin/bash
#SBATCH --exclusive

set -euoo posix pipefail

make clean && make CONFIG=armpl_mp
time ./gemm_test hpc7g

make clean && make CONFIG=nvpl
time ./gemm_test hpc7g

make clean && make CONFIG=openblas
time ./gemm_test hpc7g