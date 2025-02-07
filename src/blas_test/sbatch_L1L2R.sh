#!/bin/bash
#SBATCH --exclusive

set -euoo posix pipefail

if [[ "${2:-}" == "" ]]; then
    NAME=${SLURM_JOB_PARTITION:-""}
    NAME=${NAME:-graviton}
else
    NAME=$2
fi

CONFIG_LIST="$1"
# armpl_mp nvpl openblas
# mkl openblas

NUMA_NODE_ZERO_CPUS=$((1 + $(lscpu | awk -F- '/NUMA node0 CPU/ {print $2}')))
ALL_CPUS=$((1 + $(lscpu | awk -F- '/On-line CPU/ {print $3}')))

for CONFIG in $CONFIG_LIST; do
    make clean && make CONFIG=$CONFIG BIN_SUFFIX=$NAME
    echo $CONFIG build succeeds.
done
# exit 0
for CONFIG in $CONFIG_LIST; do
    make clean && make CONFIG=$CONFIG BIN_SUFFIX=$NAME

    mpirun --map-by core ./gemm_test${NAME} --cache-smashes lin:0,8,1024 -T 20 -r 1,$ALL_CPUS --tag ${NAME}-L1
    mpirun --map-by core ./gemm_test${NAME} --cache-smashes lin:0,128,32768 -T 20 -r 1,$ALL_CPUS --tag ${NAME}-L2
    mpirun --map-by core ./gemm_test${NAME} --cache-smashes 0,8,128,32768 -T 20 --tag ${NAME}-Ranks

    mpirun --map-by core ./gemm_test${NAME} --cache-smashes lin:0,8,1024 -T 2000 -r 1,$ALL_CPUS --tag ${NAME}-L1
    mpirun --map-by core ./gemm_test${NAME} --cache-smashes lin:0,128,32768 -T 2000 -r 1,$ALL_CPUS --tag ${NAME}-L2
    mpirun --map-by core ./gemm_test${NAME} --cache-smashes 0,8,128,32768 -T 2000 --tag ${NAME}-Ranks
done


echo "Complete."