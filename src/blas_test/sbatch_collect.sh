#!/bin/bash
#SBATCH --exclusive

set -euoo posix pipefail

if [[ "$2" == "" ]]; then
    NAME=${SLURM_JOB_PARTITION:-""}
    NAME=${NAME:-graviton}
else
    NAME=$2
fi

CONFIG_LIST="$1"
# armpl_mp nvpl openblas
# mkl openblas

NUMA_NODE_ZERO_CPUS=$(lscpu | awk '/NUMA node0 CPU/ {print $4}')
ALL_CPUS=$(lscpu | awk '/On-line CPU/ {print $4}')

for CONFIG in $CONFIG_LIST; do
    make clean && make CONFIG=$CONFIG BIN_SUFFIX=$NAME
    echo $CONFIG build succeeds.
done

if [[ $NUMA_NODE_ZERO_CPUS != $ALL_CPUS ]]; then
    echo "Starting tasks for numa 0 only: $NUMA_NODE_ZERO_CPUS"
    for CONFIG in $CONFIG_LIST; do
        make clean && make CONFIG=$CONFIG BIN_SUFFIX=$NAME
        time taskset -c ${NUMA_NODE_ZERO_CPUS} ./gemm_test ${NAME}-1socket
    done
fi

echo "Starting tasks for full-host cpus: $ALL_CPUS"
for CONFIG in $CONFIG_LIST; do
    make clean && make CONFIG=$CONFIG BIN_SUFFIX=$NAME
    time ./gemm_test ${NAME}
done

echo "Complete."