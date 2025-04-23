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

NUMA_NODE_ZERO_CPUS=$((1 + $(lscpu | awk -F[-,] '/NUMA node0 CPU/ {print $2}')))
ALL_CPUS=$((1 + $(lscpu | awk -F[-,] '/On-line CPU/ {print $3}')))

for CONFIG in $CONFIG_LIST; do
    TEST_BIN=./gemm_test-${NAME}-${CONFIG}
    rm -f $TEST_BIN
    make clean && make CONFIG=$CONFIG BIN_SUFFIX=-${NAME}-${CONFIG}
    echo $CONFIG build succeeds.
    $TEST_BIN --help | grep $CONFIG
    echo $TEST_BIN reports correct library.
done

for CONFIG in $CONFIG_LIST; do
    TEST_BIN=./gemm_test-${NAME}-${CONFIG}
    # make clean && make CONFIG=$CONFIG BIN_SUFFIX=$NAME

    mpirun --map-by core ${TEST_BIN} --cache-smashes lin:0,8,1024 -T 20 -r 1,$ALL_CPUS --tag ${NAME}-L1
    mpirun --map-by core ${TEST_BIN} --cache-smashes lin:0,128,32768 -T 20 -r 1,$ALL_CPUS --tag ${NAME}-L2
    mpirun --map-by core ${TEST_BIN} --cache-smashes 0,8,128,32768 -T 20 --tag ${NAME}-Ranks

    mpirun --map-by core ${TEST_BIN} --cache-smashes lin:0,8,1024 -T 2000 -r 1,$ALL_CPUS --tag ${NAME}-L1
    mpirun --map-by core ${TEST_BIN} --cache-smashes lin:0,128,32768 -T 2000 -r 1,$ALL_CPUS --tag ${NAME}-L2
    mpirun --map-by core ${TEST_BIN} --cache-smashes 0,8,128,32768 -T 2000 --tag ${NAME}-Ranks
done


echo "Complete."