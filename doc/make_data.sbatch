#!/bin/bash
#SBATCH --exclusive

echo "Run started from `hostname` at `date`"



for OMPI in ompi4 ompi5; do
module is-loaded ompi_main && module unload ompi_main
module is-loaded libfabric-profile && module unload libfabric-profile
module list

if [ $OMPI == "ompi5" ]; then
    module load ompi_main/develop
fi
module load libfabric-profile/develop

make clean
CC=mpicc make
sleep 1;

for NAVG in 1 100; do
for MSG in 8 32768; do
for TEST in gather scatter; do
for PML in ob1smcuda ob1 cm; do
# for PML in ob1 cm ob1smcuda; do
TAG="test_${TEST}_nodes${SLURM_JOB_NUM_NODES}_ppn${SLURM_NTASKS_PER_NODE}_avg${NAVG}_msg${MSG}_${OMPI}_${PML}"
if [ $TEST == "gather" ]; then IO_OPT=-i; else IO_OPT=-o; fi
if [ $PML == "ob1" ]; then
    PML_OPT="--mca pml ob1 --mca btl self,vader,ofi";
elif [ $PML == "ob1smcuda" ]; then
    PML_OPT="--mca pml ob1 --mca btl self,smcuda,ofi";
else
    PML_OPT="--mca pml cm --mca mtl ofi";
fi

    echo "Doing $TAG"
    mpirun -- ./bench_isend $IO_OPT -n $NAVG -t 100 -w 100 -f ${TAG}_ii.dat -m $MSG -s -r
    mpirun -- ./bench_isend $IO_OPT -n $NAVG -t 100 -w 100 -f ${TAG}_bb.dat -m $MSG

    if [ $TEST == "gather" ]; then
        mpirun -- ./bench_isend $IO_OPT -n $NAVG -t 100 -w 100 -f ${TAG}_bi.dat -m $MSG -r
    fi
    if [ $TEST == "scatter" ]; then
        mpirun -- ./bench_isend $IO_OPT -n $NAVG -t 100 -w 100 -f ${TAG}_ib.dat -m $MSG -s
    fi
done
done
done
done
done

# srun --mpi=pmix --nodes 4 ./bench_isend -i  -n 1 -t 100 -w 1 -f gather_4node_is_ir.dat -m 128 -s -r
# srun --mpi=pmix --nodes 4 ./bench_isend -i  -n 1 -t 100 -w 1 -f gather_4node_bs_br.dat -m 128

echo "--FINISHED.  All tests complete at `date`."