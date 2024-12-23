##### PICK ONE:
#------------
# BLAS_LIB = /fsx/lrbison/code/OpenBLAS/libopenblas.a
# BLAS_LDFLAGS = -pthread -fopenmp $(BLAS_LIB)
#------------
# BLAS_LIB = blas
# BLAS_LDFLAGS = -pthread -fopenmp -l$(BLAS_LIB)
#------------
# BLAS_LIB = openblas
# RPATH = /fsx/lrbison/code/OpenBLAS/
# BLAS_LDFLAGS = -pthread -fopenmp -l$(BLAS_LIB) -Wl,-rpath,$(RPATH) -L$(RPATH)
#------------
# BLAS_LIB = armpl
# RPATH = /fsx/spack/opt/spack/linux-ubuntu20.04-neoverse_v1/gcc-12.2.0/armpl-gcc-24.04-cgkexqg6vtcdc7k3di6lglrrpnbxej67/armpl_24.04_gcc/lib/
# BLAS_LDFLAGS = -pthread -fopenmp -l$(BLAS_LIB) -Wl,-rpath,$(RPATH) -L$(RPATH)
#------------
# BLAS_LIB = armpl_mp
# RPATH = /fsx/spack/opt/spack/linux-ubuntu20.04-neoverse_v1/gcc-12.2.0/armpl-gcc-24.04-cgkexqg6vtcdc7k3di6lglrrpnbxej67/armpl_24.04_gcc/lib/
# BLAS_LDFLAGS = -pthread -fopenmp -l$(BLAS_LIB) -Wl,-rpath,$(RPATH) -L$(RPATH)
#------------
### arm_compute: not a real BLAS, need to try something else...
# BLAS_LIB = arm_compute
# RPATH = /fsx/acl/arm_compute-v24.12-linux-aarch64-cpu-bin/lib/armv8a-neon
# BLAS_LDFLAGS = -pthread -fopenmp -l$(BLAS_LIB) -Wl,-rpath,$(RPATH) -L$(RPATH)
#------------
# BLAS_LIB = mkl_intel_lp64
# RPATH = /fsx/spack/opt/spack/linux-ubuntu20.04-icelake/gcc-13.3.0/intel-oneapi-mkl-2024.2.0-n342ly76imrkxwbclf7lf5woxbwaco6k/mkl/latest/lib/
# BLAS_LDFLAGS = -pthread -fopenmp -lmkl_intel_thread -lmkl_rt -lmkl_core -l$(BLAS_LIB) -Wl,-rpath,$(RPATH) -L$(RPATH)
##############

# For building OpenBLAS on Neoverse-v1:
# NUM_THREADS=256 CC=armclang FC=armflang make -j 32 target=neoverse-v1

# cflags won't really matter -- all the work is done in the library.
CFLAGS += -g -Og -Wall -fopenmp
CC ?= gcc

UTILS_PATH = ../utils
CFLAGS += -I$(UTILS_PATH) -DBLAS_LIB=$(notdir $(BLAS_LIB))
LDFLAGS += $(BLAS_LDFLAGS)

# %.o: %.c Makefile utils.h
# 	$(CC) $(CFLAGS) -c -o $@ $^

BIN_LIST := gemm_test

gemm_test: gemm_test.o $(UTILS_PATH)/json-builder.o
	$(CC) -o $@ $(CFLAGS) $^ $(LDFLAGS)

all: $(BIN_LIST)

$(BIN_LIST): $(UTILS_PATH)/utils.h

clean:
	rm -f *.o $(BIN_LIST) $(UTILS_PATH)/*.o