##### PICK ONE:
#------------
#------------
#------------
### arm_compute: not a real BLAS, need to try something else...
# BLAS_LIB = arm_compute
# RPATH = /fsx/acl/arm_compute-v24.12-linux-aarch64-cpu-bin/lib/armv8a-neon
# BLAS_LDFLAGS = -pthread -fopenmp -l$(BLAS_LIB) -Wl,-rpath,$(RPATH) -L$(RPATH)
#------------
#------------
##############
CONFIG ?= openblas
BIN_SUFFIX ?= ""

include blas_mk_shards/$(CONFIG).mk

# For building OpenBLAS on Neoverse-v1:
# NUM_THREADS=256 CC=armclang FC=armflang make -j 32 target=neoverse-v1

BLAS_TXT ?= $(notdir $(BLAS_LIB))

# cflags won't really matter -- all the work is done in the library.
CFLAGS += -g -Og -Wall -fopenmp -DDO_LOG_AX
CC ?= gcc

UTILS_PATH = ../utils
CFLAGS += -I$(UTILS_PATH) -DBLAS_LIB=$(BLAS_TXT)
LDFLAGS += $(BLAS_LDFLAGS)

# %.o: %.c Makefile utils.h
# 	$(CC) $(CFLAGS) -c -o $@ $^

BIN_LIST := gemm_test gemm_test$(BIN_SUFFIX)

gemm_test$(BIN_SUFFIX): gemm_test.o $(UTILS_PATH)/json-builder.o
	$(CC) -o $@ $(CFLAGS) $^ $(LDFLAGS)
	@echo "==> Built $@ with config from $(CONFIG): $(BLAS_TXT)"

all: $(BIN_LIST)

$(BIN_LIST): $(UTILS_PATH)/utils.h

clean:
	rm -f *.o $(BIN_LIST) $(UTILS_PATH)/*.o