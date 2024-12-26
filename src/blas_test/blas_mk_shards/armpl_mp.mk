BLAS_LIB = armpl_mp
ARMPL_VER := 24.10
INSTALL_ROOT := $(shell spack location --first -i armpl-gcc @ $(ARMPL_VER) )
RPATH = $(wildcard $(INSTALL_ROOT)/armpl_*_gcc/lib/)
BLAS_LDFLAGS = -pthread -fopenmp -l$(BLAS_LIB) -Wl,-rpath,$(RPATH) -L$(RPATH)
BLAS_TXT = $(notdir $(BLAS_LIB))-$(ARMPL_VER)