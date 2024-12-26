BLAS_LIB = mkl_intel_lp64
MKL_VER := 2024.2.0
INSTALL_ROOT := $(shell spack location --first -i intel-oneapi-mkl @ $(MKL_VER) )
RPATH = $(INSTALL_ROOT)/mkl/latest/lib/
BLAS_LDFLAGS = -pthread -fopenmp -lmkl_intel_thread -lmkl_rt -lmkl_core -l$(BLAS_LIB) -Wl,-rpath,$(RPATH) -L$(RPATH)
BLAS_TXT = $(notdir $(BLAS_LIB))-$(MKL_VER)
