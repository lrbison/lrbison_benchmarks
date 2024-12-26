BLAS_LIB = nvpl_blas_lp64_gomp
RPATH = /fsx/nvpl/nvpl-linux-sbsa-24.7/lib
BLAS_LDFLAGS = -pthread -fopenmp -lnvpl_blas_core -l$(BLAS_LIB) -Wl,-rpath,$(RPATH) -L$(RPATH)
