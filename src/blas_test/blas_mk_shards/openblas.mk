BLAS_LIB = openblas
RPATH = /fsx/lrbison/code/OpenBLAS/
BLAS_LDFLAGS = -pthread -fopenmp -l$(BLAS_LIB) -Wl,-rpath,$(RPATH) -L$(RPATH)