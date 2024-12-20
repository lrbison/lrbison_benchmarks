CFLAGS += -fopenmp -g -Og
#CFLAGS += -DDEBUG=1
CC = mpicc

# %.o: %.c Makefile utils.h
# 	$(CC) $(CFLAGS) -c -o $@ $^

BIN_LIST := simple_cas bench_isend bench_pipeline

bench_pipeline: bench_pipeline.o json-builder.o

all: $(BIN_LIST)

$(BIN_LIST): utils.h

clean:
	rm -f *.o $(BIN_LIST)