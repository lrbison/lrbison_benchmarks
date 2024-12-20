CFLAGS += -fopenmp -g -Og
#CFLAGS += -DDEBUG=1
CC = mpicc

UTILS_PATH = ../utils
CFLAGS += -I$(UTILS_PATH)

# %.o: %.c Makefile utils.h
# 	$(CC) $(CFLAGS) -c -o $@ $^

BIN_LIST := simple_cas bench_isend bench_pipeline

bench_pipeline: bench_pipeline.o $(UTILS_PATH)/json-builder.o

all: $(BIN_LIST)

$(BIN_LIST): $(UTILS_PATH)/utils.h

clean:
	rm -f *.o $(BIN_LIST) $(UTILS_PATH)/*.o