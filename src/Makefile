CFLAGS += -fopenmp -g -Og

all:

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $^

simple_cas: simple_cas.o
	$(CC) $(CFLAGS) -o $@ $^

bench_isend: bench_isend.o
	$(CC) $(CFLAGS) -o $@ $^

BIN_LIST += simple_cas bench_isend

all: $(BIN_LIST)

clean:
	rm -f *.o $(BIN_LIST)