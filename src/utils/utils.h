#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <string.h>

#include <unistd.h>
extern char *optarg;
extern int optind, opterr, optopt;

#ifdef __aarch64__
#define asm_instruction_barrier asm volatile ("ISB")
#else
#define asm_instruction_barrier do {} while(0)
#endif

#define MIN(a,b) ((a < b) ? a : b)
#define MAX(a,b) ((a > b) ? a : b)

// call this function to start a nanosecond-resolution timer
void static inline timer_start(struct timespec *start_time){
    asm_instruction_barrier;
    clock_gettime(CLOCK_MONOTONIC, start_time);
}

// call this function to end a timer, returning usec elapsed as a double
double static inline timer_stop(struct timespec *start_time){
    struct timespec end_time;
    asm_instruction_barrier;
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    long diffInNanos = (end_time.tv_sec - start_time->tv_sec) * (long)1e9 + (end_time.tv_nsec - start_time->tv_nsec);
    return diffInNanos*1e-3;
}

// call this function to end a timer, and start a new one, returning usec elapsed as a double
double static inline timer_restart(struct timespec *start_time){
    struct timespec end_time;
    asm_instruction_barrier;
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    long diffInNanos = (end_time.tv_sec - start_time->tv_sec) * (long)1e9 + (end_time.tv_nsec - start_time->tv_nsec);
    start_time->tv_nsec = end_time.tv_nsec;
    start_time->tv_sec = end_time.tv_sec;
    return diffInNanos*1e-3;
}
