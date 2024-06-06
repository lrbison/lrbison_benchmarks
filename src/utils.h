#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <string.h>

#include <unistd.h>
extern char *optarg;
extern int optind, opterr, optopt;

#define MIN(a,b) ((a < b) ? a : b)
#define MAX(a,b) ((a > b) ? a : b)

// call this function to start a nanosecond-resolution timer
void static inline timer_start(struct timespec *start_time){
    clock_gettime(CLOCK_MONOTONIC, start_time);
}

// call this function to end a timer, returning usec elapsed as a double
double static inline timer_stop(struct timespec *start_time){
    struct timespec end_time;
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    long diffInNanos = (end_time.tv_sec - start_time->tv_sec) * (long)1e9 + (end_time.tv_nsec - start_time->tv_nsec);
    return diffInNanos*1e-3;
}

// call this function to end a timer, and start a new one, returning usec elapsed as a double
double static inline timer_restart(struct timespec *start_time){
    struct timespec end_time;
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    long diffInNanos = (end_time.tv_sec - start_time->tv_sec) * (long)1e9 + (end_time.tv_nsec - start_time->tv_nsec);
    start_time->tv_nsec = end_time.tv_nsec;
    start_time->tv_sec = end_time.tv_sec;
    return diffInNanos*1e-3;
}

int static comp_dbls(const void * elem1, const void * elem2)
{
    double a = *((double*)elem1);
    double b = *((double*)elem2);
    if (a > b) return  1;
    if (a < b) return -1;
    return 0;
}
