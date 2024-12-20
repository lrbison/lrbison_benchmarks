#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <unistd.h>

#include <omp.h>

#define MIN(a,b) ((a < b) ? a : b);
#define MAX(a,b) ((a > b) ? a : b);

// call this function to start a nanosecond-resolution timer
void timer_start(struct timespec *start_time){
    clock_gettime(CLOCK_MONOTONIC, start_time);
}

// call this function to end a timer, returning nanoseconds elapsed as a long
long timer_stop(struct timespec *start_time){
    struct timespec end_time;
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    long diffInNanos = (end_time.tv_sec - start_time->tv_sec) * (long)1e9 + (end_time.tv_nsec - start_time->tv_nsec);
    return diffInNanos;
}

int comp_dbls(const void * elem1, const void * elem2)
{
    double a = *((double*)elem1);
    double b = *((double*)elem2);
    if (a > b) return  1;
    if (a < b) return -1;
    return 0;
}

int race_test(int nthread_use, int nsteps, double *usec_per_steps ) {

    volatile int shared_var;

#pragma omp parallel default(shared)
    {
        double elapsed_sec;
        int jthread = omp_get_thread_num();

        int count_to = nsteps/nthread_use;
        if (count_to * nthread_use < nsteps && jthread < nsteps%nthread_use)
            count_to++;

        if (jthread == 0) {
            shared_var = 0;
        }

        int times_counted = 0;
        struct timespec timer;
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
#pragma omp barrier
        if (jthread < nthread_use) {
            timer_start(&timer);
            int cas_done;
            int expect0;

            do {
                expect0 = shared_var;
                cas_done = __atomic_compare_exchange_n(&shared_var, &expect0, expect0+1,
                    0, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED);
                if (cas_done) {
                    times_counted++;
                }
            } while (times_counted < count_to);

            usec_per_steps[jthread] = (timer_stop(&timer) * 1e-3) / count_to;
            // printf("Race [%d/%d] counted [%d==%d].  Shared:%d\n",jthread,nthread_use, times_counted, nsteps, shared_var);
            assert(times_counted == count_to );
        } else { usec_per_steps[jthread] = -1.0; }
    }
}

int ring_test(int nthread_use, int nsteps, double *usec_per_steps ) {

    volatile int shared_var;
    const int count_to = nsteps;

#pragma omp parallel default(shared)
    {
        double elapsed_sec;

        // int nthread = omp_get_num_threads();
        int jthread = omp_get_thread_num();

        if (jthread == 0) {
            shared_var = 0;
        }

        int times_counted = 0;
        int expect_value = jthread;
        struct timespec timer;
        __atomic_thread_fence(__ATOMIC_SEQ_CST);
#pragma omp barrier
        if (jthread < nthread_use) {
            timer_start(&timer);

            while (expect_value < count_to) {
                int cas_done;
                int expect0;
                do {
                    expect0 = expect_value;
                    cas_done = __atomic_compare_exchange_n(&shared_var, &expect0, expect_value+1,
                        0, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED);
                        // usleep(1);
                } while (!cas_done);

                times_counted++;
                expect_value += nthread_use;
                // __atomic_thread_fence(__ATOMIC_SEQ_CST);
            }
            usec_per_steps[jthread] = (timer_stop(&timer) * 1e-3) / count_to;
            // printf("Thread [%d/%d] counted [%d==%d].  Shared:%d\n",jthread,nthread_use, times_counted, nsteps/nthread_use, shared_var);
            int count_expect = count_to/nthread_use;
            if (count_expect * nthread_use < count_to && jthread < count_to%nthread_use)
                count_expect++;
            assert(times_counted == count_expect );
        }
    }
}

int main(int argc, char *argv[]) {

    int nthreads_avail = 0;
    int niters = 1000;
    double *usec_per_steps;
    int nsets = 1000;
    double *trial_medians;
    double shortest;
    double longest;

    int opt;
    int thread_lower, thread_upper, user_threads=0;

    while ((opt = getopt(argc, argv, "t:s:i:")) != -1) {
        switch (opt) {
        case 't': user_threads = atoi(optarg); break;
        case 's': nsets = atoi(optarg); break;
        case 'i': niters = atoi(optarg); break;
        default:
            fprintf(stderr, "Usage: %s [-t nthreads] [-s nsets] [-i iters-per-set]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }

    #pragma omp parallel default(shared)
    #pragma omp single
    {
        nthreads_avail = omp_get_num_threads();
    }

    if (user_threads) {
        thread_lower = user_threads;
        thread_upper = user_threads;
        omp_set_num_threads(user_threads);
    } else {
        thread_lower = 1;
        thread_upper = nthreads_avail;
    }

    usec_per_steps = malloc(sizeof(*usec_per_steps) * nthreads_avail);
    trial_medians = malloc(sizeof(*trial_medians) * nsets);
    int thread_incr = 1;

    // for (int jthreads_use=thread_lower; jthreads_use <= thread_upper; jthreads_use+=thread_incr) {
    //     shortest = 99999999;
    //     longest = -1;
    //     for (int jtrial=0; jtrial < nsets; jtrial++) {
    //         race_test(jthreads_use, ncount, usec_per_steps );
    //         qsort(usec_per_steps, jthreads_use, sizeof(*usec_per_steps), comp_dbls);
    //         trial_medians[jtrial] = usec_per_steps[jthreads_use/2];
    //         shortest = MIN(shortest, usec_per_steps[0]);
    //         longest = MAX( longest, usec_per_steps[jthreads_use-1]);
    //     }
    //     qsort(trial_medians, nsets, sizeof(*trial_medians), comp_dbls);
    //     printf("Race %3d threads: %8.3f %8.3f %8.3f usec/step\n",
    //         jthreads_use, shortest, trial_medians[nsets/2], longest);
    //     if (jthreads_use == 4) { thread_incr = 4; }
    //     if (jthreads_use == 8) { thread_incr = 8; }
    // }

    thread_incr = 1;
    for (int jthreads_use=thread_lower; jthreads_use <= thread_upper; jthreads_use+=thread_incr) {
        shortest = 99999999;
        longest = -1;
        int jworst = 99;
        for (int jwarmup=0; jwarmup<10; jwarmup++) {
            ring_test(jthreads_use, niters, usec_per_steps );
        }
        for (int jtrial=0; jtrial < nsets; jtrial++) {
            ring_test(jthreads_use, niters, usec_per_steps );
            qsort(usec_per_steps, jthreads_use, sizeof(*usec_per_steps), comp_dbls);
            trial_medians[jtrial] = usec_per_steps[jthreads_use/2];
            shortest = MIN(shortest, usec_per_steps[0]);
            if (trial_medians[jtrial] > longest ) jworst = jtrial;
            longest = MAX( longest, usec_per_steps[jthreads_use-1]);
        }
        qsort(trial_medians, nsets, sizeof(*trial_medians), comp_dbls);
        printf("Ring %3d threads: %8.3f %8.3f %8.3f usec/step (jworst is %d)\n",
            jthreads_use, shortest, trial_medians[nsets/2], longest, jworst);
        if (jthreads_use == 4) { thread_incr = 4; }
        if (jthreads_use == 8) { thread_incr = 8; }
    }

    free(usec_per_steps);
    free(trial_medians);
}