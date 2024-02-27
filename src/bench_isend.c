#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <unistd.h>

#include <mpi.h>

#define MIN(a,b) ((a < b) ? a : b)
#define MAX(a,b) ((a > b) ? a : b)

struct timing_result {
    double submit_time_avg;
    double completion_time_avg;
    double total_time_avg;
};

struct run_config {
    int ntrials;
    int nsends_avg;
    int msg_size;
    int nranks;
    int nwarmup;

    int do_isend;
    int do_irecv;
    int do_barrier;
    int do_leader_exchange;
    const char *timing_file;
    char test;
};



// call this function to start a nanosecond-resolution timer
void timer_start(struct timespec *start_time){
    clock_gettime(CLOCK_MONOTONIC, start_time);
}

// call this function to end a timer, returning nanoseconds elapsed as a long
double timer_stop(struct timespec *start_time){
    struct timespec end_time;
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    long diffInNanos = (end_time.tv_sec - start_time->tv_sec) * (long)1e9 + (end_time.tv_nsec - start_time->tv_nsec);
    return diffInNanos*1e-3;
}

// call this function to end a timer, and start a new one, returning nanoseconds elapsed as a long
double timer_restart(struct timespec *start_time){
    struct timespec end_time;
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    long diffInNanos = (end_time.tv_sec - start_time->tv_sec) * (long)1e9 + (end_time.tv_nsec - start_time->tv_nsec);
    start_time->tv_nsec = end_time.tv_nsec;
    start_time->tv_sec = end_time.tv_sec;
    return diffInNanos*1e-3;
}

int comp_dbls(const void * elem1, const void * elem2)
{
    double a = *((double*)elem1);
    double b = *((double*)elem2);
    if (a > b) return  1;
    if (a < b) return -1;
    return 0;
}

void gather_send(struct run_config *config, int dest,
    char *buf, MPI_Request *req_list, MPI_Status *stat_list,
    struct timing_result *timing) {

    struct timespec start_time;

    timer_start(&start_time);

    for (int jsend=0; jsend< config->nsends_avg; jsend++) {
        buf[0] =  jsend & 127;
        if (config->do_isend) {
            MPI_Isend(buf, config->msg_size, MPI_BYTE, dest, 0, MPI_COMM_WORLD, &req_list[jsend]);
        } else {
            MPI_Send(buf, config->msg_size, MPI_BYTE, dest, 0, MPI_COMM_WORLD);
        }
    }
    if (config->do_isend) {
        timing->submit_time_avg = timer_restart(&start_time) / config->nsends_avg;
        MPI_Waitall(config->nsends_avg, req_list, MPI_STATUS_IGNORE);
        timing->completion_time_avg = timer_stop(&start_time) / config->nsends_avg;
        timing->total_time_avg = timing->completion_time_avg + timing->submit_time_avg;

    } else {
        timing->submit_time_avg = 0;
        timing->completion_time_avg = 0;
        timing->total_time_avg = timer_stop(&start_time) / config->nsends_avg;
    }
}

void gather_recv(struct run_config *config, int skip_rank, int comm_size,
    char *buf, MPI_Request *req_list, MPI_Status *stat_list, struct timing_result *timing) {
    MPI_Status stat;
    struct timespec start_time;

    timer_start(&start_time);
    int nrecvs = 0;

    for (int jrank=0; jrank < comm_size; jrank++) {
        if (jrank == skip_rank) { continue; }
        for (int jsend=0; jsend< config->nsends_avg; jsend++) {
            if (config->do_irecv) {
                MPI_Irecv(buf, config->msg_size, MPI_BYTE, jrank, 0, MPI_COMM_WORLD, &req_list[nrecvs]);
            } else {
                MPI_Recv(buf, config->msg_size, MPI_BYTE, jrank, 0, MPI_COMM_WORLD, &stat);
            }
            nrecvs++;
        }
    }
    if (config->do_irecv) {
        timing->submit_time_avg = timer_restart(&start_time) / config->nsends_avg;
        MPI_Waitall(nrecvs, req_list, MPI_STATUS_IGNORE);
        timing->completion_time_avg = timer_stop(&start_time) / config->nsends_avg;
        timing->total_time_avg = timing->completion_time_avg + timing->submit_time_avg;
    } else {
        timing->total_time_avg = timer_stop(&start_time) / config->nsends_avg;
        timing->submit_time_avg = 0;
        timing->completion_time_avg = 0;
    }
}

void scatter_recv(struct run_config *config, int root_rank,
    char *buf, MPI_Request *req_list, MPI_Status *stat_list, struct timing_result *timing) {
    MPI_Status stat;
    struct timespec start_time;

    timer_start(&start_time);
    int nrecvs = 0;

    for (int jsend=0; jsend< config->nsends_avg; jsend++) {
        if (config->do_irecv) {
            MPI_Irecv(buf, config->msg_size, MPI_BYTE, root_rank, 0, MPI_COMM_WORLD, &req_list[nrecvs]);
        } else {
            MPI_Recv(buf, config->msg_size, MPI_BYTE, root_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        nrecvs++;
    }
    if (config->do_irecv) {
        timing->submit_time_avg = timer_restart(&start_time) / config->nsends_avg;
        MPI_Waitall(nrecvs, req_list, MPI_STATUS_IGNORE);
        timing->completion_time_avg = timer_stop(&start_time) / config->nsends_avg;
        timing->total_time_avg = timing->completion_time_avg + timing->submit_time_avg;
    } else {
        timing->total_time_avg = timer_stop(&start_time) / config->nsends_avg;
        timing->submit_time_avg = 0;
        timing->completion_time_avg = 0;
    }
}

void scatter_send(struct run_config *config, int root_rank, int comm_size,
    char *buf, MPI_Request *req_list, MPI_Status *stat_list,
    struct timing_result *timing) {

    struct timespec start_time;

    timer_start(&start_time);
    int ksend = 0;
    for (int jrank=0; jrank < comm_size; jrank++) {
        if (jrank == root_rank) { continue; }
        for (int jsend=0; jsend< config->nsends_avg; jsend++) {
            buf[0] =  jsend & 127;
            if (config->do_isend) {
                MPI_Isend(buf, config->msg_size, MPI_BYTE, jrank, 0, MPI_COMM_WORLD, &req_list[ksend]);
            } else {
                MPI_Send(buf, config->msg_size, MPI_BYTE, jrank, 0, MPI_COMM_WORLD);
            }
            ksend++;
        }
    }
    if (config->do_isend) {
        timing->submit_time_avg = timer_restart(&start_time) / config->nsends_avg;
        MPI_Waitall(ksend, req_list, MPI_STATUS_IGNORE);
        timing->completion_time_avg = timer_stop(&start_time) / config->nsends_avg;
        timing->total_time_avg = timing->completion_time_avg + timing->submit_time_avg;

    } else {
        timing->submit_time_avg = 0;
        timing->completion_time_avg = 0;
        timing->total_time_avg = timer_stop(&start_time) / config->nsends_avg;
    }
}

void remove_last_two_chars(FILE *fd) {
    fseek(fd,-2, SEEK_CUR); /* */
    fprintf(fd, " \n");
}

void dump_json_doubles_array(FILE *fd, const char *name, int nvals, int stride_bytes, void *vals) {
    fprintf(fd, "  \"%s\" : [", name);
    char *bytes = vals;
    for (int jval=0; jval < nvals; jval++) {
        double val = *(double*)&bytes[jval * stride_bytes];
        fprintf(fd, " %f%c",val, jval<nvals-1 ? ',' : ']');
    }
    fprintf(fd, ",\n");
}

void dump_times(struct run_config *config, struct timing_result *trial_times) {
    int ntrials, nsends, msg_size;
    int comm_rank, comm_size;
    struct timing_result *all_trials = NULL;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    double *values;
    ntrials = config->ntrials;
    nsends = config->nsends_avg;
    msg_size = config->msg_size;

    double *tmp_buf = NULL;
    if (comm_rank == 0) {
        all_trials = malloc(sizeof(*all_trials)*comm_size*ntrials);
        // values = malloc(sizeof(*values)*comm_size*ntrials)
    }
    int bytes_send = sizeof(*all_trials) * ntrials;
    MPI_Gather(trial_times, bytes_send, MPI_BYTE, all_trials, bytes_send, MPI_BYTE, 0, MPI_COMM_WORLD);

    if (comm_rank != 0) { return; }

    double sum_time_other = 0;
    double sum_time_root = 0;
    double sum_diff_time = 0;
    int jroot = 0;
    for (int jtrail=0; jtrail<ntrials; jtrail++) {
        int jfollow = 0;
        double prev_tot_time;
        for (int jrank=0; jrank < comm_size; jrank++) {
            if (jrank == jroot) {
                sum_time_root += all_trials[jtrail + jroot*ntrials].total_time_avg;
                continue;
            }
            sum_time_other += all_trials[jtrail + jrank*ntrials].total_time_avg;

            if (jfollow == 0) {
                prev_tot_time = all_trials[jtrail + jrank*ntrials].total_time_avg;
            } else {
                sum_diff_time += all_trials[jtrail + jrank*ntrials].total_time_avg - prev_tot_time;
                prev_tot_time = all_trials[jtrail + jrank*ntrials].total_time_avg;
            }
            jfollow++;
        }
        if (config->do_leader_exchange) { jroot = (jroot+1) % config->nranks; }
    }

    printf("Mean time followers:       %8.3f usec/exchange\n",sum_time_other/(comm_size-1)/ntrials);
    printf("Mean delta-time followers: %8.3f usec/exchange\n",sum_diff_time/(comm_size-2)/ntrials);
    printf("Mean time root:            %8.3f usec/exchange\n",sum_time_root/ntrials);

    /* printing with this stride will extract rank N's data for one particular trial */
    int stride = sizeof(*trial_times) * ntrials;

    if (config->timing_file) {
        FILE *fd = fopen(config->timing_file, "w");
        const char *test_name;
        fprintf(fd, "{\n  \"test_config\" : {\n");
            fprintf(fd, "\"ntrials\" : %d,\n", config->ntrials);
            fprintf(fd, "\"nsends_avg\" : %d,\n", config->nsends_avg);
            fprintf(fd, "\"msg_size\" : %d,\n", config->msg_size);
            fprintf(fd, "\"nranks\" : %d,\n", config->nranks);
            fprintf(fd, "\"nwarmup\" : %d,\n", config->nwarmup);
            fprintf(fd, "\"ntrials\" : %d,\n", config->ntrials);

            fprintf(fd, "\"do_isend\" : %d,\n", config->do_isend);
            fprintf(fd, "\"do_irecv\" : %d,\n", config->do_irecv);
            fprintf(fd, "\"do_barrier\" : %d,\n", config->do_barrier);
            fprintf(fd, "\"do_leader_exchange\" : %d,\n", config->do_leader_exchange);
            switch (config->test) {
                case 'i': test_name = "gather"; break;
                case 'o': test_name = "scatter"; break;
            }
            fprintf(fd, "\"test_name\" : \"%s\",\n", test_name);

            remove_last_two_chars(fd);
        fprintf(fd, "},\n");

        fprintf(fd, "  \"test_result\" : [\n");

        for (int jtrail=0; jtrail<ntrials; jtrail++) {
            fprintf(fd, "{ \"trail\" : %d,\n",jtrail);
            dump_json_doubles_array(fd, "total_time_avg", comm_size, stride, &all_trials[jtrail].total_time_avg);
            dump_json_doubles_array(fd, "submit_time_avg", comm_size, stride, &all_trials[jtrail].submit_time_avg);
            dump_json_doubles_array(fd, "completion_time_avg", comm_size, stride, &all_trials[jtrail].completion_time_avg);
            remove_last_two_chars(fd);
            fprintf(fd, "},\n");
        }
        remove_last_two_chars(fd);
        fprintf(fd, "],\n");

        fprintf(fd, "  \"test_summary\" : {\n");
        fprintf(fd,"    \"avg_follower_total\" :  %8.3f,\n",sum_time_other/(comm_size-1)/ntrials);
        fprintf(fd,"    \"avg_delta_follower_delta\" :  %8.3f,\n",sum_diff_time/(comm_size-2)/ntrials);
        fprintf(fd,"    \"avg_root_total\" :  %8.3f,\n",sum_time_root/ntrials);
        remove_last_two_chars(fd);
        fprintf(fd, "  }\n}\n");

        fclose(fd);
        printf("Timing was written to \"%s\"\n",config->timing_file);
    } else {
        printf("No timing written\n");
    }

    free(all_trials);
}

void test_gather(struct run_config *config) {
    int comm_size, comm_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

    int nsends = config->nsends_avg;
    int root_rank = 0;
    int ntrials = config->ntrials;
    int msg_size = config->msg_size;
    double elapsed_usec;
    struct timing_result *trial_times;

    MPI_Status *stats;
    MPI_Request *reqs;
    char *buf;
    int root_multiplier = 1;
    if (config->do_irecv) { root_multiplier = comm_size; }
    stats = malloc(sizeof(*stats)*nsends * root_multiplier);
    reqs = malloc(sizeof(*reqs)*nsends * root_multiplier);
    buf  = malloc(msg_size * root_multiplier);
    trial_times = malloc(sizeof(*trial_times)*ntrials);


    root_rank = 0;
    for (int jtrial_warm=-config->nwarmup; jtrial_warm<ntrials; jtrial_warm++) {

        int jtrial = jtrial_warm > 0 ? jtrial_warm : 0;
        if (config->do_barrier || jtrial_warm==0) MPI_Barrier(MPI_COMM_WORLD);
        if (comm_rank == root_rank) {
            gather_recv(config, root_rank, comm_size, buf, reqs, stats, &trial_times[jtrial]);
        } else {
            gather_send(config, root_rank, buf, reqs, stats, &trial_times[jtrial]);
        }
        if (config->do_leader_exchange) root_rank = (root_rank+1)%comm_size;
    }

    dump_times(config, trial_times);

    free(stats);
    free(reqs);
    free(buf);
    free(trial_times);
}

void test_scatter(struct run_config *config) {
    int comm_size, comm_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

    int nsends = config->nsends_avg;
    int root_rank = 0;
    int ntrials = config->ntrials;
    int msg_size = config->msg_size;
    double elapsed_usec;
    struct timing_result *trial_times;

    MPI_Status *stats;
    MPI_Request *reqs;
    char *buf;
    int root_multiplier = 1;
    if (config->do_isend) { root_multiplier = comm_size; }
    // stats = malloc(sizeof(*stats)*nsends * root_multiplier);
    if (!config->do_isend && !config->do_irecv) {
        reqs = malloc(sizeof(*reqs));
    } else {
        reqs = malloc(sizeof(*reqs)*nsends * root_multiplier);
    }
    buf  = malloc(msg_size * root_multiplier);
    trial_times = malloc(sizeof(*trial_times)*ntrials);


    root_rank = 0;
    for (int jtrial_warm=-config->nwarmup; jtrial_warm<ntrials; jtrial_warm++) {

        int jtrial = jtrial_warm > 0 ? jtrial_warm : 0;
        if (config->do_barrier || jtrial_warm==0) MPI_Barrier(MPI_COMM_WORLD);
        if (comm_rank != root_rank) {
            scatter_recv(config, root_rank, buf, reqs, stats, &trial_times[jtrial]);
        } else {
            scatter_send(config, root_rank, comm_size, buf, reqs, stats, &trial_times[jtrial]);
        }
        if (config->do_leader_exchange) root_rank = (root_rank+1)%comm_size;
    }

    dump_times(config, trial_times);

    // free(stats);
    free(reqs);
    free(buf);
    free(trial_times);
}

int main(int argc, char **argv) {



    int opt;
    int msg_size = 4;
    int ii_scale = 0;
    int comm_size, comm_rank;
    struct run_config config;

    config.do_irecv = 0;
    config.do_isend = 0;
    config.msg_size = 8;
    config.nsends_avg = 100;
    config.ntrials  = 100;
    config.timing_file = NULL;
    config.nwarmup = 100;
    config.do_barrier = 1;
    config.do_leader_exchange = 0;

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);


    while ((opt = getopt(argc, argv, "bf:him:n:orst:w:x")) != -1) {
        switch (opt) {
        case 'b': config.do_barrier = 0; break;
        case 'f': config.timing_file = optarg; break;
        case 'i': config.test = 'i'; break;
        case 'm': config.msg_size = atoi(optarg); break;
        case 'n': config.nsends_avg = atoi(optarg); break;
        case 'o': config.test = 'o'; break;
        case 'r': config.do_irecv = 1; break;
        case 's': config.do_isend = 1; break;
        case 't': config.ntrials = atoi(optarg); break;
        case 'w': config.nwarmup = atoi(optarg); break;
        case 'x': config.do_leader_exchange = 1; break;
        case 'h':
        default:
            if (comm_rank==0) {
                fprintf(stderr, "Usage: %s [options...]\n", argv[0]);
                fprintf(stderr, "  -b                   Skip barriers between trials\n");
                fprintf(stderr, "  -m msg_size          Size of message in bytes\n");
                fprintf(stderr, "  -n nsends_avg        Time N iterations together.\n");
                fprintf(stderr, "  -t trials            Trials, each of which uses nsends.\n");
                fprintf(stderr, "  -s                   send as non-blocking (isend).\n");
                fprintf(stderr, "  -r                   recv as non-blocking (irecv).\n");
                fprintf(stderr, "  -o                   outcast test (scatter).\n");
                fprintf(stderr, "  -i                   incast test (gather).\n");
                fprintf(stderr, "  -f filename          Dump all data to file.\n");
                fprintf(stderr, "  -w nwarmup           number of warmup trials.\n");
                fprintf(stderr, "  -x                   exchange leader each trial.\n");
            }
            goto end;
        }
    }
    config.nranks = comm_size;
    if (config.test == 'o') { test_scatter(&config); };
    if (config.test == 'i') { test_gather(&config); };

    if (comm_rank == 0) {
        printf("Test complete.\n");
    }
end:
    MPI_Finalize();
    return 0;
}
