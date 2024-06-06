#include "utils.h"

#include <mpi.h>


struct pipeline_work {
    char **bounce_bufs;
    char *big_buf;
    int nbounce;
    int bounce_length;
    int bigbuf_length;

    MPI_Request **requests;
};

struct pipeline_result {
    int jsize;
    int jdepth;
    int jtrail;
    double elapsed_time;
    double cum_copy_time;
    double cum_wait_time;
    double cum_send_time;
};

struct pipeline_config {
    int ntrials;
    int bigsize;

    int nsizes;
    int *size_list;

    int ndepths;
    int *depth_list;

    char *timing_file;

    // int nwarmup;

    struct pipeline_work work;

    // results are indexed by jsize*ndepth*ntrail + jdepth*ntrail + jtrail;
    // ie: slow-to-fast dimensions are: [ size x depth x trail ]
    struct pipeline_result *results;
};

void pipeline_send(
        struct pipeline_config *config,
        int jsize, int jdepth )
{
    struct timespec start_time;
    struct timespec tictoc;
    struct pipeline_work *work = &config->work;

    int jrank_dest = 1;


    for (size_t jtrail=0; jtrail<config->ntrials; jtrail++) {
        size_t buf_size = config->size_list[jsize];
        size_t niter = config->bigsize / buf_size;
        size_t nbytes_sent = 0;
        size_t depth = config->depth_list[jdepth];
        // niter += niter*config->size_list[jsize] < config->bigsize;

        double waiting_time = 0;
        double copying_time = 0;
        double sending_time = 0;
        double total_time = 0;

        timer_start(&start_time);
        int jbuf = 0;
        for (size_t jiter=0; jiter<niter; jiter++) {

            if (jiter >= depth ) {
                /* complete the send for this buffer before we fill */
                timer_start(&tictoc);
                MPI_Wait(work->requests[jbuf], MPI_STATUS_IGNORE);
                waiting_time += timer_stop(&tictoc);
            }

            if (jiter < niter) {
                /* fill the bounce buffer */
                timer_start(&tictoc);
                memcpy(work->bounce_bufs[jbuf], &work->big_buf[nbytes_sent], buf_size);
                copying_time += timer_restart(&tictoc);

                /* issue the send */
                MPI_Isend(work->bounce_bufs[jbuf], buf_size, MPI_BYTE, jrank_dest, 0, MPI_COMM_WORLD, work->requests[jbuf]);
                sending_time += timer_stop(&tictoc);
            }
            jbuf = (jbuf + 1) % depth;
            nbytes_sent += buf_size;
        }
        total_time = timer_stop(&start_time);

        size_t jresult = jsize*config->ndepths*config->ntrials + jdepth*config->ntrials + jtrail;
        config->results[jresult].cum_copy_time = copying_time;
        config->results[jresult].cum_wait_time = waiting_time;
        config->results[jresult].cum_send_time = sending_time;
        config->results[jresult].jdepth = jdepth;
        config->results[jresult].jsize = jsize;
        config->results[jresult].jtrail = jtrail;
    }
}

void pipeline_recv(
        struct pipeline_config *config,
        int jsize, int jdepth )
{
    struct timespec start_time;
    struct timespec tictoc;
    struct pipeline_work *work = &config->work;

    int jrank_src = 0;


    for (size_t jtrail=0; jtrail<config->ntrials; jtrail++) {
        size_t buf_size = config->size_list[jsize];
        size_t niter = config->bigsize / buf_size;
        size_t nbytes_recv = 0;
        size_t depth = config->depth_list[jdepth];
        // niter += niter*config->size_list[jsize] < config->bigsize;

        double waiting_time = 0;
        double copying_time = 0;
        double xfer_time = 0;
        double total_time = 0;

        timer_start(&start_time);
        int jbuf = 0;
        for (size_t jiter=0; jiter<niter; jiter++) {

            if (jiter >= depth ) {
                /* complete the send for this buffer before we fill */
                timer_start(&tictoc);
                MPI_Wait(work->requests[jbuf], MPI_STATUS_IGNORE);
                waiting_time += timer_restart(&tictoc);

                /* drain the bounce buffer */
                memcpy(&work->big_buf[nbytes_recv], work->bounce_bufs, buf_size);
                copying_time += timer_stop(&tictoc);
            }

            if (jiter < niter) {
                /* post the recv */
                timer_start(&tictoc);
                MPI_Irecv(work->bounce_bufs[jbuf], buf_size, MPI_BYTE, jrank_src, 0, MPI_COMM_WORLD, work->requests[jbuf]);
                xfer_time += timer_stop(&tictoc);
            }
            jbuf = (jbuf + 1) % depth;
            nbytes_recv += buf_size;
        }
        total_time = timer_stop(&start_time);

        size_t jresult = jsize*config->ndepths*config->ntrials + jdepth*config->ntrials + jtrail;
        config->results[jresult].cum_copy_time = copying_time;
        config->results[jresult].cum_wait_time = waiting_time;
        config->results[jresult].cum_send_time = xfer_time;
        config->results[jresult].jdepth = jdepth;
        config->results[jresult].jsize = jsize;
        config->results[jresult].jtrail = jtrail;
    }
}


int main(int argc, char **argv) {


    int opt;
    int msg_size = 4;
    int ii_scale = 0;
    int retval = 1;
    int comm_size, comm_rank;
    struct pipeline_config config;

    config.ntrials = 5;
    config.bigsize = 1024*1024;
    config.nsizes = 1;
    config.ndepths = 1;


    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);


    while ((opt = getopt(argc, argv, "m:b:t:d:f:h")) != -1) {
        switch (opt) {
        case 'f': config.timing_file = optarg; break;
        case 'h':
            retval=0;
        default:
            if (comm_rank==0) {
                fprintf(stderr, "Usage: %s [options...]\n", argv[0]);
                fprintf(stderr, "  -m size              Size of the full message in bytes\n");
                fprintf(stderr, "  -b bufsizes          Size of the buffer to use.  See LIST_FORMAT\n");
                fprintf(stderr, "  -t trials            Trials. See LIST_FORMAT\n");
                fprintf(stderr, "  -d depths            Pipeline Depth. See LIST_FORMAT\n");
                fprintf(stderr, "  -f filename          Dump all data to file.\n");
                fprintf(stderr, "  -h                   Print this help.\n");
                fprintf(stderr, "\n");
                fprintf(stderr, "Careful: the message size will be sent a total of nbufsizes*ndepths*ntrails times!\n");
                fprintf(stderr, "\n");
                fprintf(stderr, "LIST_FORMAT: lists can be specified either as increasing factors or as increasing steps.");
                fprintf(stderr, "     min*factor*max:  e.g. 8*2*128 produces 8,16,32,64,128\n");
                fprintf(stderr, "     min+step+max:    e.g. 10+5+25 produces 10,15,20,25\n");
                fprintf(stderr, "Currently requires exactly 2 ranks.\n");
            }
            goto end;
        }
    }

    if (comm_rank == 0) {
        printf("Test complete.\n");
    }
    retval = 0;
end:
    MPI_Finalize();
    return retval;
}
