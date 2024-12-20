#include "utils.h"

#include <mpi.h>

#include "json-builder.h"

#ifdef DEBUG
#define DEBUG_PRINT(...) do{ fprintf( stdout, __VA_ARGS__ ); } while( 0 )
#else
#define DEBUG_PRINT(...) do{ } while ( 0 )
#endif



struct pipeline_work {
    char **bounce_bufs;
    char *big_buf;
    // int nbounce;
    // int bounce_length;
    // int bigbuf_length;

    MPI_Request *requests;
};

struct pipeline_result {
    int jsize;
    int jdepth;
    int jtrial;
    int jrank;
    double elapsed_time;
    double cum_copy_time;
    double cum_wait_time;
    double cum_send_time;
    double bandwidth;
};

struct pipeline_config {
    int ntrials;
    size_t bigsize;

    int nsizes;
    size_t *size_list;

    int ndepths;
    size_t *depth_list;

    char *timing_file;

    // int nwarmup;

    struct pipeline_work work;

    // results are indexed by jsize*ndepth*ntrail + jdepth*ntrail + jtrial;
    // ie: slow-to-fast dimensions are: [ size x depth x trail ]
    struct pipeline_result *results;
};

int create_buffers(struct pipeline_config *config) {
    size_t nresults = config->nsizes * config->ndepths * config->ntrials;
    size_t max_buf = config->size_list[config->nsizes-1];
    size_t nbufs = config->depth_list[config->ndepths-1];

    config->work.big_buf     = calloc(config->bigsize, 1);
    if (!config->work.big_buf) goto oom;
    config->work.bounce_bufs = calloc(nbufs,    sizeof(*config->work.bounce_bufs) );
    if (!config->work.bounce_bufs) goto oom;
    config->work.requests    = calloc(nbufs,    sizeof(*config->work.requests) );
    if (!config->work.requests) goto oom;
    config->results          = calloc(nresults, sizeof(*config->results) );
    if (!config->results) goto oom;

    for (int jbuf=0; jbuf<nbufs; jbuf++) {
        config->work.bounce_bufs[jbuf] = calloc(max_buf, 1);
        if (!config->work.bounce_bufs[jbuf]) goto oom;
    }
    return 0;
oom:
    fprintf(stderr, "OUT OF MEMORY! problem during buffer allocation\n");
    return 1;
}

void destroy_buffers(struct pipeline_config *config) {
    size_t nbufs = config->depth_list[config->ndepths-1];

    free(config->work.big_buf);
    free(config->work.requests);
    free(config->results);
    for (int jbuf=0; jbuf<nbufs; jbuf++) {
        free(config->work.bounce_bufs[jbuf]);
    }
    free(config->work.bounce_bufs);
    free(config->depth_list);
    free(config->size_list);
}

void pipeline_send(
        struct pipeline_config *config,
        int jsize, int jdepth, int jrank_dest )
{
    struct timespec start_time;
    struct timespec tictoc;
    struct pipeline_work *work = &config->work;
    int comm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

    double cumcum_copy_time = 0;
    double cumcum_wait_time = 0;
    double cumcum_send_time = 0;
    double cum_elapsed_time = 0;


    for (size_t jtrial=0; jtrial<config->ntrials; jtrial++) {
        size_t buf_size = config->size_list[jsize];
        size_t niter = MAX(1, config->bigsize / buf_size);
        size_t bigsize = config->bigsize;
        size_t nbytes_send_posted = 0;
        size_t nbytes_send_completed = 0;
        size_t depth = config->depth_list[jdepth];
        // niter += niter*config->size_list[jsize] < config->bigsize;
        DEBUG_PRINT("Send starting trial %ld\n",jtrial);


        double waiting_time = 0;
        double copying_time = 0;
        double sending_time = 0;
        double total_time = 0;
        size_t nbytes_xfer;


        timer_start(&start_time);
        int jbuf = 0;
        for (size_t jiter=0; jiter<niter + depth; jiter++) {

            nbytes_xfer = MIN(bigsize - nbytes_send_completed, buf_size);
            if ( nbytes_xfer > 0 && jiter >= depth) {
                DEBUG_PRINT("Waiting for send %d\n",jbuf);
                /* complete the send for this buffer before we fill */
                timer_start(&tictoc);
                MPI_Wait(&work->requests[jbuf], MPI_STATUS_IGNORE);
                waiting_time += timer_stop(&tictoc);

                nbytes_send_completed += nbytes_xfer;
            }

            nbytes_xfer = MIN(bigsize - nbytes_send_posted, buf_size);
            if (nbytes_xfer > 0) {
                DEBUG_PRINT("Filling buffer %d\n",jbuf);
                /* fill the bounce buffer */
                timer_start(&tictoc);
                memcpy(work->bounce_bufs[jbuf], &work->big_buf[nbytes_send_posted], nbytes_xfer);
                copying_time += timer_restart(&tictoc);

                /* issue the send */
                DEBUG_PRINT("Sending buffer %d (%ld bytes)\n",jbuf, nbytes_xfer);
                MPI_Isend(work->bounce_bufs[jbuf], nbytes_xfer, MPI_BYTE, jrank_dest, 0, MPI_COMM_WORLD, &work->requests[jbuf]);
                sending_time += timer_stop(&tictoc);
                nbytes_send_posted += nbytes_xfer;
            }
            jbuf = (jbuf + 1) % depth;
        }
        total_time = timer_stop(&start_time);

        size_t jresult = jsize*config->ndepths*config->ntrials + jdepth*config->ntrials + jtrial;
        config->results[jresult].cum_copy_time = copying_time;
        config->results[jresult].cum_wait_time = waiting_time;
        config->results[jresult].cum_send_time = sending_time;
        config->results[jresult].elapsed_time  = total_time;
        config->results[jresult].jdepth = jdepth;
        config->results[jresult].jsize = jsize;
        config->results[jresult].jtrial = jtrial;
        config->results[jresult].jrank = comm_rank;
        config->results[jresult].bandwidth = bigsize/(total_time*1e-6);


        cumcum_copy_time += copying_time;
        cumcum_wait_time += waiting_time;
        cumcum_send_time += sending_time;
        cum_elapsed_time += total_time;
    }

    DEBUG_PRINT("Sending (%8ld with %3ld x %8ld b): Copy: %12.3f   |   Wait: %12.3f  |  Post: %12.3f.  Elapsed %12.3f\n",
        config->bigsize, config->depth_list[jdepth], config->size_list[jsize],
        cumcum_copy_time/config->ntrials,
        cumcum_wait_time/config->ntrials,
        cumcum_send_time/config->ntrials,
        cum_elapsed_time/config->ntrials);
}

void pipeline_recv(
        struct pipeline_config *config,
        int jsize, int jdepth, int jrank_src )
{
    struct timespec start_time;
    struct timespec tictoc;
    struct pipeline_work *work = &config->work;
    int comm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);


    double cumcum_copy_time = 0;
    double cumcum_wait_time = 0;
    double cumcum_xfer_time = 0;
    double cum_elapsed_time = 0;

    for (size_t jtrial=0; jtrial<config->ntrials; jtrial++) {
        size_t buf_size = config->size_list[jsize];
        size_t bigsize = config->bigsize;
        size_t niter = MAX(1, config->bigsize / buf_size);
        size_t nbytes_recv_posted = 0;
        size_t nbytes_recv_completed = 0;
        size_t depth = config->depth_list[jdepth];
        // niter += niter*config->size_list[jsize] < config->bigsize;
        DEBUG_PRINT("\t\t\tRecv starting trial %ld\n",jtrial);


        double waiting_time = 0;
        double copying_time = 0;
        double xfer_time = 0;
        double total_time = 0;
        size_t nbytes_xfer;

        timer_start(&start_time);
        int jbuf = 0;
        for (size_t jiter=0; jiter<niter + depth; jiter++) {

            nbytes_xfer = MIN(bigsize - nbytes_recv_completed, buf_size);
            if (nbytes_xfer > 0 && jiter >= depth) {
                DEBUG_PRINT("\t\t\tWaiting for recv %d\n",jbuf);

                /* complete the send for this buffer before we fill */
                timer_start(&tictoc);
                MPI_Wait(&work->requests[jbuf], MPI_STATUS_IGNORE);
                waiting_time += timer_restart(&tictoc);

                DEBUG_PRINT("\t\t\tDraining buffer %d\n",jbuf);
                /* drain the bounce buffer */
                memcpy(&work->big_buf[nbytes_recv_completed], work->bounce_bufs[jbuf], nbytes_xfer);
                copying_time += timer_stop(&tictoc);
                nbytes_recv_completed += nbytes_xfer;
            }

            nbytes_xfer = MIN(bigsize - nbytes_recv_posted, buf_size);
            if (nbytes_xfer > 0) {
                DEBUG_PRINT("\t\t\tPosting Recv %d\n",jbuf);

                /* post the recv */
                timer_start(&tictoc);
                MPI_Irecv(work->bounce_bufs[jbuf], nbytes_xfer, MPI_BYTE, jrank_src, 0, MPI_COMM_WORLD, &work->requests[jbuf]);
                xfer_time += timer_stop(&tictoc);
                nbytes_recv_posted += nbytes_xfer;
            }
            jbuf = (jbuf + 1) % depth;
        }
        total_time = timer_stop(&start_time);

        size_t jresult = jsize*config->ndepths*config->ntrials + jdepth*config->ntrials + jtrial;
        config->results[jresult].cum_copy_time = copying_time;
        config->results[jresult].cum_wait_time = waiting_time;
        config->results[jresult].cum_send_time = xfer_time;
        config->results[jresult].elapsed_time  = total_time;
        config->results[jresult].jdepth = jdepth;
        config->results[jresult].jsize = jsize;
        config->results[jresult].jtrial = jtrial;
        config->results[jresult].jrank = comm_rank;
        config->results[jresult].bandwidth = bigsize/(total_time*1e-6);


        cumcum_copy_time += copying_time;
        cumcum_wait_time += waiting_time;
        cumcum_xfer_time += xfer_time;
        cum_elapsed_time += total_time;
    }
    DEBUG_PRINT("Recving (%8ld with %3ld x %8ld b): Copy: %12.3f   |   Wait: %12.3f  |  Post: %12.3f.  Elapsed %12.3f\n",
    config->bigsize, config->depth_list[jdepth], config->size_list[jsize],
    cumcum_copy_time/config->ntrials,
    cumcum_wait_time/config->ntrials,
    cumcum_xfer_time/config->ntrials,
    cum_elapsed_time/config->ntrials);
}


void run_test(struct pipeline_config *config) {
    int comm_rank;
    int comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    int is_sender = 1;
    int jrank_dest = comm_rank + comm_size/2;
    if (jrank_dest >= comm_size) {
        is_sender = 0;
        jrank_dest -= comm_size;
    }

    printf("Rank %d is %s %d\n",comm_rank, is_sender? "sending to" : "recving from", jrank_dest);

    for (int jsize=0; jsize < config->nsizes; jsize++) {
        for (int jdepth=0; jdepth < config->ndepths; jdepth++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (comm_rank == 0) {
                printf("Running test: %ld * %ld byte buffers with %ld byte message (%ld sends)\n",
                    config->depth_list[jdepth], config->size_list[jsize],
                    config->bigsize, config->bigsize/config->size_list[jsize]);
            }
            if (is_sender) {
                pipeline_send(config, jsize, jdepth, jrank_dest);
            } else {
                pipeline_recv(config, jsize, jdepth, jrank_dest);
            }
        }
    }
}


void save_results(struct pipeline_config *config) {
    int comm_rank;
    int comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    MPI_Datatype dtype;
    MPI_Type_contiguous(sizeof(*config->results), MPI_BYTE, &dtype);
    MPI_Type_commit(&dtype);
    struct pipeline_result *allresults;
    if (comm_rank == 0) {
        allresults = calloc(config->ntrials*comm_size, sizeof(*allresults));
    } else {
        allresults = NULL;
    }
    json_value *root;
    json_value *res_array;

    if (comm_rank == 0) {
        root = json_object_new(0);
        json_object_push(root, "ntrials", json_integer_new(config->ntrials));
        json_object_push(root, "nbuf_sizes", json_integer_new(config->nsizes));
        json_value *arr = json_array_new(config->nsizes);
        for (int j=0; j<config->nsizes; j++) {
            json_array_push( arr, json_integer_new(config->size_list[j]) );
        }
        json_object_push(root, "buf_sizes", arr);

        json_object_push(root, "ndepths", json_integer_new(config->ndepths));
        arr = json_array_new(config->ndepths);
        for (int j=0; j<config->nsizes; j++) {
            json_array_push( arr, json_integer_new(config->depth_list[j]) );
        }
        json_object_push(root, "buf_depth", arr);
        json_object_push(root, "message_size", json_integer_new(config->bigsize));
        json_object_push(root, "nranks", json_integer_new(comm_size));
        res_array = json_array_new( config->ndepths * config->nsizes );
    }

    for (int jsize=0; jsize < config->nsizes; jsize++) {
        for (int jdepth=0; jdepth < config->ndepths; jdepth++) {
            size_t jresult = jsize*config->ndepths*config->ntrials + jdepth*config->ntrials + 0;

            MPI_Gather(&config->results[jresult], config->ntrials, dtype,
                    allresults, config->ntrials, dtype, 0, MPI_COMM_WORLD);

            if (comm_rank == 0) {
                json_value *json_res_bndw, *json_res_copy, *json_res_wait, *json_res_post, *json_res_elap;
                json_value *js_jrank, *js_jtrial;
                json_value *res_root;
                size_t nitems = config->ntrials*comm_size;
                json_res_bndw = json_array_new(nitems);
                json_res_copy = json_array_new(nitems);
                json_res_wait = json_array_new(nitems);
                json_res_post = json_array_new(nitems);
                json_res_elap = json_array_new(nitems);
                js_jrank = json_array_new(nitems);
                js_jtrial = json_array_new(nitems);
                res_root = json_object_new(0);
                for (int j=0; j<nitems; j++) {
                    json_array_push(json_res_bndw, json_double_new(allresults[j].bandwidth));
                    json_array_push(json_res_copy, json_double_new(allresults[j].cum_copy_time));
                    json_array_push(json_res_wait, json_double_new(allresults[j].cum_wait_time));
                    json_array_push(json_res_post, json_double_new(allresults[j].cum_send_time));
                    json_array_push(json_res_elap, json_double_new(allresults[j].elapsed_time));
                    json_array_push(js_jrank,  json_integer_new(allresults[j].jrank));
                    json_array_push(js_jtrial, json_integer_new(allresults[j].jtrial));
                }
                json_object_push( res_root, "buffer_size", json_integer_new(config->size_list[jsize]));
                json_object_push( res_root, "buffer_depth", json_integer_new(config->depth_list[jdepth]));
                json_object_push( res_root, "bandwidths", json_res_bndw);
                json_object_push( res_root, "copys", json_res_copy);
                json_object_push( res_root, "waits", json_res_wait);
                json_object_push( res_root, "posts", json_res_post);
                json_object_push( res_root, "totals", json_res_elap);
                json_object_push( res_root, "jranks", js_jrank);
                json_object_push( res_root, "jtrials", js_jtrial);

                json_array_push(res_array, res_root);
            }
        }
    }

    if (comm_rank == 0) {
        json_object_push(root, "results", res_array);

        char *jsonbuf = malloc(json_measure(root));
        json_serialize(jsonbuf, root);
        FILE *fp = fopen(config->timing_file, "w");
        fprintf(fp, "%s\n", jsonbuf);
        fclose(fp);
        free(jsonbuf);
    }

    MPI_Type_free(&dtype);
}

void print_help() {
    int comm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    if (comm_rank==0) {
        fprintf(stderr, "Usage: bench_pipeline [options...]\n");
        fprintf(stderr, "  -m size              Size of the full message in bytes\n");
        fprintf(stderr, "  -b bufsizes          Size of the buffer to use.  See LIST_FORMAT\n");
        fprintf(stderr, "  -t trials            Number of Trials.\n");
        fprintf(stderr, "  -d depths            Pipeline Depth. See LIST_FORMAT\n");
        fprintf(stderr, "  -f filename          Dump all data to file.\n");
        fprintf(stderr, "  -h                   Print this help.\n");
        fprintf(stderr, "\n");
        fprintf(stderr, "Careful: the message size will be sent a total of nbufsizes*ndepths*ntrails times!\n");
        fprintf(stderr, "\n");
        fprintf(stderr, "LIST_FORMAT: lists can be specified as single value, or a arithmetic or geometric progression\n");
        fprintf(stderr, "     =value          e.g. =1024    produces  [1024]\n");
        fprintf(stderr, "     min+step+max    e.g. 10+5+25  produces  [10,15,20,25]\n");
        fprintf(stderr, "     min*factor*max  e.g. 8*2*128  produces  [8,16,32,64,128]\n");
        fprintf(stderr, "This test requires an even number of ranks.  The lower half will send to the upper half.\n");
    }
}

int parse_list_format(char* arg, int *nitems, size_t **items) {
    char char1, char2;
    size_t min,middle,max;
    size_t *xms;

    if (arg[0] == '=') {
        *nitems = 1;
        (*items) = malloc(*nitems * sizeof(*items));
        (*items)[0] = atol(arg+1);
        return 0;
    }

    sscanf(arg, "%ld%c%ld%c%ld",&min, &char1, &middle, &char2, &max);

    if (char1 != char2) {
        fprintf(stderr, "While parsing %s, chars don't match in LIST_FORMAT.  Expected matching * or +\n",arg);
        return 1;
    }
    if (char1 == '+') {
        *nitems = (max-min) / middle + 1;
        xms = malloc(*nitems * sizeof(*items));
        xms[0] = min;
        for (int jitem=1; jitem<*nitems; jitem++) {
            xms[jitem] = xms[jitem-1] + middle;
        }
    }
    if (char1 == '*') {
        *nitems=0;
        size_t tmp;
        for (tmp=min; tmp<=max; tmp*=middle) {
            (*nitems)++;
        }
        xms = malloc(*nitems * sizeof(*items));
        xms[0] = min;
        for (int jitem=1; jitem<*nitems; jitem++) {
            xms[jitem] = xms[jitem-1] * middle;
        }
    }
    *items = xms;
    return 0;
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
    config.nsizes = 0;
    config.ndepths = 0;
    config.timing_file = "results_bench_pipeline.json";


    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    {
        static int attach_done=0;
        int comm_rank;
        if (attach_done) goto resume;
        char *dbg_rank = getenv("DEBUG_RANK");
        if (!dbg_rank) goto resume;
        // if (atoi(dbg_rank) != ompi_comm_rank(MPI_COMM_WORLD)) goto resume;
        MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
        if (atoi(dbg_rank) != comm_rank) goto resume;

        volatile int i = 0;
        char hostname[256];
        sleep(1);
        gethostname(hostname, sizeof(hostname));
        printf("PID %d on %s ready for attach\n", getpid(), hostname);
        fflush(stdout);
        while (0 == i) {
            sleep(1);
        }
resume: attach_done = 1;
    }


    while ((opt = getopt(argc, argv, "m:b:t:d:f:h")) != -1) {
        switch (opt) {
        case 'f': config.timing_file = optarg; break;
        case 'd':
            retval = parse_list_format(optarg, &config.ndepths, &config.depth_list);
            if (retval) goto end;
            break;
        case 'b':
            retval = parse_list_format(optarg, &config.nsizes, &config.size_list);
            if (retval) goto end;
            break;
        case 't': config.ntrials = atol(optarg); break;
        case 'm': config.bigsize = atol(optarg); break;
        case 'h':
            retval=0;
        default:
            print_help();
            goto end;
        }
    }

    if (config.nsizes == 0) {
        parse_list_format("=1024", &config.nsizes, &config.size_list);
    }
    if (config.ndepths == 0) {
        parse_list_format("=2", &config.ndepths, &config.depth_list);
    }

    if (comm_size%2 != 0) {
        fprintf(stderr, "This test requires even number of ranks.\n");\
        goto end;
    }
    retval = create_buffers(&config);

    if (retval == 0) {
        run_test(&config);
        save_results(&config);
    }

    if (retval) {
        printf("Rank %d experienced error!\n",comm_rank);
    } else if (comm_rank == 0) {
        printf("Test completed: results in %s\n",config.timing_file);
    }

    destroy_buffers(&config);
end:
    MPI_Finalize();
    return retval;
}
