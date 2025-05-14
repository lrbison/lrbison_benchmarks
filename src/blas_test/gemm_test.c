#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <omp.h>
#include "utils.h"
#include <getopt.h>

// #include "cblas.h"

#include <mpi.h>

#include "json-builder.h"

/*  
    Why the two functions? See:
    https://gcc.gnu.org/onlinedocs/cpp/Stringizing.html 
    https://gcc.gnu.org/onlinedocs/cpp/Argument-Prescan.html
*/
#define xstr(s) str(s)
#define str(s) #s

#define BLASFUNC(FUNC) FUNC ## _
#define MACRO_BLASFUNC(FUNC) BLASFUNC(FUNC)

#define OPT_DTYPE_F32 1
#define OPT_DTYPE_F64 2
#define OPT_DTYPE_Z32 3
#define OPT_DTYPE_Z64 3

#define SELECTED_DTYPE OPT_DTYPE_F32

/* ==== 32-bit float ==== */
#if SELECTED_DTYPE == OPT_DTYPE_F32
#define GEMM_FUN sgemm
#define DTYPE float
#define DTYPE_PREFIX f32
#endif

/* ==== 64-bit float ==== */
#if SELECTED_DTYPE == OPT_DTYPE_F64
#define GEMM_FUN dgemm
#define DTYPE double
#define DTYPE_PREFIX f64
#endif


extern void MACRO_BLASFUNC(GEMM_FUN)();

uint64_t xorshift64star(void);

int count_cpus();
static void print_help() {
    int comm_size;
    printf("Command line Options:\n");
    printf(" --ranks|-r <list>         \t\t Run test while scanning through this many Ranks\n");
    printf(" --mat-sizes|-m <list>     \t\t Test these matrix sizes.  See -x\n");
    printf(" --all-mat-combos|-x       \t\t Without -x M=N=K at all test points.  With -x scan each matrix dimension independently.\n");
    printf("                           \t\t WARNING: with -x, test length grows by length of mat-sizes^3 power!\n");
    printf(" --cache-smashes|-c <list> \t\t Increase working set size of each test by given size in KiB.\n");
    printf(" --target_milisec|-T <MSEC>\t\t Repeat each test until MSEC many miliseconds have elapsed.\n");
    printf(" --help|-h                 \t\t Print this help.\n");
    printf(" --tag|-t <tag>            \t\t write results to file with <tag> in the name.\n");
    printf("\n");
    printf("For each argument taking a <list>, it may take the following forms:\n");
    printf("  lin:<start>,<increment>,<stop> \t\t A linearly incrementing list\n");
    printf("  list:<a>,<b>,...<n>            \t\t A user-defined list of values\n");
    printf("  <value>                        \t\t Just a single value.\n");
    printf("\n");
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    printf("This run was invoked with %d MPI Ranks.  Rank 0 sees %d OMP threads (OMP_NUM_THREADS).\n", comm_size, count_cpus());
    printf("Open MPI Note: --map-by-core will fill the node with single-threaded ranks bound to each core.\n");
    printf("Open MPI Note: --map-by ppr:<NRANKS-PER-NODE>:node:pe=<NTHREADS-PER-RANK> can be used for multi-threading.\n");
    printf("This binary was compiled using the %s library using %s precision.\n",xstr(BLAS_LIB),xstr(DTYPE));
}

struct dgemm_parms {
    char *transa;
    char *transb;
    int *m, *n, *k;
    DTYPE *alpha, *beta;
    DTYPE *a, *b, *c;
    int *lda, *ldb, *ldc;
};

static struct dgemm_parms dgemm_parms;

static uint64_t dgemm_max_shift=0;

void call_dgemm_once() {
    int a_off=0, b_off=0, c_off=0;
    if (dgemm_max_shift > 0) {
        a_off = (xorshift64star() % dgemm_max_shift) / sizeof(DTYPE);
        b_off = (xorshift64star() % dgemm_max_shift) / sizeof(DTYPE);
        c_off = (xorshift64star() % dgemm_max_shift) / sizeof(DTYPE);
    }
    
    MACRO_BLASFUNC(GEMM_FUN)(dgemm_parms.transa, dgemm_parms.transb,
        dgemm_parms.m, dgemm_parms.n, dgemm_parms.k,
        dgemm_parms.alpha,
        &dgemm_parms.a[a_off], dgemm_parms.lda,
        &dgemm_parms.b[b_off], dgemm_parms.ldb,
        dgemm_parms.beta,
        &dgemm_parms.c[c_off],
        dgemm_parms.ldc );
}

int count_cpus() {
    int nthreads = 1;
    #pragma omp parallel
    {
        #pragma omp master
        {
            nthreads = omp_get_num_threads();
        }
    }
    return nthreads;
}

void run_many_times( void (*func_ptr)(), double target_usec, int warmup, int64_t *count, double *avg_usec ) {
    struct timespec t0;
    double t_tot;

    for (int kbatch=0; kbatch<warmup; kbatch++) {
        func_ptr();
    }
    timer_start(&t0);
    *count = 0;
    int next_batch = 1;
    while (1) {
        for (int kbatch=0; kbatch<next_batch; kbatch++) {
            func_ptr();
        }
        *count += next_batch;
        t_tot = timer_stop(&t0);
        *avg_usec = t_tot / *count;
        next_batch = 0.8 * (target_usec - t_tot)/ *avg_usec;
        if (next_batch < 1) break;
    }
}

static uint64_t xorshift_state = 1;
/* use this simple xorshift for a dead-simple and repeatable fast PRNG. */
/* xorshift64s, variant A_1(12,25,27) with multiplier M_32 from line 3 of table 5 */
uint64_t xorshift64star(void) {
    /* initial seed must be nonzero, don't use a static variable for the state if multithreaded */
    xorshift_state ^= xorshift_state >> 12;
    xorshift_state ^= xorshift_state << 25;
    xorshift_state ^= xorshift_state >> 27;
    return xorshift_state * 0x2545F4914F6CDD1DULL;
}

static int parse_opts_list(char* optarg, int *num_ax, int **ax) {
	int i, ret;
	char *token;
	char *saveptr;

	*num_ax = 1;
	for (i = 0; optarg[i] != '\0'; i++) {
		*num_ax += optarg[i] == ',';
	}
	*ax = calloc(*num_ax, sizeof(**ax));
	token = strtok_r(optarg, ",", &saveptr);
	*num_ax = 0;
	while (token != NULL) {
		ret = sscanf(token, "%u", &((*ax)[*num_ax]));
		if (ret != 1) {
			fprintf(stderr, "Cannot parse integer \"%s\" in list.\n",token);
			return 1;
		}
        (*num_ax)++;
		token = strtok_r(NULL, ",", &saveptr);
	}
    return 0;
}

static int parse_opts_range(char* optarg, int *num_ax, int**ax) {
	int start, inc, end;
	int i, ret;

	ret = sscanf(optarg, "%d,%d,%d", &start, &inc, &end);
	if (ret != 3) {
		perror("sscanf");
		return 1;
	}
	assert(end >= start && inc > 0);
	*num_ax = (end - start) / inc + 1;
	*ax = calloc(*num_ax, sizeof(**ax));

    /* linear axis */
    for (i = 0; i < *num_ax && i <= end; i++) {
        (*ax)[i] = start + (i * inc);
    }
    *num_ax = i;
    return 0;
}


static int parse_axis_string( const char *optarg_const, int* num_ax, int **ax) {
    int rc;
    char *optarg = strdup(optarg_const);
    if (!strncasecmp("lin:", optarg, 4)){
        rc = parse_opts_range(&optarg[4], num_ax, ax);
    } else if (!strncasecmp("list:", optarg, 5)){
        rc = parse_opts_list(&optarg[5], num_ax, ax);
    } else {
        rc = parse_opts_list(optarg, num_ax, ax);
    }
    
    if (rc)
        printf("Did not understand axis: %s. Expected lin:<start>,<inc>,<stop> or list:<a>,<b>,...,<n> or just a single <Value>", optarg);
    free(optarg);
    return rc;
}


static void initrands(int nelem, DTYPE *arr) {
    double u64_max = UINT64_MAX;

    for (int i=0; i<nelem; i++) {
        arr[i] = (xorshift64star()/u64_max)-0.5;
    }
}

int main(int argc, char** argv) {
    DTYPE *A, *B, *C;
    
    /* user arguments: */
    int num_mat_ax = -1;
    int num_cache_ax = -1;
    int num_rank_ax = -1;
    int *cache_ax = NULL;
    int *mat_ax = NULL;
    int *rank_ax = NULL;
    int do_all_mat_combos = 0;
    double user_msec_target = 1.0;
    double user_usec_target;



    int64_t run_count, run_count_sum;
    double avg_usec, avg_usec_sum;
    char results_file[256];
    char *user_tag = "";
    

    int comm_rank, comm_size;
    

    MPI_Init(&argc, &argv);

    comm_rank = 0;
    comm_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    

    while (1) {
        int c;
        int option_index;
        static struct option long_options[] =
        {
            {"ranks",           required_argument,  NULL,   'r'},
            {"mat-sizes",       required_argument,  NULL,   'm'},
            {"all-mat-combos",  no_argument,        NULL,   'x'},
            {"cache-smashes",   required_argument,  NULL,   'c'},
            {"help",            no_argument,        NULL,   'h'},
            {"tag",             required_argument,  NULL,   't'},
            {"target-msec",     required_argument,  NULL,   'T'},
            {NULL,              0,                  NULL,    0 }
        };

        c = getopt_long(argc, argv, "r:m:xc:ht:T:", long_options, &option_index);
        if (c == -1)
            break;
        switch (c)
        {
            case 'r':
                if (parse_axis_string( optarg, &num_rank_ax, &rank_ax ))
                    goto error_exit;
                break;
            case 'm':
                if (parse_axis_string( optarg, &num_mat_ax, &mat_ax ))
                    goto error_exit;
                break;
            case 'x':
                do_all_mat_combos = 1;
                break;
            case 'c':
                if (parse_axis_string( optarg, &num_cache_ax, &cache_ax ))
                    goto error_exit;
                break;
            case 'h':
                if (comm_rank==0)
                    print_help();
                goto happy_exit;
            case 't':
                user_tag = optarg;
                break;
            case 'T':
                user_msec_target = atof(optarg);
                break;
            default:
                break;
        }
    }

    if (num_mat_ax<1) {
        if (parse_axis_string("list:8,64,256,1024", &num_mat_ax, &mat_ax))
            goto error_exit;
    }
    if (num_cache_ax<1) {
        if (parse_axis_string("0", &num_cache_ax, &cache_ax))
            goto error_exit;
    }
    if (num_rank_ax<1) {
            char tmps[64];
            sprintf(tmps, "lin:1,1,%d", comm_size);
            if (comm_rank==0) printf("input for rank axis: %s\n",tmps);
            if (parse_axis_string(tmps, &num_rank_ax, &rank_ax))
                goto error_exit;
    }
    /* convert cache_ax from KiB to bytes*/
    for (int jc=0; jc<num_cache_ax; jc++) {
        cache_ax[jc] = cache_ax[jc]*1024;
    }

    if (user_tag && strlen(user_tag)) {
        snprintf(results_file, 255, "results-%s-%s.json", xstr(BLAS_LIB), user_tag);
    } else {
        snprintf(results_file, 255, "results-%s.json", xstr(BLAS_LIB));
    }
    results_file[255] = '\0';
    
    size_t max_dim = mat_ax[num_mat_ax-1];
    size_t max_mat = max_dim * max_dim;


    if (comm_rank == 0) {
        printf("Benchmarking gemm for BLAS: %s into %s\n", xstr(BLAS_LIB),results_file );
        if (3*max_mat/1024/1024/1024 < 4) {
            printf("Max Matrix dim: %ld.  Requires %ld KiB/rank, %ld GiB Total.\n",
                max_dim, 3*max_mat/1024, comm_size*3*max_mat/1024/1024/1024);
        } else {
            printf("Max Matrix dim: %ld.  Requires %ld GiB/rank, %ld GiB Total\n",
                max_dim, 3*max_mat/1024/1024/1024, comm_size*3*max_mat/1024/1024/1024);
        }
        printf("Max Cache smashing requires additional %ld MiB/rank, %ld GiB Total\n",
                (long)3*cache_ax[num_cache_ax-1]/1024/1024, (long)comm_size*3*cache_ax[num_cache_ax-1]/1024/1024/1024);
        double total_tests = num_rank_ax*num_cache_ax*num_mat_ax;
        if (do_all_mat_combos)
            total_tests *= num_mat_ax*num_mat_ax;
        printf("Requesting %.0f total tests at %.1f ms each, minimum time to complete: %.2f minutes\n",
            total_tests, user_msec_target, total_tests*user_msec_target/1000./60.);
    }
    sleep(1);

    /* adjust if LDA(>=M) or LDB(>=K) is modified */
    if (comm_rank == 0) {
        printf("Allocating matrix for N=(%lu) elements: NxN = %ld\n", max_dim, max_mat);
    }
    xorshift_state = comm_rank+1;
    A = (DTYPE*) malloc( max_mat * sizeof(DTYPE) + cache_ax[num_cache_ax-1] );
    B = (DTYPE*) malloc( max_mat * sizeof(DTYPE) + cache_ax[num_cache_ax-1] );
    C = (DTYPE*) malloc( max_mat * sizeof(DTYPE) + cache_ax[num_cache_ax-1] );
    initrands( max_mat + cache_ax[num_cache_ax-1]/sizeof(DTYPE), A );
    initrands( max_mat + cache_ax[num_cache_ax-1]/sizeof(DTYPE), B );
    initrands( max_mat + cache_ax[num_cache_ax-1]/sizeof(DTYPE), C );

    DTYPE alpha, beta;
    user_usec_target = user_msec_target*1000.;

    json_value *root;
    json_value *axis_arr;
    json_value *result_arr;
    json_value *result_entry;

    root = json_object_new(0);

    axis_arr = json_array_new(num_mat_ax);
    for (int j=0; j<num_mat_ax; j++) {
        json_array_push( axis_arr, json_integer_new(mat_ax[j]));
    }
    json_object_push( root, "mat_ax", axis_arr );

    axis_arr = json_array_new(num_rank_ax);
    for (int j=0; j<num_rank_ax; j++) {
        json_array_push( axis_arr, json_integer_new(rank_ax[j]));
    }
    json_object_push( root, "mpi_rank_ax", axis_arr );

    axis_arr = json_array_new(num_cache_ax);
    for (int j=0; j<num_cache_ax; j++) {
        json_array_push( axis_arr, json_integer_new(cache_ax[j]));
    }
    json_object_push( root, "cache_ax", axis_arr );

    json_object_push( root, "precision", json_string_new( xstr(DTYPE) ) );
    json_object_push( root, "blas", json_string_new( xstr(BLAS_LIB) ) );
    json_object_push( root, "blas_fun", json_string_new( xstr(GEMM_FUN) ) );
    json_object_push( root, "OMP_NUM_THREADS", json_integer_new(count_cpus()));
    json_object_push( root, "MPI_ranks", json_integer_new(comm_size));
    json_object_push( root, "user_tag", json_string_new(user_tag) );
    json_object_push( root, "do_all_mat_combos", json_integer_new(do_all_mat_combos) );
    json_object_push( root, "target_milisec", json_double_new(user_msec_target) );
    

    result_arr = json_array_new(0);
    json_object_push( root, "results", result_arr );

    



    // C := A*B
    // A is [M x K]
    // B is     [K x N]
    // C is [M   x   N]
    int jM, jN, jK;
    int num_mat_combos;
    if (do_all_mat_combos == 0) {
        num_mat_combos = num_mat_ax;
    } else {
        num_mat_combos = num_mat_ax * num_mat_ax * num_mat_ax;
    }

    for (int jmat_combo=0; jmat_combo<num_mat_combos; jmat_combo++) {
        if (do_all_mat_combos == 0) {
            jM = jmat_combo;
            jN = jmat_combo;
            jK = jmat_combo;
        } else {
            int tmp = jmat_combo;
            jM = tmp % num_mat_ax; tmp = tmp / num_mat_ax;
            jN = tmp % num_mat_ax; tmp = tmp / num_mat_ax;
            jK = tmp % num_mat_ax; tmp = tmp / num_mat_ax;
        }
        if (comm_rank==0) printf("-----------\n");
    for (int jrank_ax=0; jrank_ax<num_rank_ax; jrank_ax++) {
    for (int jcache_ax=0; jcache_ax<num_cache_ax; jcache_ax++) {
        
        int nranks_working = MIN(rank_ax[jrank_ax], comm_size);
        dgemm_parms.transa = "N";
        dgemm_parms.transb = "N";
        dgemm_parms.m = &mat_ax[jM];
        dgemm_parms.n = &mat_ax[jN];
        dgemm_parms.k = &mat_ax[jK];
        alpha = -1.0;
        beta = 1.0;
        dgemm_parms.alpha = &alpha;
        dgemm_parms.beta = &beta;
        dgemm_parms.a = A;
        dgemm_parms.b = B;
        dgemm_parms.c = C;
        dgemm_parms.lda = dgemm_parms.m;
        dgemm_parms.ldb = dgemm_parms.k;
        dgemm_parms.ldc = dgemm_parms.m;
        dgemm_max_shift = cache_ax[jcache_ax];
        MPI_Barrier( MPI_COMM_WORLD );
        if (comm_rank+1 <= nranks_working) {
            run_many_times( call_dgemm_once, user_usec_target, 0, &run_count, &avg_usec);
        } else {
            avg_usec = 0.0;
            run_count = 0;
        }
        MPI_Reduce( &avg_usec, &avg_usec_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce( &run_count, &run_count_sum, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);


        if (comm_rank == 0) {

            double avg_avg_usec = avg_usec_sum / nranks_working;
            double avg_run_count = (double)run_count_sum / (double)nranks_working;
            double est_flo = 2.0 * (double)*dgemm_parms.m * (double)*dgemm_parms.n * (double)*dgemm_parms.k;
            double est_gflops = 1e-3 * est_flo / (avg_avg_usec);
            uint64_t wss = dgemm_max_shift*3 + sizeof(DTYPE)*((mat_ax[jM] * mat_ax[jK]) + (mat_ax[jK] * mat_ax[jN]) + (mat_ax[jM] * mat_ax[jN]));
            
            printf("[%d/%d]: m=%6d n=%6d, k=%6d (%8.1f in %8.3f usec - %8.3f GFLOPS/rank). WSS=%ld MiB, Smash=%ld KiB\n",
                nranks_working, comm_size,
                *dgemm_parms.m, *dgemm_parms.n, *dgemm_parms.k, avg_run_count, avg_run_count*avg_avg_usec, est_gflops, wss/1024/1024, dgemm_max_shift/1024);

            result_entry = json_object_new(0);
            json_object_push( result_entry, "m", json_integer_new(mat_ax[jM]));
            json_object_push( result_entry, "n", json_integer_new(mat_ax[jN]));
            json_object_push( result_entry, "k", json_integer_new(mat_ax[jK]));
            json_object_push( result_entry, "cache_smash_bytes", json_integer_new(dgemm_max_shift));
            json_object_push( result_entry, "ranks_working", json_integer_new(nranks_working));
            json_object_push( result_entry, "gflops", json_double_new(est_gflops));
            json_object_push( result_entry, "latency_usec", json_double_new(avg_avg_usec));
            json_object_push( result_entry, "count", json_double_new(avg_run_count));
            json_object_push( result_entry, "wss_bytes", json_integer_new(wss));
            json_array_push( result_arr, result_entry );
        }
    }
    }
    }

    if (comm_rank == 0) {
        char *jsonbuf = malloc(json_measure(root));
        json_serialize(jsonbuf, root);
        FILE *fp = fopen(results_file, "w");
        fprintf(fp, "%s\n", jsonbuf);
        fclose(fp);
        free(jsonbuf);
        printf("json-formatted results can be found in %s\n",results_file);
    }

happy_exit:
    MPI_Finalize();
    exit(0);

error_exit:
    MPI_Finalize();
    exit(1);
}
