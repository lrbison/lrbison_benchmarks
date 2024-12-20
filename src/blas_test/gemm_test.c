#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <omp.h>
#include "utils.h"

// #include "cblas.h"

#include "json-builder.h"

/* https://gcc.gnu.org/onlinedocs/cpp/Stringizing.html */
#define xstr(s) str(s)
#define str(s) #s


#define BLASFUNC(FUNC) FUNC##_

extern void BLASFUNC(dgemm)();


struct dgemm_parms {
    char *transa;
    char *transb;
    int *m, *n, *k;
    double *alpha, *beta;
    double *a, *b, *c;
    int *lda, *ldb, *ldc;
};

static struct dgemm_parms dgemm_parms;

void call_dgemm_once() {
    BLASFUNC(dgemm)(dgemm_parms.transa, dgemm_parms.transb,
        dgemm_parms.m, dgemm_parms.n, dgemm_parms.k,
        dgemm_parms.alpha,
        dgemm_parms.a, dgemm_parms.lda,
        dgemm_parms.b, dgemm_parms.ldb,
        dgemm_parms.beta,
        dgemm_parms.c,
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

void run_many_times( void (*func_ptr)(), double target_usec, int warmup, int *count, double *avg_usec ) {
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



static void ilinspace(int nelem, int low, int high, int *arr) {
    double dlow = low;
    double dstep = (double)(high-low)/(nelem-1);
    for (int j=0; j<nelem; j++) {
        arr[j] = (int)(dlow + dstep*j);
    }
}

/* use this simple xorshift for a dead-simple and repeatable fast PRNG. */
/* xorshift64s, variant A_1(12,25,27) with multiplier M_32 from line 3 of table 5 */
uint64_t xorshift64star(void) {
    /* initial seed must be nonzero, don't use a static variable for the state if multithreaded */
    static uint64_t x = 1;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    return x * 0x2545F4914F6CDD1DULL;
}


static void initrands(int nelem, double *arr) {
    double u64_max = UINT64_MAX;

    for (int i=0; i<nelem; i++) {
        arr[i] = (xorshift64star()/u64_max)-0.5;
    }
}

int main() {
    double *A, *B, *C;
    const int nfine = 3;
    int *fine_steps;
    fine_steps = (int*)malloc( nfine * sizeof(*fine_steps));
    int coarse_taps[] = {0, nfine/2, nfine-1};
    const int ncoarse = sizeof(coarse_taps) / sizeof(int);
    ilinspace( nfine, 5, 5000, fine_steps);
    int run_count;
    double avg_usec;
    char results_file[256];
    snprintf(results_file, 255, "results-%s.json", xstr(BLAS_LIB));
    results_file[255] = '\0';

    uint64_t max_dim = fine_steps[nfine-1];
    uint64_t max_mat = max_dim * max_dim;
    int sparse_sampling = 1;

    printf("Benchmarking gemm for BLAS: %s\n", xstr(BLAS_LIB) );

    /* adjust if LDA(>=M) or LDB(>=K) is modified */
    A = (double*) malloc( max_mat * sizeof(double) );
    B = (double*) malloc( max_mat * sizeof(double) );
    C = (double*) malloc( max_mat * sizeof(double) );
    initrands( max_mat, A );
    initrands( max_mat, B );
    initrands( max_mat, C );

    double alpha, beta;

    json_value *root;
    json_value *axis_arr;
    json_value *result_arr;
    json_value *result_entry;

    root = json_object_new(0);

    axis_arr = json_array_new(nfine);
    for (int j=0; j<nfine; j++) {
        json_array_push( axis_arr, json_double_new(fine_steps[j]));
    }
    json_object_push( root, "fine_axis", axis_arr );
    json_object_push( root, "test_name", json_string_new("gemm") );
    json_object_push( root, "blas", json_string_new( xstr(BLAS_LIB) ) );
    json_object_push( root, "OMP_NUM_THREADS", json_integer_new(count_cpus()));

    result_arr = json_array_new(0);
    json_object_push( root, "results", result_arr );



    // C := A*B
    // A is [M x K]
    // B is     [K x N]
    // C is [M   x   N]

    for (int jM=0; jM<nfine; jM++) {
    for (int jN=0; jN<nfine; jN++) {
    for (int jK=0; jK<nfine; jK++) {
        int is_sparse_point = 1;
        for (int jtap=0; jtap < ncoarse; jtap++) {
            if (jM == coarse_taps[jtap])
                is_sparse_point = 0;
            if (jN == coarse_taps[jtap])
                is_sparse_point = 0;
            if (jK == coarse_taps[jtap])
                is_sparse_point = 0;
        }
        if (sparse_sampling && is_sparse_point)
            continue;

        dgemm_parms.transa = "N";
        dgemm_parms.transb = "N";
        dgemm_parms.m = &fine_steps[jM];
        dgemm_parms.n = &fine_steps[jN];
        dgemm_parms.k = &fine_steps[jK];
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
        run_many_times( call_dgemm_once, 1000.0, 0, &run_count, &avg_usec);
        double est_flo = 2.0 * (double)*dgemm_parms.m * (double)*dgemm_parms.n * (double)*dgemm_parms.k;
        double est_gflops = 1e-3 * est_flo / (avg_usec) ;
        printf("m=%6d n=%6d, k=%6d (%d in %8.3f usec - %8.3f GFLOPS)\n",
            *dgemm_parms.m, *dgemm_parms.n, *dgemm_parms.k, run_count, run_count*avg_usec, est_gflops);

        result_entry = json_object_new(0);
        json_object_push( result_entry, "m", json_integer_new(fine_steps[jM]));
        json_object_push( result_entry, "n", json_integer_new(fine_steps[jN]));
        json_object_push( result_entry, "k", json_integer_new(fine_steps[jK]));
        json_object_push( result_entry, "gflops", json_double_new(est_gflops));
        json_object_push( result_entry, "latency_usec", json_double_new(avg_usec));
        json_object_push( result_entry, "count", json_integer_new(run_count));
        json_array_push( result_arr, result_entry );
    }
    }
    }

    char *jsonbuf = malloc(json_measure(root));
    json_serialize(jsonbuf, root);
    FILE *fp = fopen(results_file, "w");
    fprintf(fp, "%s\n", jsonbuf);
    fclose(fp);
    free(jsonbuf);
    printf("json-formatted results can be found in %s\n",results_file);

}
