#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <immintrin.h>
#include "simdxorshift128plus.h"


pthread_mutex_t total_hit_mutex;
long long int total_hit = 0;

typedef struct {
    int seed_1;
    int seed_2;
    long long int total_toss;
} Thread_Arg;

// Serial version
void *single_thread_estimation_serial(void *params){
    unsigned int seed = time(NULL);

    long long int hit = 0, num_toss = (long long int) params;
    double x, y, dist;
    for (long long int toss=0; toss<num_toss; toss++){
        x = ((double) rand_r(&seed) / RAND_MAX );
        y = ((double) rand_r(&seed) / RAND_MAX );
        dist = x * x + y * y;
        if (dist <= 1){
            hit++;
        }
    }
    pthread_mutex_lock(&total_hit_mutex);
    total_hit += hit;
    pthread_mutex_unlock(&total_hit_mutex);
    return NULL;
}

// SIMD version
void *single_thread_estimation_SIMD(void *thread_arg){
    Thread_Arg* arg = (Thread_Arg *) thread_arg;
    long long int hit = 0, num_toss_simd = (arg->total_toss) / 8;
    int sub_hit[8];
    __m256i x_i_vec, y_i_vec, cmp_i_vec, cnt_i_vec;
    // TBD const 
    __m256 x_ps_vec, y_ps_vec, dist_ps_vec, intmax_ps_vec, one_ps_vec, cmp_ps_vec, mask_i_vec;

    cnt_i_vec = _mm256_set_epi32(0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000);
    intmax_ps_vec = _mm256_set_ps(INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX);
    one_ps_vec = _mm256_set_ps(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
    mask_i_vec = _mm256_set_ps(0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001);

    avx_xorshift128plus_key_t mykey;
    // simd_seed must be non-zero, zero is guarnteed barely assigned
    avx_xorshift128plus_init(arg->seed_1, arg->seed_1, &mykey);

    // single precision estimation & avx2 intrinsic (8 tosses each time)
    for (long long int toss=0; toss<num_toss_simd; toss++){
        // random number & type conversion & squaring
        x_i_vec = avx_xorshift128plus(&mykey);
        x_ps_vec = _mm256_cvtepi32_ps(x_i_vec);
        x_ps_vec = _mm256_div_ps(x_ps_vec, intmax_ps_vec);
        x_ps_vec = _mm256_mul_ps(x_ps_vec, x_ps_vec);

        y_i_vec = avx_xorshift128plus(&mykey);
        y_ps_vec = _mm256_cvtepi32_ps(y_i_vec);
        y_ps_vec = _mm256_div_ps(y_ps_vec, intmax_ps_vec);
        y_ps_vec = _mm256_mul_ps(y_ps_vec, y_ps_vec);

        // distance
        dist_ps_vec = _mm256_add_ps(x_ps_vec, y_ps_vec);
        
        // counting
        cmp_ps_vec = _mm256_cmp_ps(dist_ps_vec, one_ps_vec, _CMP_LE_OS);
        cmp_ps_vec = _mm256_and_ps(cmp_ps_vec, mask_i_vec);
        cmp_i_vec = _mm256_cvtps_epi32(cmp_ps_vec);
        cnt_i_vec = _mm256_add_epi32(cnt_i_vec, cmp_i_vec);
    }

    // intra sub_hit summation
    _mm256_store_si256((__m256i *) &sub_hit, cnt_i_vec);
    for (int i=0; i<8; i++){
        hit += sub_hit[i];
    }

    pthread_mutex_lock(&total_hit_mutex);
    total_hit += hit;
    pthread_mutex_unlock(&total_hit_mutex);
    return NULL;
}

int main(int argc, char **argv){
    // check argument format
    if (argc != 3){
        printf("Usage of command: ./pi.out #Thread #Tosses\n");
        printf("e.g. ./pi.out 8 1000000000\n");
        return 1;
    }

    // parsing arguments
    int num_thread = atoi(argv[1]);
    long long int total_toss = atoi(argv[2]);
    long long int total_thread_toss = total_toss / num_thread;
    double est_pi;

    // multi-thread execution using Pthread
    pthread_t *thread_handles = (pthread_t *) malloc(num_thread * sizeof(pthread_t));
    Thread_Arg *thread_args = (Thread_Arg *) malloc(num_thread * sizeof(Thread_Arg));
    pthread_mutex_init(&total_hit_mutex, NULL);
    srand(time(NULL));
    for (int i=0; i<num_thread; i++){
        thread_args[i].seed_1 = rand();
        thread_args[i].seed_2 = rand();
        thread_args[i].total_toss = total_thread_toss;
        pthread_create(&thread_handles[i], NULL, single_thread_estimation_SIMD, (void*) &thread_args[i]);
    }

    for (int i=0; i<num_thread; i++){
        pthread_join(thread_handles[i], NULL);
    }

    est_pi = 4 * total_hit / ((double) total_toss);
    printf("%lf\n", est_pi);

    pthread_mutex_destroy(&total_hit_mutex);
    free(thread_handles);
}