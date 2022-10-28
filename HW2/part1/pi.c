#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <immintrin.h>
#include "simdxorshift128plus.h"


pthread_mutex_t total_hit_mutex;
long long int total_hit = 0;

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
void *single_thread_estimation_SIMD(void *params){
    uint64_t simd_seed[2];
    
    long long int hit = 0, num_toss_simd = (long long int) params / 8;
    // int32_t sub_hit[8] __attribute__((aligned(32)));
    int sub_hit[8]; // TBD sizeof(int) == 4
    __m256i x_i_vec, y_i_vec, cmp_i_vec, cnt_i_vec;
    __m256 x_ps_vec, y_ps_vec, dist_ps_vec, intmax_ps_vec, one_ps_vec, cmp_ps_vec, mask_i_vec;

    mask_i_vec = _mm256_set_ps(0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001);
    cnt_i_vec = _mm256_set_epi32(0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000);
    intmax_ps_vec = _mm256_set_ps(INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX, INT32_MAX);
    one_ps_vec = _mm256_set_ps(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);

    avx_xorshift128plus_key_t mykey;
    // simd_seed must be non-zero (1 ~ 2^63)
    simd_seed[0] = rand(); // TBD
    simd_seed[1] = rand(); // TBD
    avx_xorshift128plus_init(simd_seed[0], simd_seed[1], &mykey);

    // single precision & avx2 & 8 single toss each time
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
        cmp_ps_vec = _mm256_cmp_ps(dist_ps_vec, one_ps_vec, _CMP_LE_OS); // TBD _CMP_LE_OQ
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
    pthread_mutex_init(&total_hit_mutex, NULL);
    for (int i=0; i<num_thread; i++){
        pthread_create(&thread_handles[i], NULL, single_thread_estimation_SIMD, (void*) total_thread_toss);
        // pthread_create(&thread_handles[i], NULL, single_thread_estimation_serial, (void*) total_thread_toss);
    }

    for (int i=0; i<num_thread; i++){
        pthread_join(thread_handles[i], NULL);
    }

    est_pi = 4 * total_hit / ((double) total_toss);
    printf("%lf\n", est_pi);

    pthread_mutex_destroy(&total_hit_mutex);
    free(thread_handles);
}