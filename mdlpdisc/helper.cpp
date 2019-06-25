#include <cstdint>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <omp.h>
#include "helper.h"

template <typename T, typename U>
void sort_two_vecs(T x[], U y[], size_t n, T xo[], U yo[]) {
    size_t *p = (size_t*) malloc(n * sizeof(size_t));
    std::iota(p, p + n, 0);
    std::sort(p, p + n, [&](size_t i, size_t j){return x[i] < x[j];});

    std::transform(p, p + n, xo, [&](std::size_t i){return x[i];});
    std::transform(p, p + n, yo, [&](std::size_t i){return y[i];});
    free(p);
}

template <typename T>
void get_sliced_entropy(T y[], size_t low, size_t upp,
                        size_t& k, double& entropy) {
    size_t n = upp - low;
    T *ymax = std::max_element(y + low, y + upp);
    size_t *freq = (size_t*) malloc((*ymax+1)*sizeof(size_t));
    std::fill(freq, freq + *ymax+1, 0);
    for (size_t i = low; i < upp; ++i)
        *(freq + y[i]) += 1;

    k = 0;
    entropy = 0.0;
    double temp;
    for (size_t i = 0; i <= (size_t) *ymax; ++i) {
        if (*(freq+i) > 0) {
            k++;
            temp = (double) *(freq+i) / (double) n;
            entropy -= temp * log(temp);
        }
    }
    free(freq);
}

inline double xlogx(double f) {
    return f*log(f);
}

template <typename T>
void get_sliced_entropy_para(T y[], size_t low, size_t upp,
                             size_t& k, double& entropy) {
    int64_t i;
    size_t n = upp - low;
    T *ymax = std::max_element(y + low, y + upp);
    size_t arr_len = (size_t) *ymax + 1, *freq_local;
    size_t *freq = (size_t*) malloc(arr_len*sizeof(size_t));
    std::fill(freq, freq + arr_len, 0);
    size_t count = 0;
    double ent = 0.0;

    #pragma omp parallel private(i, freq_local)
    {
        freq_local = (size_t*) malloc(arr_len*sizeof(size_t));
        std::fill(freq_local, freq_local + arr_len, 0);
        #pragma omp for
        for (i = low; i < (int64_t) upp; ++i)
            *(freq_local + y[i]) += 1;

        #pragma omp critical
        for (i = 0; i < (int64_t) arr_len; ++i)
            *(freq+i) += *(freq_local+i);
        free(freq_local);
        
        #pragma omp barrier
    	#pragma omp for reduction(+:count) reduction(-:ent)
        for (i = 0; i < (int64_t) arr_len; ++i) {
            if (*(freq+i) > 0) {
                count++;
                ent -= xlogx((double) *(freq+i) / (double) n);
            }
        }
    }
    k = count;
    entropy = ent;
    free(freq);
}
