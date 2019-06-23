#ifndef __HELPER_H__
#define __HELPER_H__

template <typename T, typename U>
void sort_two_vecs(T x[], U y[], size_t n, T xo[], U yo[]);

template <typename T>
void get_sliced_entropy2(T y[], size_t low, size_t upp,
                         size_t& k, double& entropy);

template <typename T>
void get_sliced_entropy_para(T y[], size_t low, size_t upp,
                             size_t& k, double& entropy);

#endif
