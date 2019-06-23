import numpy as np
cimport numpy as np
from cython import boundscheck, wraparound, cdivision
from cython.parallel import prange, parallel
from libc.stdlib cimport abort, malloc, free
from libc.math cimport INFINITY, isinf, log, pow
from libcpp cimport bool
from libcpp.vector cimport vector as cppvector
from libcpp.stack cimport stack as cppstack
from libcpp.algorithm cimport sort as cppsort

cdef struct interval:
    size_t low, upp 

cdef struct sliced_entropy:
    size_t uni_cnt
    double entropy
    
cdef struct cut_result:
    size_t cut_loc
    bool stop_cut

cdef extern from "helper.cpp" nogil:
    pass

cdef extern from "helper.h" nogil:
    void _sort_two_vecs "sort_two_vecs"[T, U](T*, U*, size_t, T*, U*)
    void _get_sliced_entropy "get_sliced_entropy"[T](
            T*, size_t, size_t, size_t&, double&)
    void _get_sliced_entropy_para "get_sliced_entropy_para"[T](
            T*, size_t, size_t, size_t&, double&)

cdef class mdlpdisc:
    cdef cppvector[size_t] _support
    cdef np.float64_t[::1] _x
    cdef np.int64_t[::1] _y

    @boundscheck(False)
    @wraparound(False)
    def __init__(self, np.float64_t[::1] x not None, np.int64_t[::1] y not None):
        cdef size_t n = x.shape[0]
        if y.shape[0] != n:
            raise Exception("The length of x must be equal to the length of y.")
        if np.amin(y) < 0:
            raise Exception("y must have no negative elements")

        self._x = np.empty(n, dtype=np.float64)
        self._y = np.empty(n, dtype=np.int64)
        _sort_two_vecs[np.float64_t, np.int64_t](&x[0], &y[0], n, &self._x[0], &self._y[0])

        cdef size_t i
        cdef np.float64_t* x_ptr = &self._x[0]
        for i in range(n-1):
            if (x_ptr+i)[0] != (x_ptr+i+1)[0]:
                self._support.push_back(i+1)

    def get_data(self):
        return np.asarray(self._x), np.asarray(self._y)

    def get_support(self):
        return self._support
    
    @boundscheck(False)
    @wraparound(False)
    cdef sliced_entropy _get_entropy(self, size_t low, size_t upp):
        cdef sliced_entropy se
        se.uni_cnt = 0
        se.entropy = 0.0
        if upp-low > 10000:
            _get_sliced_entropy_para(&self._y[0], low, upp, se.uni_cnt, se.entropy)
        else:
            _get_sliced_entropy(&self._y[0], low, upp, se.uni_cnt, se.entropy)
        return se

    @boundscheck(False)
    @wraparound(False)
    def get_entropy(self, size_t low, size_t upp):
        cdef sliced_entropy se = self._get_entropy(low, upp)
        return se.entropy, se.uni_cnt

    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    cdef cut_result _get_cut(self, size_t low, size_t upp):
        cdef:
            cut_result cr
            size_t i, curr_cut
            double n = <double> upp - <double> low, weight
            double curr_ent, prev_ent = INFINITY
            sliced_entropy se1, se2
        for curr_cut in self._support:
            if low < curr_cut < upp:
                weight = <double> (curr_cut-low)/n
                se1 = self._get_entropy(low, curr_cut)
                se2 = self._get_entropy(curr_cut, upp)
                curr_ent = weight * se1.entropy + (1.-weight) * se2.entropy
                if curr_ent < prev_ent:
                    prev_ent, cr.cut_loc = curr_ent, curr_cut

        cdef:
            sliced_entropy se
            double gain, delta
        if isinf(prev_ent):
            cr.cut_loc = 0
            cr.stop_cut = True
            return cr
        else:
            weight = <double> (cr.cut_loc-low)/n
            se = self._get_entropy(low, upp)
            se1 = self._get_entropy(low, cr.cut_loc)
            se2 = self._get_entropy(cr.cut_loc, upp)
            gain = se.entropy - (weight * se1.entropy + (1.-weight) * se2.entropy)
            delta = log(pow(3, <double> se.uni_cnt) - 2.) - \
                (<double> se.uni_cnt * se.entropy - \
                     <double> se1.uni_cnt * se1.entropy - \
                         <double> se2.uni_cnt * se2.entropy)
            cr.stop_cut = gain <= 1. / n * (log(n - 1.) + delta)
            return cr

    def get_cut(self, size_t low, size_t upp):
        cdef cut_result cr = self._get_cut(low, upp)
        return cr.cut_loc, cr.stop_cut

    @boundscheck(False)
    @wraparound(False)
    @cdivision(True)
    def get_partition_points(self):
        cdef:
            cppvector[size_t] cut_locations
            interval intv, intv_curr
            cppstack[interval] search_intervals
            cut_result cr

        intv.low = 0
        intv.upp = self._y.shape[0]
        search_intervals.push(intv)
        while search_intervals.size() > 0:
            intv_curr = search_intervals.top()
            search_intervals.pop()
            cr = self._get_cut(intv_curr.low, intv_curr.upp)
            if cr.stop_cut:
                continue
            intv.low = intv_curr.low
            intv.upp = cr.cut_loc
            search_intervals.push(intv)
            intv.low = cr.cut_loc
            intv.upp = intv_curr.upp
            search_intervals.push(intv)
            cut_locations.push_back(cr.cut_loc)
        cppsort(cut_locations.begin(), cut_locations.end())

        cdef:
            size_t loc
            cppvector[np.float64_t] cut_points
            np.float64_t* x_ptr = &self._x[0]
        for loc in cut_locations:
            cut_points.push_back(((x_ptr+loc)[0] + (x_ptr+loc-1)[0]) / 2.0)
        cppsort(cut_points.begin(), cut_points.end())
        return cut_locations, cut_points
