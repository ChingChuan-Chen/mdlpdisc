%%writefile DiscretizationMDLP.pyx
from libc.math cimport log, pow
from libcpp cimport bool
from libcpp.map cimport map as cppmap
from libcpp.vector cimport vector as cppvector
from libcpp.stack cimport stack as cppstack
from libcpp.algorithm cimport sort as cppsort
from libcpp.algorithm cimport upper_bound
from libcpp.pair cimport pair as cpppair

cdef bool cmp_fn(double v, cpppair[double, long] data):
    return data.first >= v

cdef class DiscretizationMDLP:
    cdef cppvector[cpppair[double, long]] data
        
    def __init__(self, cppvector[double] x, cppvector[long] y):
        if x.size() != y.size():
            raise Exception("The length of x must be equal to the length of y.")
        cdef size_t i
        for i in range(x.size()):
            self.data.push_back(cpppair[double, long](x[i], y[i]))
        cppsort(self.data.begin(), self.data.end())

    def get_data(self):
        return self.data

    def get_entropy(self, size_t low, size_t upp):
        cdef:
            cppmap[long, long] freq
            size_t n = upp-low, i
    
        for i in range(low, upp):
            freq[self.data[i].second] += 1;
    
        cdef:
            double ent = 0.0, temp
        for _, v in freq:
            temp = <double> v / n
            ent += temp * log(temp)    
        return -ent, freq.size()

    def _get_cut_loc(self, double cut_value):
        cdef long cut_loc = upper_bound(self.data.begin(), self.data.end(), cut_value, cmp_fn) - self.data.begin()
        return cut_loc

    def get_cut(self, size_t low, size_t upp):
        cdef:
            cdef double nan = float('nan')
            size_t i, n = upp - low, left_y_cnt, right_y_cnt, k1, k2
            long curr_cut, prev_cut = -1
            double whole_ent, prev_cut_value = nan, curr_cut_value, prev_ent = 9999.0, weight, curr_ent
            double left_ent, right_ent, entropy1, entropy2, w
        whole_ent, k = self.get_entropy(low, upp)
        for i in range(low, upp-1):
            if self.data[i].first != self.data[i+1].first:
                curr_cut_value = (self.data[i].first + self.data[i+1].first) / 2.0
                curr_cut = self._get_cut_loc(curr_cut_value)
                weight = <double> (curr_cut-low)/n
                left_ent, left_y_cnt = self.get_entropy(low, curr_cut)
                right_ent, right_y_cnt = self.get_entropy(curr_cut, upp)
                curr_ent = weight * left_ent + (1.-weight) * right_ent
                if curr_ent < prev_ent:
                    prev_ent, prev_cut, prev_cut_value = curr_ent, curr_cut, curr_cut_value
                    entropy1, entropy2, k1, k2, w = left_ent, right_ent, left_y_cnt, right_y_cnt, weight

        if prev_cut == -1:
            return nan, -1, nan, True
        else:
            gain = whole_ent - (w * entropy1 + (1.-w) * entropy2)
            delta = log(pow(3, <double> k) - 2.) - (<double> k * whole_ent - 
                       <double> k1 * entropy1 - <double> k2 * entropy2)
            stop_cut = gain <= 1. / (<double> n) * (log(<double> n - 1.) + delta)
            return prev_ent, prev_cut, prev_cut_value, stop_cut

    def get_partition_points(self):
        cdef:
            cppvector[size_t] cut_locations
            cppvector[double] cut_points
            cppstack[cpppair[size_t, size_t]] search_intervals
            long cut
            double entropy, cut_value
            bool stop_cut
        search_intervals.push(cpppair[size_t, size_t](0, self.data.size()))
        while search_intervals.size() > 0:
            low, upp = search_intervals.top()
            search_intervals.pop()
            entropy, cut, cut_value, stop_cut = self.get_cut(low, upp)
            if stop_cut:
                continue
            search_intervals.push(cpppair[size_t, size_t](low, <size_t> cut))
            search_intervals.push(cpppair[size_t, size_t](<size_t> cut, upp))
            cut_locations.push_back(cut)
            cut_points.push_back(cut_value)
        cppsort(cut_locations.begin(), cut_locations.end())
        cppsort(cut_points.begin(), cut_points.end())
        return cut_locations, cut_points
