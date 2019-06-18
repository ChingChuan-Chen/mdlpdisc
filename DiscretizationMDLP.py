#!/usr/bin/env python3
from math import log
import bisect
from collections import Counter

class DiscretizationMDLP(object):
    def __init__(self, x, y):
        if len(x) != len(y):
            raise Exception("The length of x must be equal to the length of y.")
        # sort first
        self.data = sorted(zip(*[x, y]), key=lambda x: x[0])
        # store keys for binary search
        self.keys = [z[0] for z in self.data]
       
    def _get_entropy(self, y):
        """function to get entropy and unique count of y"""
        # get counter of y
        counter_y = Counter(y)
        # get proportions for each y
        ps = [value/len(y) for _, value in counter_y.items()]
        if len(ps) == 1:
            return 0, len(counter_y.keys())
        else:
            return -sum([p*log(p) for p in ps]), len(counter_y.keys())

    def get_cut(self, low, upp):
        """function to get cut point"""
        # get supports to check
        support = sorted(list(set(self.keys[low:upp])))
        # initialize parameters
        n = upp - low
        prev_ent, prev_cut, prev_cut_value = 9999, None, None
        entropy1, entropy2, k1, k2, w = None, None, None, None, None
        # get whole entropy
        whole_ent, k = self._get_entropy([y for _, y in self.data[low:upp]])
        for i in range(len(support)-1):
            curr_cut_value = (support[i] + support[i+1])/2
            # get current cut point
            curr_cut = bisect.bisect_right(self.keys[low:upp], curr_cut_value)
            # calculate weight
            weight = curr_cut/n
            # get the entropies of two partitions
            left_ent, left_y_cnt = self._get_entropy([y for _, y in self.data[low:(low+curr_cut)]])
            right_ent, right_y_cnt = self._get_entropy([y for _, y in self.data[(low+curr_cut):upp]])
            # get current entropy
            curr_ent = weight * left_ent + (1-weight) * right_ent
            # compare the entropy to the history
            if curr_ent < prev_ent:
                prev_ent, prev_cut, prev_cut_value = curr_ent, curr_cut, curr_cut_value
                # keep these variables for calculating the stop criterio
                entropy1, entropy2, k1, k2, w = left_ent, right_ent, left_y_cnt, right_y_cnt, weight
        if prev_cut is None:
            return None, None, None, True
        else:
            # get entropy gain
            gain = whole_ent - (w * entropy1 + (1-w) * entropy2)
            # get stoping value
            delta = log(pow(3, k) - 2) - (k * whole_ent - k1 * entropy1 - k2 * entropy2)
            # check whether to stop or not
            stop_cut = gain <= 1 / n * (log(n - 1) + delta)
            return prev_ent, low+prev_cut, prev_cut_value, stop_cut

    def get_partition_points(self):
        cut_locations = []
        cut_points = []
        search_intervals = [(0, len(self.data))]
        while len(search_intervals) > 0:
            low, upp = search_intervals.pop()
            entropy, cut, cut_value, stop_cut = self.get_cut(low, upp)
            if stop_cut:
                continue
            search_intervals.append((low, cut))
            search_intervals.append((cut, upp))
            cut_locations.append(cut)
            cut_points.append(cut_value)
        return sorted(cut_locations), sorted(cut_points)
