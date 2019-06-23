#!/usr/bin/env python3
from mdlp2.mdlp._mdlp import MDLPDiscretize, find_cut, slice_entropy
from mdlpdisc import mdlpdisc
from DiscretizationMDLP import DiscretizationMDLP
import numpy as np

x = np.array([139., 139., 139., 139., 1490., 1490., 1490., 32456., 32456., 
     33444., 33444., 33444., 35666., 35666., 35666., 35666.]).astype(np.float64)
y = np.array([1, 1, 1, 1, 2, 2, 2, 3, 4, 3, 3, 4, 3, 3, 4, 4]).astype(np.int64)

discmdlp = DiscretizationMDLP(x, y)
discmdlp.get_partition_points() # ([4, 7], [814.5, 16973.0])

discmdlp2 = mdlpdisc(x, y)
discmdlp2.get_support() # [4, 7, 9, 12]
discmdlp2.get_data()
# (array([  139.,   139.,   139.,   139.,  1490.,  1490.,  1490., 32456.,
#        32456., 33444., 33444., 33444., 35666., 35666., 35666., 35666.]),
#  array([1, 1, 1, 1, 2, 2, 2, 3, 4, 3, 3, 4, 3, 3, 4, 4], dtype=int64))
discmdlp2.get_entropy(0, len(x)) # (1.370502389918909, 4)
discmdlp2.get_entropy(0, 7) # (0.6829081047004717, 2)
discmdlp2.get_entropy(0, 4) # (0.0, 1)
discmdlp2.get_entropy(4, 7) # (0.0, 1)
discmdlp2.get_entropy(7, len(x)) # (0.6869615765973234, 2)
discmdlp2.get_cut(0, len(x)) # (7, False)
discmdlp2.get_partition_points() # ([4, 7], [814.5, 16973.0])

MDLPDiscretize(x, y, 0) # (array([4, 7]), array([  814.5, 16973. ]))
find_cut(x, y, 0, len(x)) # 7
slice_entropy(y, 0, len(x)) # (1.370502389918909, 4)
slice_entropy(y, 0, 7) # (0.6829081047004717, 2)
slice_entropy(y, 0, 4) # (0.0, 1)
slice_entropy(y, 4, 7) # (0.0, 1)
slice_entropy(y, 7, len(x)) # (0.6869615765973234, 2)

# performance for initialization (sorting)
import timeit
data_str = """
import numpy as np
x = np.array([139., 139., 139., 139., 1490., 1490., 1490., 32456., 32456., 
     33444., 33444., 33444., 35666., 35666., 35666., 35666.] * 50000).astype(np.float64)
y = np.array([1, 1, 1, 1, 2, 2, 2, 3, 4, 3, 3, 4, 3, 3, 4, 4] * 50000).astype(np.int64)
"""
cython_setup = "from mdlpdisc import mdlpdisc" + data_str

print("mdlp2 (modified): ", timeit.timeit("""
order = np.argsort(x)
x = x[order]
y = y[order]      
""", setup=data_str, number=100), "seconds")
print("mdlpdisc: ", timeit.timeit("mdlp = mdlpdisc(x, y)", 
                                     setup=cython_setup, number=100), "seconds")
# mdlp2 (modified):  2.4831722110002374 seconds
# mdlpdisc:  1.7448171599999114 seconds

# performance for calculating entropy
module_setup = """
from mdlp2.mdlp._mdlp import slice_entropy
""" + data_str + """
order = np.argsort(x)
x = x[order]
y = y[order]
"""
cython_setup = "from mdlpdisc import mdlpdisc" + data_str + "mdlp = mdlpdisc(x, y)"
scipy_setup = "from scipy.stats import entropy" + data_str

print("mdlp2 (modified): ", timeit.timeit("slice_entropy(y, 0, y.shape[0])",
                                     setup=module_setup, number=1000), "seconds")
print("mdlpdisc: ", timeit.timeit("mdlp.get_entropy(0, y.shape[0])", 
                                     setup=cython_setup, number=1000), "seconds")
print("scipy: ", timeit.timeit("entropy(np.true_divide(np.bincount(y), len(y)))", 
                                     setup=scipy_setup, number=1000), "seconds")
# mdlp2 (modified):  2.350262768999997 seconds
# mdlpdisc:  1.6760242600000055 seconds
# scipy:  2.3572629439999986 seconds

# performance for getting the first cut point
module_setup = """
from mdlp2.mdlp._mdlp import find_cut
""" + data_str + """
order = np.argsort(x)
x = x[order]
y = y[order]
"""
cython_setup = "from mdlpdisc import mdlpdisc" + data_str + "mdlp = mdlpdisc(x, y)"

print("mdlp2 (modified): ", timeit.timeit("find_cut(x, y, 0, len(x))",
                                     setup=module_setup, number=200), "seconds")
print("mdlpdisc: ", timeit.timeit("mdlp.get_cut(0, y.shape[0])", 
                                     setup=cython_setup, number=200), "seconds")
# mdlp2 (modified):  1.9396390149999974 seconds
# mdlpdisc:  2.0471303639999974 seconds

# performance for finding partition points
module_setup = """
from mdlp2.mdlp._mdlp import MDLPDiscretize
""" + data_str
cython_setup = "from mdlpdisc import mdlpdisc" + data_str
python_setup = "from DiscretizationMDLP import DiscretizationMDLP" + data_str

print("mdlp2 (modified): ", timeit.timeit("MDLPDiscretize(x, y, 0)",
                                     setup=module_setup, number=100), "seconds")
print("mdlpdisc: ", timeit.timeit('mdlpdisc(x, y).get_partition_points()', 
                                     setup=cython_setup, number=100), "seconds")
print("DiscretizationMDLP (Pure Python): ", 
      timeit.timeit('DiscretizationMDLP(x, y).get_partition_points()', 
                    setup=python_setup, number=20), "seconds")
# mdlp2 (modified):  5.935840370000008 seconds
# mdlpdisc:  4.181983544999994 seconds
# DiscretizationMDLP (Pure Python):  58.78786561800007 seconds
