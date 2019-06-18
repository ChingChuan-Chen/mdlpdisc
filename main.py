#!/usr/bin/env python3
from mdlp.discretization import MDLP
import discretization
import DiscretizationMDLP
import numpy as np

# result checking
from mdlp.discretization import MDLP
import discretization
import DiscretizationMDLP
import numpy as np

x = [139., 139., 139., 139., 1490., 1490., 1490., 32456., 32456., 
     33444., 33444., 33444., 35666., 35666., 35666., 35666.]
y = [1, 1, 1, 1, 2, 2, 2, 3, 4, 3, 3, 4, 3, 3, 4, 4]

x_np = np.array(x).reshape([len(x), -1])
y_np = np.array(y)

discmdlp = DiscretizationMDLP.DiscretizationMDLP(x, y)
discmdlp.get_partition_points() # ([4, 7], [814.5, 16973.0])

discmdlp2 = discretization.DiscretizationMDLP(x, y)
discmdlp2.get_partition_points() # ([4, 7], [814.5, 16973.0])

discmdlp3 = MDLP()
discmdlp3.fit(x_np, y_np)
discmdlp3.cut_points_[0] # array([  814., 16973.])

# performance
import timeit
data_str = """
x = [139., 139., 139., 139., 1490., 1490., 1490., 32456., 32456., 
     33444., 33444., 33444., 35666., 35666., 35666., 35666.] * 1000
y = [1, 1, 1, 1, 2, 2, 2, 3, 4, 3, 3, 4, 3, 3, 4, 4] * 1000
"""
module_setup = """
from mdlp.discretization import MDLP
import numpy as np
""" + data_str + """
x = np.array(x).reshape([len(x), -1])
y = np.array(y)
"""
python_setup = "import DiscretizationMDLP" + data_str
cython_setup = "import discretization" + data_str

print("mdlp-discretization: ", timeit.timeit("mdlpfit = MDLP()\nmdlpfit.fit(x, y)",
                                     setup=module_setup, number=50), "seconds")
print("Python code: ", timeit.timeit('DiscretizationMDLP.DiscretizationMDLP(x, y).get_partition_points()', 
                                     setup=python_setup, number=50), "seconds")
print("Cython code: ", timeit.timeit('discretization.DiscretizationMDLP(x, y).get_partition_points()', 
                                     setup=cython_setup, number=50), "seconds")

# mdlp-discretization:  54.111572814000056 seconds
# Python code:  0.9483610649999719 seconds
# Cython code:  0.05425053800013302 seconds
