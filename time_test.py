import aeev
import numpy as np
from lazy_expr import lazy_expr

a,b,c = np.ones(1e6-64), np.ones(1e6-64), np.ones(1e6-64)
aeev.call_test(a,b,c)
aeev.call_test_chunk(a,b,c)

#%timeit aeev.call_test(a,b,c)
#%timeit a+b+c

import scipy
from scipy import linalg
#linalg.blas.daxpy(a,b)
#%timeit linalg.blas.daxpy(a,b)


def test_func(): pass

lazy_a = lazy_expr(a)
lazy_b = lazy_expr(b)
expr = lazy_a + lazy_b
tup = expr.get_tuple()
print tup
target = np.zeros(1e6)
aeev.array_eval(tup, target)


# aeev: 8 ms
# numpy: 3.52 ms
# raw c: 1.6 ms

# sexp vm gives about a 5ms overhead.
