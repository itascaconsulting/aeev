import numpy as np
from _vec import vec3 as vec
from lazy_expr import lazy_expr

_a = vec((1,2,3))
_b = np.array(((1,2,3.0),(4,5,6.0), (7,8,9.0)))
_target  = np.zeros_like(_b)

a = lazy_expr(_a)
b = lazy_expr(_b)
target = lazy_expr(_target)

expr = target == a + b
