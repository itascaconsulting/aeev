import numpy as np
from _vec import vec3 as vec
from lazy_expr import lazy_expr, dis
from ops import *

_a = vec((1,2,3))
_b = np.array(((1,2,3.0),(4,5,6.0), (7,8,9.0), (10,11,12.)))
_target  = np.zeros_like(_b)

a = lazy_expr(_a)
b = lazy_expr(_b)
target = lazy_expr(_target)


import aeev
expr = ((a+1)+vec((11,12,13)))
#dis(expr)
op, l, _ = expr.get_bytecode()
print aeev.vm_eval(tuple(op), tuple(l))
((_a+1)+vec((11,12,13))).assert_close(aeev.vm_eval(tuple(op), tuple(l)))

expr = target == -b
dis(expr)
print expr.vm_eval()

(target == b+1).vm_eval()
