from ops import *
from lazy_expr import lazy_expr
import aeev
import numpy as np

a = lazy_expr(1.0)
b = lazy_expr(3.0)
c = lazy_expr(4.0)
d = lazy_expr(16.0)
e = lazy_expr(0.1)

expr = (a + b + e)**c

print expr
print expr.get_tuple()
print expr.get_bytecode()
ops, literals = expr.get_bytecode()
print aeev.vm_eval(ops, literals)

print aeev.eval(expr.get_tuple())
assert aeev.eval(expr.get_tuple()) == (1.0 + 3.0 + 0.1)**4.0

expr = (a+a+a+a+a+a+b+b+b+c+c+c+c+c+a+a+a+a+a+a+a).get_tuple()
print (a+a+a+a+a+a+b+b+b+c+c+c+c+c+a+a+a+a+a+a+a).get_bytecode()
assert aeev.eval(expr) == 42

expr = ((a+b)**lazy_expr(2) + e*(d**-e) + a/b) / c
print expr
print expr.get_tuple()
print expr.get_bytecode()
print aeev.eval(expr.get_tuple())
res = ((1.0+3.0)**2 + 0.1*(16.0**-0.1) + 1.0/3.0) / 4.0
assert aeev.eval(expr.get_tuple()) == res


# 1+2
expr = (ss_add, (i_scalar, 1.0), (i_scalar, 2.0))
print expr
print aeev.eval(expr)
assert aeev.eval(expr) == 1.0+2.0

# (1.23+5.0)**2 - 5e2/0.1
expr = (ss_sub, (ss_pow, (ss_add, (i_scalar, 1.23), (i_scalar, 5.0)),
                 (i_scalar, 2.0)),
        (ss_div, (i_scalar, 5e2), (i_scalar, 0.1)))
print expr, (1.23+5.0)**2 - 5e2/0.1

print aeev.eval(expr)
assert aeev.eval(expr) == (1.23+5.0)**2 - 5e2/0.1

arr = np.ones(1e6)
f = lazy_expr(arr)
expr = f+1
print expr.get_tuple()
target = np.zeros(1e6)
print aeev.array_eval(expr.get_tuple(), target)
