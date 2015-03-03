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
ops, literals, arrays = expr.get_bytecode()
print aeev.vm_eval(tuple(ops), tuple(literals))
tup = expr.get_tuple()
print aeev.eval(tup)
np.testing.assert_allclose(aeev.eval(expr.get_tuple()), (1.0 + 3.0 + 0.1)**4.0)
np.testing.assert_allclose(aeev.vm_eval(tuple(ops), tuple(literals)),
                           (1.0 + 3.0 + 0.1)**4.0)

expr = (a+a+a+a+a+a+b+b+b+c+c+c+c+c+a+a+a+a+a+a+a).get_tuple()
print (a+a+a+a+a+a+b+b+b+c+c+c+c+c+a+a+a+a+a+a+a).get_bytecode()
assert aeev.eval(expr) == 42
ops, lits, arrs = (a+a+a+a+a+a+b+b+b+c+c+c+c+c+a+a+a+a+a+a+a).get_bytecode()
assert aeev.vm_eval(tuple(ops), tuple(lits)) == 42

expr = ((a+b)**lazy_expr(2) + e*(d**-e) + a/b) / c
print expr
print expr.get_tuple()
print expr.get_bytecode()
print aeev.eval(expr.get_tuple())
res = ((1.0+3.0)**2 + 0.1*(16.0**-0.1) + 1.0/3.0) / 4.0
assert aeev.eval(expr.get_tuple()) == res


# 1+2
expr = (s_s_add, (lit_s, 1.0), (lit_s, 2.0))
print expr
print aeev.eval(expr)
assert aeev.eval(expr) == 1.0+2.0

# (1.23+5.0)**2 - 5e2/0.1
expr = (s_s_sub, (s_s_pow, (s_s_add, (lit_s, 1.23), (lit_s, 5.0)),
                 (lit_s, 2.0)),
        (s_s_div, (lit_s, 5e2), (lit_s, 0.1)))
print expr, (1.23+5.0)**2 - 5e2/0.1

print aeev.eval(expr)
assert aeev.eval(expr) == (1.23+5.0)**2 - 5e2/0.1

arr = np.ones(1e6)
f = lazy_expr(arr)
expr = f+1
print expr.get_tuple()
target = np.zeros(1e6)
print aeev.array_eval(expr.get_tuple(), target)

a=lazy_expr(1.0)
_b = np.ones(3)
b=lazy_expr(_b)

assert (a+2).get_tuple() == (s_s_add, (lit_s, 1), (lit_s, 2))
assert (a+b).get_tuple() == (s_as_add, (lit_s, 1), (lit_as, _b))
assert (a+b**2).get_tuple() == (s_as_add,
                                (lit_s, 1.0 ),
                                (as_s_pow, (lit_as, _b),
                                 (lit_s, 2.0)))
