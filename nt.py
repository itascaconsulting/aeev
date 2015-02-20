from lazy_expr import lazy_expr
from ops import *
import numpy as np

a=lazy_expr(1.0)
_b = np.ones(3)
b=lazy_expr(_b)

assert (a+2).get_tuple() == (s_s_add, (i_scalar, 1), (i_scalar, 2))
assert (a+b).get_tuple() == (s_a_add, (i_scalar, 1), (ia_scalar, _b))
assert (a+b**2).get_tuple() == (s_a_add,
                                (i_scalar, 1.0 ),
                                (a_s_pow, (ia_scalar, _b),
                                 (i_scalar, 2.0)))

def dis(opcodes, doubles, arrays):
    print
    print "op codes"
    print "========"
    print

    for i, o in enumerate(opcodes):
        if o & scalar_bit:
            o = o & ~scalar_bit
            print "{}:  literal load {} ({})".format(i,o,doubles[o])
        elif o & array_scalar_bit:
            o = o & ~array_scalar_bit
            print "{}:  array load {} (id: {})".format(i,o,id(arrays[o]))
        elif o in op_hash:
            print "{}:  {}".format(i, op_hash[o])
        else:
            print "{}:  data ({})".format(i,o)

    print
    print "scalar literals"
    print "==============="
    print
    for i,d in enumerate(doubles):
        print "{}  {}".format(i,d)

    print
    print "array literals"
    print "==============="
    print
    for i,a in enumerate(arrays):
        print "{}:  shape: {} id: {}".format(i,a.shape, id(a))



expr = (a+b**2)+22.2
opcodes, doubles, arrays = expr.get_bytecode()
dis(opcodes, doubles, arrays)
