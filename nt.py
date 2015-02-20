from lazy_expr import lazy_expr
from ops import *
import numpy as np


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

a=lazy_expr(1.0)
_b = np.ones(3)
b=lazy_expr(_b)


expr = (a+b**2)+22.2
opcodes, doubles, arrays = expr.get_bytecode()
dis(opcodes, doubles, arrays)
