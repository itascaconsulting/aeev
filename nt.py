from lazy_expr import lazy_expr
from ops import *
import numpy as np
import aeev

def dis(opcodes, doubles, arrays):
    print
    print "op codes"
    print "========"
    print

    for i, o in enumerate(opcodes):
        right_heap = False
        left_heap = False
        result_target = False
        aleft, aright = o & left_array, o & right_array
        o &= ~left_array
        o &= ~right_array

        if o & right_on_heap:
            right_heap = True
            o &= ~right_on_heap
        if o & left_on_heap:
            left_heap = True
            o &= ~left_on_heap
        if o & result_to_target:
            result_target = True
            o &= ~result_to_target
        if o & scalar_bit:
            o = o & ~scalar_bit
            print "{}:  literal load {} ({})".format(i,o,doubles[o])
        elif o & array_scalar_bit:
            o = o & ~array_scalar_bit
            print "{}:  array load {} (id: {})".format(i,o,id(arrays[o]))
        elif o in op_hash:
            print "{}:  {} {}{}{}{}{}".format(i, op_hash[o],
                                           "r-target " if result_target else "",
                                           "l-heap " if left_heap else "",
                                           "r-heap " if right_heap else "",
                                           "a-left " if aleft else "",
                                           "a-right " if aright else "")
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
_b = np.linspace(0,1,8*3)
b=lazy_expr(_b)
_c = np.linspace(1,2,8*3)
c=lazy_expr(_c)


print
print "(a+b**2)+22.2"
print
expr = (a+b**2)+22.2
print
print expr

opcodes, doubles, arrays = expr.get_bytecode()

dis(opcodes, doubles, arrays)

aops = np.array(opcodes, dtype=int)
adou = np.array(doubles)
target = np.zeros_like(_b)
aeev.array_vm_eval(aops, adou, arrays, target)

print
print "="*80
print "(b**2+b)"
print

expr = b+c
opcodes, doubles, arrays = expr.get_bytecode()
dis(opcodes, doubles, arrays)

print

aops = np.array(opcodes, dtype=int)
adou = np.array(doubles)
aeev.array_vm_eval(aops, adou, arrays, target)
np.testing.assert_allclose(target, _b+_c)


print
print "="*80
print "(c+b+c)"
print

expr = c+b+c
opcodes, doubles, arrays = expr.get_bytecode()
dis(opcodes, doubles, arrays)

print

aops = np.array(opcodes, dtype=int)
adou = np.array(doubles)
aeev.array_vm_eval(aops, adou, arrays, target)
np.testing.assert_allclose(target, _c + _b + _c)


print
print "="*80
print "(b+1.23)"
print

expr = b+1.23
opcodes, doubles, arrays = expr.get_bytecode()
dis(opcodes, doubles, arrays)

print

aops = np.array(opcodes, dtype=int)
adou = np.array(doubles)
aeev.array_vm_eval(aops, adou, arrays, target)
np.testing.assert_allclose(target, _b + 1.23)
