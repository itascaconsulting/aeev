from ops import *
import array
import numpy as np
import aeev
from _vec import vec3 as vec

def dis(expr):
    """ Byte code and stack disassembler/pretty-printer"""
    if type(expr) is lazy_expr:
        opcodes, doubles, arrays = expr.get_bytecode()
    else:
        opcodes, doubles, arrays = expr.rhs.get_bytecode()

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
        if o & vector_bit:
            o = o & ~vector_bit
            print "{}:  vector load {} {}".format(i, o, doubles[o:o+3])
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

class Assignment(object):
    "an expression which can be evaluated and assigned to a result"
    def __init__(self, lhs, rhs):
        assert type(lhs) is np.ndarray
        assert type(rhs) is lazy_expr
        self.lhs = lhs
        self.rhs = rhs
        self.op_stack, self.literal_stack, self.array_stack = \
                                                self.rhs.get_bytecode()
        assert len(self.array_stack), "expression must contain array"

    def vm_eval(self):
        aeev.array_vm_eval(self.op_stack, self.literal_stack,
                           self.array_stack, self.lhs)
        return self.lhs

    def __repr__(self):
        return "Assignment lhs({}) \n=\nrhs({})".format(self.lhs,
                                                        self.rhs)

class lazy_expr(object):
    """Represents an ast node, an operator and at least one value"""
    def typecode(left, right):
        "returns 0, 1, 2 or 3 for a_a, a_s, s_a, s_s"
        return left.is_scalar()*2 + right.is_scalar()

    def handle_op(a, b, base_op_code):
        code = lazy_expr.typecode(a, b)
        if code == 3: code += 300
        return lazy_expr((base_op_code + code, a, b))

    def __init__(self, data):
        if type(data) is tuple:
            self.data = data
        elif type(data) is lazy_expr:
            self.data = data.data
        elif type(data) is float or type(data) is int:
            self.data = (i_scalar, float(data))
        elif type(data) is vec:
            self.data = (i_vec, data)
        elif type(data) is np.ndarray:
            if data.ndim == 1:
                self.data = (ia_scalar, data)
            else:
                assert data.ndim == 2
                assert data.shape[1] == 3
                self.data = (ia_vec, data)
        else:
            raise ValueError("unknown type")

    def is_scalar(self):
        "return true if this leaf should evaluate to a scalar"
        if self.data[0] >= 500:
            return 1
        return 0

    def __eq__(self, other):
        rhs = lazy_expr(other)
        assert self.data[0] == ia_scalar or self.data[0] == ia_vec, \
            "lhs must be an array"
        lhs = self.data[1]
        assert type(lhs) is np.ndarray
        return Assignment(lhs, rhs)


    def __add__(self, other):
        return lazy_expr.handle_op(self, lazy_expr(other), a_a_add)
    def __radd__(self, other):
        return lazy_expr.handle_op(lazy_expr(other), self, a_a_add)

    def __sub__(self, other):
        return lazy_expr.handle_op(self, lazy_expr(other), a_a_sub)
    def __rsub__(self, other):
        return lazy_expr.handle_op(lazy_expr(other), self, a_a_sub)

    def __mul__(self, other):
        return lazy_expr.handle_op(self, lazy_expr(other), a_a_mul)
    def __rmul__(self, other):
        return lazy_expr.handle_op(lazy_expr(other), self, a_a_mul)

    def __div__(self, other):
        return lazy_expr.handle_op(self, lazy_expr(other), a_a_div)
    def __rdiv__(self, other):
        return lazy_expr.handle_op(lazy_expr(other), self, a_a_div)

    def __pow__(self, other):
        return lazy_expr.handle_op(self, lazy_expr(other), a_a_pow)
    def __rpow__(self, other):
        return lazy_expr.handle_op(lazy_expr(other), self, a_a_pow)

    def __neg__(self):
        if self.data[0] < 500:
            return lazy_expr((a_negate, self))
        else:
            return lazy_expr((s_negate, self))


    def __repr__(self):
        return "({}, ".format(op_hash[self.data[0]]) +\
            ", ".join(map(str, self.data[1:])) +  " )"

    def get_tuple(self):
        if self.data[0] == i_scalar or self.data[0] == i_vec:
            return self.data
        elif self.data[0] == ia_scalar or self.data[0] == ia_vec:
            return self.data
        else:
            if len(self.data) == 2:
                return (self.data[0],
                        self.data[1].get_tuple())
            if len(self.data) == 3:
                return (self.data[0],
                        self.data[1].get_tuple(),
                        self.data[2].get_tuple())

    def get_bytecode(self):
        # for each operation involving an array
        # determine if the result, left or right operand is
        # on the heap or stack.

        def scalar_op(opcode):
            assert type(opcode & ~bytecode_mask) is int
            return '3' == str(opcode & ~bytecode_mask)[-1]

        def left_is_array(opcode):
            char = str(opcode & ~bytecode_mask)[-1]
            return char == '0' or char == '1'

        def right_is_array(opcode):
            char = str(opcode & ~bytecode_mask)[-1]
            return char == '0' or char == '2'


        def listit(t):
            """Convert nested tuples into nested lists"""
            return list(map(listit, t)) if isinstance(t, (list, tuple)) else t

        top_cell = listit(self.get_tuple())

        def opt_visitor(cell):
            """Add bits to bytecodes"""
            op, args = cell[0], cell[1:]
            if scalar_op(op) or op == i_scalar or op == ia_scalar or \
               op == i_vec or op == ia_vec:
                return
            flags = 0
            if args[0][0] == ia_scalar:
                flags |= left_on_heap
            if len(args) == 2:
                if args[1][0] == ia_scalar:
                    flags |= right_on_heap
            if left_is_array(op): flags |= left_array
            if right_is_array(op): flags |= right_array
            cell[0] |= flags
            for arg in args:
                opt_visitor(arg)

        opt_visitor(top_cell)

        # write final result to heap
        if not scalar_op(top_cell[0]):
            top_cell[0] |= result_to_target;

        literal_stack = []
        array_id_stack = []
        array_stack = []
        op_stack = []

        def visitor(cell, literal_stack, op_stack, array_stack):
            op, args = cell[0], cell[1:]
            if op == i_scalar:
                if args[0] in literal_stack:
                    op_stack.append(scalar_bit | literal_stack.index(args[0]))
                else:
                    literal_stack.append(args[0])
                    op_stack.append(scalar_bit | (len(literal_stack)-1))
            elif op == i_vec:
                # optimize this
                x,y,z = args[0]
                literal_stack.append(x)
                literal_stack.append(y)
                literal_stack.append(z)
                op_stack.append(vector_bit | len(literal_stack)-3)

            elif op == ia_scalar or op == ia_vec:
                if id(args[0]) in array_id_stack:
                    op_stack.append(array_scalar_bit |
                                    array_id_stack.index(id(args[0])))
                else:
                    array_stack.append(args[0])
                    array_id_stack.append(id(args[0]))
                    op_stack.append(array_scalar_bit |
                                    array_id_stack.index(id(args[0])))

            else:
                for a in args:
                    visitor(a, literal_stack, op_stack, array_stack)
                op_stack.append(op)
        visitor(top_cell, literal_stack, op_stack, array_stack)
        return np.array(op_stack, dtype=int), \
            np.array(literal_stack), \
            tuple(array_stack)
