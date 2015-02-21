from ops import *
import array
import numpy as np


class lazy_expr(object):


    def typecode(left, right):
        "returns 0, 1, 2 or 3 for a_a, a_s, s_a, s_s"
        return left.is_scalar()*2 + right.is_scalar()

    def handle_op(a, b, base_op_code):
        code = lazy_expr.typecode(a, b)
        if code == 3: code += 300
        return lazy_expr((base_op_code + code, a, b))

    def __init__(self, data):
        if isinstance(data, tuple):
            self.data = data
        elif isinstance(data, lazy_expr):
            self.data = data.data
        elif isinstance(data, float) or isinstance(data, int):
            self.data = (i_scalar, float(data))
        elif type(data) is np.ndarray:
            self.data = (ia_scalar, data)
        else:
            raise ValueError("unknown type")

    def is_scalar(self):
        "return true if this leaf should evaluate to a scalar"
        if self.data[0] >= 500:
            return 1
        return 0

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
        if self.data[0] == i_scalar:
            return self.data
        elif self.data[0] == ia_scalar:
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
            assert type(opcode) is int
            return '3' == str(opcode)[-1]

        def listit(t):
            """Convert nested tuples into nested lists"""
            return list(map(listit, t)) if isinstance(t, (list, tuple)) else t

        top_cell = listit(self.get_tuple())

        if not scalar_op(top_cell[0]):
            top_cell[0] |= result_to_target;

        def opt_visitor(cell):
            op, args = cell[0], cell[1:]
            if scalar_op(op & ~result_to_target):
                return
            # if the left operand is an array literal
            # then set the read from heap bit
            if op == i_scalar:
                return
            if op == ia_scalar:
                return
            if args[0][0] == ia_scalar:
                print "inplace l"
                cell[0] |= left_on_heap
            if len(args) == 2:
                if args[1][0] == ia_scalar:
                    print "inplace r"
                    cell[0] |= right_on_heap
            for arg in args:
                opt_visitor(arg)

        opt_visitor(top_cell)
        literal_stack = []
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
            elif op == ia_scalar:
                if args[0] in array_stack:
                    op_stack.append(array_scalar_bit |
                                    array_stack.index(args[0]))
                else:
                    array_stack.append(args[0])
                    op_stack.append(array_scalar_bit |
                                    array_stack.index(args[0]))

            else:
                for a in args:
                    visitor(a, literal_stack, op_stack, array_stack)
                op_stack.append(op)
        visitor(top_cell, literal_stack, op_stack, array_stack)
        return tuple(op_stack), tuple(literal_stack), tuple(array_stack)
