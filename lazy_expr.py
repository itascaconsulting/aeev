from ops import *
import array
import numpy as np


class lazy_expr(object):

    # typecode constants
    A_A = 0
    A_S = 1
    S_A = 2
    S_S = 3

    def typecode(left,right):
        "returns 0, 1, 2 or 3 for a_a, a_s, s_a, s_s"
        return left.is_scalar()*2 + right.is_scalar()

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
        if self.data[0] < 500:
            return 1
        return 0

    def __add__(self, other):
        code = lazy_expr.typecode(self, other)
        if code == lazy_expr.A_A:
            return lazy_expr((a_a_add, self, lazy_expr(other)))
        elif code == lazy_expr.A_S:
            return lazy_expr((a_s_add, self, lazy_expr(other)))
        elif code == lazy_expr.S_A:
            return lazy_expr((s_a_add, self, lazy_expr(other)))
        elif code == lazy_expr.S_S:
            return lazy_expr((s_s_add, self, lazy_expr(other)))
        else:
            raise ValueError("AST generation error")

    def __radd__(self, other):
        code = lazy_expr.typecode(other, self)
        if code == lazy_expr.A_A:
            return lazy_expr((a_a_add, lazy_expr(other), self))
        elif code == lazy_expr.A_S:
            return lazy_expr((a_s_add, lazy_expr(other), self))
        elif code == lazy_expr.S_A:
            return lazy_expr((s_a_add, lazy_expr(other), self))
        elif code == lazy_expr.S_S:
            return lazy_expr((s_s_add, lazy_expr(other), self))
        else:
            raise ValueError("AST generation error")



    def __sub__(self, other):
        return lazy_expr((ss_sub, self, lazy_expr(other)))

    def __mul__(self, other):
        return lazy_expr((ss_mul, self, lazy_expr(other)))

    def __div__(self, other):
        return lazy_expr((ss_div, self, lazy_expr(other)))

    def __neg__(self):
        return lazy_expr((s_negate, self))

    def __pow__(self, other):
        return lazy_expr((ss_pow, self, lazy_expr(other)))

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
        top_cell = self.get_tuple()

        literal_stack = []
        op_stack = []

        def visitor(cell, literal_stack, op_stack):
            op, args = cell[0], cell[1:]
            if op == i_scalar:
                if args[0] in literal_stack:
                    op_stack.append(-literal_stack.index(args[0]))
                else:
                    literal_stack.append(args[0])
                    op_stack.append(-(len(literal_stack)-1))
            else:
                for a in args:
                    visitor(a, literal_stack, op_stack)
                op_stack.append(op)
        visitor(top_cell, literal_stack, op_stack)
        return tuple(op_stack), tuple(literal_stack)
