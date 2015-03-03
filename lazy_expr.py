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

        if (o &~ op_mask) == lit_s:
            o = o & ~bytecode_mask
            print "{}:  literal load {} ({})".format(i,o,doubles[o])
        elif (o &~ op_mask) == lit_v:
            o = o & ~bytecode_mask
            print "{}:  vector load {} {}".format(i, o, doubles[o:o+3])
        elif (o & ~op_mask) == lit_as:
            o = o & ~bytecode_mask
            print "{}:  array scalar load {} (id: {})".format(i,o,id(arrays[o]))
        elif (o & ~op_mask) == lit_av:
            o = o & ~bytecode_mask
            print "{}:  array vector load {} (id: {})".format(i,o,id(arrays[o]))
        elif (o &~ heap_mask) in op_hash:
            template = "{}:  {} types({} <- {}.{}) flags:  {}{}{}"
            out = template.format(i, op_hash[o &~ heap_mask],
                                  r_stype(o), a_stype(o), b_stype(o),
                                  "r-heap " if r_heap(o) else "",
                                  "a-heap " if a_heap(o) else "",
                                  "b-heap " if b_heap(o) else "")
            print out
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

    def handle_op(a, b, base_op):
        a_type = a.r_code()
        b_type = b.r_code()
        op_str = "{}_{}_{}".format(string_types[a_type], string_types[b_type],
                                   base_op)
        assert op_str in rop_hash, "invalid operation"
        op_code = rop_hash[op_str]
        return lazy_expr((op_code, a, b))

    def __init__(self, data):
        if type(data) is tuple:
            self.data = data
        elif type(data) is lazy_expr:
            self.data = data.data
        elif type(data) is float or type(data) is int:
            self.data = (lit_s, float(data))
        elif type(data) is vec:
            self.data = (lit_v, data)
        elif type(data) is np.ndarray:
            if data.ndim == 1:
                self.data = (lit_as, data)
            else:
                assert data.ndim == 2
                assert data.shape[1] == 3
                self.data = (lit_av, data)
        else:
            raise ValueError("unknown type")

    def r_type(self):
        return (self.data[0] & r_type_mask) >> r_shift

    def a_code(self):
        "returns 0, 1, 2 or 3 for a type"
        return (self.data[0] & a_type_mask) >> a_shift
    def b_code(self):
        "returns 0, 1, 2 or 3 for b type"
        return (self.data[0] & b_type_mask) >> b_shift
    def r_code(self):
        "returns 0, 1, 2 or 3 for return type"
        return (self.data[0] & r_type_mask) >> r_shift

    def is_scalar(self):
        raise NotImplementedError

    def __eq__(self, other):
        rhs = lazy_expr(other)
        assert self.data[0] == lit_as or self.data[0] == lit_av, \
            "lhs must be an array"
        lhs = self.data[1]
        assert type(lhs) is np.ndarray
        return Assignment(lhs, rhs)

    def __add__(self, other):
        return lazy_expr.handle_op(self, lazy_expr(other), "add")
    def __radd__(self, other):
        return lazy_expr.handle_op(lazy_expr(other), self, "add")

    def __sub__(self, other):
        return lazy_expr.handle_op(self, lazy_expr(other), "sub")
    def __rsub__(self, other):
        return lazy_expr.handle_op(lazy_expr(other), self, "sub")

    def __mul__(self, other):
        return lazy_expr.handle_op(self, lazy_expr(other), "mul")
    def __rmul__(self, other):
        return lazy_expr.handle_op(lazy_expr(other), self, "mul")

    def __div__(self, other):
        return lazy_expr.handle_op(self, lazy_expr(other), "div")
    def __rdiv__(self, other):
        return lazy_expr.handle_op(lazy_expr(other), self, "div")

    def __pow__(self, other):
        return lazy_expr.handle_op(self, lazy_expr(other), "pow")
    def __rpow__(self, other):
        return lazy_expr.handle_op(lazy_expr(other), self, "pow")

    def __neg__(self):
        if self.r_type() == 0:
            return lazy_expr((s_negate, self))
        elif self.r_type() == 1:
            return lazy_expr((as_negate, self))
        elif self.r_type() == 2:
            return lazy_expr((v_negate, self))
        elif self.r_type() == 3:
            return lazy_expr((av_negate, self))


    def __repr__(self):
        return "({}, ".format(op_hash[self.data[0]]) +\
            ", ".join(map(str, self.data[1:])) +  " )"

    def get_tuple(self):
        """ return ast as nested tuples """
        if self.data[0] == lit_s or self.data[0] == lit_v:
            return self.data
        elif self.data[0] == lit_as or self.data[0] == lit_av:
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

        def listit(t):
            """Convert nested tuples into nested lists"""
            return list(map(listit, t)) if isinstance(t, (list, tuple)) else t

        top_cell = listit(self.get_tuple())

        def scalar_op(opcode):
            return r_itype(opcode) == 0

        def opt_visitor(cell):
            """Add bits to bytecodes"""
            op, args = cell[0], cell[1:]
            if scalar_op(op) or op == lit_s or op == lit_as or \
               op == lit_v or op == lit_av:
                return
            flags = 0
            if args[0][0] == lit_as or args[0][0] == lit_av:
                flags |= a_on_heap
            if len(args) == 2:
                if args[1][0] == lit_as or args[1][0] == lit_av:
                    flags |= b_on_heap
            cell[0] |= flags
            for arg in args:
                opt_visitor(arg)

        opt_visitor(top_cell)

        # write final result to heap
        if self.r_type() == 1 or self.r_type() == 3:
            top_cell[0] |= result_to_heap

        literal_stack = []
        array_id_stack = []
        array_stack = []
        op_stack = []

        def visitor(cell, literal_stack, op_stack, array_stack):
            op, args = cell[0], cell[1:]
            if op == lit_s:
                if args[0] in literal_stack:
                    op_stack.append(lit_s | literal_stack.index(args[0]))
                else:
                    literal_stack.append(args[0])
                    op_stack.append(lit_s | (len(literal_stack)-1))
            elif op == lit_v:
                # optimize this
                x,y,z = args[0]
                literal_stack.append(x)
                literal_stack.append(y)
                literal_stack.append(z)
                op_stack.append(lit_v | len(literal_stack)-3)

            elif op == lit_as or op == lit_av:
                if id(args[0]) in array_id_stack:
                    op_stack.append(op | array_id_stack.index(id(args[0])))
                else:
                    array_stack.append(args[0])
                    array_id_stack.append(id(args[0]))
                    op_stack.append(op | array_id_stack.index(id(args[0])))

            else:
                for a in args:
                    visitor(a, literal_stack, op_stack, array_stack)
                op_stack.append(op)
        visitor(top_cell, literal_stack, op_stack, array_stack)
        return np.array(op_stack, dtype=int), \
            np.array(literal_stack), \
            tuple(array_stack)
