import itertools
import operator

# ast codes, bytecodes and flags

# type codes
s_type  = 0  ## 00
as_type = 1  ## 01
v_type  = 2  ## 10
av_type = 3  ## 11

a_shift        = 22
b_shift        = 20
r_shift        = 18
a_s  = s_type  << a_shift
a_as = as_type << a_shift
a_v  = v_type  << a_shift
a_av = av_type << a_shift

b_s  = s_type  << b_shift
b_as = as_type << b_shift
b_v  = v_type  << b_shift
b_av = av_type << b_shift

r_s  = s_type  << r_shift
r_as = as_type << r_shift
r_v  = v_type  << r_shift
r_av = av_type << r_shift

s_load         = 1 << 30
as_load        = 1 << 29
v_load         = 1 << 28
av_load        = 1 << 27
result_to_heap = 1 << 26
a_on_heap      = 1 << 25
b_on_heap      = 1 << 24
a_type_mask    = 1 << 23 | 1 << 22
b_type_mask    = 1 << 21 | 1 << 20
r_type_mask    = 1 << 19 | 1 << 18

bytecode_mask = reduce(operator.or_, [1<<i for i in range(18,31)])
op_mask = ~bytecode_mask
heap_mask = 1 << 26 | 1 << 25 | 1 << 24


def wb(n):
    s = "{:b}".format(n)[::-1]
    res = []
    for i,c in enumerate(s):
        if c=='1':
            res.append(i)
    return res[::-1]

class OpCounter(object):
    def __init__(self,n=0):
        self.n = n-1
    def __call__(self, flags=0):
        self.n += 1
        return self.n | flags

string_types = "s","as","v","av"
op_cases = list(itertools.product(string_types, repeat=2))

def flags(atype, btype):
    """ returns the flags that should be set for this operation """
    ahash = {"s" : a_s, "as" : a_as, "v": a_v, "av" : a_av}
    bhash = {"s" : b_s, "as" : b_as, "v": b_v, "av" : b_av}
    flags = ahash[atype] | bhash[btype]
    a_base = ahash[atype] >> a_shift
    b_base = bhash[btype] >> b_shift
    rtype = 0
    vector_bit = 0b10
    array_bit = 0b01
    if vector_bit & a_base or vector_bit & b_base:
        rtype |= vector_bit
    if array_bit & a_base or array_bit & b_base:
        rtype |= array_bit
    return flags | (rtype << r_shift)

op_counter = OpCounter(10)
op_hash = {
# values
    r_s  | s_load   :  "lit_s",
    r_v  | v_load   :  "lit_v",
    r_as | as_load  :  "lit_as",
    r_av | av_load  :  "lit_av",
# unary ops
    op_counter(r_s | a_s) : "s_negate",
    op_counter(r_v | a_v) : "v_negate",
    op_counter(r_as | a_as) : "as_negate",
    op_counter(r_av | a_av) : "av_negate",
# unary functions
    op_counter(r_s | a_v) : "v_mag",
    op_counter(r_v | a_v) : "v_norm",
    op_counter(r_as | a_av) : "av_mag",
    op_counter(r_av | a_av) : "av_norm",
}


for oper in "add,sub,mul,pow".split(","):
    for op0, op1 in op_cases:
        op_hash.update( {op_counter(flags(op0, op1)) :
                         "{}_{}_{}".format(op0,op1,oper)})

rop_hash = {v:k for k,v in op_hash.iteritems()}

def a_itype(opcode): return (opcode & a_type_mask) >> a_shift
def a_stype(opcode): return string_types[a_itype(opcode)]

def b_itype(opcode): return (opcode & b_type_mask) >> b_shift
def b_stype(opcode): return string_types[b_itype(opcode)]

def r_itype(opcode): return (opcode & r_type_mask) >> r_shift
def r_stype(opcode): return string_types[r_itype(opcode)]

def a_heap(opcode): return opcode & a_on_heap
def b_heap(opcode): return opcode & b_on_heap
def r_heap(opcode): return opcode & result_to_heap


for k,v in op_hash.iteritems():
    globals()[v]=k

def _write_c_header():
    print "writing opcodes to c header file"
    with open("ops.h", "w") as f:
        for k,v in op_hash.iteritems():
            print >> f, "#define {} {}".format(v.upper(), k)



if __name__ == '__main__':
    _write_c_header()
