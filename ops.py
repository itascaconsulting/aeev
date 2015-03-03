import itertools
import operator

# ast codes, bytecodes and flags

# type codes
s_type  = 0  ## 00
as_type = 1  ## 01
v_type  = 2  ## 10
av_type = 3  ## 11

a_s  = s_type  << 23
a_as = as_type << 23
a_v  = v_type  << 23
a_av = av_type << 23

b_s  = s_type  << 21
b_as = as_type << 21
b_v  = v_type  << 21
b_av = av_type << 21

r_s  = s_type  << 19
r_as = as_type << 19
r_v  = v_type  << 19
r_av = av_type << 19

# lower bit of address is also an array type flag

vector_load       = 1 << 31
vector_array_load = 1 << 30
scalar_load       = 1 << 29
scalar_array_load = 1 << 28
result_to_heap    = 1 << 27
a_on_heap         = 1 << 26
b_on_heap         = 1 << 25
a_shift = 23
b_shift = 21
r_shift = 19
a_type_mask       = 1 << 24 | 1 << 23
b_type_mask       = 1 << 22 | 1 << 21
r_type_mask       = 1 << 20 | 1 << 19

bytecode_mask = reduce(operator.or_, [1<<i for i in range(21,32)])

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
    op_counter(r_s) : "lit_s",
    op_counter(r_v) : "lit_v",
    op_counter(r_as) : "lit_as",
    op_counter(r_av) : "lit_av",
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


for k,v in op_hash.iteritems():
    globals()[v]=k

def _write_c_header():
    print "writing opcodes to c header file"
    with open("ops.h", "w") as f:
        for k,v in op_hash.iteritems():
            print >> f, "#define {} {}".format(v.upper(), k)



if __name__ == '__main__':
    _write_c_header()
