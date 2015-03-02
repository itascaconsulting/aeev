import itertools
import operator

# ast codes, bytecodes and flags

# type codes
scalar_type       = 0  ## 00
scalar_array_type = 1  ## 01
vector_type       = 2  ## 10
vector_array_type = 3  ## 11

a_scalar_type       = 0 << 23
a_scalar_array_type = 1 << 23
a_vector_type       = 2 << 23
a_vector_array_type = 3 << 23

b_scalar_type       = 0 << 21
b_scalar_array_type = 1 << 21
b_vector_type       = 2 << 21
b_vector_array_type = 3 << 21

r_scalar_type       = 0 << 19
r_scalar_array_type = 1 << 19
r_vector_type       = 2 << 19
r_vector_array_type = 3 << 19

# lower bit of address is also an array type flag

vector_load       = 1 << 31
vector_array_load = 1 << 30
scalar_load       = 1 << 29
scalar_array_load = 1 << 28
result_to_heap    = 1 << 27
a_on_heap         = 1 << 26
b_on_heap         = 1 << 25
a_type_mask       = 1 << 24 | 1 << 23
b_type_mask       = 1 << 22 | 1 << 21
r_type_mask       = i << 20 | 1 << 19

bytecode_mask = reduce(operator.or_, [1<<i for i in range(21,32)])

class OpCounter(object):
    def __init__(self,n=0):
        self.n = n-1
    def __call__(self, flags=0):
        self.n += 1
        return self.n | flags

op_counter = OpCounter(10)

types = "s","v","as","av"
op_cases = list(itertools.product(types, repeat=2))
op_cases = ["{}_{}_".format(x,y) for x,y in op_cases]

op_hash = {
# values
    op_counter(r_s) : "s",
    op_counter(r_v) : "v",
    op_counter() : "as",
    op_counter() : "av",
    op_counter(r_s | a_s) : "s_negate",
    op_counter(r_v | a_v) : "v_negate",
    op_counter(r_as | a_sa) : "as_negate",
    op_counter(r_av | a_av) : "av_negate",
# unary functions
    op_counter() : "v_mag",
    op_counter() : "v_norm",
    op_counter() : "av_mag",
    op_counter() : "av_norm",

}

for oper in "add,sub,mul,pow".split(","):
    op_hash.update( {op_counter() : prefix + oper for prefix in op_cases})

# we want the a, b and return types in the opcode bits

1/0


for k,v in op_hash.iteritems():
    globals()[v]=k

def _write_c_header():
    print "writing opcodes to c header file"
    with open("ops.h", "w") as f:
        for k,v in op_hash.iteritems():
            print >> f, "#define {} {}".format(v.upper(), k)



if __name__ == '__main__':
    _write_c_header()
