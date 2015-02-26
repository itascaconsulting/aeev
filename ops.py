# operator codes

# less than 500 returns scalar

#ops of zero or less than zero are literal indicies

class OpCounter(object):
    def __init__(self,n=0):
        self.n = n-1
    def __call__(self, n=None):
        if not n is None:
            self.n = n-1
        self.n += 1
        return self.n

s_counter = OpCounter(200)
a_counter = OpCounter(500)

op_hash = {
    # +
    a_counter(200) : "a_a_add",
    a_counter() :    "a_s_add",
    a_counter() :    "s_a_add",
    s_counter(503) : "s_s_add",

    # -
    a_counter(210) : "a_a_sub",
    a_counter() :    "a_s_sub",
    a_counter() :    "s_a_sub",
    s_counter(513) : "s_s_sub",

    # *
    a_counter(220) : "a_a_mul",
    a_counter() :    "a_s_mul",
    a_counter() :    "s_a_mul",
    s_counter(523) : "s_s_mul",

    # /
    a_counter(230) : "a_a_div",
    a_counter() :    "a_s_div",
    a_counter() :    "s_a_div",
    s_counter(533) : "s_s_div",

    # **
    a_counter(250) : "a_a_pow",
    a_counter() :    "a_s_pow",
    a_counter() :    "s_a_pow",
    s_counter(553) : "s_s_pow",

    # unary -
    a_counter(240) : "a_negate",
    s_counter(243) : "s_negate",

# bytecode bits
    1 << 11 : "array_scalar_bit",
    1 << 12 : "scalar_bit",
    1 << 13 : "result_to_target",
    1 << 14 : "right_on_heap",
    1 << 15 : "left_on_heap",
    1 << 16 : "right_array",
    1 << 17 : "left_array",
    1 << 13 | 1 << 14 | 1 << 15 | 1 << 16 | 1 << 17 : "code_mask",
    1 << 11 | 1<<12 | 1<<13 | 1 << 14 | 1<<15 | 1<<16 | 1<<17 : "bytecode_mask",

# values
    a_counter(260) : "ia_scalar",
    s_counter(563) : "i_scalar",
}

for k,v in op_hash.iteritems():
    globals()[v]=k

def _write_c_header():
    print "writing opcodes to c header file"
    with open("ops.h", "w") as f:
        for k,v in op_hash.iteritems():
            print >> f, "#define {} {}".format(v.upper(), k)



if __name__ == '__main__':
    _write_c_header()
