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

# values

    a_counter(260) : "ia_scalar",
    s_counter(563) : "i_scalar",
    # s_counter() :     "p_scalar",
    # a_counter() :     "pa_scalar",

# proxies

    #counter() :     "proxy_rad",

#################
###  vector stuff
#################

    # counter() :     "v_mag",
    # counter() :     "v_x",
    # counter() :     "v_y",
    # counter() :     "v_z",


#     counter(500) :     "vv_add",
#     counter() :     "vv_sub",
#     counter() :     "vv_mul",
#     counter() :     "vv_pow",
#     counter() :     "v_negate",

#     counter() :     "vs_add",
#     counter() :     "vs_sub",
#     counter() :     "vs_mul",
#     counter() :     "vs_pow",

#     counter() :     "sv_add",
#     counter() :     "sv_sub",
#     counter() :     "sv_mul",
#     counter() :     "sv_pow",

#     counter() :     "v_norm",
#     counter() :     "i_vector",
#     counter() :     "ia_vector",
#     counter() :     "p_vector",
#     counter() :     "pa_vector",

# # proxy values
#     counter() :     "proxy_vel",
#     counter() :     "proxy_fapply",
#     counter() :     "proxy_pos"
}

for k,v in op_hash.iteritems():
    globals()[v]=k

def _write_c_header():
    with open("ops.h", "w") as f:
        for k,v in op_hash.iteritems():
            print >> f, "#define {} {}".format(v.upper(), k)

scalar_bit = 1 << 12
array_scalar_bit = 1 << 11


if __name__ == '__main__':
    _write_c_header()
