# operator codes

# less than 500 returns scalar

op_hash = {
    0 :     "ss_add",
    1 :     "ss_sub",
    2 :     "ss_mul",
    3 :     "ss_div",
    4 :     "s_negate",
    5 :     "ss_pow",
    6 :     "v_mag",
    7 :     "v_x",
    8 :     "v_y",
    9 :     "v_z",

# values

    10 :     "i_scalar",
    11 :     "ia__scalar",
    12 :     "p_scalar",
    13 :     "pa_scalar",

# proxies

    100 :     "proxy_rad",

#################
###  vector stuff
    #################

    500 :     "vv_add",
    501 :     "vv_sub",
    502 :     "vv_mul",
    503 :     "vv_pow",
    504 :     "v_negate",

    505 :     "vs_add",
    506 :     "vs_sub",
    507 :     "vs_mul",
    508 :     "vs_pow",

    509 :     "sv_add",
    510 :     "sv_sub",
    511 :     "sv_mul",
    512 :     "sv_pow",

    513 :     "v_norm",
    600 :     "i_vector",
    601 :     "ia_vector",
    602 :     "p_vector",
    603 :     "pa_vector",

# proxy values
    700 :     "proxy_vel",
    701 :     "proxy_fapply",
    702 :     "proxy_pos"}

for k,v in op_hash.iteritems():
    globals()[v]=k

def _write_c_header():
    with open("ops.h", "w") as f:
        for k,v in op_hash.iteritems():
            print >> f, "#define {} {}".format(v.upper(), k)

if __name__ == '__main__':
    _write_c_header()
