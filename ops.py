# operator codes
ss_add = 0
ss_sub = 1
ss_mul = 2
ss_div = 3
s_negate = 4
ss_pow = 5

vv_add = 100
vv_sub = 101
vv_mul = 102
vv_pow = 103

vs_add = 200
vs_sub = 201
vs_mul = 202
vss_pow = 203

sv_add = 300
sv_sub = 301
sv_mul = 302
sv_pow = 303

#v_mag
#v_norm

# value codes (unary)
i_scalar = 400
i_vector = 401
ia_scalar = 402
ia_vector = 403

p_scalar = 500
p_vector = 501
pa_scalar = 502
pa_vector = 503

# proxy values
proxy_vel = 600
proxy_rad = 601

op_hash = {0 : "ss_add",
 1 : "ss_sub",
 2 : "ss_mul",
 3 : "ss_div",
 4 : "s_negate",
 5 : "ss_pow",
 400 : "i_scalar"}
