import numpy as np

# operator codes
ss_add = 0
ss_sub = 1
ss_mul = 2
s_pow = 3

vv_add = 100
vv_sub = 101
vv_mul = 102
vv_pow = 103

vs_add = 200
vs_sub = 201
vs_mul = 202
vs_pow = 203

sv_add = 300
sv_sub = 301
sv_mul = 302
sv_pow = 303

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

import aeev

expr = (ss_add, (i_scalar, 1.0), (i_scalar, 2.0))
print expr
print aeev.eval(expr)

# example expression
Cd = 0.4
expr = (sv_mul,
        (i_scalar, Cd),
        (sv_mul,
         (s_pow, (pa_scalar, proxy_rad), (i_scalar, 2.0)),
         (pa_vector, proxy_vel)))

print expr

# evaluator:
# loop over the ball container, i is the number in the sequence
# i and bp are globals to start with (or passed everywhere)
# take a tuple, first value is the operator rest are the arguments
# c functions corresponding to the operators

# eval_dvect (1 or 2, dv or scalar) -> v
# eval_scalar(1 or 2, dv or scalar) -> s
# recursively call these functions based on the function
