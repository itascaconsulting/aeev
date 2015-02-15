from ops import *
import aeev

class lazy_expr(object):
    def __init__(self, data):
        if isinstance(data, tuple):
            self.data = data
        elif isinstance(data, lazy_expr):
            self.data = data.data
        elif isinstance(data, float) or isinstance(data, int):
            self.data = (i_scalar, float(data))
        else:
            raise ValueError("unknown type")

    def __add__(self, other):
        return lazy_expr((ss_add, self, lazy_expr(other)))

    def __sub__(self, other):
        return lazy_expr((ss_sub, self, lazy_expr(other)))

    def __mul__(self, other):
        return lazy_expr((ss_mul, self, lazy_expr(other)))

    def __div__(self, other):
        return lazy_expr((ss_mul, self, lazy_expr(other)))

    def __neg__(self, other):
        return lazy_expr((s_negate, self))

    def __pow__(self, other):
        return lazy_expr((ss_pow, self, lazy_expr(other)))

    def __repr__(self):
        return "({}, ".format(op_hash[self.data[0]]) +\
            ", ".join(map(str, self.data[1:])) +  " )"

    def get_tuple(self):
        if self.data[0] == i_scalar:
            return self.data
        else:
            if len(self.data) == 2:
                return (self.data[0],
                        self.data[1].get_tuple())
            if len(self.data) == 3:
                return (self.data[0],
                        self.data[1].get_tuple(),
                        self.data[2].get_tuple())

a = lazy_expr(1.0)
b = lazy_expr(3.0)
c = lazy_expr(4.0)

expr = (a + b)**c
print expr
print expr.get_tuple()
print aeev.eval(expr.get_tuple())

# 1+2
expr = (ss_add, (i_scalar, 1.0), (i_scalar, 2.0))
print expr
print aeev.eval(expr)

# (1.23+5.0)**2 - 5e2/0.1
expr = (ss_sub, (ss_pow, (ss_add, (i_scalar, 1.23), (i_scalar, 5.0)),
                 (i_scalar, 2.0)),
        (ss_div, (i_scalar, 5e2), (i_scalar, 0.1)))
print expr, (1.23+5.0)**2 - 5e2/0.1

print aeev.eval(expr)


# example expression
# Cd = 0.4
# expr = (sv_mul,
#         (i_scalar, Cd),
#         (sv_mul,
#          (ss_pow, (pa_scalar, proxy_rad), (i_scalar, 2.0)),
#          (pa_vector, proxy_vel)))

# print expr

# evaluator:
# loop over the ball container, i is the number in the sequence
# i and bp are globals to start with (or passed everywhere)
# take a tuple, first value is the operator rest are the arguments
# c functions corresponding to the operators

# eval_dvect (1 or 2, dv or scalar) -> v
# eval_scalar(1 or 2, dv or scalar) -> s
# recursively call these functions based on the function
