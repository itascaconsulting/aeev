import numpy as np
from _vec import vec3 as vec
from Lazy import Lazy, dis
from ops import *

_a = vec((1,2,3))
_b = np.array(((1,2,3.0),(4,5,6.0), (7,8,9.0), (10,11,12.)))
_target  = np.zeros_like(_b)

a = Lazy(_a)
b = Lazy(_b)
target = Lazy(_target)


import aeev
expr = ((a+1)+vec((11,12,13)))
#dis(expr)
op, l, _ = expr.get_bytecode()
print aeev.vm_eval(tuple(op), tuple(l))
((_a+1)+vec((11,12,13))).assert_close(aeev.vm_eval(tuple(op), tuple(l)))

expr = target == -b
dis(expr)
print expr.vm_eval()

np.testing.assert_allclose((target == b+1).vm_eval(), _b+1)
np.testing.assert_allclose((target == 1+b).vm_eval(), 1+_b)
np.testing.assert_allclose((target == b-1).vm_eval(), _b-1)
np.testing.assert_allclose((target == 1-b).vm_eval(), 1-_b)
np.testing.assert_allclose((target == b*3).vm_eval(), _b*3)
np.testing.assert_allclose((target == 3*b).vm_eval(), 3*_b)
np.testing.assert_allclose((target == b/a).vm_eval(), _b/_a)
np.testing.assert_allclose((target == 3*a*(0.5*b/a-a)).vm_eval(),
                           3*_a*(0.5*_b/_a-_a))


_c = np.ones((1e6,3))*0.1
_d = np.ones((1e6,3))*0.5
_e = np.ones((1e6))*2
_f = np.ones((1e6))*3
_g = vec((4,5,6))
_starget = np.zeros_like(_e)
_target = np.zeros_like(_c)
c,d,e,f,g,target,starget = map(Lazy, [_c,_d,_e,_f,_g,_target,_starget])

expr = target == c * (e+f) * (d-e)**2
dis(expr)
expr.vm_eval()

np.testing.assert_allclose(expr.vm_eval(), (_c.T * (_e+_f) * (_d.T-_e)**2).T)
np.testing.assert_allclose((starget==e**2).vm_eval(), _e**2)

# array shape checking!!

expr = target==g*(e*((c+d)**2 + (0.1*c-d*10)) +  f*((c-d)**2 + (0.5*c-d*12)))

# glitch in vec * np array
def t():
    return (_e*((_c.T+_d.T)**2 + (0.1*_c.T-_d.T*10)) +  _f*((_c.T-_d.T)**2 + (0.5*_c.T-_d.T*12))).T*_g

np.testing.assert_allclose(expr.vm_eval(),t())
