from Lazy import Lazy, dis, exp
from math import exp as _exp
import numpy as np
#import aeev

_a = 1.0
a = Lazy(1.0)
_b = np.linspace(0,1,1e6)
b = Lazy(_b)
_c = np.linspace(1,2,1e6)
c = Lazy(_c)
_target = np.zeros_like(_b)
target = Lazy(_target)

expr = target == b+c
dis(expr)
print expr.vm_eval()
np.testing.assert_allclose(expr.vm_eval(), _b+_c)

expr = target == (a+b**2)+22.2
np.testing.assert_allclose(expr.vm_eval(), (1.0+_b**2)+22.2)

expr = target == b+b+b
np.testing.assert_allclose(expr.vm_eval(), _b + _b + _b)

expr = target == b+1.23
np.testing.assert_allclose(expr.vm_eval(), _b + 1.23)

expr = target == b+1.23*c+b/2.0
np.testing.assert_allclose(expr.vm_eval(), _b+1.23*_c+_b/2.0)

expr = target == b/(c*c)+1.23*c+(b*a*c+1.223*(b+c))/2.0
np.testing.assert_allclose(expr.vm_eval(),
                           _b/(_c*_c)+1.23*_c+(_b*1.0*_c+1.223*(_b+_c))/2.0)

expr = target == -b
np.testing.assert_allclose(expr.vm_eval(), -_b)

expr = target == c * (1 + -b/c)
np.testing.assert_allclose(expr.vm_eval(), _c*(1+-_b/_c))

expr = target == exp(c)
np.testing.assert_allclose(expr.vm_eval(), np.exp(_c))
