from vec import vec
import itasca as it
from itasca import ball
from Lazy import Lazy, dis
import bp
import numpy as np
import itasca.ballarray as ba

it.command("""
new
domain extent -10 10 
cmat default model linear property kn 1e1 dp_nratio 0.2
""")

origin = vec((0.0, 0.0, 0.0))
rad = 0.1
for i in range(10):
    b =  ball.create(0.1, origin+i/5.0)

target = Lazy(np.zeros(it.ball.count()))
v_target = Lazy(np.zeros((it.ball.count(),3)))

expr = bp.force_app == 0.5 * bp.rad**2 * bp.vel
dis(expr)


expr = target == bp.rad + 1
dis(expr)
expr.vm_eval()
print target
np.testing.assert_allclose(target.data[1], ba.radius()+1)

expr = v_target == bp.pos + bp.rad + 0.1
dis(expr)
expr.vm_eval()
np.testing.assert_allclose(v_target.data[1], (ba.pos().T+ba.radius()+0.1).T)