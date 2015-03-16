import itasca
import numpy as np
from itasca import ballarray as ba
import numexpr
import timeit
rho_f = 1000.0
Cd = 0.3
factor = -0.5 * rho_f * Cd * np.pi
pipe_r = 1.0

itasca.command("""
new
set fish autocreate off

domain extent -5 5 -2 2 -2 2
cmat default model linear property kn 1e4 dp_nratio 0.2

ball generate box -3.5 3.5 -{rad} {rad} -{rad} {rad} gauss radius 0.001 0.002 number 100000

def fish_force_comp()
  local rho_f = 1000.0
  local Cd = 0.3
  local factor = -0.5 * rho_f * Cd * math.pi
  local pipe_r = 1.0
  loop foreach local ball ball.list
    local rad = ball.radius(ball)
    local pos_y = ball.pos.y(ball)
    local pos_z = ball.pos.z(ball)
    local pos_r = math.sqrt(pos_y^2 + pos_z^2)
    local ux = ball.vel.x(ball)
    local uy = ball.vel.y(ball)
    local uz = ball.vel.z(ball)
    ux = ux - (pos_r - pipe_r)^2
    local fx = factor * rad^2 * ux * math.abs(ux)
    local fy = factor * rad^2 * uy * math.abs(uy)
    local fz = factor * rad^2 * uz * math.abs(uz)
    ball.force.app.x(ball) = fx
    ball.force.app.y(ball) = fy
    ball.force.app.z(ball) = fz
  end_loop
end

def fish_force_vec()
  local rho_f = 1000.0
  local Cd = 0.3
  local factor = -0.5 * rho_f * Cd * math.pi
  local pipe_r = 1.0
  loop foreach local ball ball.list
    local rad = ball.radius(ball)
    local pos_y = ball.pos.y(ball)
    local pos_z = ball.pos.z(ball)
    local pos_r = math.sqrt(pos_y^2 + pos_z^2)
    local u = ball.vel(ball)
    local v = vector((pos_r - pipe_r)^2, 0.0, 0.0)
    local rel = u-v
    local force = factor * rad^2 * rel * math.mag(rel)
    ball.force.app(ball) = force
  end_loop
end

""".format(rad=0.707))

def py_force_vec():
    for b in itasca.ball.list():
        r = b.radius()
        pos_y, pos_z = b.pos_y(), b.pos_z()
        pos_r = math.sqrt(pos_y**2 + pos_z**2)
        u = b.vel()
        vx = (pos_r - pipe_r)**2
        u[0] -= vx
        force = factor * r**2 * u * u.mag()
        b.set_force_app(force)


def py_force_comp():
    for b in itasca.ball.list():
        r = b.radius()
        pos_y, pos_z = b.pos_y(), b.pos_z()
        pos_r = math.sqrt(pos_y**2 + pos_z**2)
        ux,uy,uz = b.vel_x(), b.vel_y(), b.vel_z()
        ux -= (pos_r - pipe_r)**2
        fx = factor * r**2 * ux * abs(ux)
        fy = factor * r**2 * uy * abs(uy)
        fz = factor * r**2 * uz * abs(uz)
        b.set_force_app_x(fx)
        b.set_force_app_y(fy)
        b.set_force_app_z(fz)


def numpy_force():
    r = ba.radius()
    pos = ba.pos()
    pos[:,0] = 0.0
    pos_r = np.linalg.norm(pos,axis=1)
    u = ba.vel()
    v = np.zeros_like(u)
    v[:,0] = (pos_r-pipe_r)**2
    rel = u-v
    force = factor * r**2 * rel.T * np.linalg.norm(rel, axis=1)
    ba.set_force_app(force.T)

def numexpr_force():
    r = ba.radius()
    pos = ba.pos()
    #pos[:,0] = 0.0
    pos_y = pos[:,1]
    pos_z = pos[:,2]
    pos_r = numexpr.evaluate("sqrt(pos_y**2+pos_z**2)")
    u = ba.vel()
    v = np.zeros_like(u)
    v[:,0] = (pos_r-pipe_r)**2
    rel = u-v
    rel_x, rel_y, rel_z = rel[:,0], rel[:,1], rel[:,2]
    rel_mag = numexpr.evaluate("sqrt(rel_x**2 + rel_y**2 + rel_z**2)")
    relt = rel.T
    force = numexpr.evaluate("factor *r**2 *relt*rel_mag")
    ba.set_force_app(force.T)


from vec import vec
import itasca as it
from itasca import ball
from Lazy import Lazy, dis, mag
import bp
import numpy as np
import itasca.ballarray as ba

mask_notx = vec((0,1,1))
mask_x = vec((1,0,0))
#r_pos = (bp.pos_x**2 + bp.pos_y**2)**0.5
r_pos = mag(bp.pos*mask_notx)
fluid_vel = (r_pos-pipe_r)**2*mask_x
rel = bp.vel-fluid_vel
expr = bp.force_app == factor * bp.rad**2*(rel) * mag(rel)

def aeev_test():
    expr.vm_eval()
    
ba.set_force_app(np.zeros((it.ball.count(),3)))
aeev_test()
v0 = ba.force_app()
numpy_force()
np.testing.assert_allclose(v0, ba.force_app())
