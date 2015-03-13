from Lazy import Lazy
from ops import as_proxy, av_proxy, bp_rad, bp_pos, bp_force_app, bp_vel
from ops import bp_pos_x, bp_pos_y, bp_pos_z

rad = Lazy((as_proxy, bp_rad))
pos = Lazy((av_proxy, bp_pos))
pos_x = Lazy((as_proxy, bp_pos_x))
pos_y = Lazy((as_proxy, bp_pos_y))
pos_z = Lazy((as_proxy, bp_pos_z))
force_app = Lazy((av_proxy, bp_force_app))
vel = Lazy((av_proxy, bp_vel))
