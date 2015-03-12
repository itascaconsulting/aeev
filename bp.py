from Lazy import Lazy
from ops import as_proxy, av_proxy, bp_rad, bp_pos, bp_force_app, bp_vel

rad = Lazy((as_proxy, bp_rad))
pos = Lazy((av_proxy, bp_pos))
force_app = Lazy((av_proxy, bp_force_app))
vel = Lazy((av_proxy, bp_vel))
