"""The problem this repository started life as, in 2024: drop a hexagonal
coin onto a grating with five vertical slots and ask how often it falls
clean through. Back then it was a hand-rolled loop. Here it is as a
limit state on the damtol reliability engine, plus the exact answer by
quadrature that the original never had.

Geometry matches the old config.py: 10 x 10 grating, five slots of
width 1.2, hexagon of circumradius 0.4, every vertex must land inside
one slot.
"""

import numpy as np

from damtol import Uniform, estimate_pof

W, H = 10.0, 10.0
N_SLOTS, SLOT_W = 5, 1.2
R, SIDES = 0.4, 6

_SPACING = (W - N_SLOTS * SLOT_W) / (N_SLOTS + 1)
SLOT_X0 = _SPACING + np.arange(N_SLOTS) * (SLOT_W + _SPACING)
VERTEX_ANGLES = np.linspace(0.0, 2.0 * np.pi, SIDES, endpoint=False)

VARIABLES = {
    "x": Uniform(0.0, W),
    "y": Uniform(0.0, H),
    "theta": Uniform(0.0, 2.0 * np.pi),
}


def coin_limit_state(s):
    """Margin < 0 means the coin falls through some slot.

    For each slot the fit margin is the smallest clearance of any vertex
    to the slot boundary; the coin falls through if its best slot has
    positive clearance, so the limit state is the negative of that.
    """
    ang = VERTEX_ANGLES[None, :] + s["theta"][:, None]
    vx = s["x"][:, None] + R * np.cos(ang)
    vy = s["y"][:, None] + R * np.sin(ang)

    best = np.full(s["x"].shape, -np.inf)
    for x0 in SLOT_X0:
        clear = np.minimum.reduce([
            vx - x0, x0 + SLOT_W - vx, vy, H - vy,
        ]).min(axis=1)
        best = np.maximum(best, clear)
    return -best


def analytic_probability(n_theta=20001):
    """Exact answer by 1-D quadrature over the coin orientation.

    At fixed theta the hexagon spans a width w and height h; the centre
    positions that fit a slot form a (SLOT_W - w) by (H - h) rectangle,
    so the conditional probability is closed form and only the theta
    average needs numerics.
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
    ang = VERTEX_ANGLES[None, :] + theta[:, None]
    w = 2.0 * R * np.max(np.abs(np.cos(ang)), axis=1)
    h = 2.0 * R * np.max(np.abs(np.sin(ang)), axis=1)
    p_theta = (N_SLOTS * np.maximum(SLOT_W - w, 0.0) / W) * ((H - h) / H)
    return float(np.trapezoid(p_theta, theta) / (2.0 * np.pi))


def main():
    exact = analytic_probability()
    res = estimate_pof(coin_limit_state, VARIABLES, n=2**17,
                       method="sobol", seed=2024)
    print(f"exact (quadrature)   : {exact:.6f}")
    print(f"monte carlo estimate : {res.pof:.6f}  "
          f"[{res.ci_low:.6f}, {res.ci_high:.6f}], n={res.n_samples:,}")


if __name__ == "__main__":
    main()
