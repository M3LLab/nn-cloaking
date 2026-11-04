"""Closed-form reference for case A (homogeneous plane-strain full space).

For mu Lap u + (lam + mu) grad div u + rho omega^2 u = -F delta(x) the Green
tensor is

    G = (1/mu) [ g_s I + (1/k_s^2) grad grad (g_s - g_p) ],
    (Lap + k^2) g = -delta,   g(r) = (-i/4) H0^(2)(k r)     (time factor e^{+i w t}).

A radially symmetric Gaussian source of unit integral multiplies each
outgoing Hankel term by exp(-k^2 sigma^2 / 2) exactly (Hankel transform of a
Gaussian), up to the Gaussian tail beyond the evaluation radius.
"""

from __future__ import annotations

import numpy as np
from scipy.special import hankel2

from rayleigh_cloak.embedded.config import EmbeddedCloakConfig


def green_gaussian(cfg: EmbeddedCloakConfig, xy: np.ndarray) -> np.ndarray:
    """Complex displacement (n, 2) of the configured Gaussian source in the
    unbounded homogeneous background."""
    m, s = cfg.material, cfg.source
    w = cfg.omega
    ks = w / m.cs
    kp = w / m.cp
    d = np.asarray(s.direction, float)
    d = s.amplitude * d / np.linalg.norm(d)
    dx = np.asarray(xy)[:, 0] - s.x
    dy = np.asarray(xy)[:, 1] - s.y
    r = np.hypot(dx, dy)
    xh = np.stack([dx / r, dy / r], axis=1)

    def parts(k):
        c = -0.25j * np.exp(-0.5 * (k * s.sigma) ** 2)
        z = k * r
        h0, h1 = hankel2(0, z), hankel2(1, z)
        g = c * h0
        gp = -c * k * h1                         # g'(r)
        gpp = -c * k * k * (h0 - h1 / z)         # g''(r)
        return g, gp, gpp

    gs, gsp, gspp = parts(ks)
    _, gpp_, gppp = parts(kp)
    # grad grad (g_s - g_p) = h'' xh xh + (h'/r)(I - xh xh)
    hp = gsp - gpp_
    hpp = gspp - gppp
    xd = xh @ d
    ggd = (hpp - hp / r)[:, None] * xh * xd[:, None] + (hp / r)[:, None] * d[None, :]
    return (gs[:, None] * d[None, :] + ggd / ks ** 2) / m.mu
