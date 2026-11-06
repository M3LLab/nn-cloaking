"""Why does the bare-hole ratio <|u|>/<|u_ref|> exceed 1 at low frequency?

Splits the field beyond the cloak into scattered energy and interference with the
incident wave, and the reference field into P and S parts, at source (-3, -3).

    .venv/bin/python scripts/circular_ratio_diagnostic.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import yaml
from scipy.special import hankel2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]

from circular_frequency_sweep import FORCES, frequency_config  # noqa: E402
from rayleigh_cloak.embedded.config import load_embedded_config  # noqa: E402
from rayleigh_cloak.embedded.linsolve import solve_linear_directions  # noqa: E402
from rayleigh_cloak.embedded.mesh import build_mesh  # noqa: E402
from rayleigh_cloak.embedded.problem import build_problem  # noqa: E402
from rayleigh_cloak.embedded.sampling import PointEvaluator, exterior_grid, magnitude  # noqa: E402

FSTARS = (0.4, 1.0)
HALF_WIDTH = 2.0   # |perpendicular distance| < this counts as "near the axis"


def p_part(cfg, P, d):
    """P part of the analytic reference field: -(1/(mu ks^2)) grad grad g_p . d."""
    m, s = cfg.material, cfg.source
    ks, kp = cfg.omega / m.cs, cfg.omega / m.cp
    d = s.amplitude * np.asarray(d, float) / np.linalg.norm(d)
    dx, dy = P[:, 0] - s.x, P[:, 1] - s.y
    r = np.hypot(dx, dy)
    xh = np.stack([dx / r, dy / r], 1)
    c = -0.25j * np.exp(-0.5 * (kp * s.sigma) ** 2)
    z = kp * r
    h0, h1 = hankel2(0, z), hankel2(1, z)
    g1, g2 = -c * kp * h1, -c * kp * kp * (h0 - h1 / z)
    ggd = (g2 - g1 / r)[:, None] * xh * (xh @ d)[:, None] + (g1 / r)[:, None] * d[None, :]
    return -ggd / (ks ** 2 * m.mu)


def main():
    sweep = yaml.safe_load(open(os.path.join(ROOT, "configs/circular_embedded/frequency_sweep.yaml")))
    base = load_embedded_config(os.path.join(ROOT, sweep["base_config"]))
    out_dir = os.path.join(ROOT, "output/circular_frequency_sweep")
    out = {}
    for fstar in FSTARS:
        cfg = frequency_config(base, sweep, fstar, sweep["sources"]["on_diagonal"], out_dir)
        c, s = cfg.cloak, cfg.source
        P = exterior_grid(cfg)
        sd = np.array([c.xc - s.x, c.yc - s.y])
        sd /= np.linalg.norm(sd)
        rel = P - [c.xc, c.yc]
        along, perp = rel @ sd, rel @ np.array([-sd[1], sd[0]])
        k = along > c.b
        full = build_mesh(cfg, msh_path=os.path.join(out_dir, "_diag.msh"), verbose=False)
        sub, _ = full.without_disk()
        F = {}
        for case, mesh in (("A", full), ("B", sub)):
            pr = build_problem(mesh, cfg, cloak_on=False)
            (uV, uH), _ = solve_linear_directions(pr, [FORCES["vertical"], FORCES["horizontal"]])
            del pr
            ev = PointEvaluator(mesh.points, mesh.cells, P)
            V, H = ev(uV[:, :2] + 1j * uV[:, 2:]), ev(uH[:, :2] + 1j * uH[:, 2:])
            F[case] = {"vertical": V, "horizontal": H, "both": (V + H) / np.sqrt(2)}
        out[f"{fstar:g}"] = {}
        for force in FORCES:
            A, B = F["A"][force], F["B"][force]
            sc = B - A
            ma, mb = magnitude(A), magnitude(B)
            up = p_part(cfg, P, FORCES[force])
            us = A - up
            near, side = k & (np.abs(perp) < HALF_WIDTH), k & (np.abs(perp) >= HALF_WIDTH)
            eP, eS = (np.abs(up[k]) ** 2).sum(), (np.abs(us[k]) ** 2).sum()
            den = np.mean(ma[k] ** 2)
            out[f"{fstar:g}"][force] = dict(
                ratio=float(mb[k].mean() / ma[k].mean()),
                ratio_near_axis=float(mb[near].mean() / ma[near].mean()),
                ratio_sides=float(mb[side].mean() / ma[side].mean()),
                ref_mean=float(ma[k].mean()), p_share=float(eP / (eP + eS)),
                scattered_over_ref=float(magnitude(sc)[k].mean() / ma[k].mean()),
                scattered_energy=float(np.mean((np.abs(sc[k]) ** 2).sum(1)) / den),
                interference=float(np.mean(2 * np.real((np.conj(A[k]) * sc[k]).sum(1))) / den),
                frac_points_above=float(np.mean(mb[k] > ma[k])))
    path = os.path.join(out_dir, "ratio_diagnostic.json")
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
