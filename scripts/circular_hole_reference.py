"""Bare-hole errors at the optimisation sources and frequencies, for a fair comparison.

    .venv/bin/python scripts/circular_hole_reference.py
"""
import json, os, sys
import numpy as np
import yaml
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "scripts")]
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from circular_optimize import config_for
from rayleigh_cloak.embedded.config import load_embedded_config
from rayleigh_cloak.embedded.memguard import release_memory
from rayleigh_cloak.embedded.optimize_pixels import FrequencyCase
from rayleigh_cloak.embedded.pixels import PolarPixelGrid
from rayleigh_cloak.materials import C_iso

S = yaml.safe_load(open(os.path.join(ROOT, "configs/circular_embedded/step2.yaml")))
base = load_embedded_config(os.path.join(ROOT, S["base_config"]))
out = os.path.join(ROOT, S["output_dir"], "hole_reference")
os.makedirs(out, exist_ok=True)
sources = {g: [tuple(p) for p in S["source_groups"][g]] for g in ("train", "val", "test")}
rows = []
for f in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
    cfg = config_for(base, f, "quadratic", out, S["optimize"])
    case = FrequencyCase(cfg, PolarPixelGrid.from_config(cfg, 1, 1), sources, msh_path=os.path.join(out, "_m.msh"))
    m = cfg.material
    C = jnp.broadcast_to(C_iso(m.lam, m.mu), (1, 2, 2, 2, 2))
    for r in case.report(C, jnp.array([m.rho])):
        r["f_star"] = f
        rows.append(r)
    q = [r["e"] for r in rows if r["f_star"] == f and r["group"] == "test"]
    print(f"f*={f:g} hole, test e median {np.median(q):.4f} worst {max(q):.4f}", flush=True)
    del case
    release_memory()
json.dump(rows, open(os.path.join(out, "hole.json"), "w"), indent=1)
