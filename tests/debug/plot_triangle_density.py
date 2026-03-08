"""Plot rho_eff along a horizontal scan through the triangular cloak.

Scans x at a fixed depth y = y_top - scan_depth, chosen so the scan
passes through the outer cloak body.  Triangle borders on the free surface
(x = x_c ± c) are drawn as vertical lines.
"""
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from rayleigh_cloak.config import SimulationConfig, DerivedParams
from rayleigh_cloak.geometry.triangular import TriangularCloakGeometry
from rayleigh_cloak.materials import rho_eff

# ── build params ──────────────────────────────────────────────────────

cfg = SimulationConfig()
params = DerivedParams.from_config(cfg)
geo = TriangularCloakGeometry.from_params(params)

# ── hardcoded scan parameters ─────────────────────────────────────────

# scan depth below free surface: midpoint between inner apex a and outer apex b
scan_depth = 0.5 * (params.a + params.b)
y_scan = params.y_top - scan_depth

# horizontal extent: ±1.5 × outer half-width, centred on cloak
x_min = params.x_c - 1.5 * params.c
x_max = params.x_c + 1.5 * params.c

# triangle base corners on the free surface (hardcoded from params)
x_left  = params.x_c - params.c
x_right = params.x_c + params.c

# inner triangle extents at scan_depth
r_inner = max(0.0, 1.0 - scan_depth / params.a) if params.a > 0 else 0.0
r_outer = max(0.0, 1.0 - scan_depth / params.b)
x_inner_l = params.x_c - r_inner * params.c
x_inner_r = params.x_c + r_inner * params.c
x_outer_l = params.x_c - r_outer * params.c
x_outer_r = params.x_c + r_outer * params.c

# ── evaluate ──────────────────────────────────────────────────────────

N = 500
xs = np.linspace(x_min, x_max, N)
pts = jnp.array([[x, y_scan] for x in xs])

rho_tensors = np.array([np.array(rho_eff(p, geo, params.rho0)) for p in pts])  # (N, 2, 2)

# ── plot ──────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 4))

# plot all four tensor components
labels = [("xx", 0, 0), ("xy", 0, 1), ("yx", 1, 0), ("yy", 1, 1)]
colors = ["steelblue", "C1", "C2", "C3"]
for (lbl, i, j), col in zip(labels, colors):
    ax.plot(xs, rho_tensors[:, i, j], color=col, lw=1.8,
            label=rf"$\rho_{{{lbl}}}$")
ax.axhline(params.rho0, color="grey", ls=":", lw=1.0, label=r"$\rho_0$")

# triangle base corners (free-surface footprint) – hardcoded
ax.axvline(x_left,  color="black", ls="--", lw=1.5, label=r"$x_c \pm c$ (surface)")
ax.axvline(x_right, color="black", ls="--", lw=1.5)

# cloak annulus boundaries at scan depth
ax.axvline(x_outer_l, color="C1", ls="-.",  lw=1.2, label="outer cloak edge")
ax.axvline(x_outer_r, color="C1", ls="-.",  lw=1.2)
if r_inner > 0:
    ax.axvline(x_inner_l, color="C3", ls=":", lw=1.2, label="inner cloak edge")
    ax.axvline(x_inner_r, color="C3", ls=":", lw=1.2)

ax.set_xlabel("x  [m]")
ax.set_ylabel(r"$\rho_\mathrm{eff}$ tensor components  [kg/m³]")
ax.set_title(
    f"Effective density along horizontal scan at depth {scan_depth:.4f} m "
    f"(y = y_top − {scan_depth:.4f})\n"
    f"a = {params.a:.4f} m,  b = {params.b:.4f} m,  c = {params.c:.4f} m"
)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = "output/triangle_density.png"
plt.savefig(out, dpi=150)
plt.show()
print(f"Saved → {out}")
