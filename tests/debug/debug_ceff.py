"""Quick diagnostic: check whether C_eff actually varies inside the cloak."""
import jax
import jax.numpy as jnp
import numpy as np

from boundaries import (
    C_eff, C0, _in_cloak, _in_defect, F_tensor, rho_eff, rho0,
    x_c, y_top, a, b, c, x_off, y_off, W, H, mesh,
)

# ── 1. Check cloak geometry  ──
print("=== Cloak geometry (extended coords) ===")
print(f"  x_c = {float(x_c):.4f}  (cloak centre)")
print(f"  y_top = {float(y_top):.4f}")
print(f"  a = {float(a):.4f}  (inner depth),  b = {float(b):.4f}  (outer depth),  c = {float(c):.4f}  (half-width)")
print(f"  type(a) = {type(a)}")
print(f"  Inner apex:  ({float(x_c):.3f}, {float(y_top) - float(a):.3f})")
print(f"  Outer apex:  ({float(x_c):.3f}, {float(y_top) - float(b):.3f})")

# ── 2. Test a few hand-picked points  ──
_a, _b, _c = float(a), float(b), float(c)
test_pts = {
    "mid-cloak (axis)":   jnp.array([x_c, y_top - (_a + _b) / 2]),
    "inner boundary":     jnp.array([x_c, y_top - _a]),
    "outer boundary":     jnp.array([x_c, y_top - _b]),
    "just inside cloak":  jnp.array([x_c, y_top - _a - 0.01]),
    "outside (deep)":     jnp.array([x_c, y_top - _b - 0.5]),
    "outside (surface)":  jnp.array([x_c, y_top - 0.01]),
    "far from cloak":     jnp.array([x_off + 1.0, y_top - 1.0]),
}

print("\n=== Point-by-point check ===")
for label, pt in test_pts.items():
    in_c = _in_cloak(pt)
    in_d = _in_defect(pt)
    F = F_tensor(pt)
    Ce = C_eff(pt)
    diff = float(jnp.max(jnp.abs(Ce - C0)))
    rho = float(rho_eff(pt))
    print(f"  {label:25s}  in_cloak={bool(in_c)}  in_defect={bool(in_d)}  "
          f"max|C_eff-C0|={diff:.6e}  rho_eff={rho:.1f}  "
          f"F=diag({float(F[0,0]):.3f},{float(F[1,1]):.3f}) F21={float(F[1,0]):.4f}")

# ── 3. Check on actual mesh quadrature points  ──
from jax_fem.problem import Problem

class Dummy(Problem):
    def custom_init(self): pass
    def get_tensor_map(self):
        def f(u_grad): return jnp.zeros_like(u_grad)
        return f

dummy = Dummy(mesh=mesh, vec=2, dim=2, ele_type='TRI3')
qpts = dummy.physical_quad_points  # (num_cells, num_quads, 2)
print(f"\n=== Quadrature points: {qpts.shape} ===")

in_cloak_flags = jax.vmap(jax.vmap(_in_cloak))(qpts)
n_in_cloak = int(jnp.sum(in_cloak_flags))
n_total = int(np.prod(in_cloak_flags.shape))
print(f"  Quad points in cloak: {n_in_cloak} / {n_total}  ({100*n_in_cloak/n_total:.2f}%)")

if n_in_cloak == 0:
    print("\n  *** WARNING: No quadrature points fall inside the cloak region! ***")
    mid_cloak = jnp.array([x_c, y_top - (_a + _b) / 2])
    dists = jnp.sqrt(jnp.sum((qpts - mid_cloak[None, None, :]) ** 2, axis=-1))
    min_dist = float(jnp.min(dists))
    idx = np.unravel_index(int(jnp.argmin(dists.flatten())), dists.shape)
    closest_pt = qpts[idx[0], idx[1]]
    print(f"  Closest quad pt to mid-cloak: ({float(closest_pt[0]):.4f}, {float(closest_pt[1]):.4f})  dist={min_dist:.6f}")
    print(f"  _in_cloak(closest) = {bool(_in_cloak(closest_pt))}")
else:
    C_all = jax.vmap(jax.vmap(C_eff))(qpts)
    diff_all = jnp.max(jnp.abs(C_all - C0[None, None, :, :, :, :]), axis=(-1, -2, -3, -4))
    max_diff_in_cloak = float(jnp.max(jnp.where(in_cloak_flags, diff_all, 0.0)))
    max_diff_outside = float(jnp.max(jnp.where(~in_cloak_flags, diff_all, 0.0)))
    print(f"  max|C_eff - C0| inside cloak:  {max_diff_in_cloak:.6e}")
    print(f"  max|C_eff - C0| outside cloak: {max_diff_outside:.6e}")

    cell_idx, quad_idx = np.where(np.array(in_cloak_flags))
    sample_pt = qpts[cell_idx[0], quad_idx[0]]
    C_sample = C_eff(sample_pt)
    print(f"\n  Sample cloak point: ({float(sample_pt[0]):.4f}, {float(sample_pt[1]):.4f})")
    print(f"  C0[0,0,0,0]     = {float(C0[0,0,0,0]):.2f}")
    print(f"  C_eff[0,0,0,0]  = {float(C_sample[0,0,0,0]):.2f}")
    print(f"  C0[0,0,1,1]     = {float(C0[0,0,1,1]):.2f}")
    print(f"  C_eff[0,0,1,1]  = {float(C_sample[0,0,1,1]):.2f}")
    print(f"  C0[1,0,1,0]     = {float(C0[1,0,1,0]):.2f}")
    print(f"  C_eff[1,0,1,0]  = {float(C_sample[1,0,1,0]):.2f}")
