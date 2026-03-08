"""
Plot heatmaps of each C_eff tensor element on the mesh.

Usage:  python plot_ceff.py [path/to/mesh.msh]
        Defaults to output/_cloak_mesh.msh
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import meshio

# ── Physical parameters (must match boundaries.py) ──────────────────
rho0 = 1600.0
cs   = 300.0
cp   = np.sqrt(3.0) * cs
mu   = rho0 * cs**2
lam  = rho0 * cp**2 - 2 * mu

H = 4.305
W = 12.5
L_pml = 1.0

a = 0.0774 * H          # inner triangle depth
b = 3 * a                # outer triangle depth
c = 0.309 * H / 2.0     # half-width at surface

W_total = 2 * L_pml + W
H_total = L_pml + H
x_off = L_pml
y_off = L_pml
x_c = x_off + W / 2.0
y_top = H_total

# ── Isotropic stiffness tensor C0 ───────────────────────────────────
C0 = np.zeros((2, 2, 2, 2))
delta = np.eye(2)
for i in range(2):
    for j in range(2):
        for k in range(2):
            for l in range(2):
                C0[i, j, k, l] = (lam * delta[i, j] * delta[k, l]
                                   + mu * (delta[i, k] * delta[j, l]
                                           + delta[i, l] * delta[j, k]))


def _in_cloak(x, y):
    """True inside the cloak annulus (extended mesh coords)."""
    depth = y_top - y
    r = np.abs(x - x_c) / c
    d2 = b * (1.0 - r)
    d1 = a * (1.0 - r)
    return (r <= 1.0) & (depth >= d1) & (depth <= d2)


def C_eff_full(x, y):
    """Compute full C_eff tensor at every node. Returns (N, 2, 2, 2, 2)."""
    in_cloak = _in_cloak(x, y)

    sign = np.sign(x - x_c)
    F = np.zeros(x.shape + (2, 2))
    F[..., 0, 0] = 1.0
    F[..., 1, 1] = np.where(in_cloak, (b - a) / b, 1.0)
    F[..., 1, 0] = np.where(in_cloak, sign * a / c, 0.0)
    F[..., 0, 1] = 0.0

    det = F[..., 0, 0] * F[..., 1, 1] - F[..., 0, 1] * F[..., 1, 0]

    # Cosserat push-forward: C_eff[i,j,k,l] = F[j,J] F[l,L] C0[i,J,k,L] / J
    Cnew = np.einsum('...jJ,...lL,iJkL->...ijkl', F, F, C0)
    Cnew = Cnew / det[..., None, None, None, None]

    # Outside cloak: use C0
    Cnew[~in_cloak] = C0[None, ...]
    return Cnew


# Index labels for display
_idx = {0: 'x', 1: 'y'}


def plot_ceff(msh_path, out_dir=None):
    msh = meshio.read(msh_path)
    pts = msh.points[:, :2]
    tris = msh.cells_dict['triangle']

    x_ext = pts[:, 0]
    y_ext = pts[:, 1]

    # Full tensor at every node: (N, 2, 2, 2, 2)
    Ceff = C_eff_full(x_ext, y_ext)

    # Physical coords
    x_phys = x_ext - x_off
    y_phys = y_ext - y_off
    tri = mtri.Triangulation(x_phys, y_phys, tris)

    xlim = (4.0, 8.5)
    ylim = (2.5, 4.5)
    x_c_phys = W / 2.0

    if out_dir is None:
        out_dir = 'output/ceff_components'
    os.makedirs(out_dir, exist_ok=True)

    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    vals = Ceff[:, i, j, k, l]

                    # Skip if uniform (all equal to C0 component)
                    if np.allclose(vals, C0[i, j, k, l]):
                        continue

                    label = f'C_{_idx[i]}{_idx[j]}{_idx[k]}{_idx[l]}'
                    fname = f'{label}.png'

                    fig, ax = plt.subplots(figsize=(10, 6))
                    tc = ax.tripcolor(tri, vals, shading='flat',
                                       cmap='RdBu_r', rasterized=True)
                    fig.colorbar(tc, ax=ax, shrink=0.8, label=f'${label}$')

                    ax.plot([x_c_phys - c, x_c_phys, x_c_phys + c],
                            [H, H - b, H],
                            ls='--', color='white', lw=1.5, label='outer cloak')
                    ax.plot([x_c_phys - c, x_c_phys, x_c_phys + c],
                            [H, H - a, H],
                            ls='--', color='yellow', lw=1.5, label='inner (defect)')

                    # ax.set_xlim(xlim)
                    # ax.set_ylim(ylim)
                    ax.set_aspect('equal')
                    ax.set_xlabel('x  (physical coords)')
                    ax.set_ylabel('y  (physical coords)')
                    ax.set_title(f'${label}$  (C0 = {C0[i,j,k,l]:.2e})')
                    ax.legend(loc='lower right', fontsize=9)

                    out_path = os.path.join(out_dir, fname)
                    fig.savefig(out_path, dpi=200, bbox_inches='tight')
                    plt.close(fig)
                    print(f"  {out_path}")

    # Also plot ALL components in a single grid (only non-trivial ones)
    # Collect non-uniform components
    components = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    vals = Ceff[:, i, j, k, l]
                    components.append((i, j, k, l, vals))

    ncols = 4
    nrows = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 18))
    idx = 0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    row = i * 2 + j
                    col = k * 2 + l
                    ax = axes[row, col]
                    vals = Ceff[:, i, j, k, l]
                    label = f'C_{_idx[i]}{_idx[j]}{_idx[k]}{_idx[l]}'

                    tc = ax.tripcolor(tri, vals, shading='flat',
                                       cmap='RdBu_r', rasterized=True)
                    fig.colorbar(tc, ax=ax, shrink=0.7)

                    ax.plot([x_c_phys - c, x_c_phys, x_c_phys + c],
                            [H, H - b, H], ls='--', color='white', lw=1.0)
                    ax.plot([x_c_phys - c, x_c_phys, x_c_phys + c],
                            [H, H - a, H], ls='--', color='yellow', lw=1.0)

                    # ax.set_xlim(xlim)
                    # ax.set_ylim(ylim)
                    ax.set_aspect('equal')
                    ax.set_title(f'${label}$', fontsize=11)
                    ax.tick_params(labelsize=7)

    fig.suptitle(r'All $C_{\mathrm{eff}}$ components — cloak region', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    grid_path = os.path.join(out_dir, 'all_components.png')
    fig.savefig(grid_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  {grid_path}")
    print(f"Done — saved to {out_dir}/")


if __name__ == "__main__":
    msh_path = sys.argv[1] if len(sys.argv) > 1 else "output/_cloak_mesh.msh"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else None
    plot_ceff(msh_path, out_dir)
