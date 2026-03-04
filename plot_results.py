"""
Plot displacement fields from saved simulation results.

Usage:  python plot_results.py [path/to/results.npz]
        Defaults to output/results.npz if no argument given.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, Normalize
import matplotlib.colors as mcolors


class SigmoidNorm(mcolors.Normalize):
    """Normalize using a sigmoid, stretching extremes and compressing the middle.

    Parameters
    ----------
    mid : float
        Data value mapped to 0.5 in colour-space (the inflection point).
    steepness : float
        Controls how sharp the transition is.  Higher = more contrast at
        the extremes, flatter middle.  Typical range: 5–30.
    vmin, vmax : float
        Data range (as usual for Normalize).
    """

    def __init__(self, mid, steepness=10.0, vmin=None, vmax=None, clip=False):
        self.mid = mid
        self.steepness = steepness
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # normalise data to [0, 1] linearly first
        x = np.ma.array(value, dtype=float)
        self.autoscale_None(x)
        x = (x - self.vmin) / (self.vmax - self.vmin)
        mid_norm = (self.mid - self.vmin) / (self.vmax - self.vmin)

        # sigmoid centred on mid_norm
        s = 1.0 / (1.0 + np.exp(-self.steepness * (x - mid_norm)))
        # rescale so that s(0)→0 and s(1)→1 exactly
        s0 = 1.0 / (1.0 + np.exp(-self.steepness * (0.0 - mid_norm)))
        s1 = 1.0 / (1.0 + np.exp(-self.steepness * (1.0 - mid_norm)))
        result = (s - s0) / (s1 - s0)
        return np.ma.array(np.clip(result, 0, 1))


class AsymSigmoidNorm(mcolors.Normalize):
    """Normalize using an asymmetric sigmoid with different steepness on each side.

    Parameters
    ----------
    mid : float
        Data value mapped to 0.5 in colour-space (the inflection point).
    steepness_left : float
        Steepness for values below *mid*.  Higher = sharper transition.
    steepness_right : float
        Steepness for values above *mid*.  Higher = sharper transition.
    vmin, vmax : float
        Data range (as usual for Normalize).
    """

    def __init__(self, mid, steepness_left=10.0, steepness_right=10.0,
                 vmin=None, vmax=None, clip=False):
        self.mid = mid
        self.steepness_left = steepness_left
        self.steepness_right = steepness_right
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x = np.ma.array(value, dtype=float)
        self.autoscale_None(x)
        x = (x - self.vmin) / (self.vmax - self.vmin)
        mid_norm = (self.mid - self.vmin) / (self.vmax - self.vmin)

        # pick steepness per element depending on which side of mid_norm
        k = np.where(x <= mid_norm, self.steepness_left, self.steepness_right)
        s = 1.0 / (1.0 + np.exp(-k * (x - mid_norm)))

        # rescale so that s(0)→0 and s(1)→1 exactly
        s0 = 1.0 / (1.0 + np.exp(-self.steepness_left * (0.0 - mid_norm)))
        s1 = 1.0 / (1.0 + np.exp(-self.steepness_right * (1.0 - mid_norm)))
        result = (s - s0) / (s1 - s0)
        return np.ma.array(np.clip(result, 0, 1))


from datetime import datetime
import os



def plot_results(data_path):
    d = np.load(data_path)

    u       = d['u']           # (num_nodes, 4)
    pts_x   = d['pts_x']
    pts_y   = d['pts_y']
    x_src   = float(d['x_src'])
    y_top   = float(d['y_top'])
    x_off   = float(d['x_off'])
    y_off   = float(d['y_off'])
    W       = float(d['W'])
    H       = float(d['H'])
    x_src_phys = float(d['x_src_phys'])
    f_star  = float(d['f_star'])

    # ── Derived quantities ───────────────────────────────────────────────
    ux_R, uy_R, ux_I, uy_I = u[:, 0], u[:, 1], u[:, 2], u[:, 3]
    mag = np.sqrt(ux_R**2 + uy_R**2 + ux_I**2 + uy_I**2)

    # ── Cloak geometry (must match boundaries.py) ──────────────────────
    a_cloak = 0.0774 * H                       # inner triangle depth
    b_cloak = 3 * a_cloak                      # outer triangle depth
    c_cloak = 0.309 * H / 2.0                  # half-width at surface
    x_c     = x_off + W / 2.0                  # cloak centre (extended coords)

    os.makedirs("output", exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # ── Full-domain plot ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    norm = AsymSigmoidNorm(mid=0.25*mag.max(), steepness_left=2, steepness_right=30,
                        vmin=mag.min(), vmax=mag.max())
    tc = ax.tricontourf(pts_x, pts_y, mag, levels=100, cmap='RdBu_r', norm=norm)
    ax.plot(x_src, y_top, 'r*', markersize=16, label='source')

    ax.axvline(x_off,       color='cyan', ls='--', lw=0.7, label='PML interface')
    ax.axvline(x_off + W,   color='cyan', ls='--', lw=0.7)
    ax.axhline(y_off,       color='cyan', ls='--', lw=0.7)

    # Cloak outer boundary: (x_c-c, y_top) → (x_c, y_top-b) → (x_c+c, y_top)
    ax.plot([x_c - c_cloak, x_c, x_c + c_cloak],
            [y_top, y_top - b_cloak, y_top],
            ls='--', color='yellow', lw=1.2)
    # Cloak inner boundary: (x_c-c, y_top) → (x_c, y_top-a) → (x_c+c, y_top)
    ax.plot([x_c - c_cloak, x_c, x_c + c_cloak],
            [y_top, y_top - a_cloak, y_top],
            ls='--', color='yellow', lw=1.2)

    fig.colorbar(tc, ax=ax, shrink=0.8, label='|u|')
    ax.set_title(f"Symmetrized Triangular Cloak with Absorbing Layers  (f*={f_star})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # ax.legend(loc='upper right', fontsize=8)
    ax.set_aspect('equal')

    fname_full = f"output/cloak_abslay_full_{stamp}.png"
    fig.savefig(fname_full, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Full-domain plot saved → {fname_full}")

    # ── Physical-domain-only plot ────────────────────────────────────────
    phys_mask = ((pts_x >= x_off - 1e-8) & (pts_x <= x_off + W + 1e-8) &
                (pts_y >= y_off - 1e-8))

    fig2, ax2 = plt.subplots(figsize=(13, 4))
    mag_phys = mag[phys_mask]
    # norm2 = SigmoidNorm(mid=0.2*mag_phys.max(), steepness=14,
    #                      vmin=mag_phys.min(), vmax=1*mag_phys.max())
    norm2 = AsymSigmoidNorm(mid=0.25*mag_phys.max(), steepness_left=2, steepness_right=30,
                        vmin=mag_phys.min(), vmax=mag_phys.max())

    tc2 = ax2.tricontourf(pts_x[phys_mask] - x_off,
                        pts_y[phys_mask] - y_off,
                        mag_phys,
                        levels=100, cmap='RdBu_r', norm=norm2)
    ax2.plot(x_src_phys, H, 'r*', markersize=16)

    # Cloak boundaries in physical coords (centre at W/2, surface at y=H)
    x_c_phys = W / 2.0
    ax2.plot([x_c_phys - c_cloak, x_c_phys, x_c_phys + c_cloak],
            [H, H - b_cloak, H],
            ls='--', color='yellow', lw=1.2)
    ax2.plot([x_c_phys - c_cloak, x_c_phys, x_c_phys + c_cloak],
            [H, H - a_cloak, H],
            ls='--', color='yellow', lw=1.2)
    # ax2.legend(loc='upper right', fontsize=8)

    fig2.colorbar(tc2, ax=ax2, shrink=0.8, label='|u|')
    ax2.set_title(f"Physical domain only  (f*={f_star})")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_aspect('equal')

    fname_crop = f"output/cloak_abslay_phys_{stamp}.png"
    # fname_crop = f"output/cloak_abslay_phys.png"
    fig2.savefig(fname_crop, dpi=200, bbox_inches='tight')
    plt.close(fig2)
    print(f"Physical-domain plot saved → {fname_crop}")


def plot_vtk_results(vtk_path):
    """Plot displacement field from a VTK file using actual mesh connectivity."""
    import vtk as _vtk
    from vtk.util.numpy_support import vtk_to_numpy
    from matplotlib.collections import PolyCollection

    reader = _vtk.vtkUnstructuredGridReader()
    reader.SetFileName(vtk_path)
    reader.ReadAllScalarsOn()
    reader.ReadAllFieldsOn()
    reader.Update()
    grid = reader.GetOutput()

    # Extract points (2D)
    pts = vtk_to_numpy(grid.GetPoints().GetData())[:, :2]

    # Extract quad cells → list of (4, 2) vertex arrays
    cells = grid.GetCells()
    cell_array = vtk_to_numpy(cells.GetConnectivityArray())
    offsets = vtk_to_numpy(cells.GetOffsetsArray())
    quads = []
    for i in range(len(offsets) - 1):
        ids = cell_array[offsets[i]:offsets[i + 1]]
        quads.append(pts[ids])

    # Point data
    mag = vtk_to_numpy(grid.GetPointData().GetArray("mag_u"))

    # Per-cell colour = mean of vertex values
    cell_vals = np.array([mag[cell_array[offsets[i]:offsets[i+1]]].mean()
                          for i in range(len(offsets) - 1)])

    # Read metadata from field data
    fd = grid.GetFieldData()
    def _fval(name):
        return vtk_to_numpy(fd.GetArray(name))[0]
    x_src, y_top = _fval("x_src"), _fval("y_top")
    x_off, y_off = _fval("x_off"), _fval("y_off")
    W, H = _fval("W"), _fval("H")
    x_src_phys = _fval("x_src_phys")
    f_star = _fval("f_star")

    a_cloak = 0.0774 * H
    b_cloak = 3 * a_cloak
    c_cloak = 0.309 * H / 2.0
    x_c = x_off + W / 2.0

    os.makedirs("output", exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # ── Full-domain plot ─────────────────────────────────────────────────
    norm = AsymSigmoidNorm(mid=0.25 * mag.max(), steepness_left=2,
                           steepness_right=30, vmin=mag.min(), vmax=mag.max())

    fig, ax = plt.subplots(figsize=(14, 5))
    pc = PolyCollection(quads, array=cell_vals, cmap='RdBu_r', norm=norm,
                        edgecolors='face', linewidths=0.1, rasterized=True)
    ax.add_collection(pc)
    ax.autoscale_view()

    ax.plot(x_src, y_top, 'r*', markersize=16, label='source')
    ax.axvline(x_off, color='cyan', ls='--', lw=0.7, label='PML interface')
    ax.axvline(x_off + W, color='cyan', ls='--', lw=0.7)
    ax.axhline(y_off, color='cyan', ls='--', lw=0.7)
    ax.plot([x_c - c_cloak, x_c, x_c + c_cloak],
            [y_top, y_top - b_cloak, y_top], ls='--', color='yellow', lw=1.2)
    ax.plot([x_c - c_cloak, x_c, x_c + c_cloak],
            [y_top, y_top - a_cloak, y_top], ls='--', color='yellow', lw=1.2)

    fig.colorbar(pc, ax=ax, shrink=0.8, label='|u|')
    ax.set_title(f"Triangular Cloak – VTK mesh  (f*={f_star})")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect('equal')

    fname_full = f"output/cloak_vtk_full.png" #_{stamp}
    fig.savefig(fname_full, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"VTK full-domain plot saved → {fname_full}")

    # ── Full-domain plot of Re(u_y) ──────────────────────────────────────
    u_full = vtk_to_numpy(grid.GetPointData().GetArray("u"))  # (num_nodes, 4)
    re_uy = u_full[:, 1]  # Re(u_y)

    cell_vals_re_uy_full = np.array([re_uy[cell_array[offsets[i]:offsets[i+1]]].mean()
                                     for i in range(len(offsets) - 1)])

    vlim_full = max(abs(re_uy.min()), abs(re_uy.max()))
    norm_re_full = AsymSigmoidNorm(mid=0, steepness_left=100, steepness_right=100,
                                   vmin=-vlim_full, vmax=vlim_full)

    fig_rf, ax_rf = plt.subplots(figsize=(14, 5))
    pc_rf = PolyCollection(quads, array=cell_vals_re_uy_full, cmap='RdBu_r',
                           norm=norm_re_full, edgecolors='face', linewidths=0.1,
                           rasterized=True)
    ax_rf.add_collection(pc_rf)
    ax_rf.autoscale_view()

    ax_rf.plot(x_src, y_top, 'r*', markersize=16, label='source')
    ax_rf.axvline(x_off, color='cyan', ls='--', lw=0.7, label='PML interface')
    ax_rf.axvline(x_off + W, color='cyan', ls='--', lw=0.7)
    ax_rf.axhline(y_off, color='cyan', ls='--', lw=0.7)
    ax_rf.plot([x_c - c_cloak, x_c, x_c + c_cloak],
               [y_top, y_top - b_cloak, y_top], ls='--', color='yellow', lw=1.2)
    ax_rf.plot([x_c - c_cloak, x_c, x_c + c_cloak],
               [y_top, y_top - a_cloak, y_top], ls='--', color='yellow', lw=1.2)

    fig_rf.colorbar(pc_rf, ax=ax_rf, shrink=0.8, label='Re(u_y)')
    ax_rf.set_title(f"Re(u_y) – Full domain  (f*={f_star})")
    ax_rf.set_xlabel("x"); ax_rf.set_ylabel("y")
    ax_rf.set_aspect('equal')

    fname_re_full = f"output/cloak_vtk_re_uy_full.png"
    fig_rf.savefig(fname_re_full, dpi=200, bbox_inches='tight')
    plt.close(fig_rf)
    print(f"VTK Re(u_y) full-domain plot saved → {fname_re_full}")

    # ── Physical-domain-only plot ────────────────────────────────────────
    # Keep cells whose centroid is inside physical domain
    centroids = np.array([q.mean(axis=0) for q in quads])
    phys_mask = ((centroids[:, 0] >= x_off - 1e-8) &
                 (centroids[:, 0] <= x_off + W + 1e-8) &
                 (centroids[:, 1] >= y_off - 1e-8))

    phys_quads = [quads[i] - np.array([x_off, y_off]) for i in range(len(quads)) if phys_mask[i]]
    phys_vals = cell_vals[phys_mask]

    norm2 = AsymSigmoidNorm(mid=0.25 * phys_vals.max(), steepness_left=2,
                            steepness_right=30, vmin=phys_vals.min(), vmax=phys_vals.max())

    fig2, ax2 = plt.subplots(figsize=(13, 4))
    pc2 = PolyCollection(phys_quads, array=phys_vals, cmap='RdBu_r', norm=norm2,
                         edgecolors='face', linewidths=0.1, rasterized=True)
    ax2.add_collection(pc2)
    ax2.autoscale_view()

    ax2.plot(x_src_phys, H, 'r*', markersize=16)
    x_c_phys = W / 2.0
    ax2.plot([x_c_phys - c_cloak, x_c_phys, x_c_phys + c_cloak],
             [H, H - b_cloak, H], ls='--', color='yellow', lw=1.2)
    ax2.plot([x_c_phys - c_cloak, x_c_phys, x_c_phys + c_cloak],
             [H, H - a_cloak, H], ls='--', color='yellow', lw=1.2)

    fig2.colorbar(pc2, ax=ax2, shrink=0.8, label='|u|')
    ax2.set_title(f"Physical domain only – VTK mesh  (f*={f_star})")
    ax2.set_xlabel("x"); ax2.set_ylabel("y")
    ax2.set_aspect('equal')

    fname_crop = f"output/cloak_vtk_phys.png" #_{stamp}
    fig2.savefig(fname_crop, dpi=200, bbox_inches='tight')
    plt.close(fig2)
    print(f"VTK physical-domain plot saved → {fname_crop}")

    # ── Real part of u_y (physical domain) ──────────────────────────────
    cell_vals_re = cell_vals_re_uy_full
    phys_vals_re = cell_vals_re[phys_mask]

    vlim = max(abs(phys_vals_re.min()), abs(phys_vals_re.max()))
    # norm3 = Normalize(vmin=-vlim, vmax=vlim)
    norm3 = AsymSigmoidNorm(mid=0, steepness_left=50, steepness_right=50, 
                            vmin=-vlim, vmax=vlim)

    fig3, ax3 = plt.subplots(figsize=(13, 4))
    pc3 = PolyCollection(phys_quads, array=phys_vals_re, cmap='RdBu_r', norm=norm3,
                         edgecolors='face', linewidths=0.1, rasterized=True)
    ax3.add_collection(pc3)
    ax3.autoscale_view()

    ax3.plot(x_src_phys, H, 'r*', markersize=16)
    ax3.plot([x_c_phys - c_cloak, x_c_phys, x_c_phys + c_cloak],
             [H, H - b_cloak, H], ls='--', color='yellow', lw=1.2)
    ax3.plot([x_c_phys - c_cloak, x_c_phys, x_c_phys + c_cloak],
             [H, H - a_cloak, H], ls='--', color='yellow', lw=1.2)

    fig3.colorbar(pc3, ax=ax3, shrink=0.8, label='Re(u_y)')
    ax3.set_title(f"Re(u_y) – Physical domain  (f*={f_star})")
    ax3.set_xlabel("x"); ax3.set_ylabel("y")
    ax3.set_aspect('equal')

    fname_re = f"output/cloak_vtk_re_uy.png"  #_{stamp}
    fig3.savefig(fname_re, dpi=200, bbox_inches='tight')
    plt.close(fig3)
    print(f"VTK Re(u_y) plot saved → {fname_re}")

    # ── Total real displacement magnitude (physical domain) ─────────────
    re_ux = u_full[:, 0]  # Re(u_x)
    re_mag = np.sqrt(re_ux**2 + re_uy**2)

    cell_vals_re_mag = np.array([re_mag[cell_array[offsets[i]:offsets[i+1]]].mean()
                                 for i in range(len(offsets) - 1)])
    phys_vals_re_mag = cell_vals_re_mag[phys_mask]

    norm4 = AsymSigmoidNorm(mid=0.25 * phys_vals_re_mag.max(), steepness_left=2,
                            steepness_right=30, vmin=phys_vals_re_mag.min(),
                            vmax=phys_vals_re_mag.max())

    fig4, ax4 = plt.subplots(figsize=(13, 4))
    pc4 = PolyCollection(phys_quads, array=phys_vals_re_mag, cmap='RdBu_r', norm=norm4,
                         edgecolors='face', linewidths=0.1, rasterized=True)
    ax4.add_collection(pc4)
    ax4.autoscale_view()

    ax4.plot(x_src_phys, H, 'r*', markersize=16)
    ax4.plot([x_c_phys - c_cloak, x_c_phys, x_c_phys + c_cloak],
             [H, H - b_cloak, H], ls='--', color='yellow', lw=1.2)
    ax4.plot([x_c_phys - c_cloak, x_c_phys, x_c_phys + c_cloak],
             [H, H - a_cloak, H], ls='--', color='yellow', lw=1.2)

    fig4.colorbar(pc4, ax=ax4, shrink=0.8, label='|Re(u)|')
    ax4.set_title(f"|Re(u)| – Physical domain  (f*={f_star})")
    ax4.set_xlabel("x"); ax4.set_ylabel("y")
    ax4.set_aspect('equal')

    fname_re_mag = f"output/cloak_vtk_re_mag.png" #_{stamp}
    fig4.savefig(fname_re_mag, dpi=200, bbox_inches='tight')
    plt.close(fig4)
    print(f"VTK |Re(u)| plot saved → {fname_re_mag}")


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "output/results.npz"
    if data_path.endswith('.vtk'):
        plot_vtk_results(data_path)
    else:
        plot_results(data_path)
