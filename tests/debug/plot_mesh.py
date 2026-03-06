"""
Plot the gmsh-generated mesh from the .msh file as a PNG.

Usage:  python plot_mesh.py [path/to/mesh.msh]
        Defaults to output/_cloak_mesh.msh
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import meshio


def plot_mesh(msh_path, out_png=None):
    msh = meshio.read(msh_path)
    pts = msh.points[:, :2]
    tris = msh.cells_dict['triangle']

    tri = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.triplot(tri, lw=0.3, color='k')
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Mesh  ({len(tris)} triangles, {len(pts)} nodes)')

    if out_png is None:
        out_png = msh_path.rsplit('.', 1)[0] + '_mesh.png'
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Mesh plot saved → {out_png}")


if __name__ == "__main__":
    msh_path = sys.argv[1] if len(sys.argv) > 1 else "output/_cloak_mesh.msh"
    out_png = sys.argv[2] if len(sys.argv) > 2 else None
    plot_mesh(msh_path, out_png)
