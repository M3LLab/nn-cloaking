"""Gmsh (OpenCASCADE) mesh for the fully embedded circular cloak.

The domain is fragmented into four tagged regions so that every element lies
entirely in one of them and the circles r = a (or r = eps) and r = b are
conforming element edges (quadratic, on-curve mid-side nodes for TRI6):

    REGION_DISK     r < r_in            (removed for cases B, C, D)
    REGION_ANNULUS  r_in < r < b        (cloak material in case C)
    REGION_BOX      physical box minus the disk r < b
    REGION_ABSORB   absorbing frame around the physical box
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import gmsh
import meshio
import numpy as np

from rayleigh_cloak.embedded.config import EmbeddedCloakConfig
from rayleigh_cloak.geometry.circular import RadialMap

REGION_DISK, REGION_ANNULUS, REGION_BOX, REGION_ABSORB = 1, 2, 3, 4
REGION_NAMES = {REGION_DISK: "disk", REGION_ANNULUS: "annulus",
                REGION_BOX: "box", REGION_ABSORB: "absorbing"}


@dataclass
class TaggedMesh:
    points: np.ndarray        # (n, 2)
    cells: np.ndarray         # (m, 6) gmsh node order for TRI6, (m, 3) for TRI3
    region: np.ndarray        # (m,) region id per element
    ele_type: str
    sizes: dict               # target element sizes used to build it

    @property
    def n_nodes(self) -> int:
        return len(self.points)

    @property
    def n_cells(self) -> int:
        return len(self.cells)

    def submesh(self, keep: np.ndarray) -> tuple["TaggedMesh", np.ndarray]:
        """Keep elements where ``keep`` is True; drop orphan nodes.

        Returns ``(submesh, kept_nodes)`` with ``u_full[kept_nodes]`` the
        restriction of a full-mesh nodal field to the submesh.
        """
        cells = self.cells[keep]
        kept = np.unique(cells)
        old_to_new = np.full(self.n_nodes, -1, dtype=np.int64)
        old_to_new[kept] = np.arange(len(kept))
        return TaggedMesh(self.points[kept], old_to_new[cells], self.region[keep],
                          self.ele_type, dict(self.sizes)), kept

    def without_disk(self) -> tuple["TaggedMesh", np.ndarray]:
        return self.submesh(self.region != REGION_DISK)

    def to_jaxfem(self):
        from jax_fem.generate_mesh import Mesh
        return Mesh(self.points, self.cells, ele_type=self.ele_type)

    def save(self, path: str) -> None:
        np.savez_compressed(path, points=self.points, cells=self.cells,
                            region=self.region, ele_type=self.ele_type)


def mesh_sizes(cfg: EmbeddedCloakConfig, hole_radius: float, hole_is_eps: bool) -> dict:
    """Resolve target element sizes (all multiplied by ``mesh.scale``)."""
    m, c = cfg.mesh, cfg.cloak
    s = m.scale
    h_bg = cfg.lambda_s / m.elems_per_wavelength * s
    h_cloak = h_bg / m.cloak_refine
    if hole_is_eps:
        # case D: the virtual problem with a traction-free hole of radius eps
        h_in = min(h_cloak, max(m.inner_h_min, m.hole_h_factor * hole_radius) * s)
        delta = hole_radius
    else:
        radial = RadialMap(a=c.a, b=c.b, eps=c.eps, kind=c.radial_map)
        delta = float(c.eps * radial.drdR(c.eps))   # physical boundary-layer thickness
        h_raw = m.inner_h_max if delta <= 0 else min(m.inner_h_max, m.inner_bl_factor * delta)
        h_in = min(h_cloak, max(m.inner_h_min, h_raw) * s)
    h_src = min(h_bg, m.source_h * cfg.source.sigma * s)
    return dict(h_bg=h_bg, h_cloak=h_cloak, h_inner=h_in, h_src=h_src,
                inner_grading=m.inner_grading, bl_delta=delta, hole_radius=hole_radius)


def build_mesh(
    cfg: EmbeddedCloakConfig,
    hole_radius: float | None = None,
    hole_is_eps: bool = False,
    msh_path: str | None = None,
    verbose: bool = True,
) -> TaggedMesh:
    """Build the tagged mesh. ``hole_radius`` defaults to the cavity radius a.

    ``hole_is_eps=True`` builds the mesh for the *virtual* problem (case D):
    inner circle at r = eps, refined to resolve that small hole.
    """
    c, d, src = cfg.cloak, cfg.domain, cfg.source
    r_in = c.a if hole_radius is None else float(hole_radius)
    if not (0 < r_in < c.b):
        raise ValueError(f"hole radius {r_in} must be in (0, b)")
    sz = mesh_sizes(cfg, r_in, hole_is_eps)
    X0, X1, Y0, Y1 = cfg.outer_box
    from rayleigh_cloak.embedded import memguard
    memguard.check_mesh(cfg, sz, r_in)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Verbosity", 1)
        gmsh.model.add("embedded_cloak")
        occ = gmsh.model.occ
        outer = occ.addRectangle(X0, Y0, 0, X1 - X0, Y1 - Y0)
        box = occ.addRectangle(d.x_min, d.y_min, 0, d.x_max - d.x_min, d.y_max - d.y_min)
        disk_b = occ.addDisk(c.xc, c.yc, 0, c.b, c.b)
        disk_a = occ.addDisk(c.xc, c.yc, 0, r_in, r_in)
        occ.fragment([(2, outer)], [(2, box), (2, disk_b), (2, disk_a)])
        occ.synchronize()

        groups = {k: [] for k in REGION_NAMES}
        tol = 1e-6 * max(X1 - X0, Y1 - Y0)
        for _, tag in gmsh.model.getEntities(2):
            xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(2, tag)
            w = xmax - xmin
            if abs(w - 2 * r_in) < 1e-3 * r_in + tol:
                groups[REGION_DISK].append(tag)
            elif abs(w - 2 * c.b) < 1e-3 * c.b + tol:
                groups[REGION_ANNULUS].append(tag)
            elif abs(w - (d.x_max - d.x_min)) < 1e-3 * w + tol:
                groups[REGION_BOX].append(tag)
            elif abs(w - (X1 - X0)) < 1e-3 * w + tol:
                groups[REGION_ABSORB].append(tag)
            else:
                raise RuntimeError(f"cannot classify surface {tag} (width {w})")
        for k, tags in groups.items():
            if len(tags) != 1:
                raise RuntimeError(f"expected one surface for {REGION_NAMES[k]}, got {tags}")
            gmsh.model.addPhysicalGroup(2, tags, tag=k)

        # ── size field ───────────────────────────────────────────────
        f = gmsh.model.mesh.field
        r_expr = f"Sqrt((x-({c.xc}))^2+(y-({c.yc}))^2)"
        f_r = f.add("MathEval")
        f.setString(f_r, "F", r_expr)

        # annulus (and everything inside r = b): h_cloak, back to h_bg outside
        f_cl = f.add("Threshold")
        f.setNumber(f_cl, "InField", f_r)
        f.setNumber(f_cl, "SizeMin", sz["h_cloak"])
        f.setNumber(f_cl, "SizeMax", sz["h_bg"])
        f.setNumber(f_cl, "DistMin", c.b)
        f.setNumber(f_cl, "DistMax", c.b + 4 * sz["h_bg"])

        # graded refinement at the inner circle
        f_din = f.add("MathEval")
        f.setString(f_din, "F", f"Abs({r_expr}-({r_in}))")
        f_in = f.add("Threshold")
        f.setNumber(f_in, "InField", f_din)
        f.setNumber(f_in, "SizeMin", sz["h_inner"])
        f.setNumber(f_in, "SizeMax", sz["h_cloak"])
        f.setNumber(f_in, "DistMin", 0.0)
        f.setNumber(f_in, "DistMax", sz["inner_grading"])
        f.setNumber(f_in, "StopAtDistMax", 1)   # local field: no effect beyond DistMax

        # source refinement
        f_ds = f.add("MathEval")
        f.setString(f_ds, "F", f"Sqrt((x-({src.x}))^2+(y-({src.y}))^2)")
        f_s = f.add("Threshold")
        f.setNumber(f_s, "InField", f_ds)
        f.setNumber(f_s, "SizeMin", sz["h_src"])
        f.setNumber(f_s, "SizeMax", sz["h_bg"])
        f.setNumber(f_s, "DistMin", 4 * src.sigma)
        f.setNumber(f_s, "DistMax", 4 * src.sigma + 10 * sz["h_src"] + 2 * sz["h_bg"])
        f.setNumber(f_s, "StopAtDistMax", 1)

        f_min = f.add("Min")
        f.setNumbers(f_min, "FieldsList", [f_cl, f_in, f_s])
        f.setAsBackgroundMesh(f_min)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)   # Frontal-Delaunay

        gmsh.model.mesh.generate(2)
        if cfg.mesh.ele_type == "TRI6":
            gmsh.model.mesh.setOrder(2)

        if msh_path is None:
            os.makedirs(cfg.output_dir, exist_ok=True)
            msh_path = os.path.join(cfg.output_dir, "_embedded_mesh.msh")
        os.makedirs(os.path.dirname(msh_path) or ".", exist_ok=True)
        gmsh.write(msh_path)
    finally:
        gmsh.finalize()

    msh = meshio.read(msh_path)
    key = "triangle6" if cfg.mesh.ele_type == "TRI6" else "triangle"
    cells = np.asarray(msh.cells_dict[key], dtype=np.int64)
    region = np.asarray(msh.cell_data_dict["gmsh:physical"][key], dtype=np.int64)
    points = np.asarray(msh.points[:, :2], dtype=np.float64)
    full = TaggedMesh(points, cells, region, cfg.mesh.ele_type, sz)
    full, _ = full.submesh(np.ones(len(cells), dtype=bool))   # drop orphan nodes
    if verbose:
        counts = {REGION_NAMES[k]: int((full.region == k).sum()) for k in REGION_NAMES}
        print(f"  mesh r_in={r_in:.4g}: {full.n_nodes} nodes, {full.n_cells} {key} "
              f"{counts}; h_bg={sz['h_bg']:.4f} h_cloak={sz['h_cloak']:.4f} "
              f"h_inner={sz['h_inner']:.2e} h_src={sz['h_src']:.4f}")
    return full
