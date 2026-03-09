"""Gmsh mesh generation for the extended domain (physical + PML)."""

from __future__ import annotations

import os

import gmsh
import meshio
import numpy as np
from jax_fem.generate_mesh import Mesh

from rayleigh_cloak.config import DerivedParams, SimulationConfig
from rayleigh_cloak.geometry.base import CloakGeometry


def generate_mesh(
    cfg: SimulationConfig,
    params: DerivedParams,
    geometry: CloakGeometry,
) -> Mesh:
    """Generate a triangular (or quad) mesh of the full domain.

    When ``cfg.is_reference`` is True the domain is a plain rectangle (no
    defect).  Otherwise the geometry's ``build_gmsh_geometry`` cuts out the
    defect and adds refinement fields.

    Returns a ``jax_fem.generate_mesh.Mesh`` instance.
    """
    p = params
    h_elem = min(p.W_total / p.nx_total, p.H_total / p.ny_total)
    h_fine = h_elem / cfg.mesh.refinement_factor
    h_surf = h_fine

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("cloak_domain")
    geo = gmsh.model.geo

    # Corner points
    p1 = geo.addPoint(0.0, 0.0, 0.0, h_elem)
    p2 = geo.addPoint(p.W_total, 0.0, 0.0, h_elem)
    p3 = geo.addPoint(p.W_total, p.H_total, 0.0, h_elem)
    p4 = geo.addPoint(0.0, p.H_total, 0.0, h_elem)

    if cfg.is_reference:
        # Plain rectangle — no defect
        l_bot = geo.addLine(p1, p2)
        l_right = geo.addLine(p2, p3)
        l_top = geo.addLine(p3, p4)
        l_left = geo.addLine(p4, p1)

        outer_loop = geo.addCurveLoop([l_bot, l_right, l_top, l_left])
        geo.addPlaneSurface([outer_loop])
        gmsh.model.geo.synchronize()

        top_lines = [l_top]
    else:
        # Delegate defect cutout + refinement to the geometry object
        top_lines = geometry.build_gmsh_geometry(
            geo, (p1, p2, p3, p4), h_fine, h_elem,
        )

    # ── surface refinement (common to all geometries) ──
    f_dist_surf = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist_surf, "CurvesList", top_lines)
    gmsh.model.mesh.field.setNumber(f_dist_surf, "Sampling", 200)

    f_thresh_surf = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thresh_surf, "InField", f_dist_surf)
    gmsh.model.mesh.field.setNumber(f_thresh_surf, "SizeMin", h_surf)
    gmsh.model.mesh.field.setNumber(f_thresh_surf, "SizeMax", h_elem)
    gmsh.model.mesh.field.setNumber(f_thresh_surf, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(f_thresh_surf, "DistMax", p.lambda_star)

    # ── compose background field ──
    if cfg.is_reference:
        gmsh.model.mesh.field.setAsBackgroundMesh(f_thresh_surf)
    else:
        cloak_field = getattr(geometry, "_cloak_field_tag", None)
        if cloak_field is not None:
            f_final = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(
                f_final, "FieldsList", [cloak_field, f_thresh_surf])
            gmsh.model.mesh.field.setAsBackgroundMesh(f_final)
        else:
            gmsh.model.mesh.field.setAsBackgroundMesh(f_thresh_surf)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    gmsh.model.mesh.generate(2)

    os.makedirs(cfg.output_dir, exist_ok=True)
    msh_path = os.path.join(cfg.output_dir, "_cloak_mesh.msh")
    gmsh.write(msh_path)
    gmsh.finalize()

    msh = meshio.read(msh_path)
    points = msh.points[:, :2]
    cells = msh.cells_dict["triangle"]

    return Mesh(points, cells, ele_type=cfg.mesh.ele_type)


def generate_mesh_full(
    cfg: SimulationConfig,
    params: DerivedParams,
    geometry: CloakGeometry,
) -> Mesh:
    """Generate a mesh of the full domain (no defect cutout).

    The cloak geometry vertices are embedded so the mesh refines properly,
    but no hole is cut.  This mesh can be used for both the reference solve
    (homogeneous material) and, after ``extract_submesh``, for the cloak solve.
    """
    p = params
    h_elem = min(p.W_total / p.nx_total, p.H_total / p.ny_total)
    h_fine = h_elem / cfg.mesh.refinement_factor
    h_surf = h_fine

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("cloak_domain_full")
    geo = gmsh.model.geo

    p1 = geo.addPoint(0.0, 0.0, 0.0, h_elem)
    p2 = geo.addPoint(p.W_total, 0.0, 0.0, h_elem)
    p3 = geo.addPoint(p.W_total, p.H_total, 0.0, h_elem)
    p4 = geo.addPoint(0.0, p.H_total, 0.0, h_elem)

    top_lines = geometry.build_gmsh_geometry_full(
        geo, (p1, p2, p3, p4), h_fine, h_elem,
    )

    # ── surface refinement (same as generate_mesh) ──
    f_dist_surf = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist_surf, "CurvesList", top_lines)
    gmsh.model.mesh.field.setNumber(f_dist_surf, "Sampling", 200)

    f_thresh_surf = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thresh_surf, "InField", f_dist_surf)
    gmsh.model.mesh.field.setNumber(f_thresh_surf, "SizeMin", h_surf)
    gmsh.model.mesh.field.setNumber(f_thresh_surf, "SizeMax", h_elem)
    gmsh.model.mesh.field.setNumber(f_thresh_surf, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(f_thresh_surf, "DistMax", p.lambda_star)

    cloak_field = getattr(geometry, "_cloak_field_tag", None)
    if cloak_field is not None:
        f_final = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(
            f_final, "FieldsList", [cloak_field, f_thresh_surf])
        gmsh.model.mesh.field.setAsBackgroundMesh(f_final)
    else:
        gmsh.model.mesh.field.setAsBackgroundMesh(f_thresh_surf)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    gmsh.model.mesh.generate(2)

    os.makedirs(cfg.output_dir, exist_ok=True)
    msh_path = os.path.join(cfg.output_dir, "_cloak_mesh_full.msh")
    gmsh.write(msh_path)
    gmsh.finalize()

    msh = meshio.read(msh_path)
    points = msh.points[:, :2]
    cells = msh.cells_dict["triangle"]

    return Mesh(points, cells, ele_type=cfg.mesh.ele_type)


def extract_submesh(
    mesh: Mesh,
    geometry: CloakGeometry,
) -> tuple[Mesh, np.ndarray]:
    """Remove elements whose centroid lies inside the defect region.

    Orphan nodes (not referenced by any remaining element) are removed and
    cell indices are renumbered accordingly.

    Returns
    -------
    submesh : Mesh
        Clean mesh with no orphan nodes.
    kept_nodes : (n_sub_nodes,) int array
        Original node indices that were kept.  Useful for mapping
        full-mesh solutions to the submesh:
        ``u_submesh = u_full[kept_nodes]``.
    """
    import jax.numpy as jnp

    points = np.asarray(mesh.points)
    cells = np.asarray(mesh.cells)

    # Compute element centroids
    centroids = points[cells].mean(axis=1)  # (n_elem, 2)

    # Keep elements outside the defect
    keep = np.array([
        not bool(geometry.in_defect(jnp.array(c)))
        for c in centroids
    ])
    cells_sub = cells[keep]

    # Remove orphan nodes and renumber
    kept_nodes = np.unique(cells_sub)
    new_points = points[kept_nodes]
    old_to_new = np.full(len(points), -1, dtype=int)
    old_to_new[kept_nodes] = np.arange(len(kept_nodes))
    new_cells = old_to_new[cells_sub]

    return Mesh(new_points, new_cells, ele_type=mesh.ele_type), kept_nodes
