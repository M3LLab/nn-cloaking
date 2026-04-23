"""Triangular cloak geometry (Chatzopoulos et al., 2023).

The triangular cloak maps a V-shaped region beneath the free surface onto an
annulus between an inner (defect) and outer triangle, compressing waves around
the hidden void.
"""

from __future__ import annotations

from dataclasses import dataclass

import gmsh
import jax.numpy as jnp

from rayleigh_cloak.config import DerivedParams


@dataclass
class TriangularCloakGeometry:
    """Triangular cloak with parameters taken from :class:`DerivedParams`."""

    a: float      # inner triangle depth
    b: float      # outer triangle depth
    c: float      # surface half-width
    x_c: float    # cloak centre x (extended-mesh coords)
    y_top: float  # free-surface y (extended-mesh coords)

    @staticmethod
    def from_params(p: DerivedParams) -> TriangularCloakGeometry:
        return TriangularCloakGeometry(
            a=p.a, b=p.b, c=p.c, x_c=p.x_c, y_top=p.y_top,
        )

    # ── region membership (JAX-traceable) ────────────────────────────

    def in_cloak(self, x: jnp.ndarray) -> jnp.ndarray:
        """True inside the annular cloak region (excludes inner defect void)."""
        depth = self.y_top - x[1]
        r = jnp.abs(x[0] - self.x_c) / self.c
        d1 = self.a * (1.0 - r)
        d2 = self.b * (1.0 - r)
        return (r <= 1.0) & (depth >= d1) & (depth <= d2)

    def in_defect(self, x: jnp.ndarray) -> jnp.ndarray:
        depth = self.y_top - x[1]
        r = jnp.abs(x[0] - self.x_c) / self.c
        d1 = self.a * (1.0 - r)
        return (r <= 1.0) & (depth >= 0.0) & (depth <= d1)

    # ── coordinate transformation ────────────────────────────────────

    def F_tensor(self, x: jnp.ndarray) -> jnp.ndarray:
        sign = jnp.where(x[0] >= self.x_c, 1.0, -1.0)
        F21 = sign * self.a / self.c
        F22 = (self.b - self.a) / self.b
        F_cloak = jnp.array([[1.0, 0.0],
                              [F21, F22]])
        # F_cloak applies only in the annular region (outer minus inner triangle).
        # Compute inline to avoid composing in_cloak/in_defect in a traced context.
        depth = self.y_top - x[1]
        r = jnp.abs(x[0] - self.x_c) / self.c
        d1 = self.a * (1.0 - r)
        d2 = self.b * (1.0 - r)
        in_annulus = (r <= 1.0) & (depth >= d1) & (depth <= d2)
        return jnp.where(in_annulus, F_cloak, jnp.eye(2))

    # ── gmsh geometry construction ───────────────────────────────────

    def build_gmsh_geometry(
        self,
        geo,
        rect_points: tuple[int, int, int, int],
        h_fine: float,
        h_elem: float,
    ) -> list[int]:
        """Cut out the triangular defect and add cloak refinement fields.

        Parameters
        ----------
        geo : ``gmsh.model.geo``
        rect_points : (p1_bl, p2_br, p3_tr, p4_tl)
        h_fine, h_elem : mesh sizes

        Returns
        -------
        top_lines : line tags forming the top boundary
        """
        p1, p2, p3, p4 = rect_points

        # Triangle vertices on the free surface and apex
        pt_left = geo.addPoint(self.x_c - self.c, self.y_top, 0.0, h_fine)
        pt_right = geo.addPoint(self.x_c + self.c, self.y_top, 0.0, h_fine)
        pt_apex = geo.addPoint(self.x_c, self.y_top - self.a, 0.0, h_fine)

        # Outer cloak apex (for refinement)
        oc_apex = geo.addPoint(self.x_c, self.y_top - self.b, 0.0, h_fine)

        # Rectangle edges (top edge split around the triangle opening)
        l_bot = geo.addLine(p1, p2)
        l_right = geo.addLine(p2, p3)
        l_top1 = geo.addLine(p3, pt_right)
        l_top2 = geo.addLine(pt_left, p4)
        l_left = geo.addLine(p4, p1)

        # Triangle slopes
        tl_right = geo.addLine(pt_right, pt_apex)
        tl_left = geo.addLine(pt_apex, pt_left)

        outer_loop = geo.addCurveLoop([
            l_bot, l_right, l_top1, tl_right, tl_left, l_top2, l_left
        ])
        surf = geo.addPlaneSurface([outer_loop])

        gmsh.model.geo.synchronize()
        gmsh.model.mesh.embed(0, [oc_apex], 2, surf)

        # ── refinement fields ──
        f_dist_inner = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(
            f_dist_inner, "CurvesList", [tl_right, tl_left])
        gmsh.model.mesh.field.setNumber(f_dist_inner, "Sampling", 100)

        f_dist_outer = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(
            f_dist_outer, "PointsList", [pt_left, pt_right, oc_apex])

        f_dist_cloak = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(
            f_dist_cloak, "FieldsList", [f_dist_inner, f_dist_outer])

        f_thresh_cloak = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(f_thresh_cloak, "InField", f_dist_cloak)
        gmsh.model.mesh.field.setNumber(f_thresh_cloak, "SizeMin", h_fine)
        gmsh.model.mesh.field.setNumber(f_thresh_cloak, "SizeMax", h_elem)
        gmsh.model.mesh.field.setNumber(f_thresh_cloak, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(f_thresh_cloak, "DistMax", self.b * 2.0)

        # Store the cloak threshold field tag so mesh.py can compose it
        self._cloak_field_tag = f_thresh_cloak

        return [l_top1, l_top2]

    def build_gmsh_geometry_full(
        self,
        geo,
        rect_points: tuple[int, int, int, int],
        h_fine: float,
        h_elem: float,
    ) -> list[int]:
        """Build full domain mesh (no defect cutout) with cloak vertices embedded.

        The inner triangle vertices are embedded as mesh points so that the
        mesh refines around the cloak region, but no hole is cut.  This lets
        both reference and cloak solves share identical node positions.
        """
        p1, p2, p3, p4 = rect_points

        # Triangle vertices on the free surface and apices
        pt_left = geo.addPoint(self.x_c - self.c, self.y_top, 0.0, h_fine)
        pt_right = geo.addPoint(self.x_c + self.c, self.y_top, 0.0, h_fine)
        pt_apex = geo.addPoint(self.x_c, self.y_top - self.a, 0.0, h_fine)
        oc_apex = geo.addPoint(self.x_c, self.y_top - self.b, 0.0, h_fine)

        # Full rectangle — top edge split at cloak opening points for
        # consistent node placement with the cutout mesh
        l_bot = geo.addLine(p1, p2)
        l_right = geo.addLine(p2, p3)
        l_top1 = geo.addLine(p3, pt_right)
        l_top_mid = geo.addLine(pt_right, pt_left)
        l_top2 = geo.addLine(pt_left, p4)
        l_left = geo.addLine(p4, p1)

        outer_loop = geo.addCurveLoop([
            l_bot, l_right, l_top1, l_top_mid, l_top2, l_left
        ])
        surf = geo.addPlaneSurface([outer_loop])

        gmsh.model.geo.synchronize()

        # Embed cloak vertices so the mesh refines around them
        geo.synchronize()
        gmsh.model.mesh.embed(0, [pt_apex, oc_apex], 2, surf)

        # Inner triangle edges as embedded lines for mesh conformity
        tl_right = geo.addLine(pt_right, pt_apex)
        tl_left = geo.addLine(pt_apex, pt_left)
        geo.synchronize()
        gmsh.model.mesh.embed(1, [tl_right, tl_left], 2, surf)

        # ── refinement fields (same as cutout version) ──
        f_dist_inner = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(
            f_dist_inner, "CurvesList", [tl_right, tl_left])
        gmsh.model.mesh.field.setNumber(f_dist_inner, "Sampling", 100)

        f_dist_outer = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(
            f_dist_outer, "PointsList", [pt_left, pt_right, oc_apex])

        f_dist_cloak = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(
            f_dist_cloak, "FieldsList", [f_dist_inner, f_dist_outer])

        f_thresh_cloak = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(f_thresh_cloak, "InField", f_dist_cloak)
        gmsh.model.mesh.field.setNumber(f_thresh_cloak, "SizeMin", h_fine)
        gmsh.model.mesh.field.setNumber(f_thresh_cloak, "SizeMax", h_elem)
        gmsh.model.mesh.field.setNumber(f_thresh_cloak, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(f_thresh_cloak, "DistMax", self.b * 2.0)

        self._cloak_field_tag = f_thresh_cloak

        return [l_top1, l_top_mid, l_top2]
