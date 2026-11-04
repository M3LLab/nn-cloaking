"""Circular annular cloak geometry with a regularised radial transformation.

The cloak maps the virtual annulus ``eps <= R <= b`` onto the physical annulus
``a <= r <= b`` by a purely radial map ``r = f(R)``, ``theta = Theta``
(Chatzopoulos et al., IJES 2023, Eqs. 11-15 and Appendix Eqs. 30-34; Kohn et
al. 2014 for the ``eps`` regularisation). Outside ``r = b`` the map is the
identity. ``eps = 0`` is the singular ideal (Pendry-type) cloak; ``eps > 0`` is
a *near*-cloak whose exterior field equals that of a traction-free hole of
radius ``eps`` in the homogeneous background.

Supported maps (``radial_map``):

``linear``    (C1-eps, N = 1)  r = a + (b - a)(R - eps)/(b - eps)
``quadratic`` (C2-eps)         r = A2 R^2 + A1 R + A0,  dr/dR(b) = 1

Deformation gradient in the local polar frame (F = dx/dX, both frames share
e_r, e_theta because theta = Theta)::

    F = F_r e_r (x) e_r + F_t e_theta (x) e_theta,   F_r = dr/dR,  F_t = r/R

Index convention used throughout the FEM: ``sigma_ij = C_ijkl du_k/dx_l`` and
equilibrium ``d sigma_ij / dx_j + rho omega^2 u_i = 0`` (derivative on the 2nd
and 4th index).  The BGM / Cosserat push-forward with gauge U(X) = u(x) is then
``C'_ijkl = F_jJ F_lL C_iJkL / det F``,  ``rho' = rho / det F``
(``materials.C_eff``). The paper's Eq. (5) writes the same law with the
derivative on the 1st/3rd index; the two are transposes of each other.
"""

from __future__ import annotations

from dataclasses import dataclass

import gmsh
import jax.numpy as jnp
import numpy as np


RADIAL_MAPS = ("linear", "quadratic")


@dataclass(frozen=True)
class RadialMap:
    """Monotone radial map ``r = f(R)`` from ``[eps, b]`` onto ``[a, b]``.

    All methods are JAX-traceable and accept arrays.
    """

    a: float
    b: float
    eps: float = 0.0
    kind: str = "linear"

    def __post_init__(self):
        a, b, eps = self.a, self.b, self.eps
        if self.kind not in RADIAL_MAPS:
            raise ValueError(f"radial_map must be one of {RADIAL_MAPS}, got {self.kind!r}")
        if not (0.0 <= eps < a < b):
            raise ValueError(f"need 0 <= eps < a < b, got eps={eps}, a={a}, b={b}")
        if self.kind == "quadratic" and not (b - eps > 2.0 * (a - eps)):
            raise ValueError(
                "quadratic map is non-monotone unless b - eps > 2 (a - eps); "
                f"got a={a}, b={b}, eps={eps}")

    # quadratic coefficients (Appendix Eq. 32)
    @property
    def _A(self) -> tuple[float, float, float]:
        a, b, e = self.a, self.b, self.eps
        d2 = (b - e) ** 2
        A2 = (a - e) / d2
        A1 = (d2 - 2.0 * b * (a - e)) / d2
        A0 = b * b * (a - e) / d2
        return A2, A1, A0

    def r_of_R(self, R):
        if self.kind == "linear":
            s = (self.b - self.a) / (self.b - self.eps)
            return self.a + s * (R - self.eps)
        A2, A1, A0 = self._A
        return A2 * R * R + A1 * R + A0

    def R_of_r(self, r):
        if self.kind == "linear":
            s = (self.b - self.a) / (self.b - self.eps)
            return self.eps + (r - self.a) / s
        A2, A1, A0 = self._A
        disc = jnp.maximum(A1 * A1 - 4.0 * A2 * (A0 - r), 0.0)
        return (-A1 + jnp.sqrt(disc)) / (2.0 * A2)

    def drdR(self, R):
        if self.kind == "linear":
            return (self.b - self.a) / (self.b - self.eps) + 0.0 * R
        A2, A1, _ = self._A
        return 2.0 * A2 * R + A1

    def stretches(self, r):
        """Return ``(R, F_r, F_t)`` at physical radius ``r`` (a <= r <= b)."""
        R = self.R_of_r(r)
        return R, self.drdR(R), r / R


@dataclass
class CircularCloakGeometry:
    """Circular annular cloak centred at (x_c, y_c).

    Parameters
    ----------
    ri : Inner (cavity) radius ``a``.
    rc : Outer (cloak/background interface) radius ``b``.
    x_c, y_c : Centre coordinates.
    y_top : Legacy carpet-cloak field (top-of-domain y), unused by the
            fully embedded cloak.
    eps : Regularisation radius of the virtual hole (``0 <= eps < ri``).
          ``eps = 0`` reproduces the singular map used previously.
    radial_map : ``"linear"`` or ``"quadratic"``.
    """

    ri: float
    rc: float
    x_c: float
    y_c: float
    y_top: float | None = None
    eps: float = 0.0
    radial_map: str = "linear"

    def __post_init__(self):
        self.radial = RadialMap(a=self.ri, b=self.rc, eps=self.eps, kind=self.radial_map)

    @property
    def a(self) -> float:
        return self.ri

    @property
    def b(self) -> float:
        return self.rc

    # ── region membership (JAX-traceable) ─────────────────────────────

    def _polar(self, x: jnp.ndarray):
        dx = x[0] - self.x_c
        dy = x[1] - self.y_c
        r = jnp.sqrt(dx**2 + dy**2)
        return dx, dy, r

    def in_cloak(self, x: jnp.ndarray) -> jnp.ndarray:
        _, _, r = self._polar(x)
        return (r > self.ri) & (r < self.rc)

    def in_defect(self, x: jnp.ndarray) -> jnp.ndarray:
        _, _, r = self._polar(x)
        return r < self.ri

    # ── coordinate transformation ─────────────────────────────────────

    def polar_frame(self, x: jnp.ndarray, clamp: bool = False):
        """Return ``(r, R, F_r, F_t, Q)`` at *x*.

        ``Q = [e_r | e_theta]`` (columns) rotates polar components to
        Cartesian ones. With ``clamp=True`` the physical radius is clamped to
        ``[a, b]`` so that points that belong to the cloak *by mesh region*
        but sit a hair outside the analytic annulus (isoparametric boundary
        approximation) still receive the cloak material.
        """
        dx, dy, r = self._polar(x)
        r_safe = jnp.maximum(r, 1e-300)
        e_r = jnp.array([dx, dy]) / r_safe
        e_t = jnp.array([-e_r[1], e_r[0]])
        Q = jnp.stack([e_r, e_t], axis=1)
        r_eval = jnp.clip(r, self.ri, self.rc) if clamp else jnp.maximum(r, self.ri)
        R, F_r, F_t = self.radial.stretches(r_eval)
        return r, R, F_r, F_t, Q

    def F_cloak(self, x: jnp.ndarray, clamp: bool = True) -> jnp.ndarray:
        """Cartesian deformation gradient of the cloak map (no region test)."""
        _, _, F_r, F_t, Q = self.polar_frame(x, clamp=clamp)
        e_r, e_t = Q[:, 0], Q[:, 1]
        return F_r * jnp.outer(e_r, e_r) + F_t * jnp.outer(e_t, e_t)

    def F_tensor(self, x: jnp.ndarray) -> jnp.ndarray:
        """Deformation gradient ``F = dx/dX`` in the Cartesian frame.

        Equal to the identity outside the open annulus ``a < r < b``.
        For ``eps = 0`` and the linear map this is identical to the previous
        implementation ``F = (r I - a x̂ x̂) / R``.
        """
        F = self.F_cloak(x, clamp=False)
        return jnp.where(self.in_cloak(x), F, jnp.eye(2))

    def virtual_point(self, x: jnp.ndarray) -> jnp.ndarray:
        """Inverse map ``X = Xi^{-1}(x)`` (identity outside the cloak)."""
        dx, dy, r = self._polar(x)
        r_c = jnp.clip(r, self.ri, self.rc)
        R = self.radial.R_of_r(r_c)
        scale = jnp.where(self.in_cloak(x), R / jnp.maximum(r, 1e-300), 1.0)
        return jnp.array([self.x_c + scale * dx, self.y_c + scale * dy])

    def physical_point(self, X: jnp.ndarray) -> jnp.ndarray:
        """Forward map ``x = Xi(X)`` for ``eps <= |X - c| <= b`` (identity beyond b)."""
        dX = X[0] - self.x_c
        dY = X[1] - self.y_c
        R = jnp.sqrt(dX**2 + dY**2)
        inside = (R >= self.eps) & (R < self.rc)
        r = self.radial.r_of_R(jnp.clip(R, self.eps, self.rc))
        scale = jnp.where(inside, r / jnp.maximum(R, 1e-300), 1.0)
        return jnp.array([self.x_c + scale * dX, self.y_c + scale * dY])

    # ── bounding box for CellDecomposition ────────────────────────────

    def bbox(self) -> tuple[float, float, float, float]:
        """Return (x_min, x_max, y_min, y_max) of the cloak region."""
        return (
            self.x_c - self.rc,
            self.x_c + self.rc,
            self.y_c - self.rc,
            self.y_c + self.rc,
        )

    # ── gmsh geometry construction ────────────────────────────────────

    def build_gmsh_geometry(
        self,
        geo,
        rect_points: tuple[int, int, int, int],
        h_fine: float,
        h_elem: float,
        h_outside: float | None = None,
        top_eval_xs=None,
        h_surf: float | None = None,
    ) -> list[int]:
        """Cut out the circular void and add cloak refinement fields.

        ``h_outside`` (default = ``h_elem``) is the target mesh size beyond
        the cloak's distance threshold; pass a value > h_elem to coarsen
        the bulk mesh away from cloak/surface.

        ``top_eval_xs`` / ``h_surf``: when provided, the top edge is built
        as a chain of lines through these forced x-positions so each is a
        real triangle vertex on the free surface.
        """
        if h_outside is None:
            h_outside = h_elem
        if h_surf is None:
            h_surf = h_fine
        p1, p2, p3, p4 = rect_points
        from rayleigh_cloak.mesh import _chain_top_edge

        # Centre and circle points
        pc = geo.addPoint(self.x_c, self.y_c, 0.0, h_fine)
        # Four points on the inner circle (void boundary)
        pr = geo.addPoint(self.x_c + self.ri, self.y_c, 0.0, h_fine)
        pt = geo.addPoint(self.x_c, self.y_c + self.ri, 0.0, h_fine)
        pl = geo.addPoint(self.x_c - self.ri, self.y_c, 0.0, h_fine)
        pb = geo.addPoint(self.x_c, self.y_c - self.ri, 0.0, h_fine)

        # Four quarter-circle arcs (void boundary)
        a1 = geo.addCircleArc(pr, pc, pt)
        a2 = geo.addCircleArc(pt, pc, pl)
        a3 = geo.addCircleArc(pl, pc, pb)
        a4 = geo.addCircleArc(pb, pc, pr)

        # Rectangle edges (top edge optionally chained through eval xs)
        l_bot = geo.addLine(p1, p2)
        l_right = geo.addLine(p2, p3)
        if top_eval_xs is not None:
            top_lines = _chain_top_edge(
                geo, p3, p4, np.asarray(top_eval_xs), self.y_top, h_surf,
                descending=True,
            )
        else:
            top_lines = [geo.addLine(p3, p4)]
        l_left = geo.addLine(p4, p1)

        # Outer boundary loop
        outer_loop = geo.addCurveLoop(
            [l_bot, l_right] + top_lines + [l_left]
        )
        # Inner (void) loop — reversed orientation
        inner_loop = geo.addCurveLoop([a1, a2, a3, a4])
        surf = geo.addPlaneSurface([outer_loop, inner_loop])

        gmsh.model.geo.synchronize()

        # Points on the outer cloak circle for refinement
        por = geo.addPoint(self.x_c + self.rc, self.y_c, 0.0, h_fine)
        pot = geo.addPoint(self.x_c, self.y_c + self.rc, 0.0, h_fine)
        pol = geo.addPoint(self.x_c - self.rc, self.y_c, 0.0, h_fine)
        pob = geo.addPoint(self.x_c, self.y_c - self.rc, 0.0, h_fine)
        geo.synchronize()
        gmsh.model.mesh.embed(0, [por, pot, pol, pob], 2, surf)

        # ── refinement fields ──
        # Distance from inner circle
        f_dist_inner = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(
            f_dist_inner, "CurvesList", [a1, a2, a3, a4])
        gmsh.model.mesh.field.setNumber(f_dist_inner, "Sampling", 200)

        # Distance from outer cloak circle points
        f_dist_outer = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(
            f_dist_outer, "PointsList", [por, pot, pol, pob])

        f_dist_cloak = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(
            f_dist_cloak, "FieldsList", [f_dist_inner, f_dist_outer])

        f_thresh_cloak = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(f_thresh_cloak, "InField", f_dist_cloak)
        gmsh.model.mesh.field.setNumber(f_thresh_cloak, "SizeMin", h_fine)
        gmsh.model.mesh.field.setNumber(f_thresh_cloak, "SizeMax", h_outside)
        gmsh.model.mesh.field.setNumber(f_thresh_cloak, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(f_thresh_cloak, "DistMax", self.rc)

        self._cloak_field_tag = f_thresh_cloak

        return top_lines

    def build_gmsh_geometry_full(
        self,
        geo,
        rect_points: tuple[int, int, int, int],
        h_fine: float,
        h_elem: float,
        h_outside: float | None = None,
        top_eval_xs=None,
        h_surf: float | None = None,
    ) -> list[int]:
        """Build full domain mesh (no void cutout) with circle points embedded.

        The inner circle points are embedded so the mesh refines around the
        cloak region. Elements inside the void can be removed later via
        ``extract_submesh``.

        See ``build_gmsh_geometry`` for ``top_eval_xs`` / ``h_surf`` semantics.
        """
        if h_outside is None:
            h_outside = h_elem
        if h_surf is None:
            h_surf = h_fine
        p1, p2, p3, p4 = rect_points
        from rayleigh_cloak.mesh import _chain_top_edge

        # Rectangle edges (top edge optionally chained through eval xs)
        l_bot = geo.addLine(p1, p2)
        l_right = geo.addLine(p2, p3)
        if top_eval_xs is not None:
            top_lines = _chain_top_edge(
                geo, p3, p4, np.asarray(top_eval_xs), self.y_top, h_surf,
                descending=True,
            )
        else:
            top_lines = [geo.addLine(p3, p4)]
        l_left = geo.addLine(p4, p1)

        outer_loop = geo.addCurveLoop(
            [l_bot, l_right] + top_lines + [l_left]
        )
        surf = geo.addPlaneSurface([outer_loop])

        gmsh.model.geo.synchronize()

        # Embed inner circle as constraint lines for mesh conformity
        pc = geo.addPoint(self.x_c, self.y_c, 0.0, h_fine)
        pr = geo.addPoint(self.x_c + self.ri, self.y_c, 0.0, h_fine)
        pt = geo.addPoint(self.x_c, self.y_c + self.ri, 0.0, h_fine)
        pl = geo.addPoint(self.x_c - self.ri, self.y_c, 0.0, h_fine)
        pb = geo.addPoint(self.x_c, self.y_c - self.ri, 0.0, h_fine)

        a1 = geo.addCircleArc(pr, pc, pt)
        a2 = geo.addCircleArc(pt, pc, pl)
        a3 = geo.addCircleArc(pl, pc, pb)
        a4 = geo.addCircleArc(pb, pc, pr)

        # Outer cloak circle points
        por = geo.addPoint(self.x_c + self.rc, self.y_c, 0.0, h_fine)
        pot = geo.addPoint(self.x_c, self.y_c + self.rc, 0.0, h_fine)
        pol = geo.addPoint(self.x_c - self.rc, self.y_c, 0.0, h_fine)
        pob = geo.addPoint(self.x_c, self.y_c - self.rc, 0.0, h_fine)

        geo.synchronize()
        gmsh.model.mesh.embed(0, [pc, por, pot, pol, pob], 2, surf)
        gmsh.model.mesh.embed(1, [a1, a2, a3, a4], 2, surf)

        # ── refinement fields ──
        f_dist_inner = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(
            f_dist_inner, "CurvesList", [a1, a2, a3, a4])
        gmsh.model.mesh.field.setNumber(f_dist_inner, "Sampling", 200)

        f_dist_outer = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(
            f_dist_outer, "PointsList", [por, pot, pol, pob])

        f_dist_cloak = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(
            f_dist_cloak, "FieldsList", [f_dist_inner, f_dist_outer])

        f_thresh_cloak = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(f_thresh_cloak, "InField", f_dist_cloak)
        gmsh.model.mesh.field.setNumber(f_thresh_cloak, "SizeMin", h_fine)
        gmsh.model.mesh.field.setNumber(f_thresh_cloak, "SizeMax", h_outside)
        gmsh.model.mesh.field.setNumber(f_thresh_cloak, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(f_thresh_cloak, "DistMax", self.rc)

        self._cloak_field_tag = f_thresh_cloak

        return top_lines
