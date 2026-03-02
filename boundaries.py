"""
Symmetrized triangular elastic cloak with Rayleigh-damping absorbing layers.

Based on:
  - "Cloaking Rayleigh waves via symmetrized elastic tensors" (Chatzopoulos et al., 2023)
  - "A Simple Numerical Absorbing Layer Method in Elastodynamics" (Semblat et al., 2010)

Absorbing-layer strategy
========================
The frequency-domain elastodynamic equation with Rayleigh damping is:

    (K + iωC − ω²M) u = f,   where  C = a₀M + a₁K   (Rayleigh damping)

Expanding:
    (1 + iωa₁)K·u − ω²(1 − ia₀/ω)M·u = f

Since JAX-FEM operates on real-valued fields only, we split u = u_R + i·u_I
into real and imaginary parts.  This gives the coupled real system:

    K·u_R − ω a₁ K·u_I  −  ω²ρ u_R − ω a₀ ρ u_I  =  f_R
    K·u_I + ω a₁ K·u_R  −  ω²ρ u_I + ω a₀ ρ u_R  =  f_I

We encode this with vec=4:  DOFs = [Re(uₓ), Re(u_y), Im(uₓ), Im(u_y)].

Choosing a₀ = ξω and a₁ = ξ/ω (so both give damping ratio ξ at ω) simplifies
the coupling coefficient to just ξ(x) everywhere, with ξ increasing from 0 at
the PML/physical interface to ξ_max at the outer boundary (quadratic ramp).

JAX-FEM hooks used
==================
  • custom_init   – precompute C_eff, ρ_eff, ξ at every quadrature point
  • get_tensor_map – stiffness + stiffness-proportional damping coupling
  • get_mass_map   – inertia + mass-proportional damping coupling
  • get_surface_maps – Rayleigh-wave point source (real part only)
"""

import jax
import jax.numpy as jnp 
import numpy as np
from jax_fem.problem import Problem
from jax_fem.solver import solver
from jax_fem.generate_mesh import Mesh
import gmsh
import meshio

# ═══════════════════════════════════════════════════════════════════════
# Physical parameters  (identical to the original triangle.py)
# ═══════════════════════════════════════════════════════════════════════

IS_REFERENCE = False   # set True to disable cloak region (for reference field)

rho0 = 1600.0                          # mass density  [kg/m³]
cs   = 300.0                           # shear wave speed  [m/s]
cp   = np.sqrt(3.0) * cs               # pressure wave speed

mu   = rho0 * cs**2
lam  = rho0 * cp**2 - 2 * mu
nu   = lam / (2 * (lam + mu))          # Poisson's ratio

cR   = cs * (0.826 + 1.14 * nu) / (1 + nu)   # Rayleigh wave speed

# Normalized frequency / wavelength
f_star      = 2.0
lambda_star = 1.0
omega       = 2 * np.pi * f_star * cR / lambda_star

# Physical domain size  (same as the paper)
H = 4.305 * lambda_star                # depth
W = 12.5  * lambda_star                # width

# Cloak geometry
a   = 0.0774 * H                       # inner triangle depth
b   = 3 * a                            # outer triangle depth
c   = 0.309 * H / 2.0                  # half-width at surface

# ═══════════════════════════════════════════════════════════════════════
# Absorbing-layer (PML-region) parameters
# ═══════════════════════════════════════════════════════════════════════

L_pml    = 1.0 * lambda_star           # thickness of each absorbing layer
xi_max   = 4.0                         # peak damping ratio at outer edge
pml_pow  = 2                           # ramp exponent  (quadratic = 2)

# Extended domain with absorbing layers on left, right, bottom
#   Physical domain occupies  x ∈ [L_pml, L_pml+W],  y ∈ [L_pml, L_pml+H]
#   Free surface at y = L_pml + H  (top edge — no PML)
W_total = 2 * L_pml + W
H_total = L_pml + H

# Offsets so that physical coordinates map back to original system
x_off = L_pml
y_off = L_pml

# Source location in the EXTENDED mesh coordinate system
x_src_phys = 0.05 * W                  # in original coords
x_src      = x_off + x_src_phys        # in extended-mesh coords
y_top      = H_total                   # free-surface y-coordinate

# Cloak centre in extended coordinates
x_c = x_off + W / 2.0

# ═══════════════════════════════════════════════════════════════════════
#  Mesh
# ═══════════════════════════════════════════════════════════════════════

n_pml_x = 32                           # elements across each lateral PML
n_pml_y = 32                           # elements across the bottom PML
nx_phys = 200
ny_phys = 60

nx_total = n_pml_x + nx_phys + n_pml_x
ny_total = n_pml_y + ny_phys

# ═══════════════════════════════════════════════════════════════════════
# Triangular-cloak coordinate transformation
# ═══════════════════════════════════════════════════════════════════════

def _in_cloak(x, is_reference=IS_REFERENCE):
    """True inside the cloak annulus (uses EXTENDED mesh coords)."""
    depth = y_top - x[1]                # depth from free surface
    r     = jnp.abs(x[0] - x_c) / c
    d2    = b * (1.0 - r)
    d1    = a * (1.0 - r)
    if is_reference:
        return False
    else:
        return (r <= 1.0) & (depth >= d1) & (depth <= d2)

def _in_defect(x):
    """True inside the hidden void (uses EXTENDED mesh coords)."""
    depth = y_top - x[1]
    r     = jnp.abs(x[0] - x_c) / c
    d1    = a * (1.0 - r)
    return (r <= 1.0) & (depth >= 0.0) & (depth <= d1)

def F_tensor(x):
    """Deformation gradient of the triangular transformation."""
    sign = jnp.sign(x[0] - x_c)
    F21  = sign * a / c
    F22  = (b - a) / b
    F_cloak = jnp.array([[1.0, 0.0],
                          [F21, F22]])
    return jnp.where(_in_cloak(x), F_cloak, jnp.eye(2))


def generate_gmsh_mesh():
    """Generate a TRI3 mesh of the domain.

    When IS_REFERENCE is False (cloak simulation):
      The triangular defect is cut out as a true hole.
    When IS_REFERENCE is True (reference simulation):
      Plain rectangular domain — no defect, no cloak, homogeneous space.

    The triangle vertices (in extended-mesh coords):
      - left  surface corner: (x_c - c, y_top)
      - right surface corner: (x_c + c, y_top)
      - apex (deepest point): (x_c,     y_top - a)
    """
    h_elem = min(W_total / nx_total, H_total / ny_total)
    h_fine = h_elem / 4.0       # refined size near the cloak
    h_surf = h_elem / 4.0       # refined size near the free surface

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("cloak_domain")

    # Corner points of the rectangle
    p1 = gmsh.model.geo.addPoint(0.0,     0.0,     0.0, h_elem)  # bottom-left
    p2 = gmsh.model.geo.addPoint(W_total, 0.0,     0.0, h_elem)  # bottom-right
    p3 = gmsh.model.geo.addPoint(W_total, H_total, 0.0, h_elem)  # top-right
    p4 = gmsh.model.geo.addPoint(0.0,     H_total, 0.0, h_elem)  # top-left

    if IS_REFERENCE:
        # ── Reference: simple rectangle, no defect ──
        l_bot   = gmsh.model.geo.addLine(p1, p2)
        l_right = gmsh.model.geo.addLine(p2, p3)
        l_top   = gmsh.model.geo.addLine(p3, p4)
        l_left  = gmsh.model.geo.addLine(p4, p1)

        outer_loop = gmsh.model.geo.addCurveLoop([l_bot, l_right, l_top, l_left])
        surf = gmsh.model.geo.addPlaneSurface([outer_loop])
        gmsh.model.geo.synchronize()

        top_lines = [l_top]

    else:
        # ── Cloak: triangle cut out as a hole ──
        pt_left  = gmsh.model.geo.addPoint(x_c - c, y_top, 0.0, h_fine)
        pt_right = gmsh.model.geo.addPoint(x_c + c, y_top, 0.0, h_fine)
        pt_apex  = gmsh.model.geo.addPoint(x_c, y_top - a,  0.0, h_fine)

        # Outer cloak apex (for refinement)
        oc_apex  = gmsh.model.geo.addPoint(x_c,     y_top - b, 0.0, h_fine)

        # Rectangle edges (top edge split into 3 segments around the triangle opening)
        l_bot   = gmsh.model.geo.addLine(p1, p2)          # bottom
        l_right = gmsh.model.geo.addLine(p2, p3)          # right
        l_top1  = gmsh.model.geo.addLine(p3, pt_right)    # top: right corner → tri right
        l_top2  = gmsh.model.geo.addLine(pt_left, p4)     # top: tri left → left corner
        l_left  = gmsh.model.geo.addLine(p4, p1)          # left

        # Triangle slopes (the opening between pt_left and pt_right is the free surface)
        tl_right = gmsh.model.geo.addLine(pt_right, pt_apex)  # right slope
        tl_left  = gmsh.model.geo.addLine(pt_apex, pt_left)   # left slope

        # Single boundary loop: rectangle outer edges + triangle slopes
        outer_loop = gmsh.model.geo.addCurveLoop([
            l_bot, l_right, l_top1, tl_right, tl_left, l_top2, l_left
        ])

        surf = gmsh.model.geo.addPlaneSurface([outer_loop])

        # Embed oc_apex inside the surface so it becomes a connected mesh node
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.embed(0, [oc_apex], 2, surf)

        top_lines = [l_top1, l_top2]

        # ── Mesh refinement near the cloak/defect region ──
        f_dist_inner = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(f_dist_inner, "CurvesList",
                                         [tl_right, tl_left])
        gmsh.model.mesh.field.setNumber(f_dist_inner, "Sampling", 100)

        f_dist_outer = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(f_dist_outer, "PointsList",
                                         [pt_left, pt_right, oc_apex])

        f_dist_cloak = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(f_dist_cloak, "FieldsList",
                                         [f_dist_inner, f_dist_outer])

        f_thresh_cloak = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(f_thresh_cloak, "InField", f_dist_cloak)
        gmsh.model.mesh.field.setNumber(f_thresh_cloak, "SizeMin", h_fine)
        gmsh.model.mesh.field.setNumber(f_thresh_cloak, "SizeMax", h_elem)
        gmsh.model.mesh.field.setNumber(f_thresh_cloak, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(f_thresh_cloak, "DistMax", b * 2.0)

    # ── Mesh refinement near the free surface (x4) ──
    f_dist_surf = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(f_dist_surf, "CurvesList", top_lines)
    gmsh.model.mesh.field.setNumber(f_dist_surf, "Sampling", 200)

    f_thresh_surf = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(f_thresh_surf, "InField", f_dist_surf)
    gmsh.model.mesh.field.setNumber(f_thresh_surf, "SizeMin", h_surf)
    gmsh.model.mesh.field.setNumber(f_thresh_surf, "SizeMax", h_elem)
    gmsh.model.mesh.field.setNumber(f_thresh_surf, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(f_thresh_surf, "DistMax", lambda_star)

    # ── Final background mesh field ──
    if IS_REFERENCE:
        gmsh.model.mesh.field.setAsBackgroundMesh(f_thresh_surf)
    else:
        f_final = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(f_final, "FieldsList",
                                         [f_thresh_cloak, f_thresh_surf])
        gmsh.model.mesh.field.setAsBackgroundMesh(f_final)

    # Disable default size computation so the background field controls everything
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    gmsh.model.mesh.generate(2)

    # Export and read via meshio
    msh_path = "output/_cloak_mesh.msh"
    gmsh.write(msh_path)
    gmsh.finalize()

    msh = meshio.read(msh_path)
    points = msh.points[:, :2]
    cells = msh.cells_dict['triangle']
    return points, cells


points, cells = generate_gmsh_mesh()
mesh = Mesh(points, cells, ele_type='TRI3')

# ═══════════════════════════════════════════════════════════════════════
# Damping profile  ξ(x)
# ═══════════════════════════════════════════════════════════════════════

def _xi_profile(x):
    """Compute the local damping ratio ξ(x).

    ξ = 0 inside the physical domain and ramps quadratically
    into each absorbing-layer region (left, right, bottom).
    In PML corner regions both directions contribute additively.
    """
    # --- lateral (x-direction) attenuation ---
    d_left  = jnp.maximum(x_off - x[0], 0.0)          # distance into left PML
    d_right = jnp.maximum(x[0] - (x_off + W), 0.0)    # distance into right PML
    xi_x    = xi_max * (jnp.maximum(d_left, d_right) / L_pml) ** pml_pow

    # --- vertical (y-direction) attenuation – bottom only ---
    d_bot   = jnp.maximum(y_off - x[1], 0.0)           # distance into bottom PML
    xi_y    = xi_max * (d_bot / L_pml) ** pml_pow

    # additive combination (works well for corner regions)
    return xi_x + xi_y


# ═══════════════════════════════════════════════════════════════════════
# Effective material tensors
# ═══════════════════════════════════════════════════════════════════════

def C_iso():
    """4th-order isotropic stiffness tensor  C₀[i,j,k,l]."""
    C = jnp.zeros((2, 2, 2, 2))
    delta = jnp.eye(2)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    C = C.at[i, j, k, l].set(
                        lam * delta[i, j] * delta[k, l]
                        + mu * (delta[i, k] * delta[j, l]
                                + delta[i, l] * delta[j, k])
                    )
    return C

C0 = C_iso()

# paper-based symmetrization:
# augmented Voigt index map: 0->11, 1->22, 2->12, 3->21
pairs = [(0,0), (1,1), (0,1), (1,0)]

def C_to_voigt4(C):
    M = jnp.zeros((4,4))
    for I,(i,j) in enumerate(pairs):
        for J,(k,l) in enumerate(pairs):
            M = M.at[I,J].set(C[i,j,k,l])
    return M

def voigt4_to_C(M):
    C = jnp.zeros((2,2,2,2))
    for I,(i,j) in enumerate(pairs):
        for J,(k,l) in enumerate(pairs):
            C = C.at[i,j,k,l].set(M[I,J])
    return C

def symmetrize_stiffness(Ceff):
    # paper arithmetic mean in augmented Voigt:
    # Csym_IJ = (Ceff_IJ + Ceff_IJbar + Ceff_IbarJ + Ceff_IbarJbar)/4
    # where bar swaps 12 <-> 21  (indices 2 <-> 3)
    M = C_to_voigt4(Ceff)
    P = jnp.array([[1,0,0,0],
                   [0,1,0,0],
                   [0,0,0,1],
                   [0,0,1,0]])   # swaps 2 and 3

    MbarI  = P @ M
    MbarJ  = M @ P
    MbarIJ = P @ M @ P

    Msym = (M + MbarI + MbarJ + MbarIJ) / 4.0

    # enforce symmetric matrix (Cauchy): Msym = (Msym + Msym.T)/2
    Msym = 0.5*(Msym + Msym.T)
    return voigt4_to_C(Msym)


def C_eff(x):
    """Position-dependent effective stiffness tensor."""
    F   = F_tensor(x)
    J   = jnp.linalg.det(F)
    # Cosserat transformed tensor
    Cnew = jnp.einsum('iI,kK,IjKl->ijkl', F, F, C0) / J
    # Cnew = symmetrize_stiffness(Cnew)    # symmetrisation (uncomment if necessary)
    return jnp.where(_in_cloak(x), Cnew, C0)

def rho_eff(x):
    """Position-dependent effective density."""
    F   = F_tensor(x)
    J   = jnp.linalg.det(F)
    rho_cloak = rho0 / J
    return jnp.where(_in_cloak(x), rho_cloak, rho0)

# ═══════════════════════════════════════════════════════════════════════
# Source
# ═══════════════════════════════════════════════════════════════════════

sigma_src = 0.01 * lambda_star
# sigma_src = 3.0 * (W / nx_phys)        # Gaussian half-width  (~3 elements)
F0        = 1.0                         # force amplitude (Pa)

def top_surface(point):
    """Selects the free (top) surface  y = H_total."""
    return jnp.isclose(point[1], H_total)

# ═══════════════════════════════════════════════════════════════════════
# FEM Problem  (vec = 4 :  Re(uₓ), Re(u_y), Im(uₓ), Im(u_y))
# ═══════════════════════════════════════════════════════════════════════

class RayleighCloakProblem(Problem):
    """Frequency-domain elastodynamics with Rayleigh-damping absorbing layers.

    DOF ordering per node:  [Re(uₓ), Re(u_y), Im(uₓ), Im(u_y)]
    """

    def custom_init(self):
        # Precompute material data at every quadrature point
        # physical_quad_points shape: (num_cells, num_quads, dim)
        self.internal_vars = [
            jax.vmap(jax.vmap(C_eff))(self.physical_quad_points),    # (nc, nq, 2,2,2,2)
            jax.vmap(jax.vmap(rho_eff))(self.physical_quad_points),  # (nc, nq)
            jax.vmap(jax.vmap(_xi_profile))(self.physical_quad_points),  # (nc, nq)
        ]

    # -----------------------------------------------------------------
    def get_tensor_map(self):
        """σ(∇u) — stiffness + stiffness-proportional damping coupling.

        u_grad shape: (4, 2)   →  rows 0-1 = ∇u_R,  rows 2-3 = ∇u_I
        Returns  stress  shape (4, 2).
        """
        def stress(u_grad, C_q, _rho_q, xi_q):
            grad_R = u_grad[:2, :]       # (2, 2)  – ∂Re(u)/∂x
            grad_I = u_grad[2:, :]       # (2, 2)  – ∂Im(u)/∂x

            sig_R_undamped = jnp.einsum('ijkl,kl->ij', C_q, grad_R)
            sig_I_undamped = jnp.einsum('ijkl,kl->ij', C_q, grad_I)

            # With stiffness-proportional damping  (coupling coeff = ξ):
            #   σ_R = C:ε_R − ξ C:ε_I
            #   σ_I = C:ε_I + ξ C:ε_R
            sig_R = sig_R_undamped - xi_q * sig_I_undamped
            sig_I = sig_I_undamped + xi_q * sig_R_undamped

            return jnp.concatenate([sig_R, sig_I], axis=0)   # (4, 2)

        return stress

    # -----------------------------------------------------------------
    def get_mass_map(self):
        """Inertia + mass-proportional damping coupling.

        u shape: (4,)  →  u[:2] = u_R,  u[2:] = u_I
        Returns  (4,).
        """
        def inertia(u, _x, _C_q, rho_q, xi_q):
            u_R = u[:2]
            u_I = u[2:]

            # −ω²ρ u_R − ξ ω² ρ u_I   (real part)
            # −ω²ρ u_I + ξ ω² ρ u_R   (imaginary part)
            m_R = -omega**2 * rho_q * (u_R + xi_q * u_I)
            m_I = -omega**2 * rho_q * (u_I - xi_q * u_R)

            return jnp.concatenate([m_R, m_I])

        return inertia

    # -----------------------------------------------------------------
    def get_surface_maps(self):
        """Surface traction — real vertical point load, no imaginary component."""
        def traction(_u, x):
            g = F0 * jnp.exp(-0.5 * ((x[0] - x_src) / sigma_src) ** 2)
            # Only the real-part vertical component  Re(u_y)
            return jnp.array([0.0, g, 0.0, 0.0])

        return [traction]

# ═══════════════════════════════════════════════════════════════════════
# Boundary conditions
# ═══════════════════════════════════════════════════════════════════════
#
# Zero displacement (real & imaginary) on the OUTER edges of the PML:
#   • bottom  (y = 0)
#   • left    (x = 0)
#   • right   (x = W_total)
# The top surface (y = H_total) remains FREE (stress-free for Rayleigh waves).
# ═══════════════════════════════════════════════════════════════════════

def bc_bottom(point):
    return jnp.isclose(point[1], 0.0)

def bc_left(point):
    return jnp.isclose(point[0], 0.0)

def bc_right(point):
    return jnp.isclose(point[0], W_total)

def zero(point):
    return 0.0

# All 4 DOFs fixed on bottom, left, and right outer PML boundaries
dirichlet_bc_info = [
    # location_fns  (one per constrained DOF)
    [bc_bottom, bc_bottom, bc_bottom, bc_bottom,
     bc_left,   bc_left,   bc_left,   bc_left,
     bc_right,  bc_right,  bc_right,  bc_right],
    # DOF indices
    [0, 1, 2, 3,
     0, 1, 2, 3,
     0, 1, 2, 3],
    # prescribed values
    [zero, zero, zero, zero,
     zero, zero, zero, zero,
     zero, zero, zero, zero],
]

# ═══════════════════════════════════════════════════════════════════════
# Assemble and solve
# ═══════════════════════════════════════════════════════════════════════

problem = RayleighCloakProblem(
    mesh          = mesh,
    vec           = 4,
    dim           = 2,
    ele_type      = 'TRI3',
    dirichlet_bc_info = dirichlet_bc_info,
    location_fns  = [top_surface],
)

print("Solving frequency-domain system with absorbing layers …")
sol_list = solver(problem, solver_options={'umfpack_solver': {}})
u = sol_list[0]            # shape (num_nodes, 4)

# ═══════════════════════════════════════════════════════════════════════
# Save results for plotting
# ═══════════════════════════════════════════════════════════════════════

import os
os.makedirs("output", exist_ok=True)
from datetime import datetime

# stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
stamp = ''
np.savez(f"output/results_{f_star:.2f}_{stamp}.npz",
         u=np.asarray(u),
         pts_x=np.asarray(mesh.points[:, 0]),
         pts_y=np.asarray(mesh.points[:, 1]),
         x_src=x_src, y_top=y_top,
         x_off=x_off, y_off=y_off,
         W=W, H=H,
         x_src_phys=x_src_phys,
         f_star=f_star)
print("Results saved → output/results_{f_star:.2f}_{stamp}.npz")
# print("Run `python plot_results.py` to generate plots.")

from plot_results import plot_results, plot_vtk_results
plot_results(f"output/results_{f_star:.2f}_{stamp}.npz")

# ── Save VTK with mesh connectivity (void is naturally visible) ──
import vtk
from vtk.util.numpy_support import numpy_to_vtk

pts_np = np.asarray(mesh.points)   # (num_nodes, 2 or 3)
cells_np = np.asarray(mesh.cells)  # (num_cells, 3)
u_np = np.asarray(u)               # (num_nodes, 4)

vtk_pts = vtk.vtkPoints()
pts3d = np.zeros((pts_np.shape[0], 3))
pts3d[:, :pts_np.shape[1]] = pts_np
vtk_pts.SetData(numpy_to_vtk(pts3d, deep=True))

grid = vtk.vtkUnstructuredGrid()
grid.SetPoints(vtk_pts)

for cell_nodes in cells_np:
    tri = vtk.vtkTriangle()
    for j in range(3):
        tri.GetPointIds().SetId(j, int(cell_nodes[j]))
    grid.InsertNextCell(tri.GetCellType(), tri.GetPointIds())

mag = np.sqrt(u_np[:, 0]**2 + u_np[:, 1]**2 + u_np[:, 2]**2 + u_np[:, 3]**2)
arr_mag = numpy_to_vtk(mag, deep=True)
arr_mag.SetName("mag_u")
grid.GetPointData().AddArray(arr_mag)

arr_sol = numpy_to_vtk(u_np, deep=True)
arr_sol.SetName("u")
grid.GetPointData().AddArray(arr_sol)

for name, val in [("x_src", x_src), ("y_top", y_top),
                  ("x_off", x_off), ("y_off", y_off),
                  ("W", W), ("H", H),
                  ("x_src_phys", x_src_phys), ("f_star", f_star)]:
    vtk_arr = vtk.vtkDoubleArray()
    vtk_arr.SetName(name)
    vtk_arr.InsertNextValue(val)
    grid.GetFieldData().AddArray(vtk_arr)

vtk_path = f"output/results_{f_star:.2f}.vtk"
writer = vtk.vtkUnstructuredGridWriter()
writer.SetFileName(vtk_path)
writer.SetInputData(grid)
writer.Write()
print(f"VTK saved → {vtk_path}")

plot_vtk_results(vtk_path)

# ═══════════════════════════════════════════════════════════════════════
# Autograd-compatible loss function  (example)
# ═══════════════════════════════════════════════════════════════════════

def cloaking_loss(problem_instance, u_sol):
    """Example loss: mean |u|² on the surface downstream of the cloak.

    This function is differentiable w.r.t. any JAX parameter that
    flows into the Problem (e.g. cloak geometry a, b, c, or material
    properties) because the entire forward solve is built on JAX ops.
    """
    # Surface nodes downstream of cloak  (right half of physical domain)
    surface_mask = (
        jnp.isclose(mesh.points[:, 1], y_top) &
        (mesh.points[:, 0] > x_c + c)
    )
    # Sum of squared displacement magnitudes on those nodes
    u_R_sq = u_sol[:, 0]**2 + u_sol[:, 1]**2
    u_I_sq = u_sol[:, 2]**2 + u_sol[:, 3]**2
    energy  = jnp.where(surface_mask, u_R_sq + u_I_sq, 0.0)
    return jnp.sum(energy) / jnp.maximum(jnp.sum(surface_mask.astype(float)), 1.0)

print("\n✓  Script finished.  Absorbing layers active on left / right / bottom.")
print(f"   Domain:  {W_total:.2f} × {H_total:.2f}   (physical {W:.2f} × {H:.2f})")
print(f"   PML thickness = {L_pml:.3f},  ξ_max = {xi_max},  ramp power = {pml_pow}")
print(f"   Mesh:  {mesh.cells.shape[0]} triangles (Gmsh-generated)")