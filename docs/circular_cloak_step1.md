# Fully embedded circular cloak — Step 1: continuous (regularised) ideal cloak

Branch `circular_cloak`. Reference: Chatzopoulos et al., *Cloaking Rayleigh waves via
symmetrized elastic tensors*, IJES 191 (2023) — `docs/symmetrized_elastic_tensors.pdf`.
Here the carpet geometry is replaced by a complete annulus embedded in an unbounded
medium and excited by bulk P and S waves.

## 1. Problem

| item | value |
|---|---|
| physics | 2-D time-harmonic plane-strain elastodynamics, time factor e^{+iωt} |
| background | λ = μ = 144 MPa, ρ = 1600 kg/m³ → c_S = 300 m/s, c_P = 519.6 m/s |
| frequency | f = 200 Hz → λ_S = 1.50 m, λ_P = 2.60 m |
| cloak | annulus a < r < b, a = 0.5 m, b = 1.5 m (= λ_S), centre (x_c, y_c) = (0, 0), configurable |
| cavity | r < a removed; σ·n = 0 on the entire circle r = a (natural BC, no surface term) |
| interface | r = b bonded: conforming mesh, displacement and traction continuity are built into the weak form |
| domain | physical box [−5, 5]² m, absorbing layer of thickness L on **all four sides**, outer edge clamped |
| absorbing layer | C → (1 + iξ)C, ρ → (1 − iξ)ρ, ξ = ξ_max[(d_x/L)^p + (d_y/L)^p]; base L = 6 m (2.3 λ_P), ξ_max = 1.5, p = 3 |
| source | Gaussian body force f = F₀ d̂ g(x)/∫g, g = exp(−‖x − x_s‖²/2σ²), x_s = (−3, −3) (below-left), d̂ = (0, 1), F₀ = 1000 N/m, σ = 0.075 m |
| elements | TRI6 (isoparametric, mid-side nodes on the circles), 6-point quadrature; h_bg = λ_S/12, h_cloak = h_bg/2, h(r=a) = 2ε dr/dR(ε), h_src = σ/2 |

The source's normalising integral ∫g is evaluated with the **same quadrature as the
solve**, so the discrete total force is exactly F₀ on every mesh (`source.ratio` in each
case's `info` reports ∫g_discrete / 2πσ² ≈ 1.0000).

Strong form solved:

    ∂_j[(1 + iξ) C_ijkl ∂_l u_k] + ω²(1 − iξ) ρ u_i + f_i = 0.

## 2. Transformation and material (audit)

**Map.** Radial, θ = Θ, from the virtual annulus ε ≤ R ≤ b onto a ≤ r ≤ b, identity
outside b (`rayleigh_cloak/geometry/circular.py`, `RadialMap`):

* `linear` (paper App. Eq. 30, N = 1): r = a + (b − a)(R − ε)/(b − ε)
* `quadratic` (paper App. Eqs. 31–32): r = A₂R² + A₁R + A₀ with dr/dR(b) = 1;
  monotone only if b − ε > 2(a − ε).

**Jacobian.** Because θ = Θ, the virtual and physical polar frames coincide and

    F = ∂x/∂X = F_r e_r⊗e_r + F_θ e_θ⊗e_θ,   F_r = dr/dR,  F_θ = r/R,   J = F_r F_θ.

`F_cloak` builds this Cartesian tensor directly; `virtual_point`/`physical_point`
implement Ξ⁻¹ and Ξ.

**Tensor convention.** The FEM (jax-fem `get_tensor_map`) uses σ_ij = C_ijkl ∂u_k/∂x_l
with equilibrium ∂_jσ_ij, i.e. the derivative sits on indices 2 and 4. The BGM / Cosserat
push-forward with gauge U(X) = u(x) is then

    C'_ijkl = F_jJ F_lL C_iJkL / J,     ρ' = ρ / J        (materials.C_pushforward)

which is the paper's Eq. (5) written in its transposed (derivative-first) convention.
The full tensor is kept: major symmetry holds, minor symmetry does not.

**Polar closed form** (`materials.C_radial_polar`, our convention, g = F_r/F_θ):

    C_rrrr = g(λ+2μ)   C_θθθθ = (λ+2μ)/g   C_rrθθ = λ
    C_rθrθ = μ/g       C_θrθr = μ g        C_rθθr = C_θrrθ = μ

It is rotated to the Cartesian FEM frame with the **full fourth-order rotation**
C_ijkl = Q_ia Q_jb Q_kc Q_ld C^pol_abcd, Q = [e_r | e_θ] (`materials.rotate_C`).

**Automated audit** — `tests/test_circular_transform.py` (50 checks, both maps,
ε ∈ {0, 0.005, 0.05, 0.2}) plus the existing `tests/test_ceff_nassar_eq25.py`:

1. r(ε) = a, r(b) = b, monotone, R(r(R)) = R; dr/dR equals autodiff.
2. Closed-form Cartesian F equals `jax.jacfwd` of the point map X → x at X = Ξ⁻¹(x).
3. det F = F_r F_θ, ρ' = ρ/J.
4. Cartesian push-forward equals the rotated polar closed form (and back-rotation recovers it).
5. Major symmetry holds; minor symmetry is violated (Cosserat).
6. Strain-energy invariance J·C'(x):G_x:G_x = C:G_X:G_X for random non-symmetric gradients,
   G_x = G_X F⁻¹ — an index-convention-free check of the push-forward.
7. ε = 0, linear reproduces the previous singular-map code and Nassar et al. (2018) Eq. 2.5.

```
.venv/bin/python -m pytest tests/test_circular_transform.py tests/test_ceff_nassar_eq25.py -q   # 52 passed
```

**Material assignment.** Gmsh (OCC) fragments the domain into disk / annulus / box /
absorbing surfaces; every element carries its region tag. Annulus elements receive C', ρ'
at each quadrature point (radius clamped to [a, b] to absorb the O(h³) isoparametric
boundary approximation); all other elements get (C₀, ρ). No pixels, no cell averaging.

## 3. Regularisation ε

The singular map (ε = 0) sends a point to the circle r = a (F_θ → ∞). With ε > 0 the cloak
is the image of the homogeneous plane **with a traction-free hole of radius ε**: its
exterior field is *exactly* that of the ε-hole (case D), and inside the cloak
u_C(x) = u_D(Ξ⁻¹(x)). The finite-ε cloak is therefore a near-cloak; its exterior error is
the scattering of an ε-hole, expected O((kε)²). Results at finite ε are a regularised
approximation to the ideal limit.

Near r = a the ε-hole's near field is compressed into a radial layer of physical thickness
δ = ε·dr/dR(ε); the mesh is graded to h(r = a) = 2δ (`mesh.inner_bl_factor`).

## 4. Cases and metrics

| case | mesh | material |
|---|---|---|
| A homogeneous | full mesh incl. disk | C₀, ρ everywhere |
| B hole | A's mesh with disk elements removed | C₀, ρ |
| C hole + cloak | same as B | C', ρ' in the annulus |
| D virtual ε-hole | own mesh, hole r < ε | C₀, ρ |

A, B and C share all nodes outside r = a. All metrics are evaluated on a **common,
mesh-independent** set of exterior points (uniform grid, spacing 0.05 m, r > 1.02b,
‖x − x_s‖ > 0.75 m = λ_S/2, physical box only), by exact isoparametric interpolation
(`embedded/sampling.py`):

    e_X = ‖u_X − u_A‖₂ / ‖u_A‖₂   (complex, both components).

Also reported: rings r = 1.25b and 2b; C vs D outside (implementation + discretisation);
u_C(x) vs u_D(Ξ⁻¹x) inside the annulus; A vs the closed-form Green tensor
(`embedded/analytic.py`, Gaussian source exact through the factor e^{−k²σ²/2}).

## 5. Results (base: ε = 0.005 m = 0.01a, linear map, TRI6, 12 elements/λ_S)

Figures: `docs/figures/circular_step1/` (copies of `output/circular_step1/figures/`);
full tables: `docs/figures/circular_step1/summary.md`.

| quantity | value |
|---|---|
| **B** hole, no cloak, vs A | **0.312** (rings 1.25b: 0.441, 2b: 0.328) |
| **C** regularised ideal cloak, vs A | **1.33 × 10⁻⁴** (rings: 1.89e-4, 1.41e-4) |
| D virtual ε-hole vs A (intrinsic ε error) | 1.51 × 10⁻⁴ |
| C vs D, exterior (implementation + discretisation) | 3.8 × 10⁻⁵ |
| u_C(x) vs u_D(Ξ⁻¹x), inside the annulus | 1.7 × 10⁻⁴ |
| A vs closed-form Green tensor | 9.5 × 10⁻⁴ (absorbing-layer floor ≈ 7 × 10⁻⁴) |
| mesh / DOFs (case A) | 88 470 TRI6, 710 580 real DOFs, 6.6 GiB peak, ~1 min for A–D |

* `01_setup_bc_source.png` — domain, absorbing frame (ξ map), clamped outer edge, cavity,
  bonded interface, source and the exterior sampling set.
* `02_mesh.png` — full mesh, cloak, graded inner boundary (h = 6.7 mm at r = a), source.
* `03_material.png` — polar components and ρ'/ρ vs r; Cartesian C'_xxxx; minor-symmetry
  violation map (up to 45 % at r = a).
* `04_fields_ABC.png` — Re u_x, Re u_y, |u| for A, B, C on common colour scales.
* `05_scattered.png` — B − A and C − A (C masked for r < b; the field there is the
  transformed one); C − A shown again with its own scale (×2500).
* `06_lines.png` — |u| on rings 1.25b, 2b; Re u_y along the source–centre line.

### Regularisation (ε) — `07_eps_convergence.png`

| ε/a | C vs A | D vs A | C vs D | interior |
|---|---|---|---|---|
| 0.2   | 6.27e-2 | 6.23e-2 | 3.9e-4 | 9.9e-4 |
| 0.1   | 1.57e-2 | 1.56e-2 | 2.0e-5 | 1.8e-4 |
| 0.05  | 3.73e-3 | 3.83e-3 | 1.1e-4 | 4.0e-4 |
| 0.02  | 5.46e-4 | 6.05e-4 | 7.6e-5 | 3.0e-4 |
| 0.01  | 1.33e-4 | 1.51e-4 | 3.8e-5 | 1.7e-4 |
| 0.005 | 4.01e-5 | 3.76e-5 | 3.2e-5 | 1.5e-4 |

D vs A falls exactly as ε² (×4 per halving), the scattering of a small traction-free hole.
C follows D until it meets the discretisation floor C vs D ≈ 3 × 10⁻⁵, i.e. the
finite-ε cloak behaves as the regularised near-cloak it is supposed to be, converging to
the ideal limit as ε → 0. (h at r = a is tied to δ = ε·dr/dR(ε), so the mesh near the
cavity refines with ε.)

### Mesh — `08_mesh_sensitivity.png` (ε/a = 0.01, all sizes scaled together)

| scale | h_bg [m] | el/λ_S | elements | A vs analytic | C vs A | C vs D | A vs finest |
|---|---|---|---|---|---|---|---|
| 2.0  | 0.250 | 6  | 23 128  | 6.9e-3 | 4.1e-4 | 4.0e-4 | 6.8e-3 |
| 1.4  | 0.175 | 8.6 | 46 016  | 2.2e-3 | 1.5e-4 | 1.1e-4 | 2.0e-3 |
| 1.0  | 0.125 | 12 | 88 470  | 9.5e-4 | 1.3e-4 | 3.8e-5 | 5.8e-4 |
| 0.75 | 0.094 | 16 | 154 820 | 7.4e-4 | 1.4e-4 | 2.0e-5 | (ref) |
| 0.6  | 0.075 | 20 | ~241 000 | skipped by memory guard — run on the large server |

A converges at ≈ h^3.5 until it reaches the absorbing-layer floor; C vs A saturates at the
ε-error (D vs A ≈ 1.5e-4) from 12 el/λ_S on; C vs D keeps decreasing.

### Absorbing layers (ε/a = 0.01, scale 1)

| layer | A vs analytic | A vs base A | C vs A |
|---|---|---|---|
| L = 4 m, ξ_max = 4, p = 2 (original repo setting) | 3.5e-2 | 3.5e-2 | 1.32e-4 |
| L = 4 m, ξ_max = 1.5, p = 3 | 3.6e-3 | 3.0e-3 | 1.33e-4 |
| **L = 6 m, ξ_max = 1.5, p = 3 (base)** | 9.5e-4 | — | 1.33e-4 |
| L = 8 m, ξ_max = 1.5, p = 3 | 7.1e-4 | 4.2e-4 | 1.33e-4 |
| L = 6 m, ξ_max = 0.75 | 2.4e-3 | 2.3e-3 | 1.33e-4 |
| L = 6 m, ξ_max = 3.0 | 1.7e-3 | 9.6e-4 | 1.33e-4 |

The steep original ramp reflected ≈3.5 % back into the box; the gentle cubic ramp brings
the absolute error below 10⁻³. The **cloak metric is insensitive** (1.32–1.33e-4) because
A, B and C share the same layer and its reflections.

### Radial map

| map | C vs A | D vs A | C vs D | interior |
|---|---|---|---|---|
| linear (base) | 1.33e-4 | 1.51e-4 | 3.8e-5 | 1.7e-4 |
| quadratic (C2-ε, dr/dR(b) = 1) | 1.23e-4 | 1.51e-4 | 3.9e-5 | 1.4e-4 |

Both maps give the same ideal-cloak performance (same virtual problem). The quadratic map
removes the stiffness jump at r = b, which matters for symmetrised/pixelated designs (Step 2).

## 6. Reproduce

```bash
# environment (this workstation: no conda / PETSc)
python3 -m venv .venv
.venv/bin/pip install "jax[cpu]==0.8.0" "jax-fem==0.0.11" "fenics-basix>=0.10.0" \
    "gmsh==4.14.0" "meshio==5.3.5" "pydantic==2.13.2" PyYAML numpy scipy matplotlib pypardiso pyfiglet pytest

.venv/bin/python -m pytest tests/test_circular_transform.py tests/test_ceff_nassar_eq25.py -q
.venv/bin/python scripts/circular_step1.py --config configs/circular_embedded/step1.yaml --study all
# or one at a time: --study base | eps | mesh | absorbing | map | summary
```

Outputs: `output/circular_step1/{base,eps/*,mesh/*,absorbing/*,map/*}/` (`case_{A,B,C,D}.npz`
with mesh + complex field, `metrics.json`, `config.json`), `figures/`, `summary.md`.
Each study point runs in a fresh subprocess and is reused if already solved with an identical
config (`--force` re-solves).

**Memory guard** (`rayleigh_cloak/embedded/memguard.py`): before meshing, the element count
is estimated from the size field, and before each assembly/solve the actual count is used;
peak RAM is taken as 75 kB/TRI6 element (measured 66–78 kB) × 1.25, plus a 1.5 GiB reserve,
against `MemAvailable`. A refused point is logged as skipped and the sweep continues.
On this 31 GB machine the limit is ≈ 200k elements. On the large server, add finer levels
to `studies.mesh.values` (e.g. 0.6, 0.5, 0.35) and run `--study mesh`; nothing else changes.
The PETSc-free solver (`embedded/linsolve.py`, MKL PARDISO) is CPU-only; for very large
meshes on the server, MUMPS/cuDSS can be plugged in there.

## 7. Step 2 plan (pixelated cloak — not implemented, awaiting review)

1. **Pixel layout.** ≈ 500 *active* pixels covering a ≤ r ≤ b, defined independently of the
   FEM mesh: polar pixels (N_θ × N_r, e.g. 50 × 10 = 500) are natural for a radial map, and a
   Cartesian alternative (pixels kept if their centre is in the annulus, side ≈ 0.11 m for
   500) for comparison. Elements are assigned by quadrature point → pixel lookup; the cavity
   and r = b stay exact (mesh-conforming circles); the pixel grid never moves the geometry.
   Report the actual active-pixel count.
2. **Pixelisation error alone.** Sample the validated ideal C', ρ' into piecewise-constant
   pixels (cell average over quadrature points, and centre-point sampling) with the same
   source / metrics; compare with C (continuous) and B. ε is held at the base value.
3. **Admissible material.** Adapt the neural-field optimisation (`neural_reparam.py`) to
   output per-pixel Cauchy stiffness (minor + major symmetric, positive definite via a
   Cholesky parametrisation of the 3 × 3 Voigt matrix) and ρ > 0 (softplus/exp).
   Initialise from the symmetrised pixel-sampled ideal tensor (paper Eq. 26) and minimise
   the exterior L2 error against A on a training subset of the exterior grid, validating on
   the rest. The adjoint will use the same PARDISO factorisation (transpose solve).
4. **Pixel-count sweep.** ≈ 500 → 250 → 120 → 60 active pixels, same excitation and
   metrics; table of error vs actual pixel count, continuous-cloak and B as bounds.
5. Microstructure realisation is out of scope for Step 2.

