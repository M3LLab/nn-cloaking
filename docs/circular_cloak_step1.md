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
| domain | physical box [−5, 5]² m, PML of thickness L on **all four sides**, outer edge clamped |
| absorbing layer | true PML by complex coordinate stretching, s_j = 1 − iσ_j(x_j)/ω, σ = σ_max (d/L)^p, σ_max = (p+1) c_P ln(1/R) / 2L; base L = 2.6 m (≈ λ_P), p = 2, R = 10⁻⁶. The earlier graded damping layer remains available as `absorbing.kind: damping` |
| source | Gaussian body force f = F₀ d̂ g(x)/∫g, g = exp(−‖x − x_s‖²/2σ²), x_s = (−3, −3) (below-left), F₀ = 1000 N/m; three directions d̂, each a separate solve: vertical (0, 1), horizontal (1, 0), both (1, 1)/√2, σ = 0.075 m |
| elements | TRI6 (isoparametric, mid-side nodes on the circles), 6-point quadrature; h_bg = λ_S/16, h_cloak = h_bg/2, h(r=a) = 2ε dr/dR(ε), h_src = σ/2 |

The source's normalising integral ∫g is evaluated with the **same quadrature as the
solve**, so the discrete total force is exactly F₀ on every mesh (`source.ratio` in each
case's `info` reports ∫g_discrete / 2πσ² ≈ 1.0000).

Strong form solved:

    ∂_j[C^c_ijkl ∂_l u_k] + ω² ρ^c u_i + f_i = 0,
    C^c_ijkl = C_ijkl s_x s_y / (s_j s_l),   ρ^c = ρ s_x s_y,

with s_x = s_y = 1 in the physical box, so C^c, ρ^c reduce to the real background or cloak
values there. Indices j and l carry the stretch because the derivative sits on them.

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
(`embedded/sampling.py`).

**Primary metric: total displacement magnitude**, ‖u‖ = √(|u_x|² + |u_y|²):

    e_X = ‖ ‖u_X‖ − ‖u_A‖ ‖₂ / ‖ ‖u_A‖ ‖₂

**Secondary metric: complex displacement**, e^c_X = ‖u_X − u_A‖₂ / ‖u_A‖₂. The magnitude
metric cannot see phase or the direction of u, so two different fields with equal ‖u‖ score
zero; e^c is the stricter test and is roughly twice e_X here. The superposition and mirror
checks use e^c because they compare fields, and magnitudes do not superpose.

Also reported: rings r = 1.25b and 2b; C vs D outside (implementation + discretisation);
u_C(x) vs u_D(Ξ⁻¹x) inside the annulus; A vs the closed-form Green tensor
(`embedded/analytic.py`, Gaussian source exact through the factor e^{−k²σ²/2}).

## 5. Results (base: ε = 0.005 m = 0.01a, linear map, TRI6, 16 elements/λ_S, PML)

Tables in 5.0–5.4 are for the **vertical** force; 5.5 compares all force directions and source
positions. Figures: `docs/figures/circular_step1/{vertical,horizontal,both}/` and
`docs/figures/circular_step1_offdiag/{vertical,horizontal,both}/`; component plots
(Re u_x, Re u_y) are in each `components/` subfolder. Full tables: `summary.md` in each folder.

### 5.0 Base case

| quantity | ‖u‖ error | complex error |
|---|---|---|
| **B** hole, no cloak, vs A | **0.158** | 0.312 |
| **C** regularised ideal cloak, vs A | **6.68e-05** | 0.00013 |
| C vs A on ring 1.25b | 0.000101 | 0.000185 |
| C vs A on ring 2b | 7.68e-05 | 0.000138 |
| D virtual ε-hole vs A (error from ε alone) | 7.67e-05 | 0.000151 |
| C vs D, exterior (implementation + discretisation) | 1.18e-05 | 2.61e-05 |
| u_C(x) vs u_D(Ξ⁻¹x), inside the annulus | 4.65e-05 | 0.000125 |
| A vs closed-form Green tensor | 5e-05 | 0.000249 |

Mesh: 82 738 TRI6, 664 516 real DOFs, about 6 GiB peak, about 1 min for cases A–D.

* `01_setup_bc_source.png` — domain, PML (stretching map), clamped outer edge, cavity,
  bonded interface, source and the exterior sampling set.
* `02_mesh.png` — full mesh, cloak, graded inner boundary (h = 6.7 mm at r = a), source.
* `03_material.png` — polar components and ρ'/ρ vs r; Cartesian C'_xxxx; minor-symmetry
  violation map (up to 45 % at r = a).
* `04_fields_ABC.png` — ‖u‖ for A, B, C on one colour scale.
* `05_scattered.png` — ‖u_B‖ − ‖u_A‖ and ‖u_C‖ − ‖u_A‖ (C masked for r < b; the field there
  is the transformed one), then C alone on its own scale.
* `06_lines.png` — ‖u‖ on rings 1.25b, 2b and along the source–centre line.

### 5.1 Regularisation (ε) — `07_eps_convergence.png`

| ε/a | C vs A | D vs A | C vs D | interior | C vs A, complex |
|---|---|---|---|---|---|
| 0.2   | 3.12e-2 | 3.11e-2 | 1.2e-4 | 3.4e-4 | 6.27e-2 |
| 0.1   | 7.87e-3 | 7.88e-3 | 5.3e-6 | 3.7e-5 | 1.56e-2 |
| 0.05  | 1.88e-3 | 1.94e-3 | 5.8e-5 | 1.9e-4 | 3.72e-3 |
| 0.02  | 2.83e-4 | 3.07e-4 | 2.7e-5 | 1.0e-4 | 5.55e-4 |
| 0.01  | 6.68e-5 | 7.67e-5 | 1.2e-5 | 4.6e-5 | 1.30e-4 |
| 0.005 | 1.62e-5 | 1.91e-5 | 5.1e-6 | 2.1e-5 | 3.14e-5 |

D vs A falls as ε² (×4 per halving), the scattering of a small traction-free hole. C follows
D, i.e. the finite-ε cloak behaves as the regularised near-cloak it is supposed to be and
approaches the ideal limit as ε → 0. C vs D is small but not monotone in ε: h at r = a is
tied to δ = ε·dr/dR(ε) only below its cap (`inner_h_max`), so the mesh changes along the sweep.

### 5.2 Mesh — `08_mesh_sensitivity.png` (ε/a = 0.01, all sizes scaled together)

| scale | h_bg [m] | el/λ_S | elements | A vs analytic | C vs A | C vs D | A vs finest | C vs D, complex |
|---|---|---|---|---|---|---|---|---|
| 2.0  | 0.188 | 8    | 21 562  | 4.84e-4 | 7.51e-5 | 4.76e-5 | 4.78e-4 | 1.34e-4 |
| 1.4  | 0.131 | 11.4 | 43 022  | 1.52e-4 | 6.58e-5 | 1.84e-5 | 1.47e-4 | 4.31e-5 |
| 1.0  | 0.094 | 16   | 82 738  | 5.00e-5 | 6.68e-5 | 1.18e-5 | 4.70e-5 | 2.61e-5 |
| 0.75 | 0.070 | 21.3 | 144 878 | 1.92e-5 | 6.83e-5 | 9.60e-6 | (ref) | 2.02e-5 |
| 0.6  | 0.056 | 26.7 | 225 204 | skipped by memory guard — run on the large server |

A converges to the analytic solution at ≈ h^3.3 with no floor seen. C vs A saturates at the
ε-error from 11 el/λ_S. C vs D decreases but is flattening near 1e-5 (2e-5 complex) for the
vertical force; whether it keeps converging needs the finer server runs.

### 5.3 Absorbing layers (ε/a = 0.01, 16 el/λ_S)

| layer | elements | A vs analytic | A vs base A | C vs A |
|---|---|---|---|---|
| **PML L = 2.6 m, p = 2, R = 1e-6 (base)** | 82 738 | 5.00e-5 | — | 6.68e-5 |
| PML L = 1.5 m, R = 1e-4 | 66 080 | 1.08e-4 | 9.7e-5 | 6.71e-5 |
| PML L = 4.0 m | 107 256 | 5.01e-5 | 1.4e-5 | 6.71e-5 |
| PML R = 1e-4 | 82 738 | 6.60e-5 | 4.4e-5 | 6.68e-5 |
| PML R = 1e-8 | 82 738 | 4.99e-5 | 1.0e-6 | 6.68e-5 |
| PML p = 3 | 82 738 | 5.01e-5 | 1.0e-6 | 6.68e-5 |
| damping L = 6 m, ξ_max = 1.5, p = 3 (previous base) | 149 054 | 4.43e-4 | 4.4e-4 | 6.71e-5 |
| damping L = 4 m, ξ_max = 4, p = 2 (original repo layer) | 107 256 | 1.80e-2 | 1.8e-2 | 6.64e-5 |

The PML is more accurate with a far thinner layer. The **cloak metric is insensitive to the
layer** (6.6–6.7e-5) because A, B and C share the same layer and its reflections; absolute
field accuracy is not.

### 5.4 Radial map

| map | C vs A | D vs A | C vs D | interior |
|---|---|---|---|---|
| linear (base) | 6.68e-5 | 7.67e-5 | 1.18e-5 | 4.65e-5 |
| quadratic (C2-ε, dr/dR(b) = 1) | 6.38e-5 | 7.67e-5 | 1.47e-5 | 4.30e-5 |

Both maps give the same ideal-cloak performance (same virtual problem). The quadratic map
removes the stiffness jump at r = b, which matters for symmetrised/pixelated designs (Step 2).

### 5.5 Force direction and source position (‖u‖ error against A)

| source | force | B: bare hole | C: cloak | D: ε-hole | C vs D | A vs analytic |
|---|---|---|---|---|---|---|
| (−3, −3), on y = x | vertical | 0.158 | 6.68e-5 | 7.67e-5 | 1.18e-5 | 5.00e-5 |
| (−3, −3) | horizontal | 0.158 | 6.67e-5 | 7.67e-5 | 1.19e-5 | 5.06e-5 |
| (−3, −3) | both | 0.104 | 3.36e-5 | 3.77e-5 | 5.12e-6 | 5.08e-5 |
| (−3.5, −2), off-diagonal | vertical | 0.188 | 7.95e-5 | 9.13e-5 | 1.40e-5 | 4.90e-5 |
| (−3.5, −2) | horizontal | 0.129 | 5.36e-5 | 6.13e-5 | 8.99e-6 | 4.97e-5 |
| (−3.5, −2) | both | 0.111 | 4.02e-5 | 4.55e-5 | 6.24e-6 | 5.07e-5 |

"both" is one source with direction (1, 1)/√2, in phase, with the same total force.

* **Force direction.** The cloak works for every direction tested. At (−3.5, −2) the vertical
  and horizontal problems are genuinely different (bare-hole error 0.188 vs 0.129), and C vs A
  stays about 2400 times below B in both.
* **Superposition.** The `both` solve equals (u_V + u_H)/√2 to 1.4e-14–9.0e-14 at (−3, −3) and
  1.5e-14–5.7e-14 at (−3.5, −2) (complex error), confirming linearity to round-off. So `both`
  carries no new field information beyond the two separate solves.
* **Mirror check.** At (−3, −3) the source lies on y = x, and cavity, cloak and box are symmetric
  about it, so in the continuum the horizontal field is the reflected vertical field. They agree
  to 2.6e-4 (complex; the mesh is not symmetric), so **vertical and horizontal are not independent
  tests at that position**.
* **`both` scatters less at (−3, −3).** There the force points straight at the cloak. This is
  expected to send mainly P waves (longer wavelength) towards it, though the P/S split at
  the cloak was not measured.

### 5.6 Frequency sweep

`docs/figures/circular_frequency_sweep/`: `frequency_sweep_ratio.png`, `frequency_sweep_error.png`,
`summary.md`. Cases A, B, C for the three forces at both source positions, 25 frequencies,
f* = b f / c_S from 0.1 to 2.5 (20 to 500 Hz), ε/a = 0.01 fixed. h_bg = min(0.094 m, λ_S/10): the base
mesh up to f* = 1.6, then 10 elements/λ_S. Vertical and horizontal share one factorisation;
`both` is their superposition. At f* = 1 the sweep reproduces the base-case values exactly.

Ratio metric: ⟨‖u‖⟩/⟨‖u_ref‖⟩ over exterior points beyond the cloak, (x − c)·ŝ > b, with ŝ from
source to centre. A is 1 by definition.

| quantity over the whole sweep | value |
|---|---|
| C ratio, largest deviation from 1 | 1.5e-5 |
| B ratio, range | 0.796 to 1.062 |
| C vs A, L2 of ‖u‖, range across the six cases | about 2e-6 (f* = 0.1) to 3.2e-4 (f* = 2.5) |
| B vs A, L2 of ‖u‖, at f* ≥ 0.4 | about 0.07 to 0.20 |
| A vs analytic, L2 of ‖u‖, largest | 2.3e-4 |
| A vs analytic, complex, largest | 2.7e-3 |

* The cloak holds across the sweep. C vs A grows with frequency, since ε is fixed while the wavelength
  shrinks. Above about f* = 1 it is similar in size to A vs analytic, so the sweep does not separate that
  growth from mesh error; finer meshes would be needed.
* **The ratio is a weak metric here.** B stays within about 20 % of 1, and even exceeds 1 at low frequency
  for the `both` force (1.06 at f* = 0.4), although B vs A shows real scattering there. A mean of ‖u‖ can
  hide scattering that moves energy around. Use the L2 figure as the main evidence.
* At (−3, −3), vertical and horizontal coincide because of the mirror symmetry.
* The x axis differs from the Rayleigh-wave plots in the paper's convention (f* = c_R/λ*): with
  c_R ≈ 0.889 c_S, the value f* = 2.5 here corresponds to about 2.8 there. Going to 4 here needs the large
  server: at 10 elements/λ_S it is roughly 430 000 elements.

### 5.7 Not covered by Step 1

Two source positions only, and frequencies only up to f* = 2.5. All element sizes were scaled together, so
refinement at r = a and quadrature order were not varied on their own. Case B has no independent
analytic check (only A does). The PML has no dedicated unit test; it is validated by case A
against the analytic solution.

## 6. Reproduce

```bash
# environment (this workstation: no conda / PETSc)
python3 -m venv .venv
.venv/bin/pip install "jax[cpu]==0.8.0" "jax-fem==0.0.11" "fenics-basix>=0.10.0" \
    "gmsh==4.14.0" "meshio==5.3.5" "pydantic==2.13.2" PyYAML numpy scipy matplotlib pypardiso pyfiglet pytest

.venv/bin/python -m pytest tests/test_circular_transform.py tests/test_ceff_nassar_eq25.py -q
.venv/bin/python scripts/circular_step1.py --config configs/circular_embedded/step1.yaml --study all
# or one at a time: --study base | eps | mesh | absorbing | map | summary
.venv/bin/python scripts/circular_step1.py --config configs/circular_embedded/step1_offdiag.yaml --study base
```

One force direction only: add `--polarization vertical | horizontal | both`
(the list is `polarizations` in the YAML).

Outputs: `output/circular_step1/<direction>/{base,eps/*,mesh/*,absorbing/*,map/*}/` (`case_{A,B,C,D}.npz`
with mesh + complex field, `metrics.json`, `config.json`), `<direction>/figures/`, `<direction>/summary.md`, and the
comparison `output/circular_step1/summary.md`.
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

