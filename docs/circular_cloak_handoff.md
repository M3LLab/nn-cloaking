# Circular cloak: project handoff

Status as of 2026-11-06, branch `circular_cloak`. Written so that a person or another agent can
continue the work, or draft a paper, from this file alone. Detailed derivations and all result
tables are in [`circular_cloak_step1.md`](circular_cloak_step1.md).

## 1. Status

| stage | state |
|---|---|
| Step 1: continuous regularised ideal cloak | done locally, awaiting final approval |
| Finest mesh level (scale 0.6, 225k elements) | not run: exceeds the memory guard on the 31 GB workstation |
| Step 2: pixelated cloak and optimisation | **plan only; do not start before the Step 1 review is approved** |
| Microstructure realisation | later stage, out of scope |

## 2. Goal and problem

Adapt `nn-cloaking` from a carpet cloak for Rayleigh waves to a **fully embedded circular cloak for
bulk P and S waves**, based on Chatzopoulos et al., IJES 191 (2023), `docs/symmetrized_elastic_tensors.pdf`.

- 2-D time-harmonic plane strain, time factor e^{+iωt}.
- Background: λ = μ = 144 MPa, ρ = 1600 kg/m³ (c_S = 300 m/s, c_P = 519.6 m/s).
- f = 200 Hz, so λ_S = 1.5 m and λ_P = 2.6 m.
- Annulus a < r < b with a = 0.5 m, b = 1.5 m. The cavity r < a is removed and traction-free; r = b is
  a bonded interface.
- Physical box [−5, 5]² m, PML on all four sides, clamped outer edge.
- Source: normalised Gaussian body force, F₀ = 1000 N/m, σ = 0.075 m, below-left of the cloak.

Comparison cases on common exterior sampling points:

| case | definition |
|---|---|
| A | homogeneous background, no hole |
| B | traction-free hole, no cloak |
| C | hole + continuous regularised BGM/Cosserat cloak |
| D | homogeneous background with a traction-free hole of radius ε (added as a diagnostic) |

The regularised cloak must reproduce D exactly outside r = b, so C vs D isolates implementation and
mesh error, and D vs A is the error due to ε alone. **Finite-ε results are a regularised approximation
to the ideal limit, not a perfect cloak.**

## 3. Decisions and the reason for each

| decision | reason |
|---|---|
| Full tensor C'_ijkl = F_jJ F_lL C_iJkL / J, ρ' = ρ/J, no minor symmetry | BGM/Cosserat gauge U(X) = u(x), as requested. The paper's Eq. 5 is the same law with the derivative on indices 1 and 3 |
| Radial map from ε ≤ R ≤ b onto a ≤ r ≤ b; `linear` is base, `quadratic` is available | requested regularisation; paper appendix Eqs. 30–32 |
| Material assigned by mesh region tag at quadrature points | conforming circles; no element straddles r = a or r = b |
| **True PML** (s_j = 1 − iσ_j/ω; L = 2.6 m, p = 2, R = 1e-6) | requested. Replaced a graded damping layer, which stalled near 7e-4 (complex) against the analytic solution. Still available as `absorbing.kind: damping` |
| 16 elements per λ_S, TRI6, 6-point quadrature | case A against the analytic solution converges at about h^3.3 |
| **Total displacement magnitude ‖u‖ is the primary metric and plot** | requested. The complex error is kept as a secondary column because magnitude is blind to phase and direction. Component plots are in `components/` |
| Force directions vertical, horizontal, both, as separate solves | requested |
| Direct solve with MKL PARDISO instead of PETSc | no PETSc here; `jax_fem.solver` cannot be imported |
| Memory check before meshing and before every solve | requested after an out-of-memory kill. Refused points are logged as skipped |

## 4. Key results (ε/a = 0.01, ‖u‖ error against A)

| source | force | B: bare hole | C: cloak | D: ε-hole | C vs D |
|---|---|---|---|---|---|
| (−3, −3) | vertical | 0.158 | 6.68e-5 | 7.67e-5 | 1.18e-5 |
| (−3, −3) | horizontal | 0.158 | 6.67e-5 | 7.67e-5 | 1.19e-5 |
| (−3, −3) | both | 0.104 | 3.36e-5 | 3.77e-5 | 5.12e-6 |
| (−3.5, −2) | vertical | 0.188 | 7.95e-5 | 9.13e-5 | 1.40e-5 |
| (−3.5, −2) | horizontal | 0.129 | 5.36e-5 | 6.13e-5 | 8.99e-6 |
| (−3.5, −2) | both | 0.111 | 4.02e-5 | 4.55e-5 | 6.24e-6 |

Complex error for the vertical force at (−3, −3): B 0.312, C 1.30e-4.

- **ε convergence:** the cloak error falls as ε² from 3.1e-2 at ε/a = 0.2 to 1.6e-5 at ε/a = 0.005.
- **Layer:** the cloak error stays at 6.6–6.7e-5 across all absorbing layers, because A, B and C share them.
- **Radial map:** linear 6.68e-5, quadratic 6.38e-5.
- **Frequency sweep, f* = b f / c_S from 0.1 to 2.5:** C vs A (‖u‖) ranges from about 2e-6 to 3.2e-4, and B vs A reaches about
  0.07 to 0.20 at f* ≥ 0.4. Figures: `docs/figures/circular_frequency_sweep/`.
- **Case A vs analytic:** 5.0e-5 in ‖u‖ and 2.5e-4 complex at the base mesh.

## 5. Validation evidence

1. **Transformation audit:** `tests/test_circular_transform.py` and `tests/test_ceff_nassar_eq25.py`, 52 tests
   passing: Jacobian against autodiff, Cartesian against rotated polar tensor, major but not minor
   symmetry, strain-energy invariance, Nassar et al. Eq. 2.5.
2. **Case A against the closed-form 2-D Green's tensor:** confirms source normalisation, force sign, time
   convention and PML absorption.
3. **C against D:** outside the cloak and, after mapping, inside it.
4. **Superposition:** `both` equals (u_V + u_H)/√2 to about 1e-14 (complex).
5. **Mirror symmetry:** at (−3, −3) horizontal matches the reflected vertical to 2.6e-4 (complex).

## 6. Limits and open questions

- **Bandwidth tested only for f* = 0.1 to 2.5** (20 to 500 Hz, ε/a = 0.01). Above about f* = 1, C vs A is similar in size
  to the mesh error of A vs analytic, so its growth with frequency is not yet separated from mesh error.
- **The ⟨‖u‖⟩ ratio is weak:** it can stay near 1 despite real scattering. Use the L2 error as the main evidence.
- **Two source positions.** At (−3, −3) vertical and horizontal are mirror images, so only (−3.5, −2)
  tests force direction independently.
- **C vs D on fine meshes:** flattening near 1e-5 in ‖u‖ for the vertical force. It is not yet known
  whether this is a floor; scale 0.6 and finer should settle it.
- **Not varied separately:** refinement at r = a, and quadrature order.
- **No independent check of case B**, such as a cavity series solution.
- **No dedicated PML unit test;** it is validated through case A.
- **`rayleigh_cloak.solver` and the old optimisation workflows** have not been run on this branch and
  need PETSc. Step 2 needs a new adjoint solve or a PETSc install.
- **Not verified:** the explanation that `both` scatters less at (−3, −3) because it sends mainly P
  waves at the cloak. The P/S split was not measured.

## 7. Code map

| path | content |
|---|---|
| `rayleigh_cloak/geometry/circular.py` | `RadialMap`, `CircularCloakGeometry` with ε |
| `rayleigh_cloak/materials.py` | `C_pushforward`, `C_radial_polar`, `rotate_C` |
| `rayleigh_cloak/embedded/config.py` | YAML schema |
| `rayleigh_cloak/embedded/mesh.py` | gmsh mesh with region tags |
| `rayleigh_cloak/embedded/problem.py` | jax-fem weak form, PML, source |
| `rayleigh_cloak/embedded/linsolve.py` | PETSc-free direct solve |
| `rayleigh_cloak/embedded/sampling.py` | point evaluation, `rel_l2`, `rel_l2_mag` |
| `rayleigh_cloak/embedded/analytic.py` | closed-form Green's tensor |
| `rayleigh_cloak/embedded/memguard.py` | memory checks |
| `rayleigh_cloak/embedded/run.py`, `plots.py` | cases A–D and metrics; figures |
| `scripts/circular_step1.py` | driver for all studies |
| `configs/circular_embedded/step1.yaml`, `step1_offdiag.yaml` | source at (−3, −3); source at (−3.5, −2) |
| `scripts/circular_frequency_sweep.py`, `configs/circular_embedded/frequency_sweep.yaml` | frequency sweep; shares one factorisation per case |
| `docs/figures/circular_step1*/` | committed figures and summary tables |

`output/` is gitignored. Solved fields (`case_*.npz`) are not in the repository; the commands below
regenerate them.

## 8. Run it

```bash
python3 -m venv .venv
.venv/bin/pip install "jax[cpu]==0.8.0" "jax-fem==0.0.11" "fenics-basix>=0.10.0" \
    "gmsh==4.14.0" "meshio==5.3.5" "pydantic==2.13.2" PyYAML numpy scipy matplotlib pypardiso pyfiglet pytest

.venv/bin/python -m pytest tests/test_circular_transform.py tests/test_ceff_nassar_eq25.py -q
.venv/bin/python scripts/circular_step1.py --config configs/circular_embedded/step1.yaml --study all
.venv/bin/python scripts/circular_step1.py --config configs/circular_embedded/step1_offdiag.yaml --study base
.venv/bin/python scripts/circular_frequency_sweep.py --config configs/circular_embedded/frequency_sweep.yaml
```

- Options: `--study base|eps|mesh|absorbing|map|summary`, `--polarization vertical|horizontal|both`, `--force`.
- Solved points with an identical config are reused, so sweeps resume, and metrics and figures can be
  rebuilt without re-solving.
- Peak RAM is about 75 kB per TRI6 element. A full `--study all` for one direction takes roughly 15 minutes
  on 12 cores.

## 9. Environment notes

- This workstation has no conda or PETSc and 31 GB of RAM. The venv uses Python 3.13, while
  `environment.yml` targets another setup.
- A large server with an H100 is available for bigger jobs. `linsolve.py` runs on the CPU, so more RAM is
  what helps; using the GPU would need a GPU sparse solver.
- Git: this machine has no global identity. Commits so far use `dav-pil <davitpiliposyan@gmail.com>`
  through `git -c`. Do not push without being asked.
- Gmsh `Threshold` fields used as local refinement need `StopAtDistMax`, otherwise they impose their
  `SizeMax` across the whole domain. This bug was hit and fixed once.
- Wait loops must not use `pgrep -f` on a string in their own command line; they will never exit.

## 10. Next steps

1. On the large server, run the mesh study at scales 0.6, 0.5 and 0.35 (edit `studies.mesh.values`) to settle
   the C vs D question.
2. Optional before Step 2: other frequencies; refinement at r = a and quadrature on their own.
3. After approval, Step 2 as planned in section 7 of `circular_cloak_step1.md`: about 500 **active** pixels (not a
   500 × 500 grid); sample the ideal tensor first to measure pixelisation error alone; then neural-field
   optimisation with admissible Cauchy stiffness and positive density; keep material-pixel resolution separate
   from FEM resolution; keep the cavity and r = b geometrically exact; reduce the active pixel count and report
   actual counts.

For Step 2, include both force directions in the loss, because a design tuned to one force may not hold for another.
One factorisation can serve both right-hand sides.

## 11. Paper material

The results support these statements, for the tested conditions only:

- A regularised BGM/Cosserat circular cloak, fully embedded and excited by a point force radiating both P and S
  waves, reduces exterior scattering by about 2400–3100 times at ε/a = 0.01 relative to a bare cavity, in both ‖u‖ and
  complex error terms.
- The residual scales as ε² and matches a traction-free hole of radius ε in the background.
- The performance holds for all tested in-plane force directions.

Suggested outline: introduction; transformation and regularisation; numerical method (PML, TRI6, mesh); validation
(analytic case A, C vs D, superposition); cloaking results; sensitivity; pixelated and optimised cloaks once Step 2
exists.

Needed before submission: the finer mesh levels, frequency results separated from mesh error, an independent check of case B, and a
literature check. The references in the source paper were not independently reviewed for this work.
Equations and tables should be re-derived from the code and data, not copied from this summary alone.
