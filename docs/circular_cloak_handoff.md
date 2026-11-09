# Circular cloak: project handoff

Status as of 2026-11-09, branch `circular_cloak`. Written so that a person or another agent can
continue the work, or draft a paper, from this file alone. Detailed derivations and all result
tables are in [`circular_cloak_step1.md`](circular_cloak_step1.md).

## 1. Status

| stage | state |
|---|---|
| Step 1: continuous regularised ideal cloak | done locally, awaiting final approval |
| Finest mesh level (scale 0.6, 225k elements) | not run: exceeds the memory guard on the 31 GB workstation |
| Step 2: pixelated cloak and optimisation | baselines and optimisation done locally; report `docs/circular_cloak_step2.md`, paper Section 7; next runs on the H100 |
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
  0.07 to 0.20 at f* ≥ 0.4. Figures: `docs/figures/circular_frequency_sweep/`. The sweep saves its fields in `output/`, so new plots need no new solve (`--plots-only`).
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
| `scripts/circular_ratio_diagnostic.py` | decomposition behind the ratio exceeding 1 |
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

1. Finer meshes for the ideal cloak are **not** needed (decided 2026-11-06). The flattening of C vs D near 1e-5 stays an
   open note, not a task.
2. Optional before Step 2: other frequencies; refinement at r = a and quadrature on their own.
3. After approval, Step 2 as planned in section 7 of `circular_cloak_step1.md`: about 500 **active** pixels (not a
   500 × 500 grid); sample the ideal tensor first to measure pixelisation error alone; then neural-field
   optimisation with admissible Cauchy stiffness and positive density; keep material-pixel resolution separate
   from FEM resolution; keep the cavity and r = b geometrically exact; reduce the active pixel count and report
   actual counts.

For Step 2, include both force directions in the loss, because a design tuned to one force may not hold for another.
One factorisation can serve both right-hand sides.

## 12. Step 2 status

Code: `embedded/pixels.py` (polar pixels, independent of the mesh), `embedded/adjoint.py` (one factorisation; the adjoint uses
K^T = S K S, and `LinearBackend` is where a GPU solver such as cuDSS would go), `embedded/neural_field.py` (admissible Cauchy
material by Cholesky factor; reuses the MLP and Adam of the old pipeline), `embedded/optimize_pixels.py`, `scripts/circular_pixel_baseline.py`,
`scripts/circular_optimize.py` (checkpoints and resumes), `configs/circular_embedded/step2.yaml`, `tests/test_circular_adjoint.py`.
Results: `docs/figures/circular_step2/`.

Gradient: matches central finite differences within 1e-5 relative (complex and magnitude losses) and JAX-FEM reverse mode.

Baselines at f* = 1, eps/a = 0.01, worst ‖u‖ error over the six source and force cases (bare hole 0.188, continuous cloak 8e-5):

| material, 500 pixels | linear | quadratic |
|---|---|---|
| ideal, centre sampling | 4.2e-3 | 1.6e-2 |
| ideal, cell average | 1.3e-2 | 3.2e-2 |
| symmetrised, cell average | 0.222 | 0.202 |
| symmetrised continuous (no pixels) | 0.222 | 0.200 |

Symmetrising costs far more than pixels, and leaves the cloak no better than the bare hole at this frequency.

Pilot: 500 pixels, quadratic map, f* = 1, trained on source (-3.5, -2) with both forces, complex loss, 150 Adam steps at about 7 s each, CPU.

| source | vertical | horizontal | both |
|---|---|---|---|
| (-3.5, -2), trained | 5.9e-3 | 6.1e-3 | 5.6e-3 |
| (-3, -3), not trained | 9.6e-2 | 0.119 | 4.7e-2 |

The trained source improves about 30 times from the symmetrised start (0.202), and was still improving at step 150. **The unseen source improves much less (0.159 to about 0.10): the design is tuned to its training source.** Held-out exterior points do not reveal this,
as they come from the same field. So future training needs several sources in the loss, and validation must use an unseen source.

For the H100: PARDISO runs on the CPU, so the GPU only helps JAX work until a GPU solver is added behind `LinearBackend`; multi-case training is
sequential for now.

### Multi-source and broadband runs (2026-11-06 to 2026-11-08)

One mesh is refined around every source, so one factorisation per frequency serves all sources and forces
(`FrequencyCase`). Sources are split into train (8, ring r = 3.5 m), validation (4, selects the best design)
and test (4, including the two Step 1 sources, never used for any decision). `BandCase` switches frequency on one
problem by rescaling density and recomputing the PML, so a pool of frequencies costs one factorisation per drawn
frequency. All gradient tests pass (`tests/test_circular_adjoint.py`, 8 tests).

Full table: `docs/figures/circular_step2/optimisation_results.md`; histories in `docs/figures/circular_step2/runs/`.

- f* = 1, 500 pixels: test median 0.042 against the bare hole's 0.158 at f* = 1; quadratic beats linear (0.049).
- Pixel count at f* = 1: 0.042 (500), 0.057 (252), 0.065 (120), 0.099 (60).
- Single-frequency designs are worse than the bare hole at f* = 1.5 and 2.
- Three fixed training frequencies leave gaps between them; a pool of 11 frequencies in [0.5, 1.5], 2 per step,
  removes them: test median 0.073 to 0.119 across the band against 0.138 to 0.168 for the bare hole, but 0.174 at
  f* = 1.75 outside it.
- Every run except 60 pixels was still improving when stopped. The pool run takes about 40 s per step locally.
- The pool run was continued to 3000 steps: test medians 0.067–0.119 across [0.5, 1.5]; best validation 0.2053.

### Running on the H100 server

Changes for speed (2026-11-08):
- `embedded/fast_assembly.py`: the matrix outside the cloak is assembled once per frequency and cached; each step adds
  only the annulus element matrices. `tests/test_circular_fast_assembly.py` checks it against full jax-fem assembly.
- All adjoint right-hand sides are solved together, and the gradient is one batched JAX call.
- `CuDSSBackend` in `embedded/adjoint.py` (cuDSS via nvmath-python, the same calls as `rayleigh_cloak_3d/cudss_solver.py`):
  one plan, then values updated in place and refactorised each step. **It has not run anywhere yet** (no GPU here).
- Each line of `history.jsonl` has a `timing` entry per phase (assemble, factorize, solve, loss, adjoint, sensitivity).

Measured here on a 27k-element mesh while another run used the CPU: assembly fell from about 0.9 s to below 0.1 s
per frequency, but the step stayed about 5-7 s because factorisation and solves dominate. So the GPU solver is the main gain.

Setup and first run on the server:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install "jax[cuda12]==0.8.0" "jax-fem==0.0.11" "fenics-basix>=0.10.0" "gmsh==4.14.0" "meshio==5.3.5" \
    "pydantic==2.13.2" PyYAML numpy scipy matplotlib pypardiso pyfiglet pytest cupy-cuda12x "nvmath-python[cu12]"
python -m pytest tests/test_circular_transform.py tests/test_circular_fast_assembly.py tests/test_circular_adjoint.py -q
python scripts/benchmark_step.py --backend cudss --compare --scale 1.0     # must print "[compare] OK"
```

Only train with `backend: cudss` (in `configs/circular_embedded/step2.yaml`) after `--compare` reports OK: it checks the
loss and both gradients against PARDISO, which also re-checks the sign-flip adjoint on the GPU. If it fails, train with
`backend: pardiso` on the server's CPUs; the fast assembly still helps there. To continue a local run there, copy its
folder `output/circular_step2/optimize/<run>/` and run the same command; it resumes from `checkpoint.pkl`.

## 11. Paper material

The draft is `docs/circular-cloak-paper/paper.tex` (Wave Motion template, `./build.sh`; tables come from `make_tables.py`). Authors, CRediT and funding are still to be confirmed.

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
