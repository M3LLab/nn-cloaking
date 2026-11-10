# Circular cloak: start here

Branch `circular_cloak` is a copy of `main` plus the embedded circular cloak
work. Most of the repository (triangular/carpet cloak, 3D cloak, datasets,
diffusion, surrogate models) comes from `main` and is **not** used here. This
page lists exactly which files belong to the circular cloak.

Rule of thumb: a path belongs to this work if it contains `embedded/`,
`circular_embedded`, `circular_cloak`, `circular-cloak-paper`, `circular_step`,
`circular_frequency_sweep`, or is one of the `scripts/circular_*.py` /
`tests/test_circular_*.py` files listed below.

## Read in this order

1. [`circular_cloak_handoff.md`](circular_cloak_handoff.md): one-page status,
   problem, decisions, key results, limits, Step 2 status, how to run.
2. [`circular-cloak-paper/paper.pdf`](circular-cloak-paper/paper.pdf): full
   write-up (problem statement, proofs, figures, tables, Step 1 and Step 2).
3. Detail, if needed:
   [`circular_cloak_step1.md`](circular_cloak_step1.md) (ideal cloak),
   [`circular_cloak_step2.md`](circular_cloak_step2.md) (pixels, neural field).

## Files of this work

### Code: `rayleigh_cloak/embedded/` (all new)

| File | Role |
|---|---|
| `config.py` | Configuration (material, cloak, source, domain, PML, mesh, metrics) |
| `mesh.py` | gmsh mesh with region tags (cavity, annulus, box, absorbing layer) |
| `problem.py` | Time-harmonic elastodynamics, BGM cloak tensors, PML, Gaussian source |
| `linsolve.py` | Direct sparse solve (PARDISO), several force directions at once |
| `sampling.py` | Point evaluation of FE fields, error metrics |
| `analytic.py` | Closed-form Green tensor (check of case A) |
| `run.py` | Cases A (homogeneous), B (hole), C (cloak), D (virtual hole) |
| `plots.py` | Step 1 figures |
| `memguard.py` | Memory checks before meshing and solving |
| `pixels.py` | Step 2 polar pixel grid, sampling of the ideal cloak |
| `neural_field.py` | Neural field producing admissible Cauchy tensors and density |
| `adjoint.py` | Adjoint gradients, PARDISO / cuDSS backends |
| `fast_assembly.py` | Fast reassembly of only the annulus elements |
| `optimize_pixels.py` | Loss, gradient and validation for one frequency or a band |

### Shared code from `main`, changed on this branch

| File | Change |
|---|---|
| `rayleigh_cloak/materials.py` | Added `C_pushforward`, `C_radial_polar`, `rotate_C` |
| `rayleigh_cloak/geometry/circular.py` | Added `RadialMap` and the regularised circular geometry |

Three small helpers are imported from old code and must stay:
`rayleigh_cloak/optimize.py` (Adam optimiser),
`rayleigh_cloak/neural_reparam.py` (MLP init/forward),
`rayleigh_cloak_3d/cudss_solver.py` (cuDSS library path, H100 only).

### Scripts

| Script | Purpose |
|---|---|
| `scripts/circular_step1.py` | Step 1: cases A–D, ε/mesh/PML/map studies, figures |
| `scripts/circular_frequency_sweep.py` | Frequency sweep, fields and plots |
| `scripts/circular_ratio_diagnostic.py` | Why ‖u‖/‖u_A‖ > 1 for the hole |
| `scripts/circular_hole_reference.py` | Hole reference for Step 2 |
| `scripts/circular_pixel_baseline.py` | Step 2 baseline: ideal cloak sampled into pixels |
| `scripts/circular_optimize.py` | Step 2 neural-field optimisation (`--run NAME`) |
| `scripts/circular_step2_figures.py` | Step 2 figures |
| `scripts/benchmark_step.py` | Timing of one optimisation step; cuDSS check on the H100 |

### Configs: `configs/circular_embedded/`

`step1.yaml`, `step1_offdiag.yaml`, `frequency_sweep.yaml`, `step2.yaml`.

### Tests

`tests/test_circular_transform.py`, `tests/test_circular_adjoint.py`,
`tests/test_circular_fast_assembly.py`. Run with
`pytest tests/test_circular_*.py`.

### Documents and figures

- `docs/circular_cloak_handoff.md`, `docs/circular_cloak_step1.md`,
  `docs/circular_cloak_step2.md`
- `docs/circular-cloak-paper/` (LaTeX source, tables, figures, PDF)
- `docs/figures/circular_step1/`, `docs/figures/circular_step1_offdiag/`,
  `docs/figures/circular_frequency_sweep/`, `docs/figures/circular_step2/`

## Not part of this work (from `main`)

- `configs/circular_optimize.yaml`, `configs/circular_optimize_neural.yaml`:
  an **older** circular test (Nassar 2018 geometry, 40 kHz, cell-based).
  Same word "circular", unrelated to this branch's work.
- `run.py`, `run_optimize.py`, `run_optimize_3d.py` and all other top-level
  `configs/*.yaml`: triangular/carpet cloak and 3D cloak.
- `rayleigh_cloak/` outside `embedded/` (except the files listed above),
  `rayleigh_cloak_3d/`, `dataset/`, `dataset_gen/`, `latent_ae/`,
  `surrogate/`, `microstructure_generation_2d/`, `multiscale_generation/`,
  `gauge_opt/`.
- `docs/wavemotion-paper/` (triangular-cloak paper), `docs/metamat2026/`,
  `docs/dataset/`, `docs/multifreq/` and the other docs not listed above.
