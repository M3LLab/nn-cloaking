# Circular cloak, Step 2: pixelated cloaks with admissible Cauchy materials

Branch `circular_cloak`, results as of 2026-11-09. Continues `docs/circular_cloak_step1.md` (the validated continuous cloak).
The same content, with proofs, is Section 7 of `docs/circular-cloak-paper/paper.pdf`. Figures and data:
`docs/figures/circular_step2/`; regenerate the figures with `.venv/bin/python scripts/circular_step2_figures.py`.

All errors are the normalised L2 error of the total displacement magnitude ‖u‖ against case A, `e`, unless marked `e_c`
(complex error), on the Step 1 exterior sampling grid. Radial map quadratic, ε/a = 0.01, base mesh, unless stated.

## 1. Setup

- **Pixels:** polar grid uniform in θ and r over a < r < b: 50×10 = 500, 36×7 = 252, 24×5 = 120, 15×4 = 60 pixels, all active.
  Pixels are independent of the FEM mesh; the cavity and r = b stay exact mesh edges.
- **Admissible material:** Cauchy (minor and major symmetry), Voigt matrix M positive definite, density positive.
- **Neural field:** 73 polar Fourier features of the pixel centre, MLP 3×128 tanh, 7 outputs, 43 399 parameters (MLP and Adam reused
  from `rayleigh_cloak/neural_reparam.py` and `optimize.py`). Output: M = μ L Lᵀ with L lower triangular, positive diagonal (exp), and
  ρ = ρ₀ exp(·). Admissible for every parameter value (paper, Proposition 6). Starts exactly at the symmetrised pixel cloak.
- **Gradient:** adjoint through the JAX-FEM weak form. K is complex symmetric, so Kᵀ = S K S and each adjoint reuses the forward
  factorisation (paper, Proposition 7). Checked: finite differences within 1e-5 relative (complex and magnitude losses), JAX-FEM reverse
  mode within 1e-8, transpose system residual 1e-8 (`tests/test_circular_adjoint.py`, 8 tests).
- **Sources:** train: 8 on the ring r = 3.5 m at 22.5° + 45°k; validation (selects the best design): (±2.5, 0), (0, ±4.5);
  test (never used for any decision): (−3, −3), (−3.5, −2), (3, 3), (2.2, −1.5). Both force directions. One mesh refined around all
  sources, so one factorisation per frequency serves every source, force and adjoint.
- **Loss:** mean over training right-hand sides of the squared relative complex error on a fixed random half of the exterior points
  (the other half held out).
- **Optimiser:** Adam, cosine decay 2e-3 → 2e-4, 600 steps (single frequency and three-frequency runs). The pool run starts from the
  three-frequency design: 1500 steps 1e-3 → 1e-4, then continued to 3000 at 1.3e-4 → 5e-5.
- **Cost on this workstation (12-core CPU, PARDISO):** about 7 s/step with one source, 18–20 s with 8 sources at one frequency,
  51 s at three frequencies, 33–40 s per pool step (two frequencies).

## 2. Pixels and symmetry without optimisation

Worst e over the two Step 1 sources × three forces, f* = 1 (`docs/figures/circular_step2/baseline.json`):

| material | pixels | linear map | quadratic map |
|---|---|---|---|
| hole, no cloak | – | 0.188 | 0.188 |
| ideal, continuous | – | 7.9e-05 | 7.6e-05 |
| symmetrised, continuous | – | 0.222 | 0.200 |
| ideal pixels, centre | 500 | 4.18e-03 | 1.57e-02 |
| ideal pixels, average | 500 | 1.34e-02 | 3.21e-02 |
| symmetrised pixels, average | 500 | 2.22e-01 | 2.02e-01 |
| ideal pixels, centre | 252 | 9.31e-03 | 3.37e-02 |
| ideal pixels, average | 252 | 2.28e-02 | 4.79e-02 |
| symmetrised pixels, average | 252 | 2.23e-01 | 2.05e-01 |
| ideal pixels, centre | 120 | 2.01e-02 | 6.40e-02 |
| ideal pixels, average | 120 | 3.83e-02 | 6.65e-02 |
| symmetrised pixels, average | 120 | 2.25e-01 | 2.08e-01 |
| ideal pixels, centre | 60 | 3.13e-02 | 1.03e-01 |
| ideal pixels, average | 60 | 6.60e-02 | 8.81e-02 |
| symmetrised pixels, average | 60 | 2.30e-01 | 2.10e-01 |

**Pixels cost little; symmetry costs everything.** Ideal (Cosserat) pixels reach 4e-3 to 3e-2 at 500 pixels, but the arithmetic-mean
symmetrisation is as bad as or worse than the bare hole, with or without pixels.

## 3. One training source overfits

Pilot: 500 pixels, f* = 1, trained on one source only, 150 steps (`pilot_evaluation.json`):

| source | force | e | e_c |
|---|---|---|---|
| (-3, -3), not trained | vertical | 9.61e-02 | 1.92e-01 |
| (-3, -3), not trained | horizontal | 1.19e-01 | 2.38e-01 |
| (-3, -3), not trained | both | 4.69e-02 | 1.16e-01 |
| (-3.5, -2), trained | vertical | 5.91e-03 | 1.16e-02 |
| (-3.5, -2), trained | horizontal | 6.14e-03 | 1.13e-02 |
| (-3.5, -2), trained | both | 5.62e-03 | 1.19e-02 |

Held-out points of the trained source did not show this. All later runs use 8 training sources and report test sources.

## 4. Optimised designs

At f* = 1 (`runs/*/evaluation.json`). The bare hole gives median 0.158, worst 0.263 on the test sources.

| run | training frequencies | pixels | map | steps | train median | **test median** | test worst | Step 1 sources worst |
|---|---|---|---|---|---|---|---|---|
| `f1_500` | f* = 1 | 500 | quadratic | 600 | 0.026 | **0.042** | 0.053 | 0.044 |
| `f1_500_linear` | f* = 1 | 500 | linear | 600 | 0.031 | **0.049** | 0.075 | 0.062 |
| `f1_252` | f* = 1 | 252 | quadratic | 600 | 0.033 | **0.057** | 0.072 | 0.061 |
| `f1_120` | f* = 1 | 120 | quadratic | 600 | 0.037 | **0.065** | 0.083 | 0.072 |
| `f1_60` | f* = 1 | 60 | quadratic | 600 | 0.090 | **0.099** | 0.131 | 0.102 |
| `band_500` | f* = 0.5, 1, 1.5 | 500 | quadratic | 600 | 0.090 | **0.086** | 0.170 | 0.097 |
| `pool_500` | pool of 11 in [0.5, 1.5] | 500 | quadratic | 3000 | 0.077 | **0.069** | 0.140 | 0.105 |

- 500 pixels at f* = 1: test median 0.042, worst 0.053 — 3.8× and 5.0× below the hole.
- Pixel count: 0.042 (500), 0.057 (252), 0.065 (120), 0.099 (60).
- Quadratic beats linear after optimisation (0.042 vs 0.049), the reverse of the unoptimised ranking.
- Against the baselines (`fig_pixels.png`): optimisation recovers most of the symmetrisation loss (0.202 → 0.044 worst at the Step 1
  sources, 500 pixels) but stays above the non-admissible ideal pixels (0.016 centre, 0.032 average). At 60 pixels they match.
- Not converged: every run except 60 pixels was still improving (`fig_training.png`).

## 5. Frequency

Median / worst e on the test sources (`fig_frequency.png`):

| f* | hole, no cloak | single frequency (`f1_500`) | three frequencies (`band_500`) | frequency pool (`pool_500`) |
|---|---|---|---|---|
| 0.5 | 0.138 / 0.222 | 0.106 / 0.189 | 0.074 / 0.120 | 0.071 / 0.115 |
| 0.6 | – | – | – | 0.079 / 0.124 |
| 0.75 | 0.155 / 0.278 | – | 0.126 / 0.239 | 0.075 / 0.132 |
| 0.9 | – | – | – | 0.067 / 0.103 |
| 1 | 0.158 / 0.263 | 0.042 / 0.053 | 0.086 / 0.170 | 0.069 / 0.140 |
| 1.1 | – | – | – | 0.077 / 0.178 |
| 1.25 | 0.165 / 0.266 | – | 0.141 / 0.236 | 0.092 / 0.179 |
| 1.4 | – | – | – | 0.109 / 0.156 |
| 1.5 | 0.168 / 0.253 | 0.214 / 0.353 | 0.106 / 0.211 | 0.119 / 0.175 |
| 1.75 | – | – | – | 0.180 / 0.307 |
| 2 | 0.168 / 0.255 | 0.235 / 0.380 | 0.221 / 0.343 | – |

- The single-frequency design is worse than the hole at f* = 1.5 and 2.
- Three fixed training frequencies leave peaks between them (0.126 at 0.75, 0.141 at 1.25).
- The frequency pool removes them: median 0.067–0.119 and worst 0.103–0.179 across [0.5, 1.5], against 0.138–0.168 and 0.222–0.278 for
  the hole, a median ratio of 1.4–2.3. Cost: 0.069 instead of 0.042 at f* = 1. Continuing the pool run from 1500 to 3000 steps improved
  almost every test frequency by about 3–12 % (median), with best validation 0.2148 → 0.2053.
- Outside the band: 0.180 at f* = 1.75 (hole not computed there); f* = 2 not evaluated for the pool design (memory limit).

## 6. Designs and fields

- `fig_materials.png`: polar components of the symmetrised start and two optimised designs. The designs keep the large tangential
  stiffness near the cavity, add strong coupling terms, and vary with angle although the problem is radially symmetric (nothing imposes
  that symmetry).
- Contrast: Voigt eigenvalues 0.13 μ to 311 μ (single frequency) and 0.04 μ to 786 μ (pool); density 0.70–2.1 ρ₀ and 0.21–3.2 ρ₀.
- `fig_fields.png`: at the unseen source (−3.5, −2), both designs suppress most of the hole's shadow at f* = 1; at f* = 1.5 only the
  pool design does.

## 7. Limitations

- Not converged; networks, learning rates and source layout not tuned.
- One geometry, one mesh, ε/a = 0.01; four test sources at radii 2.7–4.2 m only.
- Pixel materials are unconstrained in contrast and anisotropy; their realisation by microstructures has not been studied.
- A gap of more than two orders of magnitude remains to the continuous (Cosserat) cloak.
- The H100 path (cuDSS) is written but untested; see the handoff document.

## 8. Reproduce

```bash
.venv/bin/python scripts/circular_pixel_baseline.py --config configs/circular_embedded/step2.yaml
.venv/bin/python scripts/circular_hole_reference.py
.venv/bin/python scripts/circular_optimize.py --run f1_500        # also f1_500_linear, f1_252, f1_120, f1_60, band_500, pool_500
.venv/bin/python scripts/circular_step2_figures.py
.venv/bin/python docs/circular-cloak-paper/make_tables.py && docs/circular-cloak-paper/build.sh
```
