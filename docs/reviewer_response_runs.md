# Wave Motion revision: runs that would close the open reviewer items

Companion to `docs/reviewer_response_draft.md`. Nothing here has been run.

The draft response can be sent **without** any of these runs, provided:

- the limitations are worded as in the draft, and
- the provenance fix in §0.1 of the draft is applied. That fix needs no run.

The runs are ordered by value for effort.

**Reproducibility rule** (see CLAUDE.md, 2026-10-22):

- Every run gets a committed config and goes through the project runner.
- A new capability, such as field export or complex-surface output, goes behind a new option whose default keeps the current behaviour. Existing results must reproduce unchanged.

**Machine note:**

- The Sec. 4 outputs, i.e. the trained weights for the 14×10 single-frequency and broadband designs, are on the m3l machine (configs point to `/home/m3l/workspace/...`). Runs that re-evaluate those designs (R2, R3a) must run there or copy the weights over first.
- The 20×15 / microstructure items (R1, R3b, R4) can run on falcon. The cuDSS pixel validation takes about 90 s per solve at 1 element/pixel on the A6000.

| # | What | Answers | Cost (estimate) | Needs |
|---|---|---|---|---|
| R4 | Fully resolved field plot + uncloaked η for the 20×15 setup | R2.3 (explicitly requested), R1.1 | ~3 forward solves, < 15 min; small code change | falcon |
| R1 | Fully resolved validation of the paper's 0.967 cell set | §0.1 provenance; R1.1, R2.3 | 2–3 pixel solves (~2 min each) + 100-cell hi-fi re-homogenisation (~10 min on 8 cores) | falcon |
| R3 | Macro mesh convergence of the *final* designs | R1.3 | ~3 meshes × 2–3 designs, forward only | falcon + m3l weights |
| R6 | Broadband training with 3 and 5 frequencies | R2.2 | 2 optimisations, ≈ 3/7 and 5/7 of the 7-frequency run | either machine |
| R2 | Time-domain pulse / waveform fidelity | R2.1, R1.2 | ~80 frequencies × 4 configurations, forward only; new post-processing script | m3l weights |
| R5 | Initialisation and seed sensitivity, 14×10 | R1.1(b) | 6 optimisations × 1000 iterations | either machine |

---

## R4: displacement field of the fully resolved cloak

**Why.** Reviewer 2 "strongly suggests" this plot. No such image exists: `design-validation/run_validation.py` writes only `cloak_structure_triangle.png` and `validation_result.csv`.

**What:**

1. Add an option to `design-validation/run_validation.py`, e.g. `save_fields: false` by default, that writes:
   - the complex displacement of the reference solve and the cloak solve at nodes, as npz;
   - the surface displacement on the fixed evaluation points.
2. Rerun `configs/_cloak_cudss.yaml`, which uses the 2m cells at refinement 44 and gives 0.7550, with `save_fields: true`.
   - Check that it reproduces u_ratio = 0.754965 bit-for-bit. Determinism was confirmed earlier.
3. Produce the same fields for the **homogenised** cloak built from the same cells' `pred_flat4` / `pred_rho`, i.e. the `val_diffusion_2m` setup, on a comparable mesh.
4. Also produce the **uncloaked notch** η for this exact 20×15 geometry at f\* = 2. This value is currently missing; the draft uses ≈0.34 read off another figure.
5. Make the figure:
   - |Re u| or |u| for reference / homogenised / fully resolved, on a shared colour scale, zoomed on the cloak and the downstream strip;
   - plus a line plot of |u|/|u_ref| along the free surface.

   Point out where the loss occurs: in the cloak near the surface, at cell interfaces, or downstream.

**Optional.** Add a fourth panel for the pixel-optimised design (`canvas_best_20x15.npy`, η = 0.866). Comparing it with the as-generated design would show *where* the pixel changes recover transmission, which speaks directly to the attribution question. The canvases already exist (`output_flip_*/canvas_published.npy`, `output_levelset_20x15/canvas_best_20x15.npy`), so only the field export is new.

**Watch for.** max|u|/p95 is about 7 on these meshes. Hot spots in the field image may be numerical, so check them against the refinement-55 run before interpreting them.

## R1: validate the paper's 0.967 cells fully resolved

**Why.** The paper's 0.967 cells (`output/cell20_phys_flat4_materialreg4/cell_designs_diffusion`, mirrored as `design-validation/data/cell_designs_diffusion_TRI3`) have never been run through the corrected fully resolved pipeline (cement ligaments, traction-free voids, cuDSS). The 0.76 belongs to a different cell set.

Only needed if the authors want to keep 0.967 as the headline instead of switching to the 0.934 / 0.76 pair.

**What:**

1. A new config, e.g. `design-validation/configs/_cloak_tri3cells_rf44.yaml`, copied from `_cloak_cudss.yaml` with `cell_designs: data/cell_designs_diffusion_TRI3`. Run it at refinement 22, 44 and 55.
2. Re-homogenise the same 100 canvases with the hi-fi static TRI6 homogeniser (`compute_stiffness_hifi`, N = 100). Then rerun the matched-homogenised sweep at f\* = 2.
   - Their current 0.967 used the legacy TRI3 homogeniser at 1 element/pixel, which over-predicts thin-cell stiffness by up to more than 130%.

**Outcome:**

- This either gives a like-for-like pair for the 0.967 cells,
- or shows that 0.967 does not survive accurate homogenisation, in which case switch to 0.934.

## R3: macro mesh convergence of the final designs

**Why.** The existing study (`docs/mesh_refinement_selection.md` §3) was done on a 10×8 optimum, and its raw outputs are not on falcon. Reviewer 1 asks for a check of the results reported in the paper.

**What.** Forward-only evaluation of η at f\* = 2 on 3 refinement levels, for example cloak refinement 2/3/4 with the far field fixed, plus one finer far field. Designs:

- (a) 14×10 single-frequency;
- (a′) broadband 14×10 at f\* = 1, 2, 3;
- (b) `cell20_phys` targets, which give 0.985;
- (b′) the 2m realised cells, which give 0.934.

Report a table of nodes vs η. The finest mesh must stay positive-definite and pass the max|u|/p95 check.

## R6: number of training frequencies (3 and 5 vs 7)

**Why.** Reviewer 2 asks for exactly this. No comparable run exists: older 5- and 9-frequency runs use a different parameterisation and loss.

**What:**

- Two new configs copied from the variant the paper actually used (`A_multifreq_14x10_nogmm.yaml` or `_gmm`; confirm which), changing only `loss.multi_freq.f_stars` and `weights`:
  - |S| = 3: f\* = [1, 2, 3]
  - |S| = 5: f\* = [1, 1.5, 2, 2.5, 3]
- Keep everything else identical, including the seed, 3000 iterations and the learning rate.
- Evaluate each on the same dense sweep as Fig. 9 (f\* = 0–4, step 0.1).
- Report:
  - η at the training frequencies;
  - the in-band mean, minimum, and mean |1 − η|;
  - the minimum *between* training frequencies.

**Cost.** Per-step cost scales linearly with |S|, so about 3/7 and 5/7 of the original run's wall time. That time is not recorded on falcon; check the m3l logs.

## R2: time-domain pulse response / waveform fidelity

**Why.** Reviewer 2 asks for post-hoc time-domain or waveform metrics. It also supports R1.2. The objective is phase-blind, and the broadband design's phase has never been measured.

**What:**

1. Add an option to the frequency-sweep script to save the **complex** surface displacement (both components) at fixed receiver points downstream of the cloak, e.g. at the surface points used by η plus 2–3 single receivers.
2. Run a sweep f\* ∈ (0, 4], step 0.05 (80 frequencies), for:
   - the reference half-space;
   - the uncloaked notch;
   - the ideal polar cloak;
   - the broadband 14×10 design;
   - optionally the single-frequency 14×10 design, as the contrast case.

   The source must be identical for all of them, as it is now.
3. Post-processing script (new, committed): multiply by a Ricker spectrum centred at f\* = 2, chosen so most of its energy lies in [1, 3], and inverse-FFT to get the time traces at the receivers.
4. Metrics, each relative to the reference trace:
   - peak normalised cross-correlation **after** optimal time shift (waveform shape fidelity);
   - the lag at that peak (arrival delay);
   - correlation **without** shift (overall fidelity, including delay);
   - transmitted energy ratio ∫u²dt / ∫u_ref²dt.
5. Also report the unwrapped phase difference φ(f) − φ_ref(f) across the band.
   - If it is close to linear, the design behaves as a pure delay with little distortion.
   - The 0.18λ claim can be re-checked from the same data.

**Caveat.** The frequency step must resolve the travel time across the domain, W = 12.5 b. Check for time-domain aliasing by halving the step once.

## R5: initialisation and seed sensitivity (14×10, single frequency)

**Why.** Paper line 459 claims insensitivity to initialisation, but no runs supporting it were found. The 1×1 exploration shows multiple basins. Skip this if the authors retrieve the original comparison from the m3l machine, or soften the sentence instead.

**What:**

- 2 initialisations: `dataset_centroid`, and `pushforward` with `symmetrize_init: true`.
- 3 network seeds: e.g. 42, 7, 123.
- 6 runs of `configs/A_single_frequency/cells_14x10.yaml` with only those keys changed, 1000 iterations each.
- Report the mean, min and max of η at f\* = 2 and of the final loss.
- Optionally, show the per-cell material spread across runs, which indicates whether the optimum is unique or degenerate.

---

## Items that need no run but need someone with access to the m3l machine

- Loss histories for `output/A_single_frequency/cells_14x10`, the broadband 14×10 run and the 1×1 / 2×1 / 2×2 runs (R1.1a).
- Which broadband variant is the paper's: `nogmm`, `gmm`, `nogmm_restart` or `C_multifreq_14x10_gmm`.
- The dense-sweep CSV behind Fig. 9, for exact numbers in the R1.2 and R2.2 replies.
- The script and method behind the 0.18λ / 65° phase-lag figure (`single_freq_14x10_zoom_phase.png`).
- Whether the 1×1 0.857 came from a κ = 0.99 / R = 30 "ceiling" run (draft §0.2).
- The configs for the 2×1 and 2×2 runs (`A_partition_pushed`); they are not on falcon.
