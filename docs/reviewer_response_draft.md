# Wave Motion: draft response to reviewers

**Paper:** *Neural-field design of broadband Rayleigh-wave carpet cloaks under microstructure realisability constraints*
**Submitted source:** `docs/wavemotion-paper_submitted/Neural-field-design-broadband-Rayleigh-wave-cloaks.tex`
**Comments:** `docs/Comments_for_author.md`
**Drafted:** 2026-11-20, from the repository, run outputs and earlier analysis notes only. No new simulations were run.
**Companion files:**
- `docs/reviewer_response_runs.md` describes the runs that would be needed to close the remaining items.
- **`docs/reviewer_response_final.md` holds the final letter and the main problems.** This file is the evidence appendix behind it. Where the two differ, the final file is the corrected one.

## How to read this file

Each reviewer comment has four parts:

- **Draft reply:** text we can adapt for the response letter.
- **Evidence:** where each number comes from, for us only. Do not paste this into the letter.
- **Status:**
  - ✅ answerable now
  - ⚠️ answerable, but an author must confirm or decide something first
  - 🔁 needs a new run (see the runs file)
  - ❓ the information is not available in this repository
- **Manuscript changes:** edits to the .tex file.

Some of the runs behind Sec. 4 were made on another machine. Their configs point to `/home/m3l/workspace/...`, and none of the `output/A_*` directories (14×10 single-frequency, 1×1/2×1/2×2, broadband 14×10, grid sweep) exist on falcon. Where an answer depends on those outputs it is marked ❓, and the numbers should be pulled from that machine.

---

## 0. Read first: inconsistencies found while answering

These came up while gathering evidence. Several of them directly affect what we can tell the reviewers, so they should be settled before the letter is written.

### 0.1 The 0.967 and the 0.76 come from different microstructure sets. This is the most important one.

The paper presents η = 0.967 (homogenised) and η ≈ 0.76 (fully resolved) as two views of the same diffusion cloak. They are not the same cloak.

| | Cells | Diffusion model | Homogeniser used for the η | Homogenised η at f\*=2 | Fully resolved η |
|---|---|---|---|---|---|
| **Paper's 0.967** | `output/cell20_phys_flat4_materialreg4/cell_designs_diffusion` | v1_ortho, trained on `ca_bulk_squared` (156,672 cells), best-of-16 | legacy TRI3, 1 element/pixel, evaluated dynamically at f\* | **0.9667** (`val_diffusion/frequency_sweep_matched.csv`) | **never validated** with the corrected solver |
| **Cells behind the 0.76** | `.../cell_designs_diffusion_2m` | v2, trained on `ca_squared_2m` (905,728 cells), best-of-15 | static hi-fi TRI6, 2 elements/pixel (same homogeniser as the dataset) | **0.9342** (`val_diffusion_2m/frequency_sweep_matched.csv`) | **0.7550** at 1.01 e/px; **0.7733** at refinement 55 |

How this was checked:

- `design-validation/configs/_cloak_cudss.yaml` and `_cloak_rf55.yaml` both set `cell_designs: data/cell_designs_diffusion_2m`.
- The canvases are md5-identical to `output/.../cell_designs_diffusion_2m`. For example, `cell_014` hashes to `66ed74…`, while the 0.967 set's cell_014 hashes to `b5dd3d…`.

Two further problems follow from this:

- **The legacy homogeniser is known to be too stiff.** The 0.967 cells were scored with the legacy TRI3 homogeniser at 1 element/pixel. The docstring of `dataset/cellular_chiral/bulk_stiffness.py` (lines 28–31, citing `results/mesh_artifact/SUMMARY.md`) records that it **over-predicts stiffness by about 12% for bulky cells and by more than 130% for thin cells**. So the 0.967 is not on the same accuracy footing as the 0.934.
- **Older pixel runs on the 0.967 cells are void.** Pixel-level runs on the 0.967 cell set exist (`conv_tri3`, `conv_tri6`, `val_diffusion/frequency_sweep_validated.csv`, η ≈ 0.04–0.1). They predate two fixes in the validation code. First, the ligaments were assigned soil instead of cement, 83× too soft (fixed 2026-08-01). Second, ersatz voids were used instead of traction-free holes. Those results must not be used.

**Options:**

- **(a) Recommended, no new run.** Report the like-for-like pair: homogenised **0.934**, fully resolved **0.755 at 1.01 e/px and 0.773 at refinement 55 (≈1.27 e/px)**. Keep 0.967 only as the v1_ortho per-cell comparison in Table 3. The headline gap then becomes about **16–18 points, not 21**. The abstract and conclusion (97% / 76%) need rewording.
- **(b) New run.** Validate the 0.967 cells fully resolved, run R1 in the runs file. Running them through the hi-fi homogeniser would probably also move the 0.967; by how much is unknown.

### 0.2 The paper's hyper-parameters do not match the configs of the paper runs

| Item | Paper | Configs of the paper runs | Source |
|---|---|---|---|
| MLP | "four-layer MLP with 256 hidden units (~200,000 weights)" (lines 432, 454) | `hidden_size: 512`, `n_layers: 6`, `n_fourier: 64`, which is about **1.18 M** weights | `configs/A_single_frequency/cells_14x10.yaml:64-66`; `configs/A_multifreq_14x10_nogmm.yaml:109-111`; `output/cell20_phys_flat4_materialreg4/config.yaml:75-77` |
| Anisotropy cap R | 30 (line 505) | `anisotropy_ratio: 12.0` | same files |
| κ | 0.99 (line 516) | `kappa: 0.95` | same files |
| Fourier features | "Fourier positional encoding of Tancik" | Deterministic integer frequencies `linspace(1, n_fourier)`, not random Gaussian features | `rayleigh_cloak/neural_reparam.py:64-77` |

Only the 1×1 "ceiling" exploration configs (`configs/A_1x1_ceiling/E3_kappa99.yaml` and others) use R = 30 and κ = 0.99.

**Decision needed:** which run produced the 1×1 value of 0.857? If it is a ceiling run, the paper should state per experiment which R and κ were used.

### 0.3 Initialisation claim not supported by the runs found

Line 459 says two initialisations, arithmetic-mean symmetrisation and the dataset mean, made no appreciable difference.

- Every paper-run config found uses `init: dataset_centroid` with `symmetrize_init: false`. No matched comparison with a symmetrised start was found.
- For the 1×1 case, `configs/A_1x1_ceiling/_gen_basins.py` (lines 4–9) states that seeds 7/42/123 "landed in qualitatively different basins" of a "degenerate, multimodal" objective.
- The 0.85 level was reached by perturbed warm restarts: "E3 … trans 0.821" and "warm_s101_e02 lifted 0.821 -> 0.850" (`_gen_warm_seeds.py`, `_gen_warm2.py`).

So either the comparison exists on the other machine, or the sentence must be softened. See R1.1.

### 0.4 Smaller text and number mismatches

1. **Table 3 "continuous optimum 0.990".** For the run whose targets were actually realised (`cell20_phys_flat4_materialreg4`), feeding the optimiser's own targets back into the evaluation sweep gives **0.9853** (`val_ideal_targets/frequency_sweep_matched.csv`). The 0.990 matches the 20×15 entry of the single-frequency grid sweep (`docs/wavemotion-paper/figures/cell_sweep_histogram.png`), which is a different run. Check which one is meant; 0.985 is the correct ceiling for the realised cloak.
2. **Per-cell errors (Fig. 7 caption).** 6.3% is the best-of(NN, diffusion) error. **Diffusion alone is 7.0%**, and the inverse best-of is 5.9% (inverse chosen for 81 of 100 cells). Source: `scripts/plot_bestof_dumbbell.py` → `bestof_dumbbell_nn.png`.
3. **Connectivity penalty.** Sec. 5.2 says inverse design uses a connectivity penalty. `scripts/cell_inverse_design_sweep.py` defaults to `--weight-conn 0` (off), and the command actually used was not logged. Check this, or drop the sentence.
4. **GMM penalty, Eq. 31.** The paper writes a sum over cells. The code takes the mean (`rayleigh_cloak/material_prior.py:215-216`), with weight 0.1. This is equivalent up to the weight, but should be stated consistently.
5. **Rayleigh speed used to define f\*.** The paper says c_R ≈ 0.92 c_s. The code computes the frequency with the Viktorov approximation c_R = c_s(0.826 + 1.14ν)/(1 + ν) = **0.889 c_s** (`rayleigh_cloak/config.py:551,562`). The exact value for ν = 1/4 is 0.9194 c_s. So the true b/λ_R is 3.3% below the f\* label: the nominal f\* = 2 is b/λ_R = 1.93, and the band [1, 3] is really [0.97, 2.90]. Either restate the definition of f\* or add the factor.
6. **Dataset size.** "N ~ 10⁶ cells". 2,000,000 cells were generated (`output/ca_bulk_squared/cells.npy`). The GMM and v1_ortho were trained on the **156,672** unique cells after fuzzy deduplication. The v2 model used 905,728 cells (exact deduplication, hi-fi homogenisation). Say which set each step uses.
7. **The 14×10 single-frequency config also has the GMM prior on** (`material_cement_GMM weight 0.1`, `cells_14x10.yaml:52`). The paper introduces the GMM only in Sec. 5. Check whether the 0.990 design used it, and say so.
8. **Mesh size.** The paper quotes "~246,000 elements, ~124,000 nodes" for all runs. The broadband run used about 75k TRI6 nodes (`A_multifreq_14x10_nogmm.yaml:23-30`), and `cell20_phys` used TRI3 at rf = 5. State the mesh per experiment, or say the 124k mesh is the one used for evaluation.

---

# Reviewer 1

## R1.1 Optimisation procedure, sensitivity to initialisation, GMM/diffusion details, and the 0.967 → 0.76 gap

### (a) Learning rate, iterations, convergence and stopping

**Status:** ✅ for the settings; ❓ for the loss curves of the Sec. 4 runs, which are on the other machine; ⚠️ the network-size mismatch in 0.2 must be fixed first.

**Draft reply:**

> All neural-field optimisations use Adam (β₁ = 0.9, β₂ = 0.999, ε = 10⁻⁸) with a cosine learning-rate decay from 5·10⁻⁴ to 10⁻⁵ over a fixed budget of iterations: 1000 for the single-frequency 14×10 design, 3000 for the broadband 14×10 design and 2000 for the GMM-constrained 20×15 design. No early stopping is used. The iterate with the lowest loss is retained. Convergence is judged from the loss history. For the constrained 20×15 design the loss falls from 0.553 at initialisation to 7.3·10⁻³ after 250 iterations, 8.6·10⁻⁴ after 1000, and 7.08·10⁻⁴ at iteration 2000, with only a 4% reduction over the last 500 iterations. We have added these settings, the network size (5 hidden layers of 512 units, 64 Fourier frequencies per coordinate) and the loss histories to the revised manuscript [Appendix/Table X].

**Evidence:**

- Optimiser: `rayleigh_cloak/optimize.py:118-135`.
- Schedule and best-iterate retention: `rayleigh_cloak/neural_reparam.py:547-553, 590-592`.
- Iteration counts: `cells_14x10.yaml:56`, `A_multifreq_14x10_nogmm.yaml:101`, `cell20_phys…/config.yaml:66`.
- Loss values: `output/cell20_phys_flat4_materialreg4/loss_history.csv`.
- The final layer is initialised ×0.01, so the network starts at the initial material: `neural_reparam.py:406-407`.

**Still missing (❓):** loss histories for the 14×10 single-frequency and broadband runs (`output/A_single_frequency/cells_14x10/`, `output/A_multifreq_14x10_*/`) on the m3l machine. Also confirm which broadband variant the paper used: `_nogmm`, `_gmm`, `_nogmm_restart` or `C_multifreq_14x10_gmm`.

### (b) Sensitivity to initialisation

**Status:** ⚠️ The current claim is not backed by anything found here (see 0.3). Choose one:

- **Honest version, no run.** Replace the sentence at line 459 with something like:

  > The objective is non-convex. For the single-material (1×1) case, which has only five design variables, different network seeds converge to different local optima, and we obtained the reported design with perturbed warm restarts. For the cell-resolved designs (46 and 100 cells) all reported results start from the dataset-mean material.

  Then give the reviewer the same text, plus a statement that a systematic multi-seed study is added (if run R5 is done) or left as a limitation.

- **Supported version.** Run R5 (a few seeds × two initialisations for the 14×10 design) and report the spread of η.

### (c) GMM and diffusion details

**Status:** ✅, once 0.1 is decided, because the details differ between the two diffusion models.

**Draft reply:**

> **GMM prior.** K = 16 full-covariance components fitted with scikit-learn's EM (max. 200 iterations, seed 0) to the standardised 5-D descriptor (C₁₁, C₂₂, C₆₆, C₁₂, ρ) of the 156,672 unique homogenised catalogue cells. The threshold τ is the 25th percentile of the catalogue log-density (τ = −5.07). The penalty enters the loss as the cell-average of max(0, τ − log p), with weight 0.1.
>
> **Diffusion model.** A conditional U-Net denoiser on 64×64 padded binary cells: base width 128, one residual block per level, self-attention at the 8×8 and 4×4 resolutions with 4 heads, dropout 0.1, and 50% self-conditioning. It is a continuous-time model with a cosine log-SNR noise schedule. It is conditioned on the 5-vector (C₁₁, C₂₂, C₁₂, C₆₆, volume fraction) by classifier-free guidance: the condition is dropped with probability 0.1 during training, and the guidance scale at sampling is w = 2. Training used Adam at 10⁻⁴ with cosine decay, batch 256 and EMA 0.9999. Sampling uses 50 steps. For each cloak cell we draw N = 16 [v1_ortho] / 15 [v2] candidates, homogenise each, and keep the one with the lowest mean relative error over (C₁₁, C₂₂, C₁₂, C₆₆, ρ).
>
> **Inverse design.** A per-cell 4×128 Fourier-feature MLP density field, initialised from the nearest catalogue cell. Adam runs for 350 iterations (learning rate 2·10⁻³ → 3·10⁻⁴). The projection sharpness β is ramped from 1 to 32, followed by a straight-through binary tail, with SIMP exponent 3 and a density-match weight of 10.

**Evidence:**

- GMM: `dataset/gmm_flat4.npz` (`n_components=16`, `covariance_type=full`, `threshold=-5.0719`, `threshold_percentile=0.25`, `n_samples=156672`), `dataset/cellular_chiral/fit_gmm.py:102,158`, run config key `loss.regularizations.material_cement_GMM`.
- Diffusion v2: `output/diffusion_ca_squared/v2/logs/version_0/hparams.yaml` and `microstructure_generation_2d/network/model.py`.
- Sampling defaults: `scripts/cell_diffusion_design_sweep.py:381-387`.
- v2 run log: `cell_designs_diffusion_2m.log` (15 seeds, 50 steps, w = 2.0).
- v1_ortho: config only, `configs/diffusion/diffusion_ca_squared_v1_ortho.yaml`. **The checkpoint is not on disk.**
- Inverse design: `scripts/cell_inverse_design_sweep.py:469-516`.

**Gaps:**

- The epochs actually completed for v2 are unknown. Saved checkpoints go up to epoch 69, and the best validation loss is at epoch 59 according to earlier notes.
- `configs/diffusion/diffusion_ca_squared_v2.yaml` does **not** match the real run (it says 64 channels and 500 epochs). Cite the hparams file, not the yaml.
- Inverse design: the connectivity-penalty question is open (0.4.3).

### (d) The 0.967 → 0.76 gap, quantitatively

This is the same question as R2.3. See R2.3 below for the full answer. The short version for R1:

> A like-for-like decomposition for the realised cloak at f\* = 2:
> - 0.985 with the optimiser's exact per-cell targets
> - 0.934 with the homogenised properties of the generated cells (per-cell property mismatch, mean 8.7%)
> - 0.755–0.773 fully resolved (homogenisation and assembly error)
>
> The microstructure realisation therefore costs about 5 points at the effective-property level. The larger, 16–18 point loss arises when the explicit geometry replaces the effective medium.

---

## R1.2 Meaning of "broadband" and implications of the phase-invariant objective

**Status:** ✅ for the bandwidth (after fixing 0.4.5). ✅ for the qualitative phase implications; the quantitative pulse metrics are 🔁 (shared with R2.1).

**Draft reply:**

> By "broadband" we mean that one fixed, frequency-independent design is optimised and evaluated over a continuous band rather than at a single frequency. The training band is f\* = b/λ_R ∈ [1, 3]: a 3:1 frequency ratio, or about 1.6 octaves, which is a fractional bandwidth Δf/f_c of 100%. Because the problem is linear and scale-free, the band is defined relative to the cloak depth b. For the soil parameters used here (c_s = 300 m/s) and b ≈ 1 m, it corresponds to about 270–800 Hz.
>
> Over this band the design keeps η between 0.92 and 0.99 (mean 0.96). The uncloaked notch falls from about 0.65 at f\* = 1 to 0.31–0.42 over f\* ∈ [1.4, 3]. Outside the band, performance degrades to 0.78 at f\* = 3.5 and 0.60 at f\* = 4.0 (Fig. 9). We have replaced "octave-plus" with this explicit definition. Below the band the notch is increasingly small compared with the wavelength, and the uncloaked η rises towards 1 as f\* → 0 (about 0.9 at f\* = 0.5).
>
> Regarding the objective: Eq. (20) compares displacement magnitudes, so it is invariant to a phase shift of the transmitted field. Its minimiser restores the amplitude of the transmitted surface wave but does not control its phase. For the single-frequency design we observe a lag of about 0.18λ (65°) relative to the reference. A phase lag that grows linearly with frequency is a pure time delay. It shifts a pulse without distorting it, but it would reveal the defect to a time-of-flight measurement. Waveform distortion arises only from frequency-dependent amplitude, which our η measures, and from phase that is non-linear in frequency, which η does not see. We have therefore restated the scope of the work as amplitude (magnitude) restoration of the transmitted Rayleigh wave, i.e. vibration and shadow mitigation, rather than full phase cloaking. [If R2-time-domain is run: we have added a time-domain pulse test, see the reply to Reviewer 2, comment 1.]

**Evidence:**

- Band: `A_multifreq_14x10_nogmm.yaml:89` (`f_stars: [1.0, 1.3333, 1.6667, 2.0, 2.3333, 2.6667, 3.0]`, equal weights).
- Hz conversion: b = 3a = 3·0.0774·4.305·λ\* = 0.9996 m, and ω = 2π f\* c̃_R/λ\* with c̃_R = 0.889·300 = 266.6 m/s, so f\* = 1 → 267 Hz and f\* = 3 → 800 Hz.
- Uncloaked values (0.65 at f\* = 1, 0.31–0.42 for f\* ≥ 1.4, about 0.9 at 0.5): read by eye off `figures/multifreq_sweep.png`. The underlying CSV is not on this machine.
- Mean 0.96, 0.78 and 0.60: paper text.
- The "pure delay vs distortion" point is standard signal theory, not a result of ours. Do not present it as measured.
- **Note:** no script in this repository computes the 0.18λ / 65° lag, and `single_freq_14x10_zoom_phase.png` has no generator here. Confirm how it was measured before repeating it in the letter.

---

## R1.3 Mesh convergence / numerical resolution

**Status:** ✅ partial evidence for the homogenised macro model and for the fully resolved model. 🔁 a short convergence check of the *final* 14×10 and 20×15 designs (R3) would make the answer complete.

**Draft reply:**

> **Homogenised (macro) model.** The mesh was selected by a convergence study.
> - Two element types (linear TRI3, quadratic TRI6) and successive cloak refinements converge to the same cloak ratio within 1%: TRI3 gives 0.9748 / 0.9805 / 0.9806 for cloak refinement factors 2 / 3 / 4.
> - For a positive-definite optimised design (10×8 cells), η = 0.975 ± 0.001 over four refinements, from 33k to 78k nodes.
> - Because the metric is a magnitude ratio between two solves on the same far-field mesh, far-field dispersion error (a phase error) largely cancels.
> - We also found that convergence requires every cell to be positive-definite. A non-positive-definite cell produces a localised spurious mode that *grows* with refinement. This is why the constrained decoding of Sec. 3.2 is used.
>
> [Add the R3 table here if run: η of the final 14×10 and 20×15 designs at 3 refinements.]
>
> **Fully resolved microstructure.** We solve the explicit solid/void geometry with quadratic (TRI6) elements. Voids are meshed as traction-free holes and the solver is a GPU sparse direct solver. Results at three resolutions:
>
> | elements per pixel | nodes | η |
> |---|---|---|
> | 0.51 | 0.47 M | 0.712 |
> | 1.01 | 1.00 M | 0.755 |
> | ≈1.27 | 1.38 M | 0.773 |
>
> The value is still increasing at the finest resolution we could fit in memory. A run at 1.52 elements per pixel exceeded the 31 GB host memory during assembly. The fully resolved ratio should therefore be read as a lower bound, η ≥ 0.77, not a converged value. We state this explicitly in the revised manuscript.

**Evidence:**

- Macro: `docs/mesh_refinement_selection.md` §3a–3c. The raw outputs and the linked `results/mesh_tri6_validation/VALIDATION.md` are **not** on this machine.
- Pixel: `design-validation/output_cloak_cudss/convergence.csv`, `design-validation/output_cloak_rf55/validation_result.csv`; the refinement-66 run has no result (`output_cloak_rf66/run.log`).
- Determinism: 0.754965 is reproduced exactly as the zero-flip control in every flip study (`design-validation/output_flip_*/*.csv`).
- The convergence write-up, including "quote ≥0.77 and rising": `design-validation/docs/gpu_validation_and_optimization.md` §2.3–2.5.

**Caution:**

- The three pixel points do not fit u(h) = u\* − C hᵖ for any p > 0, so **no extrapolated limit is defensible**. Do not give one.
- The quantity max|u|/p95 ≈ 7 on these meshes is a warning sign that the run log itself flags. Do not claim the fields are artefact-free.

---

## R1.4 References

**Status:** see each item. In the compiled PDF the references are numbered in order of first citation, checked against the .bbl: [4] = diatta2014controlling, [9] = kadic2020elastodynamic, [23] = nakarmi2024, [30] = yang2024.

| Ref | Reviewer request | Finding | Action |
|---|---|---|---|
| [4] Diatta & Guenneau 2014 | complete it | Appl. Phys. Lett. **105** (2), **021901** (2014), **doi:10.1063/1.4887454**, verified via Crossref. | ✅ Add volume, number, pages and DOI. |
| [9] Kadic et al. 2020 | add the DOI | **10.1016/j.wavemoti.2019.102419**, verified from the Wegener group page, which links to doi.org. | ✅ Add it to the bib. |
| [23] Nakarmi et al. 2024 | does it support the specific CA procedure? | Our generator `dataset/cellular_chiral/generator.py` says in its header that it "implements the dataset generation method from Nakarmi et al., CMAME 432 (2024) 117372". It implements their **Algorithm 1** (flood fill of disconnected regions) and **Algorithm 2** (Kruskal MST bridging). Our own choices are not attributed in the code: mirror ("squared") assembly of a 25×25 quadrant into a D₂-symmetric 50×50 cell, frozen solid border with width-5 central gates, B5678/S4/D0123 rules over 7 iterations, initial live fraction drawn from U[0.2, 0.8], and bridges of width 4. **The Nakarmi paper itself could not be fetched** (OSTI and NSF-PAR refused the connection), so we could not check which of these choices are theirs. | ⚠️ An author should check the PDF. Then write in the manuscript: "following the CA generation and connectivity-repair procedure (flood fill and minimum-spanning-tree bridging, Algorithms 1–2) of [23], *modified as follows*: …" and list our changes. |
| [30] Yang et al. 2024, arXiv:2401.13570 | update if peer-reviewed | **The bib author list is wrong.** arXiv (v2, 10 Jun 2024) gives Yanyan Yang, Lili Wang, Xiaoya Zhai, Kai Chen, Wenming Wu, Yunkai Zhao, Ligang Liu, Xiao-Ming Fu. Crossref has a likely journal version with a changed title: "Guided diffusion for fast inverse design of voxel-based mechanical metamaterials", Smart Mater. Manuf. 4 (2026) 100129, doi:10.1016/j.smmf.2026.100129, adding author Falai Chen. | ⚠️ Fix the authors. An author confirms the journal version before citing it. |

**Draft reply (after the checks):**

> We thank the reviewer. Ref. [4] has been completed (Appl. Phys. Lett. 105, 021901, 2014, doi: …) and the DOI added to Ref. [9]. Regarding Ref. [23]: our generator implements the cellular-automaton evolution and the connectivity-repair steps (flood-fill identification of disconnected regions and minimum-spanning-tree bridging) of Nakarmi et al. We have now stated explicitly in Sec. 5.1 the modifications we introduced: mirror assembly of a 25×25 quadrant into a D₂-symmetric 50×50 cell, fixed border gates that guarantee tiling, the rule set and iteration count, and the sampled initial live fraction. Ref. [30] [has been updated to its published version / remains an arXiv preprint at the time of writing].

---

# Reviewer 2

## R2.1 Phase lag: "cloaking" vs "magnitude restoration", and time-domain or broadband waveform metrics

**Status:**

- ✅ Scope and limitation. We agree, and the text can be changed.
- 🔁 Quantitative pulse and waveform metrics need run R2. These are forward frequency-domain solves plus Fourier synthesis only, with no new optimisation.
- ❓ No phase, complex-field or time-domain result exists in this repository. `scripts/animate_wave_propagation.py` only animates the time-harmonic field over one period.

**Draft reply:**

> We agree with the reviewer, and we have restated the scope of the work. The objective (20) is phase-invariant. The designs presented are therefore **amplitude-restoring (magnitude) carpet cloaks** for the transmitted Rayleigh wave. They suppress the shadow and restore the vibration level behind the defect, but they are not phase-exact cloaks. The term "cloaking" in the title and abstract is now qualified accordingly, and a sentence on the application boundary has been added to the Discussion: the designs target vibration and shadow mitigation, not concealment from phase- or time-of-flight-based detection.
>
> On broadband or pulsed excitation: a phase lag that is linear in frequency is a pure delay and does not distort the waveform, whereas a non-linear phase and frequency-dependent amplitude do.
>
> [If R2 is run:] To quantify this we synthesised the time-domain response to a Ricker pulse centred at f\* = 2, with most of its energy in the training band, by Fourier superposition of the frequency-domain solutions. We report: (i) the normalised cross-correlation between the cloaked and reference surface waveforms, after optimal time alignment and without alignment; (ii) the arrival-time delay; (iii) the transmitted energy ratio. These are given for the uncloaked notch, the ideal cloak and the broadband design.
>
> [If R2 is not run:] A time-domain assessment requires the complex surface response at all frequencies of the band. We have added it to the list of limitations and future work.

**Evidence:** the phase lag of 0.18λ (65°) is from paper lines 688–695 (see the caution under R1.2). The phase behaviour of the **broadband** design has never been measured.

---

## R2.2 Why untrained frequencies perform well, and sensitivity to the number of training frequencies

**Status:**

- ✅ The explanation can be argued from existing evidence, mostly Fig. 9 and the single-frequency sweeps. Hypotheses must be stated as such.
- 🔁 The 3- and 5-frequency sensitivity study needs run R6.
- ❓ No comparable run with fewer frequencies exists. Older 5- and 9-frequency runs (`cauchy_tri_minimax*`) use a different 50×50 parameterisation and loss, have no frequency sweeps, and are **not comparable**.

**Draft reply:**

> We thank the reviewer for this question. The evidence we have points to the smoothness of the physical response in frequency, together with a sampling interval that is fine enough relative to that smoothness:
>
> 1. **The response varies smoothly with frequency.** All materials in the design are non-dispersive (frequency-independent C and ρ, no resonant inclusions), and the problem is open: energy radiates into the half-space and the PML. The response therefore has no high-Q resonances. This is visible in Fig. 9: even the uncloaked notch varies smoothly with f\*, on a scale of about 0.3–0.5 in f\*. The seven training frequencies are spaced Δf\* = 1/3 apart, which is comparable to this scale.
> 2. **The dense sweep confirms that this sampling is adequate but not lavish.** The sweep in Fig. 9 has a step of 0.1. Of its 21 in-band points, 18 are not training frequencies. The lowest in-band values, about 0.92 at f\* = 1.2 and about 0.94 at f\* = 2.2, fall *between* training frequencies, so the response does sag slightly between samples. No sharp resonance appears.
> 3. **The neural-field parameterisation alone does not produce broadband behaviour.** The single-frequency design uses the same neural field, yet its cloak ratio drops quickly away from f\* = 2 (Sec. 4.4; e.g. 0.963 at f\* = 2.0 and 0.923 at f\* = 2.1 for a single-frequency 20×15 design). The spatial smoothness of the field is therefore not what produces the broadband performance. Including the frequencies in the objective is. We have no evidence that the spatial smoothness prior suppresses narrowband resonances, and we no longer suggest it.
>
> [If R6 is run:] We have added a sensitivity study with 3 and 5 training frequencies over the same band (Table X). [Results.]

**Evidence:**

- Point 1 is an argument. The "0.3–0.5" scale is read by eye from the black curve in `figures/multifreq_sweep.png`.
- Point 2 values are read off the same figure. The CSV is ❓, on the m3l machine, and should be used for exact numbers.
- Point 3: `output/singlefreq_20x15_gmm_cold/frequency_sweep_optimized.csv` (0.9629 at f\* = 2.0, 0.923 at 2.1). This is a 20×15 GMM run, not the 14×10 single-frequency design. If the 14×10 sweep is retrieved, use it instead.
- Paper line 1104 says the network's implicit smoothness "eliminates the spurious cell-to-cell oscillations" in the multi-frequency case. No comparison run backing this was found. Soften it or support it.

---

## R2.3 Attribution of the ~20% gap, and a displacement-field plot of the fully resolved cloak

**Status:**

- ⚠️ First fix 0.1. The like-for-like gap is **0.934 → 0.755–0.773**, not 0.967 → 0.76.
- ✅ A preliminary attribution can be given from existing measurements.
- 🔁 The field plot needs run R4. It is cheap, but no fully resolved displacement-field image exists. The validation script only writes the geometry image `cloak_structure_triangle.png`, so it needs a field-export option.

**Draft reply:**

> We agree that this needed a fuller treatment. We have added a new subsection with a step-by-step decomposition at f\* = 2 for the realised 20×15 cloak, using the same 100 microstructures throughout:
>
> | Model of the cloak | η |
> |---|---|
> | Optimised continuum targets (ideal realisation) | 0.985 |
> | Homogenised properties of the generated microstructures | 0.934 |
> | Fully resolved solid/void geometry (1.0 / 1.27 elements per pixel) | 0.755 / 0.773 (not yet mesh-converged, increasing) |
> | Uncloaked notch | ≈0.34 [check, see below] |
>
> The microstructure realisation itself, i.e. imperfect matching of the target stiffness and density, costs about 5 points. Most of the loss, 16–18 points, appears when the explicit geometry replaces the effective medium. We have ruled out two candidate explanations:
>
> - **Void modelling.** Re-homogenising catalogue cells with the voids removed (traction-free holes) instead of filled with a 10⁻⁶ ersatz material changes the effective stiffness by a median of about 10⁻⁵ relative, and at most about 1% (C₆₆ of the thinnest cells).
> - **Internal resonance of the cells.** The unit cell (6.7 cm) is about 1.5% of the shear wavelength in the concrete phase at the operating frequency, so the cells respond quasi-statically.
>
> The remaining candidates are effects that first-order homogenisation does not represent:
>
> - **(i) Limited scale separation.** The cloak cells are much stiffer than the soil, so wavelengths inside the cloak are long. But the cell size relative to the *local* shear wavelength in the cloak still ranges from 0.06 to 0.32, i.e. 3–17 cells per local wavelength (median 9.5).
> - **(ii) Boundary layers at the traction-free surface.** About a fifth of the cloak cells lie within half a Rayleigh wavelength of the surface, where the homogenisation corrector is not small.
> - **(iii) Interfaces between neighbouring dissimilar cells.** Neighbouring cells always share the same fixed solid gate, so the load path is never severed. However, 34% of the selected cells carry extra material next to the gate, compared with 18% in the catalogue. Where such a cell meets one without it, the effective contact is smaller than either cell's self-tiling homogenisation assumes.
>
> We have not yet separated these contributions quantitatively, and we say so in the revised text. Three observations on the fully resolved model support this picture:
>
> - **The gap has a direction.** Randomly *removing* 2,500 concrete pixels (1% of the cloak's pixels, 1.9% of its concrete) raises the fully resolved η in each of four trials (0.763–0.780, against 0.755). Randomly *adding* the same number lowers it in each of four trials (0.732–0.750). The assembled microstructure therefore behaves as if it had more solid than the effective-medium design calls for. It is biased in one direction, not merely noisy. This is consistent with interface and boundary-layer effects that the cell-by-cell homogenisation leaves out. A pixel-optimised design shows no such bias (both directions lower η on average).
> - **The result is not fragile to small random defects.** Flipping 0.01–0.1% of the cloak pixels at random changes η by at most 0.006 (5 seeds per level). The 0.755 is therefore a property of the design, not of a particular pixel.
> - **Microscale detail matters.** Changing only 0.09% of the pixels with an adjoint-gradient-guided procedure raises the fully resolved η from 0.755 to 0.804. The resolved response therefore depends on detail that the homogenised constants do not capture.
>
> [If R4 is run:] Fig. X compares the displacement magnitude of the defect-free reference, the homogenised cloak and the fully resolved cloak, together with the surface displacement profile. [Describe where the loss appears.]

**Evidence:**

- 0.985: `val_ideal_targets/frequency_sweep_matched.csv`.
- 0.934: `val_diffusion_2m/frequency_sweep_matched.csv`.
- 0.755 / 0.773: `design-validation/output_cloak_cudss`, `output_cloak_rf55`.
- Mean per-cell error of the 2m cells, 8.67%: `cell_designs_diffusion_2m.log` and `cell_designs_diffusion_2m_finetuned/PROVENANCE.md`.
- Void model: `dataset/stiffness/results/VOIDFREE_VS_ERSATZ.md`. Over 500 cells the median relative difference is −1.6·10⁻⁵ for C₁₁ and C₂₂; the maximum over 500 + 299 thin cells is 1.3·10⁻² for C₆₆.
- Cell vs concrete wavelength: cell = 2c/20 = 0.0666 m. Concrete c_s = √(E/(2(1+ν)ρ)) = 2331 m/s, so λ_s = 4.37 m at 533 Hz.
- Ligament bending estimate (about 12.6 kHz in old notes): **dropped from the reply**, because the script is lost and the inputs are not recorded.
- Scale separation 0.06–0.32 (3.1–17.3 cells per wavelength, median 9.5): computed today from the `pred_flat4[3]` (C₆₆) and `pred_rho` of the 100 `cell_designs_diffusion_2m` cells, at ω = 2π·533 Hz.
- **Re-derived 2026-11-20:**
  - 22 of the 100 cloak cells have their centre within 0.5 λ_R (0.517 m) of the surface.
  - Exactly-gate edges: 66.0% of the selected 2m cells' 400 edges, against 82.1% of 20,000 random cells of `ca_squared_2m`. The raw 2M generated set gives 47.1%, because duplicates skew it; `ca_squared_2m` is the right comparison, being v2's training set.
  - All gate pixels are solid on 399 of 400 selected edges; cell index 9 has 9 of its 10 gate pixels on one edge.
- The 0.804 pixel optimisation: `design-validation/output_val_opt2/validation_result.csv`, 232 pixels flipped (`design-validation/docs/gpu_validation_and_optimization.md` §3.5).
- Add/remove asymmetry: `design-validation/output_flip_asymmetry/flip_asymmetry.csv`. Baseline 0.754965; remove_only ×4 gives 0.7739 / 0.7797 / 0.7633 / 0.7716; add_only ×4 gives 0.7323 / 0.7339 / 0.7503 / 0.7465; symmetric ×4 gives 0.748–0.763. All runs are at rf 44, 2,500 pixels of the 250,000 in-cloak pixels. On the level-set-optimised design (`output_flip_optimal_asym`, baseline 0.8859), both directions lower η on average, with large spread (remove 0.653–0.889, add 0.851–0.876).
- Random-flip robustness: `output_flip_sensitivity/flip_sensitivity.csv` (rf 44). At 0.01% (25 px), η is 0.7547–0.7554. At 0.1% (250 px) it is 0.7492–0.7586. At 1% (2,500 px) it is 0.7595–0.7781. At 3% it is 0.732–0.783. At rf 55 (`output_flip_rf55`), 1% gives 0.782–0.797 against a baseline of 0.7733. The "published (0 flips)" control reproduces 0.7550 in every study.
- **Caution on manufacturing-tolerance claims.** A *systematic* boundary offset is a different matter from random flips. A ±0.5 px level-set offset of the whole boundary gave η = 0.057 / 0.485 (`output_levelset_14x10/ersatz_vs_void.log`), but that offset removed or added a large fraction of the concrete, not a sub-pixel amount. Do not claim tolerance to uniform ligament thinning.

**Post-submission fully resolved improvements.** These are in `design-validation/`, all at rf 44 with voids removed and validated on the traction-free-hole model. Including them is optional and an author decision; they could support the reply to R2.3 or the outlook.

| Design | η (fully resolved) | Source |
|---|---|---|
| Paper's cells (2m), as generated | 0.755 | `output_cloak_cudss` |
| + 232 gradient-ranked pixel flips | 0.804 | `output_val_opt2` |
| Fine-tuned diffusion cells + flips | 0.834 | `output_val_opt2_finetuned` |
| + level-set boundary optimisation, native 20×15 grid | **0.866** | `output_levelset_20x15_canonrig/ersatz_vs_void.log` (`canvas_best_20x15.npy`) |
| Level-set, design regridded to 14×10 cells of 75 px | 0.886 | `output_levelset_v5` → `output_flip_optimal` baseline |

The 0.886 design is no longer a tiling of the 100 catalogue-derived 20×15 cells: the regrid kept only 24 of 74 cells whole (`configs/levelset_20x15.yaml` header). The 0.866 is the like-for-like one.
- Uncloaked notch η for the 20×15 configuration: **not found.** The notch geometry is identical across grids, so the broadband figure value at f\* = 2 (about 0.34, read off the plot) is a proxy computed on a different mesh. One forward solve gives the exact value (R4 includes it).

**Caution:**

- An earlier linearised (first-order) sensitivity analysis predicted a change of +0.012 in η from the realisation error, where the truth was −0.051. The response to per-cell error is strongly non-linear (earlier notes). Do not offer a linear error budget per cell.
- The note "0.76 converged at refinement 130, confirmed at 260" in old project notes comes from the **pre-fix** validation pipeline. Use only the cuDSS numbers above.

---

# Summary of manuscript edits

1. Fix the provenance pairing of 0.967 and 0.76 (0.1) throughout: abstract, Sec. 5.4, Discussion, Conclusion.
2. Correct the network size, R and κ (0.2). Add a hyper-parameter table covering the optimiser, schedule, iterations and stopping rule for each experiment.
3. Rewrite or support the initialisation sentence at line 459.
4. Add GMM, diffusion and inverse-design details (R1.1c). Fix the connectivity-penalty sentence and the 6.3% vs 7.0% figures.
5. Define "broadband" explicitly. Restate the scope as amplitude restoration. Fix the f\* / c_R definition.
6. Add the mesh-convergence paragraph and table, and state that the resolved η is a lower bound.
7. Add the gap decomposition subsection (R2.3). Add the field figure if R4 is run.
8. Explain the untrained-frequency behaviour (R2.2). Soften the "implicit smoothness" claim at line 1104.
9. Fix the bibliography: [4], [9], [23] wording, [30] status and authors.
10. Fix the Table 3 "0.990" vs 0.985 question, the dataset-size wording and the mesh description per experiment.
