# Wave Motion revision: final answers and main problems

**Paper:** *Neural-field design of broadband Rayleigh-wave carpet cloaks under microstructure realisability constraints*
**Reviews:** `docs/Comments_for_author.md` (R1, 6 Oct; R2, 30 Oct)
**Status:** final pass, 2026-11-20

Everything below comes from files in this repository, `design-validation/`, or verified online sources. No new simulations were run.

- **Evidence and file paths for every number:** `docs/reviewer_response_draft.md`.
- **Optional runs:** `docs/reviewer_response_runs.md`.

**This file has three parts:**

- **Part A:** the problems in the submitted paper that must be fixed before resubmitting. Several change what we can say to the reviewers.
- **Part B:** the response letter, ready to paste. It assumes the recommended fixes in Part A are applied and **no new runs** are made. Optional inserts for runs are marked `[OPTIONAL — if run Rx]`.
- **Part C:** what stays open.

---

# Part A: main problems in the submitted paper

Ranked by how much they matter.

### A1. The headline pair 0.967 / 0.76 mixes two different cloaks. Critical.

- The abstract, Sec. 5.4, the Discussion and the Conclusion present homogenised η = 0.967 and fully resolved η ≈ 0.76 as the same diffusion cloak. They are not.
- **To be clear, the problem is not that the two numbers differ.** A tensor (homogenised) value and an explicit-structure value are expected to differ; that difference is the gap Reviewer 2 asks about. The problem is that the two numbers describe **different microstructures**, so their difference is not that gap:

| Cell set | Tensor (homogenised) η | Explicit structure η |
|---|---|---|
| A: v1_ortho, `cell_designs_diffusion` | **0.967** (in the paper) | never simulated |
| B: v2, `cell_designs_diffusion_2m` | **0.934** | **0.755 / 0.773** (the paper's 0.76) |

  The homogenised-vs-explicit gap must be measured within one row: 0.934 → 0.76 for set B (about 17 points), not 0.967 → 0.76 (21 points).
- **The 0.967 cells:**
  - Directory `cell_designs_diffusion`, made by the v1_ortho model.
  - Scored with the legacy TRI3 homogeniser at 1 element/pixel, which is documented to over-predict stiffness by 12% (bulky cells) to more than 130% (thin cells).
  - Never run through the corrected fully resolved solver.
  - The v1_ortho checkpoint is no longer on disk, so these cells cannot be regenerated.
- **The 0.76 cells:**
  - Directory `cell_designs_diffusion_2m`, made by the v2 model.
  - Their homogenised η with the accurate hi-fi homogeniser is **0.934**.
  - Verified by the validation configs and by identical canvas checksums. In the `design-validation` flip studies this canvas is literally called `canvas_published`.
- **Fix, no run needed:** report the like-for-like chain:

| Step | η |
|---|---|
| Optimised targets | 0.985 |
| Homogenised realised cells | **0.934** |
| Fully resolved | **0.755** (1.0 element/pixel) / **0.773** (1.27 element/pixel) |

  - Keep 0.967 only in the method-comparison table (Table 3 / Fig. 8), labelled as the v1_ortho, legacy-homogeniser evaluation.
  - Abstract: "97% … 76%" becomes something like "93% … at least 77%".
  - The gap the reviewers ask about is **16–18 points, not 21**.

### A2. The fully resolved η is not mesh-converged. Important.

- Results: 0.712 at 0.51 e/px, 0.755 at 1.01, 0.773 at about 1.27, and still rising. The next refinement exceeds host RAM.
- No extrapolation is defensible: the three points fit no power law.
- The paper's "approximately 0.76" must become "**≥ 0.77, not yet mesh-converged**". This is also the honest answer to R1.3.

### A3. The network size and constraint constants in the paper are wrong. Important.

| | Paper | Every paper-run config |
|---|---|---|
| MLP | 4 layers × 256, ~200k weights | 5 hidden layers × 512, 64 Fourier frequencies per coordinate, about **1.18 M** weights |
| Anisotropy cap R | 30 | **12** |
| Coupling bound κ | 0.99 | **0.95** |
| Fourier features | random (Tancik) implied | **Deterministic** integer frequencies 1…64 (sin/cos of x and y) |

- R = 30 and κ = 0.99 appear only in the 1×1 "ceiling" exploration configs.
- Before writing the hyper-parameter table R1 asks for, confirm which run produced the 1×1 value 0.857.

### A4. The initialisation-insensitivity sentence (line 459) has no supporting runs. Important.

- Every paper config starts from the dataset mean (`dataset_centroid`, `symmetrize_init: false`). No run starting from the symmetrised tensor was found.
- The 1×1 exploration notes say different seeds land in "qualitatively different basins" of a "multimodal" objective, and that 0.82 → 0.85 needed perturbed warm restarts.
- R1 explicitly asks about initialisation sensitivity.
- **Fix:** replace the sentence with the honest version in Part B, unless the original comparison is found on the m3l machine.

### A5. Smaller factual mismatches. Fix in text.

1. **Rayleigh speed.** The code sets the frequency using c_R = 0.889 c_s (Viktorov approximation). The paper says 0.92 c_s; the exact value is 0.9194 c_s. The true b/λ_R is therefore 3.3% below every quoted f\*: the design frequency "f\* = 2" is b/λ_R = 1.93. Either define f\* through the 0.889 c_s used, or rescale.
2. **Table 3 "continuous optimum 0.990".** For the run whose targets were realised, the exact-target η is **0.985**. The 0.990 matches the 20×15 entry of the grid sweep, which is a different run.
3. **Fig. 7 caption.** 6.3% is best-of(NN, diffusion). **Diffusion alone is 7.0%.**
4. **Inverse design.** The paper describes a connectivity penalty, but the script default has it off (`--weight-conn 0`) and the command was not logged.
5. **GMM penalty, Eq. 31.** The paper writes a sum; the code takes the mean over cells (weight 0.1).
6. **Dataset size.** "N ~ 10⁶": 2,000,000 cells were generated. The GMM and v1_ortho used **156,672** unique cells, homogenised with the legacy TRI3 homogeniser. The v2 model used **905,728** cells with hi-fi TRI6.
   - Consequence worth knowing: the GMM realisability prior was fitted in the legacy (over-stiff) property space, while the final cells are scored with the hi-fi homogeniser.
7. **The 14×10 single-frequency config also uses the GMM prior** (weight 0.1). The paper introduces it only in Sec. 5.
8. **Mesh.** "~124k nodes" is quoted for everything. The broadband run used about 75k TRI6 nodes; the constrained 20×15 run used TRI3 at rf 5. State the mesh per experiment.
9. **Line 1104.** "Implicit smoothness eliminates the spurious cell-to-cell oscillations": no comparison run supports this. Soften it.
10. **The 0.18λ / 65° phase lag.** No script in the repo computes it. Confirm how it was measured before repeating it to the reviewers.

### A6. Bibliography. Must fix.

| Ref | Fix |
|---|---|
| [4] diatta2014controlling | Add `volume = {105}, number = {2}, pages = {021901}, doi = {10.1063/1.4887454}`. Verified via Crossref. |
| [9] kadic2020elastodynamic | Add `doi = {10.1016/j.wavemoti.2019.102419}`. Verified. |
| [23] nakarmi2024 | Our generator implements their Algorithms 1–2: flood fill of disconnected regions, and Kruskal minimum-spanning-tree bridging. The other choices are ours or unverified: mirror assembly of a 25×25 quadrant into a D₂ 50×50 cell, frozen border with central gates, B5678/S4 rules × 7 iterations, live fraction U[0.2, 0.8]. **We could not open the Nakarmi full text.** An author must check it, then reword Sec. 5.1 as in Part B. |
| [30] yang2024 | **The author list in our bib is wrong.** arXiv lists: Yanyan Yang, Lili Wang, Xiaoya Zhai, Kai Chen, Wenming Wu, Yunkai Zhao, Ligang Liu, Xiao-Ming Fu (v2, 10 Jun 2024). Crossref has a journal paper by the same group under a *different title*: "Guided diffusion for fast inverse design of **voxel-based** mechanical metamaterials", *Smart Materials in Manufacturing* 4 (2026) 100129, doi:10.1016/j.smmf.2026.100129, which adds Falai Chen as an author. It is very likely the published version. **An author should open it and confirm** before citing it in place of the arXiv entry. |

Unused bib entries (kingma, stenger, sigmund, …) do not appear in the PDF, so they are harmless.

---

# Part B: response letter

> We thank both reviewers for their careful reading and constructive comments. In preparing this revision we re-audited every reported number against the underlying run outputs. This audit led to one substantive correction (Response R2.3/R1.1d): the homogenised value previously quoted for the realised cloak (η = 0.967) came from an earlier set of generated microstructures than the one simulated in full resolution. All values for the realised cloak now refer to a single set of 100 microstructures. Changes in the manuscript are marked in blue.

## Reviewer 1

**Comment 1 (a): optimisation procedure, learning rate, iterations, convergence and stopping.**

> All neural-field optimisations use Adam (β₁ = 0.9, β₂ = 0.999) with a cosine learning-rate decay from 5·10⁻⁴ to 10⁻⁵ over a fixed iteration budget: 1000 iterations for the single-frequency 14×10 design, 3000 for the broadband design, and 2000 for the realisability-constrained 20×15 design. No early-stopping rule is used. The iterate with the lowest loss is retained, and convergence is judged from the loss history. For the constrained 20×15 design the loss decreases from 0.553 to 7.3·10⁻³ after 250 iterations and 8.6·10⁻⁴ after 1000, reaching 7.08·10⁻⁴ at iteration 2000, with only a 4% reduction over the last 500 iterations.
>
> The network is a multilayer perceptron with five hidden layers of 512 tanh units, fed by deterministic Fourier features sin(kx), cos(kx), sin(ky), cos(ky) for k = 1, …, 64. Its last layer is initialised at 1% scale, so that optimisation starts from the prescribed initial material. The anisotropy cap and coupling bound used in all reported runs are R = 12 and κ = 0.95. We correct the values previously given in the text and apologise for the inconsistency. These settings are now collected in a new Table [X].

*[If the m3l loss histories are retrieved, add the 14×10 and broadband curves to the appendix.]*

**Comment 1 (b): sensitivity to initialisation.**

> We thank the reviewer for raising this point, and we have corrected our statement. All reported cell-resolved designs (46 and 100 cells) start from the mean material of the microstructure catalogue. The objective is non-convex. For the single-material case (five design variables) we observed that different network seeds converge to different local optima, and the reported design was obtained by perturbed warm restarts. We have replaced the previous sentence, which claimed insensitivity to initialisation, with this description, and we list a systematic multi-start study as a limitation.

*[OPTIONAL — if run R5: replace the last sentence with the spread of η over 3 seeds × 2 initialisations.]*

**Comment 1 (c): GMM and diffusion details.**

> We have added the following to Sec. 5.1–5.2.
>
> **GMM prior.** K = 16 full-covariance Gaussian components fitted by EM to the standardised descriptor (C₁₁, C₂₂, C₆₆, C₁₂, ρ) of the 156,672 unique catalogue cells. The threshold τ is the 25th percentile of the catalogue log-density. The penalty is the cell-average of max(0, τ − log p) with weight 0.1.
>
> **Diffusion model.** A conditional U-Net denoiser on 64×64 (padded) binary cells: base width 128, channel multipliers (1, 2, 4, 8, 8), one residual block per level, 4-head self-attention at 8×8 and 4×4, dropout 0.1, self-conditioning. It uses a continuous-time cosine log-SNR schedule and x₀-parameterisation. It is conditioned on (C₁₁, C₂₂, C₁₂, C₆₆, volume fraction) through classifier-free guidance, with conditioning dropout 0.1 and guidance scale 2. It was trained on 905,728 unique homogenised cells with Adam (10⁻⁴, cosine), batch 256 and EMA 0.9999, with loss reweighting by inverse volume-fraction density. At inference we draw 15 candidates per cloak cell with 50 sampling steps, homogenise each, and keep the one with the smallest mean relative error over the five targets.
>
> **Inverse design.** A per-cell Fourier-feature MLP density field (4 × 128), initialised from the nearest catalogue cell and optimised with Adam for 350 iterations (2·10⁻³ → 3·10⁻⁴). The projection sharpness is ramped from 1 to 32, followed by a straight-through binary phase.

*[Check the connectivity-penalty sentence, A5.4, before sending.]*

**Comment 1 (d): the gap between η = 0.967 and η = 0.76.**

> In auditing this comparison we found that the two values referred to different sets of generated microstructures. The value 0.967 was obtained with an earlier diffusion model and a lower-fidelity cell homogenisation. The fully resolved simulation used microstructures from the final model, whose homogenised cloak ratio is 0.934. For one consistent set of 100 microstructures at f\* = 2, the chain is:
> - 0.985 with the optimised continuum targets;
> - 0.934 with the homogenised properties of the generated cells;
> - 0.755–0.773 for the fully resolved geometry, not yet mesh-converged and increasing with refinement.
>
> Imperfect property matching therefore costs about 5 points. The main loss, 16–18 points, arises when the explicit geometry replaces the effective medium. A quantitative discussion is given in our response to Reviewer 2, Comment 3, and in the new Sec. [5.5]. The abstract, the Discussion and the Conclusion have been corrected accordingly.

**Comment 2: meaning of "broadband" and implications of the phase-invariant objective.**

> By "broadband" we mean that one fixed design is optimised against, and evaluated over, a continuous frequency band rather than a single frequency. The training band is f\* ∈ [1, 3]: a 3:1 frequency ratio (about 1.6 octaves), or a fractional bandwidth Δf/f_c = 100%. Because the problem is linear and scale-invariant, the band is defined relative to the cloak depth b. For the soil considered (c_s = 300 m/s) and b = 1 m, it corresponds to about 270–800 Hz.
>
> Over this band the design keeps η between 0.92 and 0.99 (mean 0.96). The uncloaked notch falls from about 0.65 at f\* = 1 to 0.31–0.42 for f\* ≥ 1.4. This definition replaces the expression "octave-plus". We have also corrected the definition of f\*: the frequencies were set using the approximate Rayleigh speed c_R ≈ 0.889 c_s, so the nominal f\* = 2 corresponds to b/λ_R = 1.93.
>
> Regarding the objective: Eq. (20) compares displacement magnitudes and is therefore blind to the phase of the transmitted wave. The designs restore the amplitude of the transmitted surface wave but do not control its phase; at the design frequency the single-frequency cloak lags the reference by about 0.18 λ. A phase lag that grows linearly with frequency amounts to a pure time delay, which shifts a pulse without distorting it but would reveal the defect to a time-of-flight measurement. Distortion arises from frequency-dependent amplitude, which η measures, and from phase that is non-linear in frequency, which η does not see. We have therefore restated the scope of the work as restoration of the transmitted surface-wave amplitude (vibration and shadow mitigation), not phase-exact cloaking. See also our response to Reviewer 2, Comment 1.

**Comment 3: mesh convergence.**

> We have added a convergence section [Appendix X].
>
> **Homogenised model.** Linear (TRI3) and quadratic (TRI6) elements converge to the same cloak ratio within 1%. For example, TRI3 gives 0.9748, 0.9805 and 0.9806 at three successive cloak refinements, and an optimised positive-definite design gives η = 0.975 ± 0.001 for meshes from 33k to 78k nodes. Because η is a ratio of magnitudes computed on a shared far-field mesh, far-field dispersion (a phase error) largely cancels. We also found that convergence requires every cell to be positive-definite: a single non-positive-definite cell produces a localised mode that grows with refinement. This is the reason for the constrained material decoding of Sec. 3.2.
>
> **Fully resolved microstructure.** Quadratic elements, voids meshed as traction-free holes, GPU sparse direct solver:
>
> | elements per pixel | nodes | η |
> |---|---|---|
> | 0.51 | 0.47 M | 0.712 |
> | 1.01 | 1.00 M | 0.755 |
> | ≈1.27 | 1.38 M | 0.773 |
>
> The value still increases at the finest mesh that fits in memory, and no defensible extrapolated limit can be obtained from these three points. We therefore report the fully resolved cloak ratio as a lower bound, η ≥ 0.77, and say so in the abstract and Sec. 5.4. The solution is deterministic: repeated runs reproduce η = 0.754965 exactly.

*[OPTIONAL — if run R3: add the convergence table of the final 14×10 and 20×15 designs.]*

**Comment 4: references.**

> - Ref. [4] has been completed (Appl. Phys. Lett. 105 (2) 021901, 2014, doi:10.1063/1.4887454).
> - The DOI has been added to Ref. [9] (doi:10.1016/j.wavemoti.2019.102419).
> - Ref. [23]: our generator implements the cellular-automaton evolution and the connectivity-repair steps of Nakarmi et al. (flood-fill identification of disconnected regions and minimum-spanning-tree bridging). Sec. 5.1 now states explicitly the choices specific to our implementation: mirror assembly of a 25×25 quadrant into a D₂-symmetric 50×50 cell, fixed solid border gates that guarantee tiling, the birth/survival rule set and iteration count, and the sampled initial live fraction.
> - Ref. [30] has been corrected (author list) and [updated to its published version, Smart Mater. Manuf. 4 (2026) 100129 / kept as the arXiv preprint].

*[Before sending: an author checks the Nakarmi PDF for which steps are theirs, and opens the Smart Mater. Manuf. paper to confirm it is the published version of the arXiv preprint.]*

## Reviewer 2

**Comment 1: phase lag; cloaking vs magnitude restoration; time-domain metrics.**

> We agree with the reviewer, and we have revised the scope of the paper accordingly. The objective (20) is phase-invariant, so the designs are amplitude-restoring carpet cloaks for the transmitted Rayleigh wave. They remove the shadow behind the defect and restore the vibration level, but they do not restore the phase. The title and abstract now qualify the term "cloak", and the Discussion states the application boundary explicitly: the designs target vibration and shadow mitigation, not concealment from phase- or time-of-flight-based detection.
>
> Regarding pulsed or broadband excitation: a phase lag that is linear in frequency is a pure time delay and does not distort the waveform, whereas a non-linear phase response and frequency-dependent amplitude do. The broadband objective (21) controls the amplitude at every training frequency but not the phase. The waveform fidelity of the broadband design under pulsed excitation is therefore not guaranteed by the present objective. We now state this as a limitation, and we identify phase-aware objectives (comparing complex fields) as the natural extension.

*[OPTIONAL — if run R2: "We have added a time-domain test (Fig. X): a Ricker pulse centred at f\* = 2 synthesised from the frequency-domain solutions; normalised cross-correlation with the reference waveform after/without time alignment: …; arrival delay: …; transmitted energy ratio: …". **Of all the optional runs, this one is the most likely to be requested again if omitted.**]*

**Comment 2: why untrained frequencies perform well; number of training frequencies.**

> We thank the reviewer for this question. Our evidence points to the smoothness of the physical response in frequency, combined with a sampling interval that is fine enough relative to it. It does not point to the neural-field prior:
>
> 1. **The response is smooth in frequency.** All materials are non-dispersive (frequency-independent stiffness and density, no resonant inclusions), and the problem is open: energy radiates into the half-space. There are therefore no high-Q resonances, and η(f) varies smoothly. Even the uncloaked notch varies on a scale of roughly 0.3–0.5 in f\* (Fig. 9). The training frequencies are spaced Δf\* = 1/3 apart.
> 2. **The sampling is adequate but not dense.** The dense sweep in Fig. 9 (step 0.1; 18 of its 21 in-band points are untrained) shows its lowest in-band values, about 0.92 at f\* = 1.2 and 0.94 at f\* = 2.2, between training frequencies. The response sags slightly between samples, with no narrowband dips.
> 3. **The neural-field prior alone does not give bandwidth.** Using the same parameterisation, a design trained on a single frequency is strongly narrowband. For the 20×15 single-frequency design, η = 0.96 at f\* = 2.0 but 0.78 at 1.7, 0.72 at 2.3, and 0.58–0.61 at f\* = 1.5 and 2.5. The spatial smoothness of the field therefore does not produce the broadband behaviour; including the band in the objective does.
>
> This single-frequency case is also one end of the trade-off the reviewer asks about: one training frequency gives near-perfect cloaking in a narrow band, while seven give η ≥ 0.92 across [1, 3]. The cost per optimisation step grows linearly with the number of frequencies. We have added this discussion to Sec. 4.4.

*[OPTIONAL — if run R6: add "Retraining with 3 and 5 frequencies gives … (Table X)".]*

**Comment 3: attribution of the gap, and a field plot of the fully resolved cloak.**

> We agree that the previous explanation was insufficient, and we have added a new Sec. [5.5]. First, a correction (see also Reviewer 1, Comment 1d): for one consistent set of microstructures the gap is 0.934 (homogenised) → 0.755–0.773 (fully resolved, not yet mesh-converged), not 0.967 → 0.76. The decomposition at f\* = 2 is:
>
> | Model of the cloak | η |
> |---|---|
> | Optimised continuum targets | 0.985 |
> | Homogenised generated microstructures | 0.934 |
> | Fully resolved microstructures (1.0 / 1.27 el/px) | 0.755 / 0.773 |
>
> Imperfect property matching (mean per-cell error 8.7%) costs about 5 points. The main loss arises when the explicit geometry replaces the effective medium.
>
> We have **ruled out** two candidate explanations:
> - **(i) The void model.** Re-homogenising catalogue cells with the voids meshed as traction-free holes, instead of a 10⁻⁶ ersatz material, changes the effective constants by a median of 10⁻⁵ relative and at most about 1% (C₆₆ of the thinnest cells).
> - **(ii) Internal resonance.** The 6.7 cm unit cell is about 1.5% of the shear wavelength in concrete at the operating frequency, so the cells respond quasi-statically.
>
> The **remaining mechanisms** are those that first-order, cell-by-cell homogenisation does not represent:
> - **(a) Limited scale separation inside the cloak.** The cell size is 0.06–0.32 of the local shear wavelength in the cloak, i.e. 3–17 cells per wavelength.
> - **(b) The free-surface boundary layer.** 22 of the 100 cells lie within half a Rayleigh wavelength of the surface.
> - **(c) Interfaces between dissimilar neighbouring cells.** All cells share the same solid gates on their edges, so the load path is continuous; 399 of 400 edges are intact. However, 34% of the selected cells' edges carry additional material next to the gate, compared with 18% in the catalogue. Where such a cell meets one without it, the contact differs from what each cell's periodic homogenisation assumes.
>
> **Three numerical observations** support this attribution:
> - **The error has a definite sign.** Removing 2,500 randomly chosen concrete pixels (1% of the cloak) raises the fully resolved η in each of four trials (0.763–0.780 vs 0.755). Adding the same number lowers it in each of four trials (0.732–0.750). The assembled structure behaves stiffer and/or heavier than its homogenised description.
> - **The result is robust to small random defects.** Flipping 0.01–0.1% of the cloak pixels at random changes η by at most 0.006.
> - **Microscale detail controls the resolved response.** Changing 0.09% of the pixels (232) with an adjoint-gradient-ranked procedure raises the fully resolved η from 0.755 to 0.804.
>
> We have not yet separated contributions (a)–(c) quantitatively and state this explicitly. Reducing the gap through dynamic or interface-aware homogenisation is identified as the main direction for future work.

*[Field plot: the reviewer "strongly suggests" it. It needs run R4: a field-export option plus one ~90 s solve. If it is not run, add: "A displacement-field comparison requires storing the 4-million-DOF fully resolved solution, which our current validation code does not do; we will add it [in the final version / as supplementary material]." **We recommend running R4: it is cheap and directly requested.**]*

*[OPTIONAL, author decision: mention post-submission pixel-level optimisation of the resolved cloak, which reached η = 0.866 on the original 20×15 grid (fine-tuned diffusion cells → 232 pixel flips → level-set boundary optimisation). This shows the gap is largely recoverable at the microscale. It is new work, though, and invites further review.]*

---

# Part C: still open before sending

| # | Item | Who / how |
|---|---|---|
| 1 | Decide on A1 (switch to 0.934 / ≥0.77; recommended), or run R1 to validate the 0.967 cells | authors |
| 2 | Which run gave the 1×1 η = 0.857, and with which R and κ | m3l machine |
| 3 | Loss histories for 14×10 single-frequency and broadband; which broadband variant (`nogmm` / `gmm` / `restart`) is in the paper; Fig. 9 CSV | m3l machine |
| 4 | How the 0.18λ / 65° phase lag was measured | the author who made `single_freq_14x10_zoom_phase.png` |
| 5 | Was the connectivity penalty on in inverse design? | the author who ran it |
| 6 | Check the Nakarmi PDF for [23]; confirm the Smart Mater. Manuf. paper is [30]'s published version | authors |
| 7 | Run R4 (field plot), recommended; R2 (time domain), advisable | `docs/reviewer_response_runs.md` |

**If no runs are made at all:** every comment still gets a substantive answer. Two requests stay unmet: R2's field plot (R2.3) and the time-domain metrics (R2.1). Both reviewers rated the paper minor revision (R1 explicitly; R2 says "further strengthen"), so honest scope statements are defensible. R2.3's field plot is the one most worth the 90 seconds.
