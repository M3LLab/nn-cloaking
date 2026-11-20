# Response to reviewers: comment-by-comment letter

**Paper:** *Neural-field design of broadband Rayleigh-wave carpet cloaks under microstructure realisability constraints* (Wave Motion)
**Draft:** 2026-11-21

> **Internal: delete this box before sending.**
>
> - **The main assumption.** This version assumes the homogenised η ≈ 0.967 has been rerun with the hi-fi homogeniser on another server. It also assumes that run uses **the same 100 microstructures** as the fully resolved η ≈ 0.76.
>   - Check: the canvas fingerprint of that run's cell folder should be `4cd545facddc`, the same cells as the 0.76.
>   - Fill the two `[from the 0.96 run]` placeholders from that run.
>   - If the assumption fails, use the 0.934-based wording in `docs/reviewer_response_final.md` instead.
> - **What is unchanged.** All other numbers come from files in this repository. Evidence for each is in `docs/reviewer_response_draft.md`.
> - **Paper fixes.** The paper corrections that the answers announce are listed in `docs/reviewer_response_final.md`, Part A3–A6: network size, R/κ, initialisation sentence, c_R, references. They are needed in any case.
> - **Optional inserts.** `[OPTIONAL — if run Rx]` marks text to add only if that run is made; see `docs/reviewer_response_runs.md`.

---

We thank both reviewers for their careful reading and constructive comments. We have revised the manuscript accordingly; changes are marked in blue. Our point-by-point responses follow, with each comment reproduced in italics.

---

## Reviewer 1

### Comment 1.1

> *The authors should clarify the optimization procedure, including learning rate, number of iterations, convergence and stopping criteria, …*

**Response.** All neural-field optimisations use Adam (β₁ = 0.9, β₂ = 0.999) with a cosine learning-rate decay from 5·10⁻⁴ to 10⁻⁵ over a fixed iteration budget:

- 1000 iterations for the single-frequency 14×10 design;
- 3000 for the broadband design;
- 2000 for the realisability-constrained 20×15 design.

No early-stopping rule is used. The iterate with the lowest loss is retained, and convergence is judged from the loss history. For the constrained 20×15 design, the loss falls from 0.553 at the start to 7.3·10⁻³ after 250 iterations and 8.6·10⁻⁴ after 1000. It reaches 7.08·10⁻⁴ at iteration 2000, only 4% lower than 500 iterations earlier.

The network is a multilayer perceptron with five hidden layers of 512 tanh units. It is fed deterministic Fourier features sin(kx), cos(kx), sin(ky), cos(ky) for k = 1, …, 64. The last layer is initialised at 1% scale, so that optimisation starts from the prescribed initial material.

All reported runs use an anisotropy cap R = 12 and a coupling bound κ = 0.95. We have corrected the values given in the previous version of the text and apologise for the inconsistency. All settings are now collected in a new Table [X].

*[If the loss histories of the 14×10 and broadband runs are retrieved from the m3l machine, add them to the appendix.]*

### Comment 1.2

> *… and provide a brief assessment of sensitivity to initialization.*

**Response.** We thank the reviewer for this point, and we have corrected our statement.

- All reported cell-resolved designs (46 and 100 cells) start from the mean material of the microstructure catalogue.
- The objective is non-convex. In the single-material case (five design variables), different network seeds converged to different local optima, and the reported design was obtained by perturbed warm restarts.

We have replaced the earlier sentence, which claimed insensitivity to initialisation, with this description. A systematic multi-start study is now listed as a limitation.

*[OPTIONAL — if run R5: replace the last sentence with the spread of η over 3 seeds × 2 initialisations.]*

### Comment 1.3

> *The parameters and implementation details of the GMM/diffusion-based microstructure generation should be reported more explicitly, …*

**Response.** We have added the following to Sec. 5.1–5.2.

- **GMM prior.**
  - K = 16 full-covariance Gaussian components, fitted by expectation–maximisation to the standardised descriptor (C₁₁, C₂₂, C₆₆, C₁₂, ρ) of the 156,672 unique catalogue cells.
  - The threshold τ is the 25th percentile of the catalogue log-density.
  - The penalty is the cell average of max(0, τ − log p), with weight 0.1.
- **Diffusion model.**
  - Architecture: a conditional U-Net denoiser on 64×64 (padded) binary cells, with base width 128 and channel multipliers (1, 2, 4, 8, 8). It has one residual block per level, 4-head self-attention at 8×8 and 4×4, dropout 0.1, and self-conditioning.
  - Noise schedule: continuous-time cosine log-SNR, with x₀-parameterisation.
  - Conditioning: on (C₁₁, C₂₂, C₁₂, C₆₆, volume fraction) through classifier-free guidance, with conditioning dropout 0.1 and guidance scale 2.
  - Training: 905,728 unique homogenised cells; Adam (10⁻⁴, cosine decay), batch 256, EMA 0.9999, and loss reweighting by inverse volume-fraction density.
  - Inference: 15 candidates per cloak cell, 50 sampling steps each. Every candidate is homogenised, and the one with the smallest mean relative error over the five targets is kept.
- **Inverse design.**
  - A per-cell Fourier-feature MLP density field (4 × 128), initialised from the nearest catalogue cell.
  - Optimised with Adam for 350 iterations (2·10⁻³ → 3·10⁻⁴).
  - The projection sharpness is ramped from 1 to 32, followed by a straight-through binary phase.

*[Internal: check whether the connectivity penalty was on in inverse design before keeping any sentence about it. The script default is off.]*

### Comment 1.4

> *… and the discrepancy between the homogenized efficiency (η≈0.967) and the fully resolved microstructure result (η≈0.76) should be discussed quantitatively.*

**Response.** We agree, and we have added a new Sec. [5.5] with a quantitative breakdown. For the realised design at f\* = 2, the cloak ratio for each level of modelling is:

| Model of the cloak | η |
|---|---|
| Optimised continuum targets | 0.985 |
| Homogenised generated microstructures | 0.967 |
| Fully resolved microstructures, 1.0 / 1.27 elements per pixel | 0.755 / 0.773 |

What each step costs:

- **Property matching.** The generated cells match their targets with a mean per-cell error of [X% — from the 0.96 run]. This costs only about 2 points of η.
- **Replacing the effective medium with the explicit geometry.** This accounts for almost all of the loss, about 20 points.

The mechanisms responsible, and the numerical tests behind them, are given in our response to Reviewer 2, Comment 3.

### Comment 2

> *The meaning and bandwidth associated with the term "broadband" should also be clarified, together with the implications of using a phase-invariant displacement-magnitude objective.*

**Response.**

**Meaning and bandwidth.** By "broadband" we mean that a single fixed design is optimised against, and evaluated over, a continuous frequency band rather than a single frequency.

- The training band is f\* ∈ [1, 3]: a 3:1 frequency ratio (about 1.6 octaves), or a fractional bandwidth Δf/f_c of 100%.
- Because the problem is linear and scale-invariant, the band is defined relative to the cloak depth b. For the soil considered (c_s = 300 m/s) and b = 1 m, it corresponds to about 270–800 Hz.
- Over this band the design keeps η between 0.92 and 0.99, with a mean of 0.96. The uncloaked notch falls from about 0.65 at f\* = 1 to 0.31–0.42 for f\* ≥ 1.4.

This definition replaces the expression "octave-plus". We have also corrected the definition of f\*: frequencies were set using the approximate Rayleigh speed c_R ≈ 0.889 c_s, so the nominal f\* = 2 corresponds to b/λ_R = 1.93.

**Implications of the phase-invariant objective.** Eq. (20) compares displacement magnitudes, so it is blind to the phase of the transmitted wave.

- The designs restore the amplitude of the transmitted surface wave but do not control its phase. At the design frequency, the single-frequency cloak lags the reference by about 0.18 λ.
- A phase lag that grows linearly with frequency is a pure time delay. It shifts a pulse without distorting it, but it would reveal the defect to a time-of-flight measurement.
- Distortion comes from two sources: amplitude that varies with frequency, which η measures, and phase that is non-linear in frequency, which η does not see.

We have therefore restated the scope of the work as restoration of the transmitted surface-wave amplitude (vibration and shadow mitigation), not phase-exact cloaking. See also our response to Reviewer 2, Comment 1.

*[Internal: confirm how the 0.18 λ was measured before sending; no script in the repository computes it.]*

### Comment 3

> *A mesh-convergence or numerical-resolution check would strengthen the validation.*

**Response.** We have added a convergence appendix [Appendix X].

**Homogenised model.**

- Linear (TRI3) and quadratic (TRI6) elements converge to the same cloak ratio within 1%.
  - TRI3 gives 0.9748, 0.9805 and 0.9806 at three successive cloak refinements.
  - An optimised positive-definite design gives η = 0.975 ± 0.001 for meshes from 33k to 78k nodes.
- η is a ratio of magnitudes computed on a shared far-field mesh, so far-field dispersion (a phase error) largely cancels out of it.
- Convergence requires every cell to be positive-definite: a single non-positive-definite cell produces a localised mode that grows with refinement. This is the reason for the constrained material decoding of Sec. 3.2.

**Fully resolved microstructure.** The model uses quadratic elements, voids meshed as traction-free holes, and a GPU sparse direct solver:

| Elements per pixel | Nodes | η |
|---|---|---|
| 0.51 | 0.47 M | 0.712 |
| 1.01 | 1.00 M | 0.755 |
| ≈1.27 | 1.38 M | 0.773 |

- The reported value η ≈ 0.76 is the 1-element-per-pixel result.
- The value increases with refinement, so it is conservative: the finest mesh that fits in memory gives 0.773.
- The solution is deterministic; repeated runs reproduce η = 0.754965 exactly.

*[OPTIONAL — if run R3: add the convergence table for the final 14×10 and 20×15 designs.]*

### Comment 4

> *Finally, the authors should carefully cross-check the reference list and citation support, including completing Ref. [4], adding the DOI to Ref. [9], verifying that Ref. [23] directly supports the specific cellular-automaton procedure used, and updating Ref. [30] if a peer-reviewed version is available.*

**Response.** We have cross-checked the entire reference list.

- **Ref. [4]** has been completed: Appl. Phys. Lett. 105 (2) 021901, 2014, doi:10.1063/1.4887454.
- **Ref. [9]**: the DOI has been added, doi:10.1016/j.wavemoti.2019.102419.
- **Ref. [23]**:
  - Our generator implements the cellular-automaton evolution and the connectivity-repair steps of Nakarmi et al., namely flood-fill identification of disconnected regions and minimum-spanning-tree bridging.
  - Sec. 5.1 now states explicitly the choices specific to our implementation: mirror assembly of a 25×25 quadrant into a D₂-symmetric 50×50 cell, fixed solid border gates that guarantee tiling, the birth/survival rule set and iteration count, and the sampled initial live fraction.
- **Ref. [30]**: the author list has been corrected, and the entry [updated to its published version, Smart Mater. Manuf. 4 (2026) 100129 / kept as the arXiv preprint].

*[Internal: an author checks the Nakarmi PDF for which steps are theirs, and opens the Smart Mater. Manuf. paper to confirm it is the published version of the arXiv preprint.]*

---

## Reviewer 2

### Comment 1

> *Equation (20) solely constrains the displacement magnitude, which, as the authors acknowledge, results in an optimized cloak with a phase lag of approximately 0.18λ. … for broadband or pulse excitations (such as seismic waves), the phase lag will vary across frequencies, which could potentially lead to severe waveform distortion. We suggest the authors explicitly define the application boundaries of this work (i.e., whether it is strictly "cloaking" or "magnitude restoration"). Furthermore, … we recommend providing post-hoc metrics such as the time-domain impulse response or broadband waveform fidelity.*

**Response.** We agree with the reviewer and have revised the scope of the paper accordingly.

**Application boundaries.** The objective (20) is phase-invariant, so the designs are amplitude-restoring carpet cloaks for the transmitted Rayleigh wave. They remove the shadow behind the defect and restore the vibration level, but they do not restore the phase. Accordingly:

- the title and abstract now qualify the term "cloak";
- the Discussion states the application boundary explicitly: vibration and shadow mitigation, not concealment from phase- or time-of-flight-based detection.

**Pulsed or broadband excitation.**

- A phase lag that is linear in frequency is a pure time delay and does not distort the waveform. A non-linear phase response, and amplitude that varies with frequency, do distort it.
- The broadband objective (21) controls the amplitude at every training frequency but not the phase. The present objective therefore does not guarantee the waveform fidelity of the broadband design under pulsed excitation.

We now state this as a limitation, and identify phase-aware objectives that compare complex fields as the natural extension.

*[OPTIONAL — if run R2: "We have added a time-domain test (Fig. X). A Ricker pulse centred at f\* = 2 is synthesised from the frequency-domain solutions. Normalised cross-correlation with the reference waveform, after / without time alignment: …; arrival delay: …; transmitted energy ratio: …." Of the optional runs, this one is the most likely to be requested again if omitted.]*

### Comment 2

> *In Sec. 4.4, the optimization is performed using only 7 uniformly spaced frequencies, yet the dense sweep demonstrates excellent performance across the entire band. … We suggest adding a brief discussion on whether this is attributed to the physical continuity of the system's response, the spatial smoothness prior of the neural field suppressing narrowband resonances, or simply because 7 sampling points are sufficient to cover the band. Additionally, a sensitivity analysis on sampling density (e.g., how much performance drops if reduced to 3 or 5 frequencies) would greatly help readers …*

**Response.** We thank the reviewer for this question. Our evidence points to the smoothness of the physical response in frequency, combined with a sampling interval that is fine enough to resolve it. It does not point to the neural-field prior.

1. **The response is smooth in frequency.**
   - All materials are non-dispersive: stiffness and density do not depend on frequency, and there are no resonant inclusions.
   - The problem is open, since energy radiates into the half-space. There are therefore no high-Q resonances, and η(f) varies smoothly.
   - Even the uncloaked notch varies on a scale of roughly 0.3–0.5 in f\* (Fig. 9), whereas the training frequencies are spaced Δf\* = 1/3 apart.
2. **The sampling is adequate but not dense.**
   - The dense sweep in Fig. 9 has a step of 0.1; 18 of its 21 in-band points are untrained.
   - Its lowest in-band values, about 0.92 at f\* = 1.2 and 0.94 at f\* = 2.2, fall between training frequencies.
   - The response sags slightly between samples, with no narrowband dips.
3. **The neural-field prior alone does not produce bandwidth.**
   - With the same parameterisation, a design trained at a single frequency is strongly narrowband.
   - For the 20×15 single-frequency design, η = 0.96 at f\* = 2.0 but 0.78 at 1.7, 0.72 at 2.3, and 0.58–0.61 at f\* = 1.5 and 2.5.
   - So the spatial smoothness of the field does not produce the broadband behaviour; including the band in the objective does.

This single-frequency case is also one end of the trade-off the reviewer asks about. One training frequency gives near-perfect cloaking in a narrow band, while seven give η ≥ 0.92 across [1, 3]. The cost per optimisation step grows linearly with the number of frequencies. We have added this discussion to Sec. 4.4.

*[OPTIONAL — if run R6: add "Retraining with 3 and 5 frequencies gives … (Table X)."]*

### Comment 3

> *There is a significant gap of approximately 20% between the homogenized back-filling result (η=0.967) and the direct FEM simulation of the fully-resolved solid/void geometry (η=0.76). … We recommend the authors at least provide a preliminary attribution for this gap. Furthermore, we strongly suggest supplementing a displacement field plot of the fully-resolved cloak (compared against the homogenized and reference fields) to intuitively visualize where the performance loss occurs.*

**Response.** We agree that the previous explanation was insufficient, and we have added a new Sec. [5.5]. The decomposition at f\* = 2, for the same 100 microstructures, is:

| Model of the cloak | η |
|---|---|
| Optimised continuum targets | 0.985 |
| Homogenised generated microstructures | 0.967 |
| Fully resolved microstructures, 1.0 / 1.27 elements per pixel | 0.755 / 0.773 |

Imperfect property matching (mean per-cell error [X% — from the 0.96 run]) costs only about 2 points. The gap the reviewer identifies therefore arises almost entirely when the explicit geometry replaces the effective medium.

**Two candidate explanations ruled out.**

- **(i) The void model.** We re-homogenised catalogue cells with the voids meshed as traction-free holes instead of a 10⁻⁶ ersatz material. The effective constants change by a median of 10⁻⁵ (relative), and by at most about 1% (C₆₆ of the thinnest cells).
- **(ii) Internal resonance.** The 6.7 cm unit cell is about 1.5% of the shear wavelength in concrete at the operating frequency, so the cells respond quasi-statically.

**Remaining mechanisms.** These are the effects that first-order, cell-by-cell homogenisation does not represent:

- **(a) Limited scale separation inside the cloak.** The cell size is 0.06–0.32 of the local shear wavelength in the cloak, i.e. 3–17 cells per wavelength.
- **(b) The free-surface boundary layer.** 22 of the 100 cells lie within half a Rayleigh wavelength of the surface.
- **(c) Interfaces between dissimilar neighbouring cells.**
  - All cells share the same solid gates on their edges, so the load path is continuous; 399 of 400 edges are intact.
  - However, 34% of the selected cells' edges carry additional material next to the gate, compared with 18% in the catalogue.
  - Where such a cell meets one without it, the contact differs from what each cell's periodic homogenisation assumes.

**Three numerical observations support this attribution.**

- **The error has a definite sign.** Removing 2,500 randomly chosen concrete pixels (1% of the cloak) raised the fully resolved η in each of four trials (0.763–0.780, against 0.755). Adding the same number lowered it in each of four trials (0.732–0.750). The assembled structure therefore behaves stiffer and/or heavier than its homogenised description.
- **The result is robust to small random defects.** Flipping 0.01–0.1% of the cloak pixels at random changes η by at most 0.006.
- **Microscale detail controls the resolved response.** Changing 0.09% of the pixels (232) with an adjoint-gradient-ranked procedure raises the fully resolved η from 0.755 to 0.804.

We have not yet separated the contributions of (a)–(c) quantitatively, and we state this explicitly. Reducing the gap through dynamic or interface-aware homogenisation is identified as the main direction for future work.

**Field plot.** [Choose one:]

- *[If run R4 is made, recommended (one ~90 s solve plus a field-export option):]* "Fig. [X] compares the displacement magnitude of the reference, homogenised and fully resolved cloaks on a common colour scale, together with the surface profile |u|/|u_ref|. The loss is concentrated in …"
- *[If not run:]* "A displacement-field comparison requires storing the 4-million-degree-of-freedom fully resolved solution, which our current validation code does not do. We will add it [in the final version / as supplementary material]."

*[OPTIONAL, author decision: mention the post-submission pixel-level optimisation of the resolved cloak, which reached η = 0.866 on the original 20×15 grid (fine-tuned diffusion cells → 232 pixel flips → level-set boundary optimisation). It shows the gap is largely recoverable at the microscale, but it is new work and invites further review.]*

---

We thank the reviewers again for their comments, which have substantially improved the clarity and scope of the manuscript.
