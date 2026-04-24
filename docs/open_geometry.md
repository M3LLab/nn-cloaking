# Open-Geometry Cloak Optimisation

Joint optimisation of **cloak shape** (pixel mask on a fixed cell grid) and **cell material** `(C_eff, ρ_eff)`. The cloak footprint is no longer fixed to the triangular geometry — it is initialised from the triangle and then free to evolve.

The defect (the hole being hidden) stays fixed in both flows: it is cut from the FEM mesh once via `extract_submesh`. Only the *cloak-vs-background* assignment of the surviving cells is trainable.

## Usage

```bash
python run_optimize_open_geometry.py                           # configs/open_geometry.yaml
python run_optimize_open_geometry.py configs/my_config.yaml    # explicit config
```

The config is read as a normal `SimulationConfig` (same schema as `run_optimize.py`) plus an optional top-level `shape_opt:` block with shape-specific hyperparameters.

---

## The mental model

### The setup

Imagine a **grid of cells** (e.g. 4×4 below) with the triangular cloak drawn over it:

```
 . . . .
 . T T .    T = cell centre lies inside the triangle
 . T T .    . = cell centre lies outside
 . . . .
```

Each cell holds two numbers — stiffness `C` and density `ρ` — fed to the FEM. There are two "reference" values:

- **cloak material** `(C_cloak, ρ_cloak)` — the push-forward from transformation elasticity.
- **background** `(C₀, ρ₀)` — homogeneous surroundings.

### How `run_optimize.py` treats the triangle

The triangle is **hardcoded**. Cells inside it start at `(C_cloak, ρ_cloak)` and are trained; cells outside start at `(C₀, ρ₀)` and are pinned there by the L2 drift regulariser and tiny gradients.

> **Result:** only the *values* inside the triangle evolve. The *shape* of the cloaked region is frozen to the triangle forever. If the optimum is a circle, this flow can't find it.

Analogy: "the painted region is fixed to a triangle; you can only tune the paint colour."

### How `run_optimize_open_geometry.py` treats the triangle

Every cell gets **one extra trainable number** — a logit `s`. A sigmoid turns it into an **occupancy** `m = σ(β · s) ∈ [0, 1]`:

```
 s = +3   →  m ≈ 0.95   →  cell behaves like its trained (C, ρ)      (cloak)
 s = −3   →  m ≈ 0.05   →  cell behaves like (C₀, ρ₀)               (background)
 s =  0   →  m = 0.50   →  half-and-half                            (edge)
```

The FEM sees a **blend**:

```
C_effective[i] = m[i] · C[i] + (1 − m[i]) · C₀
ρ_effective[i] = m[i] · ρ[i] + (1 − m[i]) · ρ₀
```

Logits are **initialised from the triangle**:

```
 s[i] = +init_magnitude    if cell centre is inside the triangle
 s[i] = −init_magnitude    otherwise
```

so at step 0 the FEM sees (approximately) the same thing as the triangular-cloak flow. But every logit is now a knob Adam can turn.

**What the optimiser can now do that it couldn't before:**

1. Push a logit *inside* the triangle below 0 → punches a hole in the cloak.
2. Push a logit *outside* the triangle above 0 → grows the cloak beyond the triangle.
3. Fine-tune the `(C, ρ)` values of any cell with meaningful occupancy.

Analogy: "the mask decides *where* paint goes; the materials decide *what colour*; the triangle was just a starting guess — both are adjustable."

### Why a sigmoid instead of a 0/1 switch

A hard switch has no derivative, so Adam can't train it. The sigmoid provides a smooth knob: `s` can drift `0.5 → 0.6 → 0.7 → …` and the optimiser gets a continuous gradient signal the whole way. That is the entire point of "differentiable parameterisation".

---

## The per-step forward pipeline

Each training step, the raw logits are turned into an effective per-cell material through this pipeline — every stage is optional and disabled by default, but they compose cleanly:

```
   shape_logits s                            (trainable, one per cell)
         │
         ▼  smooth_sigma > 0:
   separable Gaussian (σ cells)              ← differentiable filter;
         │                                     suppresses single-cell islands
         ▼
   σ(β_t · s)           β_t = linear ramp    ← β-continuation for
         │              beta → beta_end        sharper masks without
         ▼                                     killing gradients
   occupancy m ∈ [0, 1]
         │
         ▼  simp_p > 1:
   m^p                                       ← SIMP: makes grey values
         │                                     disproportionately costly
         ▼
   blend with (C_cell, ρ_cell):
      C_eff = m^p · C_cell + (1 − m^p) · C₀
      ρ_eff = m^p · ρ_cell + (1 − m^p) · ρ₀
         │
         ▼
     FEM forward solve
         │
         ▼
   loss = L_cloak
        + λ_l2·L_drift  + λ_neighbor·L_mat_TV     (raw material regs)
        + λ_mask·L_logit_TV                       (mask smoothness)
        + λ_bin·mean(m·(1−m))                     (binarisation pull)
```

Defaults (`simp_p = 1, smooth_sigma = 0, lambda_bin = 0, beta_end = null`) collapse this to the vanilla "logit → sigmoid → convex blend" of the first version — so turning knobs on is always opt-in.

**After training** (orthogonal to the gradient path):

```
   final soft mask (from the pipeline above)
         │
         ▼  project_final = true:
   threshold at 0.5 → scipy.ndimage.label → keep largest component
         │
         ▼
   hard binary mask  →  saved as optimized_params_projected.npz
                        + shape_mask_projected.png
```

The projection is not differentiable — it's a manufacturability cleanup, not a training step.

---

## Side-by-side: what each flow does to the triangle

| | `run_optimize.py` | `run_optimize_open_geometry.py` |
|---|---|---|
| **Mesh** | Defect cut from mesh; triangle outer boundary has no special role in the mesh | Same |
| **Cells inside triangle** | Start at `(C_cloak, ρ_cloak)`; material trainable | Same material init; *plus* logit `s = +init_magnitude` |
| **Cells outside triangle** | Start at `(C₀, ρ₀)`; technically trainable but pinned by regs | Same material init; *plus* logit `s = −init_magnitude` |
| **Trainable arrays** | `cell_C_flat`, `cell_rho` | `cell_C_flat`, `cell_rho`, **`shape_logits`** |
| **What FEM sees** | `(C_cell, ρ_cell)` directly | Blended `(m·C_cell + (1−m)·C₀, m·ρ_cell + (1−m)·ρ₀)` |
| **Can move the defect?** | No | No |
| **Can change outer cloak boundary?** | **No** | **Yes** |
| **Can punch holes in the cloak?** | Not really (regs fight it) | **Yes** (logit drops) |
| **Can extend the cloak outside the triangle?** | No | **Yes** (logit rises) |
| **Role of the triangle** | A permanent, geometric constraint | A warm-start for the mask |

One line: **`run_optimize.py` trusts the triangle. `run_optimize_open_geometry.py` starts from it, then optimises around it.**

---

## Loss

```
L  =  L_cloak                                # existing — boundary L2 to reference
   +  λ_l2       · L_l2_drift                # existing — on raw cell materials
   +  λ_neighbor · L_material_tv             # existing — on raw cell materials
   +  λ_mask     · L_mask_tv                 # TV on logits (perimeter-like)
   +  λ_bin      · mean( m · (1 − m) )       # binarisation penalty (opt-in)
```

The two existing material regularisers act on the **raw** cell values (not the blended ones), so they retain their original meaning. `L_mask_tv` is a mean-squared difference of logits across 4-connected neighbour pairs — a cheap perimeter-like penalty that discourages speckled masks. `L_bin` is zero at `m ∈ {0, 1}` and peaks at `m = 0.5`; enable it (set `lambda_bin > 0`) together with a β ramp or SIMP to get sharp masks.

## Topology & binarisation

Plain sigmoid masks have two well-known failure modes. The pipeline ships with standard levers for each; defaults preserve the vanilla behaviour, and all knobs are turned on by editing `shape_opt:` — no code changes.

### Connectivity — single connected body

A naive mask can produce disconnected islands. Two mechanisms address this:

- **`smooth_sigma` — differentiable smoothing during training.** A separable Gaussian filter (σ in cell units) is applied to the logits *before* the sigmoid. This suppresses thin one-cell features that become disconnected after binarisation. Small values (1–2 cells) are usually enough. Fully differentiable; gradients flow through it.
- **`project_final` + `project_connectivity` — post-hoc cleanup.** After training, the largest 4- or 8-connected component of `mask > 0.5` is extracted and saved as `optimized_params_projected.npz` + `shape_mask_projected.png`. This is a hard projection (not differentiable), suitable for a "manufacturable" export or for warm-starting a next run that trains around the projected shape.

### Binarisation — push `m` toward 0 or 1

A mid-valued `m ≈ 0.5` cell isn't physically realisable. Three complementary levers:

- **β ramp (`beta` → `beta_end`).** β is the sigmoid temperature. At a high β, the sigmoid is nearly a step function — but its gradient vanishes everywhere except near the edge, which makes starting at high β untrainable. Standard remedy: **β-continuation** — start at `β = 1` (soft, good gradients), linearly ramp to `β = 6…16` over `n_iters`. Each intermediate β still has live gradients on the cells currently near the edge.
- **SIMP exponent (`simp_p`).** Raises occupancy to a power before blending: `C_eff = m^p · C + (1 − m^p) · C₀`. Intermediate `m` values contribute disproportionately little stiffness per unit "mass cost", so the optimiser prefers extremes on its own. `p = 3` is the classic choice; `p = 1` disables.
- **Binarisation penalty (`lambda_bin`).** Adds `λ · mean(m · (1 − m))` to the loss. Small gradients by construction — use *together* with a β ramp, not instead of one. `λ = 1e-2` to `1e-1` is a reasonable range.

The β ramp is the single biggest win. Add SIMP and `lambda_bin` if the final mask is still too grey.

### Recommended recipes

Two practical starting points:

```yaml
# A) Clean, connected mask — conservative
shape_opt:
  beta: 1.0
  beta_end: 6.0
  smooth_sigma: 1.0
  lambda_mask_smooth: 1.0e-2
  simp_p: 1.0
  lambda_bin: 0.0
  project_final: true
```

```yaml
# B) Aggressive topology-opt style (binary target)
shape_opt:
  beta: 1.0
  beta_end: 10.0
  smooth_sigma: 1.5
  lambda_mask_smooth: 5.0e-3
  simp_p: 3.0
  lambda_bin: 3.0e-2
  project_final: true
```

Chain runs via the warm-start: run (A) for 300 steps, then warm-start a second run from `optimized_params.npz` with recipe (B).

## Pipeline reuse

Nothing in `problem.py`, `cells.py`, `solver.py`, `optimize.py`, or `config.py` changes. `RayleighCloakProblem.set_params` expects `(cell_C_flat, cell_rho)` of a given shape; the open-geometry loop hands it `apply_shape_mask(...)` output with those exact shapes.

---

## Files

| File | Role |
|---|---|
| [`rayleigh_cloak/shape_mask.py`](../rayleigh_cloak/shape_mask.py) | Pure differentiable mask: init, sigmoid, blend, TV, neighbour enumeration. No FEM deps. |
| [`rayleigh_cloak/open_geometry.py`](../rayleigh_cloak/open_geometry.py) | `ShapeOptConfig`, `solve_optimization_open_geometry`, `run_optimization_open_geometry`. |
| [`run_optimize_open_geometry.py`](../run_optimize_open_geometry.py) | CLI: init, optimise, save mask, plot evolution. |
| [`configs/open_geometry.yaml`](../configs/open_geometry.yaml) | Example config based on `cell_based.yaml` + a `shape_opt:` section. |

## Config — the `shape_opt` block

Read directly by [`open_geometry.py`](../rayleigh_cloak/open_geometry.py); fields that are missing fall back to dataclass defaults. All keys are optional.

| Field | Default | Meaning |
|---|---|---|
| `beta` | `1.0` | Sigmoid sharpness at step 0 (or constant if `beta_end` is null). |
| `beta_end` | `null` | If set, β ramps linearly from `beta` → `beta_end` across `n_iters`. Standard β-continuation; the single biggest knob for sharp masks. Try `beta_end: 6`–`10`. |
| `init_magnitude` | `3.0` | \|logit\| at init. `σ(β · magnitude) ≈ 0.95` with the defaults — soft enough to keep gradients alive, decisive enough that the initial forward pass is approximately the triangular cloak. |
| `logits_lr_mult` | `1.0` | Effective logit LR = `optimization.lr · logits_lr_mult`. Useful if the materials (Pa scale) and logits (O(1) scale) want different step sizes. |
| `lambda_mask_smooth` | `1.0e-2` | Weight on the mask-TV regulariser (penalises logit variation across neighbours). Too small → speckled masks; too large → mask freezes near the initial triangle. |
| `lambda_bin` | `0.0` | Weight on `mean(m · (1 − m))`. Opt-in binarisation penalty; use together with a β ramp. |
| `simp_p` | `1.0` | SIMP exponent applied to `m` before the material blend. `p = 3` is the classic topology-opt choice; `p = 1` gives the convex blend. |
| `smooth_sigma` | `0.0` | Separable Gaussian filter σ on logits (cell units) before the sigmoid. Suppresses disconnected islands; differentiable. 1–2 is typical; `0` disables. Clamped to `(min(n_x, n_y) - 1) / 2`. |
| `plot_mask_every` | `10` | Save a `shape_steps/mask_NNNN.png` frame every N steps. `0` disables. |
| `project_final` | `true` | After training, save the largest connected component of `mask > 0.5` as a second `.npz` + PNG. Not differentiable — used as a manufacturable export. |
| `project_connectivity` | `1` | `1` = 4-connected neighbours for the projection, `2` = 8-connected. |

Material-side hyperparameters (`n_iters`, `lr`, `lambda_l2`, `lambda_neighbor`, `cells.n_x/n_y/n_C_params`, `loss.type`, mesh fields) come from the standard `SimulationConfig` — identical to `run_optimize.py`.

## Outputs

Everything lands under `output_dir` from the config (default `output/open_geometry/`):

| Path | Contents |
|---|---|
| `config.yaml` | Copy of the exact YAML used |
| `loss_history.csv` | Per-step `step, beta, total, cloak, l2, material_tv, mask_tv, bin, mask_solid_frac` |
| `loss_curves.pdf` | Three-panel: total/cloak, regularisers (L2 drift, material TV, mask TV, binarisation), β schedule (when ramping) |
| `shape_mask_final.png` | Heatmap of the final (soft) mask `σ(β · filter(s))` over the `n_x × n_y` cell grid |
| `shape_mask_projected.png` | Binary heatmap of the largest connected component, if `project_final: true` |
| `shape_steps/mask_NNNN.png` | Evolution frames — one per `plot_mask_every` steps |
| `optimized_params.npz` | **Best-loss snapshot** (overwritten on every improvement): `cell_C_flat, cell_rho, shape_logits, shape_mask, beta, n_x, n_y` |
| `optimized_params_final.npz` | Final-step snapshot, same keys (+ `beta` = final β) |
| `optimized_params_projected.npz` | Same as final but with `shape_mask` hard-thresholded to largest CC (and `shape_mask_soft` preserving the pre-projection values) — only if `project_final: true` |

The `.npz` keys match the existing cell-based optimiser where they overlap, so [`scripts/vis/plot_optimized_profiles.py`](../scripts/vis/plot_optimized_profiles.py) and other post-hoc tools that read `cell_C_flat` / `cell_rho` work unchanged. `shape_logits` and `shape_mask` are additive.

## Warm-start

Setting `optimization.init_params: "path/to/optimized_params.npz"` in the config warm-starts:

- `cell_C_flat`, `cell_rho` — always loaded if present.
- `shape_logits` — loaded if present in the `.npz`; otherwise the triangular mask seeds them.

This lets you chain runs: anneal `β` in a second run by warm-starting from the first and raising `shape_opt.beta`.

## Known caveats

- **PETSc residual tolerance.** Like the rest of the pipeline, the adjoint solve requires [`patches/solver.py`](../patches/solver.py) to be applied on top of the installed JAX-FEM (relative-error tolerance). Without the patch, the adjoint may trip the `err < 0.1` assertion on meshes where the forward solve is perfectly fine.
- **Defect-interior cells.** The FEM mesh has the defect cut out, but cells whose centres fall inside the defect remain in the cell grid (with the sentinel mapping sending defect-interior quadrature points to background). Their logits move freely but no physics depends on them.
- **`init_magnitude` interacts with the β schedule.** Initial occupancy is `σ(β_start · init_magnitude)`. If you combine a large `init_magnitude` with a high `beta_end`, the effective `|β · s|` at the end of training is large and the sigmoid saturates — gradients vanish for any cell still near the edge. Defaults (`init_magnitude = 3`, `beta_start = 1`, `beta_end = null`) stay in the well-behaved regime; when enabling a ramp, check that `beta_end · init_magnitude ≲ 30` or lower the init magnitude.
- **Smoothing is clamped on small grids.** `smooth_sigma` uses a kernel of radius `3σ`; if that exceeds `(min(n_x, n_y) − 1) / 2` the radius is silently clamped. Harmless on the default 50×50 grid; relevant if you run very coarse cell grids.
- **Projection vs warm-start.** `optimized_params_projected.npz` stores the *binary* mask under `shape_mask` and the pre-projection soft mask under `shape_mask_soft`. The warm-start path currently loads `shape_logits`, not either mask — so warm-starting from a projected run is a warm-start on the *raw logits*, and the projection itself is re-imposed only if you set `project_final: true` on the follow-up run.
