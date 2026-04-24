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
   +  λ_mask     · L_mask_tv                 # new      — TV on logits, full grid
```

The two existing material regularisers act on the **raw** cell values (not the blended ones), so they retain their original meaning. `L_mask_tv` is a mean-squared difference of logits across 4-connected neighbour pairs — a cheap perimeter-like penalty that discourages speckled masks.

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
| `beta` | `1.0` | Sigmoid sharpness. Raise during or after training to drive the mask toward binary. Watch for gradient saturation if `β · s` ≫ 1 everywhere. |
| `init_magnitude` | `3.0` | \|logit\| at init. `σ(β · magnitude) ≈ 0.95` with the defaults — soft enough to keep gradients alive, decisive enough that the initial forward pass is approximately the triangular cloak. |
| `logits_lr_mult` | `1.0` | Effective logit LR = `optimization.lr · logits_lr_mult`. Useful if the materials (Pa scale) and logits (O(1) scale) want different step sizes — Adam mostly normalises scale out, but the multiplier is available. |
| `lambda_mask_smooth` | `1.0e-2` | Weight on the mask-TV regulariser. Too small → speckled masks; too large → mask freezes near the initial triangle. |
| `plot_mask_every` | `10` | Save a `shape_steps/mask_NNNN.png` frame every N steps. `0` disables. |

Material-side hyperparameters (`n_iters`, `lr`, `lambda_l2`, `lambda_neighbor`, `cells.n_x/n_y/n_C_params`, `loss.type`, mesh fields) come from the standard `SimulationConfig` — identical to `run_optimize.py`.

## Outputs

Everything lands under `output_dir` from the config (default `output/open_geometry/`):

| Path | Contents |
|---|---|
| `config.yaml` | Copy of the exact YAML used |
| `loss_history.csv` | Per-step `step, total, cloak, l2, material_tv, mask_tv, mask_solid_frac` |
| `loss_curves.pdf` | Two-panel: total/cloak on top, regularisers (L2 drift, material TV, mask TV) on bottom |
| `shape_mask_final.png` | Heatmap of final `σ(β · s)` over the `n_x × n_y` cell grid (origin lower-left, equal aspect) |
| `shape_steps/mask_NNNN.png` | Evolution frames — one per `plot_mask_every` steps |
| `optimized_params.npz` | **Best-loss snapshot** (overwritten on every improvement): `cell_C_flat, cell_rho, shape_logits, shape_mask, beta, n_x, n_y` |
| `optimized_params_final.npz` | Final-step snapshot, same keys |

The `.npz` keys match the existing cell-based optimiser where they overlap, so [`scripts/vis/plot_optimized_profiles.py`](../scripts/vis/plot_optimized_profiles.py) and other post-hoc tools that read `cell_C_flat` / `cell_rho` work unchanged. `shape_logits` and `shape_mask` are additive.

## Warm-start

Setting `optimization.init_params: "path/to/optimized_params.npz"` in the config warm-starts:

- `cell_C_flat`, `cell_rho` — always loaded if present.
- `shape_logits` — loaded if present in the `.npz`; otherwise the triangular mask seeds them.

This lets you chain runs: anneal `β` in a second run by warm-starting from the first and raising `shape_opt.beta`.

## Known caveats

- **PETSc residual tolerance.** Like the rest of the pipeline, the adjoint solve requires [`patches/solver.py`](../patches/solver.py) to be applied on top of the installed JAX-FEM (relative-error tolerance). Without the patch, the adjoint may trip the `err < 0.1` assertion on meshes where the forward solve is perfectly fine.
- **Defect-interior cells.** The FEM mesh has the defect cut out, but cells whose centres fall inside the defect remain in the cell grid (with the sentinel mapping sending defect-interior quadrature points to background). Their logits move freely but no physics depends on them.
- **`init_magnitude` vs `β`.** These two multiply in the initial occupancy. If you raise `β` during training, the effective `|β · s|` grows too — start soft and anneal. Current code uses a fixed `β`; anneal manually by chaining warm-started runs.
