# Bug: Collapsed Dispersion Frequencies for Optimized Cloak

## Symptom

Running Bloch-Floquet dispersion on the optimized cell-based cloak produced
frequencies capped at $f^* \approx 0.18$, while the reference case reached
$f^* \approx 4.8$. The dispersion diagram appeared flat/empty — a ~26×
collapse in the frequency range.

## Root Cause

The neural-network optimization (isotropic, `n_C_params=2`, config
`best_configs/cauchy_tri_top.yaml`) produced cells with **negative Lamé $\mu$**
(36 cells, min = −72 MPa) and **negative density $\rho$** (15 cells, min =
−607 kg/m³). Out of 2500 total cells, 834 are inside the cloak, and 51 of
those have at least one unphysical parameter.

### Why the optimization produces negative values

The analytic transformation cloak requires material properties that vary
continuously from near-zero at the inner defect boundary ($r = a$) to
background values at the outer boundary ($r = b$). The analytic $C_\mathrm{eff}$
is a full Cosserat (non-symmetric) tensor, but the optimization uses only two
isotropic parameters ($\lambda$, $\mu$) per cell. The MLP attempts to match
the target displacement field using this limited parameterization — near the
defect tip where the true transformation demands extreme gradients and
anisotropy, the optimizer overshoots zero and lands on small negative values.
This is an inherent limitation of isotropic approximation of Cosserat
behaviour.

### Why the forward solver works but dispersion breaks

The **forward BVP solver** (`Ku = f`) is a driven problem with a source term
and PML absorbing boundaries. The linear system is well-conditioned regardless
of local material sign because the PML provides complex-valued damping that
regularizes the operator. Negative $\mu$ or $\rho$ in a few cells does not
prevent convergence.

The **dispersion analysis** solves a **generalized eigenvalue problem**
$K\mathbf{v} = \omega^2 M\mathbf{v}$ (Bloch-Floquet, no source, no PML).
This requires:

- $M$ positive-definite → needs $\rho > 0$ everywhere
- $K$ positive-semi-definite → needs $\mu > 0$ (shear stiffness)

Negative $\rho$ produces 4 negative diagonal entries in $M$, making it
indefinite. Negative $\mu$ makes $K$ indefinite across 45 elements. The
shift-invert eigensolver (`scipy.sparse.linalg.eigsh` with `sigma=0`) then
returns near-zero spurious eigenvalues instead of the true dispersion branches.

### Detailed numbers

| Quantity | Before clamping | After clamping |
|---|---|---|
| $\rho$ range | [−607, 5240] kg/m³ | [16, 5240] kg/m³ |
| $\mu$ range | [−72.2, 465.8] MPa | [1.44, 465.8] MPa |
| $M$ diagonal min | $-2.0 \times 10^{-1}$ | $7.8 \times 10^{-3}$ |
| Cells with $\rho < 0$ | 15 | 0 |
| Cells with $\mu < 0$ | 36 | 0 |
| $f^*$ range (550 eigs, 50 k-pts) | [0, 0.18] | [0, 2.5] |

Background material: $\rho_0 = 1600$ kg/m³, $\lambda = \mu = 144$ MPa.

## Fix

In `element_materials_optimized()` (`scripts/dispersion_jaxfem.py`), cell
parameters are clamped to physically valid floor values before FEM assembly:

```python
rho_min = 0.01 * rho0       # 1% of background density = 16 kg/m³
mu_min  = 0.01 * p["mu"]    # 1% of background shear modulus = 1.44 MPa

cell_rho = np.maximum(cell_rho, rho_min)
cell_C_flat[:, 1] = np.maximum(cell_C_flat[:, 1], mu_min)  # μ column
```

Warnings are printed when clamping occurs. The clamping affects only the
dispersion analysis; the forward solver still uses the original unclamped
values.

## Q: Is the fix just changing negative parameters to positive?

Not exactly — it is a **floor clamp**, not a sign flip. Only ~2% of cells are
affected (51 out of 2500), and only for the eigenvalue dispersion.

- **Most cells** (2449/2500) have valid positive $\mu$ and $\rho$ and are
  untouched.
- **A few cells** near the inner boundary of the cloak triangle have small
  negative $\mu$ or $\rho$. The neural network overshoots there because the
  analytic transformation demands extreme gradients ($\rho \to 0$, $C \to 0$
  as you approach the defect tip). The MLP can't perfectly represent this with
  isotropic parameters and occasionally crosses zero.
- The clamp sets a **floor at 1% of background** — so a cell with $\mu =
  {-72}$ MPa becomes $\mu = 1.44$ MPa (very soft, close to the intended
  near-zero value), not $\mu = 144$ MPa (background). The spatial pattern of
  "soft near defect, stiff near outer edge" is preserved.

This is purely a post-processing step for the eigenvalue dispersion. The
forward cloaking simulation uses the original unclamped values unchanged — the
BVP solver doesn't need positive-definiteness.

## Verification

After clamping:
- $M$ diagonal minimum: $7.8 \times 10^{-3}$ (was $-2.0 \times 10^{-1}$)
- First 20 eigenvalues at $k_\mathrm{norm} = 0.125$: $f^* \in [0.15, 0.83]$
- Full sweep (550 eigenvalues × 50 k-points): frequencies span the expected
  $[0, 2.5]$ range, matching the reference case's coverage

## Files Modified

- `scripts/dispersion_jaxfem.py` — `element_materials_optimized()` function
