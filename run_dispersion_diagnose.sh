#!/usr/bin/env bash
# Diagnose where the extra high-IPR modes in the optimized-cloak dispersion
# come from: geometry (triangle cutout) vs material optimization (cell map).
#
# Runs three cases against the same config + output dir:
#   1. inspect the optimized .npz   (shape, clamped counts, ρ/μ ranges)
#   2. reference + ideal_cloak      (--case both)
#   3. reference + optimized_cloak  (--case optimized_vs_ref)
#
# Compare the two resulting dispersion_comparison*.png files:
#   - modes in (2) that miss the Rayleigh line → geometric/Cosserat-approx effect
#   - modes in (3) that aren't in (2)         → optimization/clamping artifacts
#
# Usage:
#   ./run_dispersion_diagnose.sh
#   ./run_dispersion_diagnose.sh -f          # force re-run of cached .npz sweeps
#   ./run_dispersion_diagnose.sh <config.yaml> <params.npz>   # custom inputs

set -e
cd "$(dirname "$0")"

CONFIG="${1:-best_configs/latest/latent_ae_optimize_sweep.yaml}"
PARAMS="${2:-best_configs/latest/optimized_params.npz}"
FORCE=""
for arg in "$@"; do
    [[ "$arg" == "-f" ]] && FORCE="-f"
done

echo "=============================================================="
echo " Config:         $CONFIG"
echo " Optimized npz:  $PARAMS"
echo " Force rerun:    ${FORCE:-no}"
echo "=============================================================="

# ── 1. Inspect optimized params ───────────────────────────────────────
echo
echo "─── Inspecting $PARAMS ───"
PYTHONPATH="$(pwd)" python - <<PY
import numpy as np, yaml, pathlib
d = np.load("$PARAMS")
cfg = yaml.safe_load(open("$CONFIG"))
rho0 = float(cfg["material"]["rho0"])
cs   = float(cfg["material"]["cs"])
mu0  = rho0 * cs**2
print("keys         :", list(d.keys()))
for k in d.keys():
    print(f"  {k:20s} shape={d[k].shape}  dtype={d[k].dtype}")
C = d["cell_C_flat"]; rho = d["cell_rho"]
print()
print(f"n_C_params (from npz)       : {C.shape[1]}")
print(f"n_cells                     : {C.shape[0]}")
print(f"background μ0 = ρ·cs²       : {mu0:.1f}")
print(f"background ρ0                : {rho0:.1f}")
print()
print(f"cell_C_flat col-0 range     : [{C[:,0].min():.3e}, {C[:,0].max():.3e}]  (λ if iso)")
if C.shape[1] >= 2:
    print(f"cell_C_flat col-1 range     : [{C[:,1].min():.3e}, {C[:,1].max():.3e}]  (μ if iso)")
print(f"cell_rho range              : [{rho.min():.3e}, {rho.max():.3e}]")
print()
rho_floor = 0.01*rho0
mu_floor  = 0.01*mu0
print(f"Would-clamp at ρ_min={rho_floor:.2f}: {int((rho<rho_floor).sum())} / {rho.size}")
if C.shape[1] >= 2:
    print(f"Would-clamp at μ_min={mu_floor:.2f}: {int((C[:,1]<mu_floor).sum())} / {C.shape[0]}")
    print(f"Cells with negative μ       : {int((C[:,1]<0).sum())}")
print(f"Cells with negative λ       : {int((C[:,0]<0).sum())}")
PY

# Determine n_C_params from the npz so we don't silently truncate anisotropy
NC=$(PYTHONPATH="$(pwd)" python -c "import numpy as np; print(np.load('$PARAMS')['cell_C_flat'].shape[1])")
echo
echo "Using --n-C-params $NC"

# ── 2. reference + ideal_cloak  (isolates triangle/geometry effect) ───
echo
echo "=============================================================="
echo " Run A:  reference + ideal analytic cloak  (--case both)"
echo "         → shows effect of triangle cutout + analytic C_eff"
echo "=============================================================="
./dispersion_run_jax.sh "$CONFIG" --case both $FORCE

# Rename comparison plot so the next run doesn't overwrite it
OUTDIR=$(PYTHONPATH="$(pwd)" python -c "
import yaml, pathlib
cfg = yaml.safe_load(open('$CONFIG'))
print(pathlib.Path(cfg['output_dir']) / 'dispersion')
")
if [[ -f "$OUTDIR/dispersion_comparison.png" ]]; then
    cp "$OUTDIR/dispersion_comparison.png" "$OUTDIR/dispersion_comparison_ideal_vs_ref.png"
    echo "Saved → $OUTDIR/dispersion_comparison_ideal_vs_ref.png"
fi

# ── 3. reference + optimized_cloak  (isolates optimization effect) ────
echo
echo "=============================================================="
echo " Run B:  reference + optimized cloak  (--case optimized_vs_ref)"
echo "         → shows effect of optimized material (+ clamping)"
echo "=============================================================="
./dispersion_run_jax.sh "$CONFIG" \
    --params-npz "$PARAMS" \
    --n-C-params "$NC" \
    --case optimized_vs_ref \
    $FORCE

echo
echo "=============================================================="
echo " Done.  Compare:"
echo "   $OUTDIR/dispersion_comparison_ideal_vs_ref.png"
echo "   $OUTDIR/dispersion_comparison_optimized.png"
echo ""
echo " Interpretation:"
echo "   - high-IPR modes present in BOTH   → geometric (triangle + approx)"
echo "   - high-IPR modes ONLY in optimized → optimization / clamping artifact"
echo "=============================================================="
