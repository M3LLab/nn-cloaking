#!/usr/bin/env bash
# Wave Motion revision (2026-11-20): put the paper's 0.967 cell set on the same
# footing as the later ca_squared_2m-era results.
#
# The paper's Fig. 8 / 0.967 come from June 2026 sweeps (val_nn, val_inverse,
# val_diffusion) whose cells carry stiffnesses from the legacy TRI3 1 el/px
# homogeniser. This runner:
#   0. reproduces the June f*=2 values at the current commit (footing check),
#   1. re-homogenises the SAME canvases with the hi-fi static TRI6 N=100
#      homogeniser (new dirs; the June dirs are never touched),
#   2. reruns the matched-homogenised sweep over the Fig. 8 band (f*=1.5-2.5).
# The fully resolved 0.76 (design-validation, v2 cells) stays as published;
# no explicit-structure runs here.
#
# Every step writes to a NEW path and is skipped if its output already exists.
# Usage: scripts/run_rehomog_v1cells.sh [steps...]   (default: 0 1 2)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=${PY:-$HOME/miniconda3/envs/jax-fem-env/bin/python}
JOBS=${JOBS:-2}
# If memory runs out, the kernel should kill these runs, not the user's services.
echo 1000 > /proc/self/oom_score_adj

D=output/cell20_phys_flat4_materialreg4
R=$D/rehomog_v1
mkdir -p "$R"
STEPS=${*:-0 1 2}

sweep() {  # sweep <config> <cell-designs> <fmin> <fmax> <outdir>
  if [[ -f "$5/frequency_sweep_matched.csv" ]]; then echo "skip (exists): $5"; return; fi
  "$PY" scripts/frequency_sweep_validated.py "$1" "$D/optimized_params.npz" \
    --dataset output/ca_bulk_squared/stiffness.h5 \
    --fmin "$3" --fmax "$4" --fstep 0.1 --no-validated \
    --cell-designs "$2" -o "$5" > "$5.log" 2>&1
  echo "RESULT $(basename "$5"):"; cat "$5/frequency_sweep_matched.csv"
}

for s in $STEPS; do case $s in
0)  # NN cells have no saved dir: export them from the same catalogue match
    [[ -d $R/cell_designs_nn_bulk ]] || "$PY" scripts/export_nn_cell_designs.py \
      $D/config.yaml $D/optimized_params.npz \
      --dataset output/ca_bulk_squared/stiffness.h5 -o $R/cell_designs_nn_bulk
    # June values at f*=2: diffusion 0.966740, inverse 0.936238, NN 0.815545 (TRI3 macro mesh)
    sweep $D/config.yaml $D/cell_designs_diffusion 2.0 2.0 $R/repro_diffusion_tri3_f2
    sweep $D/config.yaml $D/cell_designs           2.0 2.0 $R/repro_inverse_tri3_f2
    sweep $D/config.yaml $R/cell_designs_nn_bulk   2.0 2.0 $R/repro_nn_tri3_f2
    ;;
1)  for name in diffusion:$D/cell_designs_diffusion inverse:$D/cell_designs nn:$R/cell_designs_nn_bulk; do
      tag=${name%%:*}; src=${name#*:}
      [[ -d $R/cell_designs_${tag}_hifi ]] && { echo "skip (exists): ${tag}_hifi"; continue; }
      "$PY" scripts/rehomogenise_cell_designs.py "$src" -o $R/cell_designs_${tag}_hifi -j "$JOBS" \
        2>&1 | grep -v "PETSC ERROR" | tee $R/cell_designs_${tag}_hifi.log
    done
    ;;
2)  for tag in diffusion inverse nn; do
      sweep $D/config.yaml $R/cell_designs_${tag}_hifi 1.5 2.5 $R/val_${tag}_hifi_tri3
    done
    # Same diffusion cells on the TRI6 macro mesh used for the v2 0.934 (val_diffusion_2m)
    sweep $D/config_tri6.yaml $D/cell_designs_diffusion  2.0 2.0 $R/val_diffusion_legacy_tri6_f2
    sweep $D/config_tri6.yaml $R/cell_designs_diffusion_hifi 2.0 2.0 $R/val_diffusion_hifi_tri6_f2
    ;;
*)  echo "unknown step $s"; exit 2;;
esac; done
