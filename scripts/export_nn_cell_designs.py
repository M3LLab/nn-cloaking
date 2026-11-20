"""Write the nearest-neighbour (NN) cloak cells as a cell-designs directory.

``frequency_sweep_validated.py`` without ``--cell-designs`` snaps each cloak
cell to its nearest dataset entry in memory and never saves the chosen
canvases. This script runs the SAME ``build_canvas`` matching and writes the
result in the ``cell_XXX/{canvas.npy, weights.npz}`` layout, so the NN baseline
can be re-homogenised (``rehomogenise_cell_designs.py``) and swept with
``--cell-designs`` exactly like the inverse-design and diffusion cells.

``pred_flat4``/``pred_rho`` are the dataset's stored values for the matched
entry (i.e. what the NN matched sweep used). ``target_*`` are the optimiser's
values. The script checks that re-tiling the written cells reproduces the
in-memory canvas and matched materials bit-for-bit.

Usage
-----
    PYTHONPATH=. python scripts/export_nn_cell_designs.py \
        output/cell20_phys_flat4_materialreg4/config.yaml \
        output/cell20_phys_flat4_materialreg4/optimized_params.npz \
        --dataset output/ca_bulk_squared/stiffness.h5 \
        -o output/cell20_phys_flat4_materialreg4/rehomog_v1/cell_designs_nn_bulk
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np


def _load_fsv():
    path = Path(__file__).resolve().parent / "frequency_sweep_validated.py"
    spec = importlib.util.spec_from_file_location("frequency_sweep_validated", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", type=Path)
    p.add_argument("params", type=Path)
    p.add_argument("--dataset", type=Path, default=Path("output/ca_bulk_squared/stiffness.h5"))
    p.add_argument("-o", "--out", type=Path, required=True, help="NEW output dir (must not exist)")
    args = p.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to write into existing {args.out}")

    fsv = _load_fsv()
    canvas, (n_x, n_y), (H, W), _bbox, mC, mrho, diag = fsv.build_canvas(
        args.params, args.dataset, args.config)
    cloak_mask, _c, _i = fsv._build_cloak_mask(args.config, n_x, n_y)
    cloak_idx = np.where(cloak_mask)[0]
    npz = np.load(args.params)
    tC, tr = npz["cell_C_flat"], npz["cell_rho"]          # flat order C11, C22, C66, C12

    args.out.mkdir(parents=True)
    for idx in cloak_idx:
        ix, iy = int(idx) // n_y, int(idx) % n_y
        row, col = n_y - 1 - iy, ix
        cell = canvas[row * H:(row + 1) * H, col * W:(col + 1) * W].astype(np.uint8)
        d = args.out / f"cell_{int(idx):03d}"
        d.mkdir()
        np.save(d / "canvas.npy", cell)
        np.savez(d / "weights.npz",
                 target_flat4=tC[idx][[0, 1, 3, 2]], target_rho=np.float64(tr[idx]),
                 pred_flat4=mC[idx][[0, 1, 3, 2]], pred_rho=np.float64(mrho[idx]))

    # Round trip: tiling the written cells must reproduce the NN canvas and materials.
    c2, _g, _hw, _b, mC2, mrho2, _d = fsv.build_canvas_from_cell_designs(
        args.params, args.dataset, args.config, args.out)
    assert np.array_equal(c2, canvas), "canvas round trip failed"
    assert np.array_equal(mC2, mC) and np.array_equal(mrho2, mrho), "materials round trip failed"
    print(f"wrote {cloak_idx.size} NN cells ({diag['n_unique_dataset_entries']} unique) -> {args.out}; "
          f"round trip OK")


if __name__ == "__main__":
    main()
