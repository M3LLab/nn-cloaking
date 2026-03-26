"""Allow running as: python -m rayleigh_cloak.dataset.cellular_chiral"""
import numpy as np
from pathlib import Path

from .visualize import main
from .generator import CAConfig, generate_chiral_unit_cell
from .visualize import plot_quadrant_and_cell


out_dir = Path("output/ca_chiral")
dataset_dir = out_dir / "dataset"
individual_dir = out_dir / "individual"
dataset_dir.mkdir(parents=True, exist_ok=True)
individual_dir.mkdir(parents=True, exist_ok=True)


def generate_dataset(N=100, seed=0):
    config = CAConfig()

    for s in range(seed, seed + N):
        uc, q = generate_chiral_unit_cell(config=config, seed=s)
        plot_quadrant_and_cell(q, uc, s, save_path=individual_dir / f"seed_{s}.png")
        save_path = dataset_dir / f"seed_{s}.npy"
        np.save(save_path, uc)


# main()
generate_dataset(N=16, seed=0)