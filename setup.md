# Setup

Practical setup notes for running the current `nn-cloacking` codebase.

## 1. Prerequisites

You need:

- Python 3.10
- Conda or Miniconda
- A working JAX/JAX-FEM stack
- `gmsh`, `meshio`, `matplotlib`, `vtk`, `pydantic`, `pyyaml`, `numpy`

The solver also depends on the PETSc support used by JAX-FEM.

## 2. Recommended approach

The checked-in `environment.yml` appears to be a fully exported environment, not a minimal cross-platform spec. It includes platform-specific packages, so it may not recreate cleanly on every machine, especially macOS.

Try the environment file first:

```bash
conda env create -f environment.yml
conda activate jax-fem-env
```

If that fails, use the manual setup below.

## 3. Manual Conda setup

Create a fresh environment:

```bash
conda create -n jax-fem-env python=3.10 -y
conda activate jax-fem-env
```

Install the core Python packages:

```bash
pip install numpy pyyaml pydantic matplotlib meshio gmsh vtk
```

Then install JAX and JAX-FEM in the way that matches your machine and accelerator setup.

Typical CPU-only JAX install:

```bash
pip install jax jaxlib
```

If your JAX-FEM install is separate:

```bash
pip install jax-fem
```

## 4. Project structure to know

Main runtime entry point:

- `run.py`

Main configs:

- `configs/default.yaml`
- `configs/reference.yaml`

Core package:

- `rayleigh_cloak/`

Outputs are written to:

- `output/`

## 5. Run the simulation

Default cloak simulation:

```bash
python run.py
```

Reference simulation:

```bash
python run.py configs/reference.yaml
```

## 6. Expected output files

A successful run writes files such as:

- `output/results_2.00.npz`
- `output/results_2.00.vtk`
- `output/_cloak_mesh.msh`
- `output/cloak_vtk_full.png`
- `output/cloak_vtk_phys.png`
- `output/cloak_vtk_re_uy_full.png`
- `output/cloak_vtk_re_uy.png`
- `output/cloak_vtk_re_mag.png`

## 7. Plot saved results again

Plot from NPZ:

```bash
python -m rayleigh_cloak.plot output/results_2.00.npz
```

Plot from VTK:

```bash
python -m rayleigh_cloak.plot output/results_2.00.vtk
```

## 8. Quick verification

A healthy setup should allow this command to finish without errors:

```bash
python run.py
```

You should see console output reporting:

- the extended and physical domain sizes,
- PML thickness and damping settings,
- triangle count.

## 9. Known caveats

- Some tests and older comments still refer to removed legacy scripts such as `boundaries.py`.
- `configs/reference.yaml` includes `material.mu_factor`, but the current `MaterialConfig` uses `rho0` and `cs`.
- If `gmsh` or `vtk` import fails, reinstall them inside the active environment.
- On macOS, using a clean Conda env plus `pip` installs is often more reliable than recreating the exported `environment.yml` exactly.

## 10. Related docs

- See `README.md` for the current codebase overview.
- See `docs/final/nn_cloaking_theory_notes.tex` for theory notes.
