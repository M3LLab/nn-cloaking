# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

This project uses a conda environment defined in `environment.yml`:

```bash
conda env create -f environment.yml
conda activate jax-fem-env
```

Key dependencies: FEniCS, JAX, JAX-FEM, gmsh, PETSc/petsc4py, SLEPc, MPI. The environment is heavy; import errors are usually due to missing conda env activation.

`patches/solver.py` is a patched version of a JAX-FEM internal file (it patches `jax_fem/solver.py` in place). If JAX-FEM is updated, re-apply the patch.

## Running Simulations

```bash
# Forward solve (continuous analytic C_eff)
python run.py                            # uses configs/default.yaml
python run.py configs/continuous.yaml    # explicit config

# Reference solve (no cloak/defect)
python run.py configs/reference.yaml

# Cell-based forward solve
python run.py configs/cell_based.yaml

# Material optimization
python run_optimize.py                       # uses configs/cell_based.yaml
python run_optimize.py configs/optimize.yaml # custom config
```

Outputs (plots, `.npz` checkpoints) go to the directory set by `output_dir` in the config (default: `output/`).

## Analysis Scripts

Standalone scripts under `scripts/` are run with `PYTHONPATH=$(pwd)` so they can import the `rayleigh_cloak` package. Wrapper shell scripts set this up:

```bash
./run_ideal_sweep.sh        # analytical frequency sweep → output/continuous/
./run_dispersion.sh         # Bloch-Floquet dispersion curves → output/dispersion/
```

Both wrappers skip work if their output files already exist; pass `-f` to force re-run. Other `scripts/` files (plotting, dataset generation, comparisons) are invoked directly with the same `PYTHONPATH` convention.

## Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run a single test file
python -m pytest tests/test_ceff.py

# Debug scripts (not pytest — run directly)
python tests/debug/plot_mesh.py
```

Tests under `tests/debug/` are standalone scripts for visual inspection, not part of the pytest suite.

## Architecture Overview

### Core simulation pipeline

1. **Config** ([rayleigh_cloak/config.py](rayleigh_cloak/config.py)): `SimulationConfig` (Pydantic, serialisable to/from YAML) holds all user parameters. `DerivedParams` (frozen dataclass) computes physical quantities (wave speeds, domain extents, PML lengths, cloak dimensions) from the config once. All downstream code accepts `DerivedParams`, not raw config fields.

2. **Geometry** ([rayleigh_cloak/geometry/](rayleigh_cloak/geometry/)): Every geometry implements the `CloakGeometry` protocol (`base.py`): `in_cloak`, `in_defect`, `F_tensor`, `build_gmsh_geometry`. Current implementations: `TriangularCloakGeometry` (default, shear-transformation cloak) and `CircularCloakGeometry`.

3. **Mesh** ([rayleigh_cloak/mesh.py](rayleigh_cloak/mesh.py)): Calls gmsh to generate a TRI3 mesh of the full domain (physical region + PML). The reference config uses a plain rectangle; the cloaked config cuts out the defect via the geometry object. `extract_submesh` removes nodes inside the defect after meshing.

4. **FEM problem** ([rayleigh_cloak/problem.py](rayleigh_cloak/problem.py)): `RayleighCloakProblem` subclasses JAX-FEM's `Problem`. DOF ordering per node: `[Re(ux), Re(uy), Im(ux), Im(uy)]`. The problem precomputes material tensors and PML damping coefficients at quadrature points in `custom_init`, then assembles stiffness and mass maps.

5. **Solver** ([rayleigh_cloak/solver.py](rayleigh_cloak/solver.py)): High-level `solve(config)` and `solve_reference(config)` functions orchestrate mesh generation, problem setup, and the JAX-FEM linear solve. `solve_cell_based` handles piecewise-constant material parameterization.

### Material parameterization

- **Continuous** ([rayleigh_cloak/materials.py](rayleigh_cloak/materials.py)): Analytic transformation elasticity — `C_eff(x)` and `rho_eff(x)` computed pointwise from the deformation gradient `F_tensor`.
- **Cell-based** ([rayleigh_cloak/cells.py](rayleigh_cloak/cells.py)): `CellDecomposition` lays a regular grid over the cloak bounding box. Each FEM quadrature point is assigned to a cell at setup time; per-cell material arrays are expanded to QP arrays via JAX fancy indexing (differentiable).
- **Neural reparameterization** ([rayleigh_cloak/neural_reparam.py](rayleigh_cloak/neural_reparam.py)): An MLP maps cell-center coordinates → `(C_flat, rho)`. The network weights become the optimization variables. Fourier feature embeddings are used as positional encoding.
- **Topology neural** ([rayleigh_cloak/neural_reparam_topo.py](rayleigh_cloak/neural_reparam_topo.py)): MLP parameterizes a pixel-level density field with SIMP penalization and Heaviside projection for binarization.

### Optimization

[rayleigh_cloak/optimize.py](rayleigh_cloak/optimize.py) implements a self-contained Adam optimizer (no optax dependency) and the loss functions. The main loop in `run_optimization` calls JAX-FEM's `ad_wrapper` for implicit adjoint differentiation through the FEM solve. Loss = cloaking term (relative L2 on boundary) + L2 regularization (drift from init) + neighbor smoothness regularization.

Multi-frequency optimization is in `neural_reparam.py` (`run_optimization_neural_multifreq`) and dispatches parallel forward/adjoint solves via `ThreadPoolExecutor` (PETSc releases the GIL).

### PML absorbing layers

[rayleigh_cloak/absorbing.py](rayleigh_cloak/absorbing.py): Rayleigh-damping PML implemented as complex coordinate stretching via a per-quadrature-point damping coefficient `xi`. Applied on left, right, and bottom boundaries (and all four sides for circular geometry / plane-wave source).

### Loss targets

[rayleigh_cloak/loss.py](rayleigh_cloak/loss.py): `resolve_loss_target` selects boundary nodes for the cloaking loss based on `LossConfig.type`: `"right_boundary"` (default), `"top_surface"` (free surface beyond cloak footprint), or `"outside_cloak"` (all physical-domain nodes outside cloak).

## Config System

All configs are YAML files under `configs/`. The main entry points (`run.py`, `run_optimize.py`) accept an optional positional argument with the config path, falling back to a default. Configs map directly onto `SimulationConfig` fields; any field not specified uses the Pydantic default. See `configs/default.yaml` for the baseline and `configs/reference.yaml` for the reference (no-cloak) setup.

Key config dimensions:
- `geometry_type`: `"triangular"` or `"circular"`
- `optimization.method`: `"raw"` (per-cell arrays), `"neural"` (MLP reparameterization), `"neural_topo"` (topology/density)
- `cells.n_C_params`: stiffness tensor complexity — `2` (isotropic), `6` (block-diagonal Cosserat, recommended), `16` (full Voigt4)
- `loss.type`: which boundary region to measure cloaking quality
