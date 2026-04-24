"""Open-geometry cloak optimisation: jointly optimise shape + material.

Sits on top of the existing 2D pipeline without modifying any physics code.
The cell-based FEM problem is built exactly as in
:func:`rayleigh_cloak.solver.solve_optimization`; the only addition is a
trainable per-cell shape logit whose sigmoid blends the material toward the
background before the arrays are handed to ``fwd_pred``.

Public entry point: :func:`solve_optimization_open_geometry`.

The shape parameterisation itself lives in :mod:`rayleigh_cloak.shape_mask`
so that geometry is kept separate from physics/model code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from jax_fem.solver import ad_wrapper

from rayleigh_cloak.cells import CellDecomposition
from rayleigh_cloak.config import DerivedParams, SimulationConfig
from rayleigh_cloak.loss import resolve_loss_target
from rayleigh_cloak.materials import C_iso, CellMaterial, _get_converters
from rayleigh_cloak.mesh import extract_submesh, generate_mesh_full
from rayleigh_cloak.optimize import (
    adam_init,
    adam_update,
    cloaking_loss,
    l2_regularization,
    neighbor_regularization,
)
from rayleigh_cloak.problem import build_problem
from rayleigh_cloak.shape_mask import (
    all_neighbor_pairs,
    apply_shape_mask,
    init_logits_from_cloak_mask,
    mask_smoothness,
    occupancy,
)
from rayleigh_cloak.solver import (
    _create_geometry,
    _petsc_opts,
    solve_reference,
)


# ── config (read from the raw YAML's ``shape_opt:`` section) ──────────


@dataclass
class ShapeOptConfig:
    """Hyperparameters for the shape-mask optimisation.

    Populated from the YAML's top-level ``shape_opt:`` block (optional).
    Kept as a plain dataclass so we don't touch :mod:`rayleigh_cloak.config`.
    """
    beta: float = 1.0                # sigmoid sharpness
    init_magnitude: float = 3.0      # |logit| at init (inside/outside cloak)
    logits_lr_mult: float = 1.0      # logits LR = material lr * this
    lambda_mask_smooth: float = 1e-2 # TV penalty on logits
    plot_mask_every: int = 10        # 0 = never

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ShapeOptConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        raw = data.get("shape_opt", {}) or {}
        known = {k: raw[k] for k in cls.__dataclass_fields__ if k in raw}
        return cls(**known)


# ── result ───────────────────────────────────────────────────────────


@dataclass
class OpenGeometryResult:
    """Outcome of a joint shape + material optimisation run."""
    cell_C_flat: jnp.ndarray
    cell_rho: jnp.ndarray
    shape_logits: jnp.ndarray
    shape_mask: jnp.ndarray          # sigmoid(beta * logits) — final occupancy
    n_x: int
    n_y: int
    loss_history: list[float] = field(default_factory=list)
    cloak_history: list[float] = field(default_factory=list)
    l2_history: list[float] = field(default_factory=list)
    neighbor_history: list[float] = field(default_factory=list)
    mask_smooth_history: list[float] = field(default_factory=list)


# ── optimisation loop ────────────────────────────────────────────────


def run_optimization_open_geometry(
    fwd_pred: Callable,
    cell_params_init: tuple[jnp.ndarray, jnp.ndarray],
    logits_init: jnp.ndarray,
    u_ref_boundary: jnp.ndarray,
    boundary_indices: np.ndarray,
    neighbor_pairs: np.ndarray,          # used for material smoothness (init cloak only)
    mask_neighbor_pairs: np.ndarray,     # used for logit TV (full grid)
    C0_flat: jnp.ndarray,
    rho0: float,
    n_x: int,
    n_y: int,
    beta: float,
    n_iters: int,
    lr: float,
    logits_lr_mult: float,
    lambda_l2: float,
    lambda_neighbor: float,
    lambda_mask_smooth: float,
    loss_fn: Callable | None = None,
    step_callback: Callable | None = None,
    mask_callback: Callable | None = None,
    mask_every: int = 10,
) -> OpenGeometryResult:
    """Adam loop over ``(cell_C_flat, cell_rho, logits)``.

    Material gradients use the mask-blended effective arrays (``m·C + (1-m)·C0``
    and similarly for rho), so cells currently "outside" the shape get naturally
    small material gradients — shape decides where material matters, then
    material fine-tunes where shape has committed.
    """
    if loss_fn is None:
        loss_fn = cloaking_loss

    C0 = jnp.asarray(C0_flat)
    boundary_idx = jnp.asarray(boundary_indices)
    nb_pairs = jnp.asarray(neighbor_pairs)
    mask_pairs = jnp.asarray(mask_neighbor_pairs)

    cell_C_init, rho_init = cell_params_init
    state: dict[str, jnp.ndarray] = {
        "C": jnp.asarray(cell_C_init),
        "rho": jnp.asarray(rho_init),
        "logits": jnp.asarray(logits_init),
    }
    material_init = (state["C"], state["rho"])  # regs measure drift from push-forward init

    def combined_loss(s: dict[str, jnp.ndarray]) -> jnp.ndarray:
        C_eff, rho_eff = apply_shape_mask(
            s["C"], s["rho"], s["logits"], C0, rho0, beta=beta,
        )
        sol_list = fwd_pred((C_eff, rho_eff))
        u_cloak = sol_list[0]
        L_cloak = loss_fn(u_cloak, u_ref_boundary, boundary_idx)
        L_l2 = l2_regularization((s["C"], s["rho"]), material_init)
        L_nb = neighbor_regularization((s["C"], s["rho"]), nb_pairs)
        L_mask = mask_smoothness(s["logits"], mask_pairs)
        return (
            L_cloak
            + lambda_l2 * L_l2
            + lambda_neighbor * L_nb
            + lambda_mask_smooth * L_mask
        )

    loss_and_grad = jax.value_and_grad(combined_loss)
    opt_state = adam_init(state)

    loss_hist: list[float] = []
    cloak_hist: list[float] = []
    l2_hist: list[float] = []
    nb_hist: list[float] = []
    mask_hist: list[float] = []

    logit_scale = logits_lr_mult  # applied per-step to logit gradients

    for step in range(n_iters):
        loss_val, grads = loss_and_grad(state)
        loss_val_f = float(loss_val)
        loss_hist.append(loss_val_f)

        # Break out component values (cheap — no forward solve)
        L_l2_f = float(l2_regularization((state["C"], state["rho"]), material_init))
        L_nb_f = float(neighbor_regularization((state["C"], state["rho"]), nb_pairs))
        L_mask_f = float(mask_smoothness(state["logits"], mask_pairs))
        L_cloak_f = (
            loss_val_f
            - lambda_l2 * L_l2_f
            - lambda_neighbor * L_nb_f
            - lambda_mask_smooth * L_mask_f
        )
        cloak_hist.append(L_cloak_f)
        l2_hist.append(L_l2_f)
        nb_hist.append(L_nb_f)
        mask_hist.append(L_mask_f)

        m_vals = np.asarray(occupancy(state["logits"], beta))
        print(
            f"  Step {step:4d} | total = {loss_val_f:.4e}"
            f"  cloak = {L_cloak_f:.4e}"
            f"  cloak_pct = {np.sqrt(max(L_cloak_f, 0.0)) * 100:.2e}"
            f"  L2 = {L_l2_f:.4e}  nb = {L_nb_f:.4e}  tv(s) = {L_mask_f:.4e}"
            f"  mask_mean = {m_vals.mean():.3f}  mask_solid = "
            f"{(m_vals > 0.5).sum()}/{m_vals.size}"
        )

        if step_callback is not None:
            step_callback(
                step, loss_val_f, L_cloak_f, L_l2_f, L_nb_f, L_mask_f, state,
            )

        if mask_callback is not None and mask_every > 0 and step % mask_every == 0:
            mask_callback(step, np.asarray(m_vals).reshape(n_x, n_y))

        if logit_scale != 1.0:
            grads = dict(grads)
            grads["logits"] = grads["logits"] * logit_scale

        updates, opt_state = adam_update(grads, opt_state, lr=lr)
        state = jax.tree.map(lambda p, u: p + u, state, updates)

    final_mask = np.asarray(occupancy(state["logits"], beta))
    if mask_callback is not None:
        mask_callback(n_iters, final_mask.reshape(n_x, n_y))

    return OpenGeometryResult(
        cell_C_flat=state["C"],
        cell_rho=state["rho"],
        shape_logits=state["logits"],
        shape_mask=jnp.asarray(final_mask),
        n_x=n_x,
        n_y=n_y,
        loss_history=loss_hist,
        cloak_history=cloak_hist,
        l2_history=l2_hist,
        neighbor_history=nb_hist,
        mask_smooth_history=mask_hist,
    )


# ── solver setup ─────────────────────────────────────────────────────


def solve_optimization_open_geometry(
    config: SimulationConfig,
    shape_cfg: ShapeOptConfig,
    step_callback: Callable | None = None,
    mask_callback: Callable | None = None,
) -> OpenGeometryResult:
    """Run joint shape + material optimisation over the cell grid.

    Mirrors :func:`rayleigh_cloak.solver.solve_optimization` (same mesh,
    reference solve, cell decomposition, FEM problem, loss target) and adds
    a per-cell shape logit on top.  The defect stays fixed — it is cut from
    the mesh up front; the mask only controls cloak-vs-background blending
    on cells that remain in the FEM domain.
    """
    params = DerivedParams.from_config(config)
    geometry = _create_geometry(config, params)

    print("=== Step 1: Generating shared mesh ===")
    full_mesh = generate_mesh_full(config, params, geometry)
    print(f"  Full mesh: {len(full_mesh.points)} nodes, "
          f"{len(full_mesh.cells)} elements")

    print("=== Step 2: Solving reference problem (on full mesh) ===")
    ref_result = solve_reference(config, mesh=full_mesh)

    print("=== Step 3: Extracting submesh (removing defect elements) ===")
    cloak_mesh, kept_nodes = extract_submesh(full_mesh, geometry)
    print(f"  Submesh: {len(cloak_mesh.points)} nodes, "
          f"{len(cloak_mesh.cells)} elements")

    print("=== Step 4: Setting up cell decomposition ===")
    cell_decomp = CellDecomposition(geometry, config.cells.n_x, config.cells.n_y)
    C0 = C_iso(params.lam, params.mu)
    cell_mat = CellMaterial(
        geometry, C0, params.rho0, cell_decomp,
        n_C_params=config.cells.n_C_params,
        symmetrize_init=config.cells.symmetrize_init,
    )
    print(f"  {cell_decomp.n_cells} total cells "
          f"({cell_decomp.n_cloak_cells} inside initial triangular cloak)")

    to_flat, _ = _get_converters(config.cells.n_C_params)
    C0_flat = jnp.asarray(to_flat(C0))

    print("=== Step 5: Building FEM problem ===")
    problem = build_problem(cloak_mesh, config, params, geometry, cell_decomp)

    boundary_indices, u_ref_boundary, loss_fn = resolve_loss_target(
        config.loss.type, np.asarray(cloak_mesh.points), geometry, params,
        kept_nodes, ref_result.u,
    )
    print(f"  {len(boundary_indices)} loss nodes ({config.loss.type})")

    material_neighbors = cell_decomp.get_neighbor_pairs()
    mask_neighbors = all_neighbor_pairs(cell_decomp.n_x, cell_decomp.n_y)
    print(f"  {len(material_neighbors)} material neighbour pairs, "
          f"{len(mask_neighbors)} shape neighbour pairs")

    print("=== Step 6: Initialising shape logits from triangular cloak ===")
    logits_init = init_logits_from_cloak_mask(
        cell_decomp.cloak_mask, magnitude=shape_cfg.init_magnitude,
    )
    print(f"  init_magnitude = {shape_cfg.init_magnitude}, "
          f"beta = {shape_cfg.beta}, "
          f"sigmoid(beta·mag) ≈ "
          f"{float(jax.nn.sigmoid(shape_cfg.beta * shape_cfg.init_magnitude)):.3f}")

    solver_opts = _petsc_opts(config)
    fwd_pred = ad_wrapper(problem, solver_opts, solver_opts)

    if config.optimization.init_params:
        data = np.load(config.optimization.init_params)
        params_init = (jnp.asarray(data["cell_C_flat"]),
                       jnp.asarray(data["cell_rho"]))
        if "shape_logits" in data.files:
            logits_init = jnp.asarray(data["shape_logits"])
            print(f"  Warm-started shape logits from {config.optimization.init_params}")
        print(f"  Warm-started cell materials from {config.optimization.init_params}")
    else:
        params_init = cell_mat.get_initial_params()

    print("=== Step 7: Optimising ===")
    opt_cfg = config.optimization
    return run_optimization_open_geometry(
        fwd_pred=fwd_pred,
        cell_params_init=params_init,
        logits_init=logits_init,
        u_ref_boundary=u_ref_boundary,
        boundary_indices=boundary_indices,
        neighbor_pairs=material_neighbors,
        mask_neighbor_pairs=mask_neighbors,
        C0_flat=C0_flat,
        rho0=float(params.rho0),
        n_x=cell_decomp.n_x,
        n_y=cell_decomp.n_y,
        beta=shape_cfg.beta,
        n_iters=opt_cfg.n_iters,
        lr=opt_cfg.lr,
        logits_lr_mult=shape_cfg.logits_lr_mult,
        lambda_l2=opt_cfg.lambda_l2,
        lambda_neighbor=opt_cfg.lambda_neighbor,
        lambda_mask_smooth=shape_cfg.lambda_mask_smooth,
        loss_fn=loss_fn,
        step_callback=step_callback,
        mask_callback=mask_callback,
        mask_every=shape_cfg.plot_mask_every,
    )
