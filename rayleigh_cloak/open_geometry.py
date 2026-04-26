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

from concurrent.futures import ThreadPoolExecutor
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
from rayleigh_cloak.neural_reparam import FreqTarget
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
    binarization_penalty,
    init_logits_from_cloak_mask,
    mask_smoothness,
    occupancy,
    smooth_logits,
)
from rayleigh_cloak.solver import (
    _create_geometry,
    _make_config_at_fstar,
    _petsc_opts,
    solve_reference,
)


# ── config (read from the raw YAML's ``shape_opt:`` section) ──────────


@dataclass
class ShapeOptConfig:
    """Hyperparameters for the shape-mask optimisation.

    Populated from the YAML's top-level ``shape_opt:`` block (optional).
    Kept as a plain dataclass so we don't touch :mod:`rayleigh_cloak.config`.

    Defaults preserve the plain-sigmoid baseline (SIMP p = 1, no filter, no
    binarisation penalty, constant β).  Turn knobs on for sharper, more
    manufacturable masks.
    """
    beta: float = 1.0                # sigmoid sharpness (β_start if _end unset)
    beta_end: float | None = None    # if set, linearly ramp β → β_end over n_iters
    init_magnitude: float = 3.0      # |logit| at init (inside/outside cloak)
    logits_lr_mult: float = 1.0      # logits LR = material lr * this
    lambda_mask_smooth: float = 1e-2 # TV penalty on logits
    lambda_bin: float = 0.0          # binarisation penalty weight (mean m(1-m))
    simp_p: float = 1.0              # SIMP exponent on occupancy (1 = convex blend)
    smooth_sigma: float = 0.0        # Gaussian filter σ on logits (0 = off)
    plot_mask_every: int = 10        # 0 = never
    project_final: bool = True       # save largest-connected-component artefacts
    project_connectivity: int = 1    # 1 = 4-conn, 2 = 8-conn (for final projection)
    # Early stop: after the minimum n_iters (from optimization config), keep
    # going until ``patience`` consecutive steps go by without improving the
    # best total loss.  ``patience = 0`` disables the early-stop path and
    # the loop terminates exactly at n_iters.
    patience: int = 0
    max_iters: int | None = None     # hard cap when patience > 0; None = uncapped
    improvement_tol: float = 0.0     # relative drop required to count as improvement

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ShapeOptConfig":
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        raw = data.get("shape_opt", {}) or {}
        known = {k: raw[k] for k in cls.__dataclass_fields__ if k in raw}
        return cls(**known)

    def resolve_beta(self, step: int, n_iters: int) -> float:
        """Current β for a given step (linear ramp if ``beta_end`` is set)."""
        if self.beta_end is None or n_iters <= 1:
            return float(self.beta)
        t = step / (n_iters - 1)
        return float(self.beta + (self.beta_end - self.beta) * t)


# ── early stop ───────────────────────────────────────────────────────


class _StopCondition:
    """Plateau-based early-stop after a guaranteed minimum of iterations.

    Loop runs at least ``min_iters``.  After that, the next call to
    :meth:`should_stop` returns a non-empty reason string when either the
    plateau window (``patience`` consecutive non-improving steps) has
    elapsed or the hard cap (``max_iters``) is hit.

    A step is "improving" when its total loss is strictly less than
    ``best * (1 - improvement_tol)``.  ``patience = 0`` disables the
    plateau check entirely — the loop then terminates exactly at
    ``min_iters``.
    """

    def __init__(
        self,
        min_iters: int,
        patience: int,
        max_iters: int | None,
        improvement_tol: float,
    ) -> None:
        self.min_iters = int(min_iters)
        self.patience = int(patience)
        self.max_iters = int(max_iters) if max_iters is not None else None
        self.improvement_tol = float(improvement_tol)
        self.best = float("inf")
        self.no_improve = 0

    def update(self, loss: float) -> None:
        if loss < self.best * (1.0 - self.improvement_tol):
            self.best = loss
            self.no_improve = 0
        else:
            self.no_improve += 1

    def should_stop(self, step: int) -> str | None:
        """Return a reason string if the loop should terminate after ``step``."""
        next_step = step + 1
        # Hard cap always wins.
        if self.max_iters is not None and next_step >= self.max_iters:
            return f"hit max_iters cap ({self.max_iters})"
        # Below the minimum: never stop.
        if next_step < self.min_iters:
            return None
        # No early-stop configured: stop exactly at min_iters.
        if self.patience == 0:
            return f"reached n_iters={self.min_iters}"
        # Past the minimum, with patience: stop when plateau window elapsed.
        if self.no_improve >= self.patience:
            return (
                f"no improvement in {self.patience} steps "
                f"(best={self.best:.4e})"
            )
        return None


# ── result ───────────────────────────────────────────────────────────


@dataclass
class OpenGeometryResult:
    """Outcome of a joint shape + material optimisation run."""
    cell_C_flat: jnp.ndarray
    cell_rho: jnp.ndarray
    shape_logits: jnp.ndarray         # post-training raw logits (pre-filter)
    shape_mask: jnp.ndarray           # final occupancy σ(β_end · filter(logits))
    n_x: int
    n_y: int
    beta_end: float = 1.0             # the β used to compute the final mask
    loss_history: list[float] = field(default_factory=list)
    cloak_history: list[float] = field(default_factory=list)
    l2_history: list[float] = field(default_factory=list)
    neighbor_history: list[float] = field(default_factory=list)
    mask_smooth_history: list[float] = field(default_factory=list)
    bin_history: list[float] = field(default_factory=list)
    beta_history: list[float] = field(default_factory=list)


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
    shape_cfg: ShapeOptConfig,
    n_iters: int,
    lr: float,
    lambda_l2: float,
    lambda_neighbor: float,
    loss_fn: Callable | None = None,
    step_callback: Callable | None = None,
    mask_callback: Callable | None = None,
) -> OpenGeometryResult:
    """Adam loop over ``(cell_C_flat, cell_rho, logits)``.

    Material gradients use the mask-blended effective arrays (``m^p·C + (1−m^p)·C₀``
    and similarly for rho), so cells currently "outside" the shape get naturally
    small material gradients — shape decides where material matters, then
    material fine-tunes where shape has committed.

    β is looked up each step from :meth:`ShapeOptConfig.resolve_beta` and
    passed to the loss as a traced scalar, so a linear β ramp does not
    trigger re-tracing.
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
    material_init = (state["C"], state["rho"])

    # Static knobs (closed over; not traced):
    simp_p = float(shape_cfg.simp_p)
    sigma = float(shape_cfg.smooth_sigma)
    lambda_mask_smooth = float(shape_cfg.lambda_mask_smooth)
    lambda_bin = float(shape_cfg.lambda_bin)
    logit_scale = float(shape_cfg.logits_lr_mult)

    def _effective_logits(raw_logits: jnp.ndarray) -> jnp.ndarray:
        return smooth_logits(raw_logits, n_x, n_y, sigma)

    def combined_loss(s: dict[str, jnp.ndarray], beta: jnp.ndarray) -> jnp.ndarray:
        s_eff = _effective_logits(s["logits"])
        C_eff, rho_eff = apply_shape_mask(
            s["C"], s["rho"], s_eff, C0, rho0, beta=beta, simp_p=simp_p,
        )
        sol_list = fwd_pred((C_eff, rho_eff))
        u_cloak = sol_list[0]
        L_cloak = loss_fn(u_cloak, u_ref_boundary, boundary_idx)
        L_l2 = l2_regularization((s["C"], s["rho"]), material_init)
        L_nb = neighbor_regularization((s["C"], s["rho"]), nb_pairs)
        L_mask = mask_smoothness(s["logits"], mask_pairs)
        m_eff = occupancy(s_eff, beta)
        L_bin = binarization_penalty(m_eff)
        return (
            L_cloak
            + lambda_l2 * L_l2
            + lambda_neighbor * L_nb
            + lambda_mask_smooth * L_mask
            + lambda_bin * L_bin
        )

    loss_and_grad = jax.value_and_grad(combined_loss, argnums=0)
    opt_state = adam_init(state)

    loss_hist: list[float] = []
    cloak_hist: list[float] = []
    l2_hist: list[float] = []
    nb_hist: list[float] = []
    mask_hist: list[float] = []
    bin_hist: list[float] = []
    beta_hist: list[float] = []

    stop = _StopCondition(
        min_iters=n_iters,
        patience=shape_cfg.patience,
        max_iters=shape_cfg.max_iters,
        improvement_tol=shape_cfg.improvement_tol,
    )

    step = 0
    stop_reason = "loop did not enter"
    while True:
        # Resolve β against the *minimum* schedule so it stops ramping at
        # n_iters; later iterations stay at β_end.
        beta_t = shape_cfg.resolve_beta(min(step, n_iters - 1), n_iters)
        beta_jnp = jnp.float32(beta_t)

        loss_val, grads = loss_and_grad(state, beta_jnp)
        loss_val_f = float(loss_val)
        loss_hist.append(loss_val_f)

        # Component breakdown — cheap (no forward solve)
        s_eff = _effective_logits(state["logits"])
        m_eff = occupancy(s_eff, beta_jnp)
        L_l2_f = float(l2_regularization((state["C"], state["rho"]), material_init))
        L_nb_f = float(neighbor_regularization((state["C"], state["rho"]), nb_pairs))
        L_mask_f = float(mask_smoothness(state["logits"], mask_pairs))
        L_bin_f = float(binarization_penalty(m_eff))
        L_cloak_f = (
            loss_val_f
            - lambda_l2 * L_l2_f
            - lambda_neighbor * L_nb_f
            - lambda_mask_smooth * L_mask_f
            - lambda_bin * L_bin_f
        )
        cloak_hist.append(L_cloak_f)
        l2_hist.append(L_l2_f)
        nb_hist.append(L_nb_f)
        mask_hist.append(L_mask_f)
        bin_hist.append(L_bin_f)
        beta_hist.append(beta_t)

        m_vals = np.asarray(m_eff)
        grey = float(((m_vals > 0.1) & (m_vals < 0.9)).mean())
        print(
            f"  Step {step:4d} | β={beta_t:4.2f}"
            f"  total = {loss_val_f:.4e}"
            f"  cloak = {L_cloak_f:.4e}"
            f"  cloak_pct = {np.sqrt(max(L_cloak_f, 0.0)) * 100:.2e}"
            f"  tv(s) = {L_mask_f:.4e}  bin = {L_bin_f:.4e}"
            f"  solid = {(m_vals > 0.5).sum()}/{m_vals.size}  grey = {grey:.2%}"
        )

        if step_callback is not None:
            step_callback(
                step, loss_val_f, L_cloak_f, L_l2_f, L_nb_f,
                L_mask_f, L_bin_f, beta_t, state,
            )

        if (mask_callback is not None and shape_cfg.plot_mask_every > 0
                and step % shape_cfg.plot_mask_every == 0):
            mask_callback(step, np.asarray(m_vals).reshape(n_x, n_y))

        stop.update(loss_val_f)
        reason = stop.should_stop(step)
        if reason is not None:
            stop_reason = reason
            break

        if logit_scale != 1.0:
            grads = dict(grads)
            grads["logits"] = grads["logits"] * logit_scale

        updates, opt_state = adam_update(grads, opt_state, lr=lr)
        state = jax.tree.map(lambda p, u: p + u, state, updates)
        step += 1

    print(f"  Stopping: {stop_reason}  (after {step + 1} steps)")

    # Final mask uses the end-of-schedule β (i.e. β at the last "scheduled"
    # step — the ramp finishes at n_iters, plateau iterations stay at β_end).
    beta_final = shape_cfg.resolve_beta(n_iters - 1, n_iters)
    final_s_eff = _effective_logits(state["logits"])
    final_mask = np.asarray(occupancy(final_s_eff, jnp.float32(beta_final)))
    if mask_callback is not None:
        mask_callback(step + 1, final_mask.reshape(n_x, n_y))

    return OpenGeometryResult(
        cell_C_flat=state["C"],
        cell_rho=state["rho"],
        shape_logits=state["logits"],
        shape_mask=jnp.asarray(final_mask),
        n_x=n_x,
        n_y=n_y,
        beta_end=beta_final,
        loss_history=loss_hist,
        cloak_history=cloak_hist,
        l2_history=l2_hist,
        neighbor_history=nb_hist,
        mask_smooth_history=mask_hist,
        bin_history=bin_hist,
        beta_history=beta_hist,
    )


# ── multi-frequency loop ─────────────────────────────────────────────


def run_optimization_open_geometry_multifreq(
    freq_targets: list[FreqTarget],
    cell_params_init: tuple[jnp.ndarray, jnp.ndarray],
    logits_init: jnp.ndarray,
    neighbor_pairs: np.ndarray,
    mask_neighbor_pairs: np.ndarray,
    C0_flat: jnp.ndarray,
    rho0: float,
    n_x: int,
    n_y: int,
    shape_cfg: ShapeOptConfig,
    n_iters: int,
    lr: float,
    lambda_l2: float,
    lambda_neighbor: float,
    step_callback: Callable | None = None,
    mask_callback: Callable | None = None,
    max_workers: int = 0,
) -> OpenGeometryResult:
    """Multi-frequency variant of :func:`run_optimization_open_geometry`.

    Mirrors :func:`rayleigh_cloak.neural_reparam.run_optimization_neural_multifreq`:
    each frequency contributes its own weighted ``L_cloak`` via a separate
    ``value_and_grad`` dispatched in a thread pool (PETSc releases the GIL).
    The frequency-independent regularisers (L2, material TV, mask TV,
    binarisation) are evaluated once per step and their gradient summed in.

    The shape mask is applied before every per-frequency forward solve, so
    a single set of ``(C, ρ, logits)`` is optimised against all frequencies
    simultaneously.
    """
    n_freq = len(freq_targets)
    if max_workers <= 0:
        max_workers = n_freq

    C0 = jnp.asarray(C0_flat)
    nb_pairs = jnp.asarray(neighbor_pairs)
    mask_pairs = jnp.asarray(mask_neighbor_pairs)

    # Pre-convert per-freq boundary indices once
    for ft in freq_targets:
        ft.boundary_indices = jnp.asarray(ft.boundary_indices)

    cell_C_init, rho_init = cell_params_init
    state: dict[str, jnp.ndarray] = {
        "C": jnp.asarray(cell_C_init),
        "rho": jnp.asarray(rho_init),
        "logits": jnp.asarray(logits_init),
    }
    material_init = (state["C"], state["rho"])

    simp_p = float(shape_cfg.simp_p)
    sigma = float(shape_cfg.smooth_sigma)
    lambda_mask_smooth = float(shape_cfg.lambda_mask_smooth)
    lambda_bin = float(shape_cfg.lambda_bin)
    logit_scale = float(shape_cfg.logits_lr_mult)

    def _effective_logits(raw_logits: jnp.ndarray) -> jnp.ndarray:
        return smooth_logits(raw_logits, n_x, n_y, sigma)

    # Per-frequency cloak loss + grad closure.  Each captures its own
    # fwd_pred / u_ref / indices / loss_fn / weight; the shape mask is
    # applied inside the closure so material *and* logit gradients flow
    # through every frequency.
    def _make_freq_loss_and_grad(ft: FreqTarget):
        _fwd = ft.fwd_pred
        _u_ref = ft.u_ref_boundary
        _idx = ft.boundary_indices
        _lfn = ft.loss_fn
        _w = ft.weight

        def _loss(s: dict[str, jnp.ndarray], beta: jnp.ndarray) -> jnp.ndarray:
            s_eff = _effective_logits(s["logits"])
            C_eff, rho_eff = apply_shape_mask(
                s["C"], s["rho"], s_eff, C0, rho0, beta=beta, simp_p=simp_p,
            )
            sol_list = _fwd((C_eff, rho_eff))
            return _w * _lfn(sol_list[0], _u_ref, _idx)

        return jax.value_and_grad(_loss, argnums=0)

    freq_loss_and_grads = [_make_freq_loss_and_grad(ft) for ft in freq_targets]

    # Frequency-independent regularisers (L2 drift, material TV, mask TV,
    # binarisation).  β must be traced because L_bin depends on it.
    def _reg_loss(s: dict[str, jnp.ndarray], beta: jnp.ndarray) -> jnp.ndarray:
        L_l2 = l2_regularization((s["C"], s["rho"]), material_init)
        L_nb = neighbor_regularization((s["C"], s["rho"]), nb_pairs)
        L_mask = mask_smoothness(s["logits"], mask_pairs)
        s_eff = _effective_logits(s["logits"])
        m_eff = occupancy(s_eff, beta)
        L_bin = binarization_penalty(m_eff)
        return (
            lambda_l2 * L_l2
            + lambda_neighbor * L_nb
            + lambda_mask_smooth * L_mask
            + lambda_bin * L_bin
        )

    reg_loss_and_grad = jax.value_and_grad(_reg_loss, argnums=0)

    opt_state = adam_init(state)

    loss_hist: list[float] = []
    cloak_hist: list[float] = []
    l2_hist: list[float] = []
    nb_hist: list[float] = []
    mask_hist: list[float] = []
    bin_hist: list[float] = []
    beta_hist: list[float] = []

    f_star_str = ", ".join(
        f"{ft.f_star:.2f}(w={ft.weight:.2f})" for ft in freq_targets
    )
    print(f"  Multi-freq optimisation: {n_freq} frequencies [{f_star_str}]")
    print(f"  Thread pool: {max_workers} workers")

    pool = ThreadPoolExecutor(max_workers=max_workers)

    stop = _StopCondition(
        min_iters=n_iters,
        patience=shape_cfg.patience,
        max_iters=shape_cfg.max_iters,
        improvement_tol=shape_cfg.improvement_tol,
    )

    step = 0
    stop_reason = "loop did not enter"
    try:
        while True:
            # Hold β at β_end once the ramp finishes (plateau iters past n_iters).
            beta_t = shape_cfg.resolve_beta(min(step, n_iters - 1), n_iters)
            beta_jnp = jnp.float32(beta_t)

            # Per-freq cloak in parallel
            futures = [pool.submit(fn, state, beta_jnp) for fn in freq_loss_and_grads]
            results = [f.result() for f in futures]

            cloak_loss = sum(float(r[0]) for r in results)
            cloak_grad = jax.tree.map(
                lambda *gs: sum(gs),
                *(r[1] for r in results),
            )

            # Frequency-independent regularisers
            reg_val, reg_grad = reg_loss_and_grad(state, beta_jnp)
            reg_val_f = float(reg_val)
            grads = jax.tree.map(lambda a, b: a + b, cloak_grad, reg_grad)

            total = cloak_loss + reg_val_f
            loss_hist.append(total)
            cloak_hist.append(cloak_loss)

            # Component breakdown — cheap (no extra FEM solve)
            s_eff = _effective_logits(state["logits"])
            m_eff = occupancy(s_eff, beta_jnp)
            L_l2_f = float(l2_regularization((state["C"], state["rho"]), material_init))
            L_nb_f = float(neighbor_regularization((state["C"], state["rho"]), nb_pairs))
            L_mask_f = float(mask_smoothness(state["logits"], mask_pairs))
            L_bin_f = float(binarization_penalty(m_eff))
            l2_hist.append(L_l2_f)
            nb_hist.append(L_nb_f)
            mask_hist.append(L_mask_f)
            bin_hist.append(L_bin_f)
            beta_hist.append(beta_t)

            m_vals = np.asarray(m_eff)
            grey = float(((m_vals > 0.1) & (m_vals < 0.9)).mean())
            per_freq = "  ".join(
                f"f*={ft.f_star:.2f}:{float(r[0]):.4e}"
                for ft, r in zip(freq_targets, results)
            )
            print(
                f"  Step {step:4d} | β={beta_t:4.2f}"
                f"  total = {total:.4e}"
                f"  cloak = {cloak_loss:.4e}"
                f"  tv(s) = {L_mask_f:.4e}  bin = {L_bin_f:.4e}"
                f"  solid = {(m_vals > 0.5).sum()}/{m_vals.size}  grey = {grey:.2%}"
                f"\n    {per_freq}"
            )

            if step_callback is not None:
                step_callback(
                    step, total, cloak_loss, L_l2_f, L_nb_f,
                    L_mask_f, L_bin_f, beta_t, state,
                )

            if (mask_callback is not None and shape_cfg.plot_mask_every > 0
                    and step % shape_cfg.plot_mask_every == 0):
                mask_callback(step, np.asarray(m_vals).reshape(n_x, n_y))

            stop.update(total)
            reason = stop.should_stop(step)
            if reason is not None:
                stop_reason = reason
                break

            if logit_scale != 1.0:
                grads = dict(grads)
                grads["logits"] = grads["logits"] * logit_scale

            updates, opt_state = adam_update(grads, opt_state, lr=lr)
            state = jax.tree.map(lambda p, u: p + u, state, updates)
            step += 1
    finally:
        pool.shutdown(wait=False)

    print(f"  Stopping: {stop_reason}  (after {step + 1} steps)")

    beta_final = shape_cfg.resolve_beta(n_iters - 1, n_iters)
    final_s_eff = _effective_logits(state["logits"])
    final_mask = np.asarray(occupancy(final_s_eff, jnp.float32(beta_final)))
    if mask_callback is not None:
        mask_callback(step + 1, final_mask.reshape(n_x, n_y))

    return OpenGeometryResult(
        cell_C_flat=state["C"],
        cell_rho=state["rho"],
        shape_logits=state["logits"],
        shape_mask=jnp.asarray(final_mask),
        n_x=n_x,
        n_y=n_y,
        beta_end=beta_final,
        loss_history=loss_hist,
        cloak_history=cloak_hist,
        l2_history=l2_hist,
        neighbor_history=nb_hist,
        mask_smooth_history=mask_hist,
        bin_history=bin_hist,
        beta_history=beta_hist,
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

    print("=== Step 2: Extracting submesh (removing defect elements) ===")
    cloak_mesh, kept_nodes = extract_submesh(full_mesh, geometry)
    print(f"  Submesh: {len(cloak_mesh.points)} nodes, "
          f"{len(cloak_mesh.cells)} elements")

    print("=== Step 3: Setting up cell decomposition ===")
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

    material_neighbors = cell_decomp.get_neighbor_pairs()
    mask_neighbors = all_neighbor_pairs(cell_decomp.n_x, cell_decomp.n_y)
    print(f"  {len(material_neighbors)} material neighbour pairs, "
          f"{len(mask_neighbors)} shape neighbour pairs")

    print("=== Step 4: Initialising shape logits from triangular cloak ===")
    logits_init = init_logits_from_cloak_mask(
        cell_decomp.cloak_mask, magnitude=shape_cfg.init_magnitude,
    )
    print(f"  init_magnitude = {shape_cfg.init_magnitude}, "
          f"beta = {shape_cfg.beta}, "
          f"sigmoid(beta·mag) ≈ "
          f"{float(jax.nn.sigmoid(shape_cfg.beta * shape_cfg.init_magnitude)):.3f}")

    solver_opts = _petsc_opts(config)

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

    opt_cfg = config.optimization
    beta_end = shape_cfg.beta if shape_cfg.beta_end is None else shape_cfg.beta_end

    # ── Multi-frequency branch ────────────────────────────────────────
    # Mirrors the dispatch in :func:`rayleigh_cloak.solver.solve_optimization_neural`.
    mf = config.loss.multi_freq
    if mf.f_min is not None and mf.f_max is not None and mf.f_step is not None:
        f_stars = list(np.arange(mf.f_min, mf.f_max + 0.5 * mf.f_step, mf.f_step))
        weights = [1.0] * len(f_stars)
    elif mf.f_stars:
        f_stars = mf.f_stars
        weights = mf.weights if mf.weights else [1.0] * len(f_stars)
        if len(weights) != len(f_stars):
            raise ValueError(
                f"multi_freq.weights length ({len(weights)}) must match "
                f"f_stars length ({len(f_stars)})"
            )
    else:
        f_stars = []
        weights = []

    if f_stars:
        print(f"=== Step 5: Building per-frequency problems "
              f"({len(f_stars)} frequencies) ===")
        freq_targets: list[FreqTarget] = []
        for f_star, weight in zip(f_stars, weights):
            cfg_f = _make_config_at_fstar(config, f_star)
            dp_f = DerivedParams.from_config(cfg_f)

            print(f"  f*={f_star:.2f}: solving reference ...", end="", flush=True)
            ref_f = solve_reference(cfg_f, mesh=full_mesh)

            problem_f = build_problem(cloak_mesh, cfg_f, dp_f, geometry, cell_decomp)
            fwd_pred_f = ad_wrapper(problem_f, solver_opts, solver_opts)

            indices_f, u_ref_f, loss_fn_f = resolve_loss_target(
                config.loss.type, np.asarray(cloak_mesh.points), geometry,
                dp_f, kept_nodes, ref_f.u,
            )
            print(f" {len(indices_f)} loss nodes, weight={weight:.2f}")

            freq_targets.append(FreqTarget(
                f_star=f_star,
                weight=weight,
                fwd_pred=fwd_pred_f,
                u_ref_boundary=u_ref_f,
                boundary_indices=indices_f,
                loss_fn=loss_fn_f,
            ))

        print("=== Step 6: Optimising (multi-freq open-geometry) ===")
        print(
            f"  β schedule: {shape_cfg.beta} → {beta_end}  |  "
            f"SIMP p = {shape_cfg.simp_p}  |  smooth σ = {shape_cfg.smooth_sigma}  |  "
            f"λ_bin = {shape_cfg.lambda_bin}"
        )
        return run_optimization_open_geometry_multifreq(
            freq_targets=freq_targets,
            cell_params_init=params_init,
            logits_init=logits_init,
            neighbor_pairs=material_neighbors,
            mask_neighbor_pairs=mask_neighbors,
            C0_flat=C0_flat,
            rho0=float(params.rho0),
            n_x=cell_decomp.n_x,
            n_y=cell_decomp.n_y,
            shape_cfg=shape_cfg,
            n_iters=opt_cfg.n_iters,
            lr=opt_cfg.lr,
            lambda_l2=opt_cfg.lambda_l2,
            lambda_neighbor=opt_cfg.lambda_neighbor,
            step_callback=step_callback,
            mask_callback=mask_callback,
            max_workers=mf.max_workers,
        )

    # ── Single-frequency path (unchanged) ─────────────────────────────
    print("=== Step 5: Solving reference problem (on full mesh) ===")
    ref_result = solve_reference(config, mesh=full_mesh)

    print("=== Step 6: Building FEM problem ===")
    problem = build_problem(cloak_mesh, config, params, geometry, cell_decomp)

    boundary_indices, u_ref_boundary, loss_fn = resolve_loss_target(
        config.loss.type, np.asarray(cloak_mesh.points), geometry, params,
        kept_nodes, ref_result.u,
    )
    print(f"  {len(boundary_indices)} loss nodes ({config.loss.type})")

    fwd_pred = ad_wrapper(problem, solver_opts, solver_opts)

    print("=== Step 7: Optimising ===")
    print(
        f"  β schedule: {shape_cfg.beta} → {beta_end}  |  "
        f"SIMP p = {shape_cfg.simp_p}  |  smooth σ = {shape_cfg.smooth_sigma}  |  "
        f"λ_bin = {shape_cfg.lambda_bin}"
    )
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
        shape_cfg=shape_cfg,
        n_iters=opt_cfg.n_iters,
        lr=opt_cfg.lr,
        lambda_l2=opt_cfg.lambda_l2,
        lambda_neighbor=opt_cfg.lambda_neighbor,
        loss_fn=loss_fn,
        step_callback=step_callback,
        mask_callback=mask_callback,
    )
