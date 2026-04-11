"""Neural reparameterization of cell-based material fields.

Instead of optimizing raw per-cell (C_flat, rho) arrays directly, a small MLP
maps cell-center coordinates to material parameters.  The MLP weights become
the optimization variables; gradients flow through the FEM adjoint and then
back through the network via standard JAX autodiff.

The MLP's smoothness bias replaces explicit neighbor regularization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from rayleigh_cloak.cells import CellDecomposition
from rayleigh_cloak.optimize import (
    AdamState,
    adam_init,
    adam_update,
    cloaking_loss,
    l2_regularization,
)


# ── MLP definition (pure JAX) ───────────────────────────────────────


def _init_layer(key, n_in, n_out):
    """Xavier-uniform initialization for a single dense layer."""
    k1, k2 = jax.random.split(key)
    bound = jnp.sqrt(6.0 / (n_in + n_out))
    W = jax.random.uniform(k1, (n_in, n_out), minval=-bound, maxval=bound)
    b = jnp.zeros(n_out)
    return {"W": W, "b": b}


def init_mlp(key, layer_sizes: list[int]) -> list[dict]:
    """Initialize an MLP as a list of {W, b} dicts."""
    params = []
    for i in range(len(layer_sizes) - 1):
        key, subkey = jax.random.split(key)
        params.append(_init_layer(subkey, layer_sizes[i], layer_sizes[i + 1]))
    return params


def mlp_forward(params: list[dict], x: jnp.ndarray) -> jnp.ndarray:
    """Forward pass: Dense → tanh → ... → Dense (no final activation)."""
    h = x
    for layer in params[:-1]:
        h = jnp.tanh(h @ layer["W"] + layer["b"])
    last = params[-1]
    return h @ last["W"] + last["b"]


def fourier_features(xy: jnp.ndarray, n_freq: int = 32) -> jnp.ndarray:
    """Map (n, 2) coordinates to (n, 4*n_freq) Fourier features.

    Frequencies are log-spaced from 1 to ``n_freq`` to capture both
    large-scale gradients and sharper spatial transitions.
    """
    freqs = jnp.linspace(1.0, float(n_freq), n_freq)  # (n_freq,)
    # (n, 2) @ (2,) → project each coord onto each freq
    proj_x = xy[:, 0:1] * freqs[None, :]  # (n, n_freq)
    proj_y = xy[:, 1:2] * freqs[None, :]  # (n, n_freq)
    return jnp.concatenate([
        jnp.sin(proj_x), jnp.cos(proj_x),
        jnp.sin(proj_y), jnp.cos(proj_y),
    ], axis=-1)  # (n, 4*n_freq)


def random_fourier_features(
    xy: jnp.ndarray,
    n_freq: int = 256,
    sigma: float = 10.0,
    key: jax.random.PRNGKey | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Map (n, 2) coordinates to (n, 2*n_freq) random Fourier features.

    Draws a random projection matrix B ~ N(0, sigma^2) of shape (2, n_freq).
    The features are [sin(2π xy B), cos(2π xy B)].  Higher sigma enables
    the network to represent higher spatial frequencies.

    Returns (features, B) so that B can be stored for reproducibility.
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    B = sigma * jax.random.normal(key, (2, n_freq))  # (2, n_freq)
    proj = 2.0 * jnp.pi * xy @ B                      # (n, n_freq)
    features = jnp.concatenate([jnp.sin(proj), jnp.cos(proj)], axis=-1)
    return features, B


# ── Weight I/O ──────────────────────────────────────────────────────


def save_theta(
    theta: list[dict],
    path: str,
    opt_state: AdamState | None = None,
) -> None:
    """Save MLP weights (and optionally Adam state) to an .npz file."""
    arrays = {}
    for i, layer in enumerate(theta):
        arrays[f"W_{i}"] = np.asarray(layer["W"])
        arrays[f"b_{i}"] = np.asarray(layer["b"])
    arrays["n_layers"] = np.array(len(theta))
    if opt_state is not None:
        arrays["adam_t"] = np.array(opt_state.t)
        for i, layer in enumerate(opt_state.m):
            arrays[f"adam_m_W_{i}"] = np.asarray(layer["W"])
            arrays[f"adam_m_b_{i}"] = np.asarray(layer["b"])
        for i, layer in enumerate(opt_state.v):
            arrays[f"adam_v_W_{i}"] = np.asarray(layer["W"])
            arrays[f"adam_v_b_{i}"] = np.asarray(layer["b"])
    np.savez(path, **arrays)


def load_theta(path: str) -> tuple[list[dict], AdamState | None]:
    """Load MLP weights (and Adam state if present) from an .npz file.

    Returns (theta, opt_state) where opt_state is None if not saved.
    """
    data = np.load(path)
    n_layers = int(data["n_layers"])
    theta = []
    for i in range(n_layers):
        theta.append({
            "W": jnp.array(data[f"W_{i}"]),
            "b": jnp.array(data[f"b_{i}"]),
        })
    opt_state = None
    if "adam_t" in data:
        t = int(data["adam_t"])
        m = []
        v = []
        for i in range(n_layers):
            m.append({
                "W": jnp.array(data[f"adam_m_W_{i}"]),
                "b": jnp.array(data[f"adam_m_b_{i}"]),
            })
            v.append({
                "W": jnp.array(data[f"adam_v_W_{i}"]),
                "b": jnp.array(data[f"adam_v_b_{i}"]),
            })
        opt_state = AdamState(m=m, v=v, t=t)
    return theta, opt_state


# ── Reparameterization ───────────────────────────────────────────────


@dataclass
class NeuralReparam:
    """Wraps an MLP that maps cell coordinates to material parameters.

    Attributes
    ----------
    cell_centers_norm : (n_cells, 2) normalized to [0, 1]
    cell_features : (n_cells, n_features) Fourier features
    C_flat_init : (n_cells, n_C_params) initial stiffness
    rho_init : (n_cells,) initial density
    C_scale : (n_C_params,) per-component std of initial C across cloak cells
    rho_scale : float std of initial rho across cloak cells
    cloak_mask : (n_cells,) bool — which cells are in the cloak
    """

    cell_features: jnp.ndarray
    C_flat_init: jnp.ndarray
    rho_init: jnp.ndarray
    cloak_mask: jnp.ndarray
    output_scale: float = 0.1

    def decode(self, theta: list[dict]) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Map MLP weights → (cell_C_flat, cell_rho).

        Uses a *relative* (multiplicative) residual so that the correction
        is proportional to the local initial value:
            C(x,y) = C_init(x,y) * (1 + output_scale * Phi(x,y))
        This handles the orders-of-magnitude variation in C_eff across the
        cloak (near-zero at r_i, ~background at r_c) without needing
        per-component normalization.
        Non-cloak cells keep their initial (background) values.
        """
        raw = mlp_forward(theta, self.cell_features)  # (n_cells, n_C_params+1)
        n_C = self.C_flat_init.shape[1]

        rel_C = raw[:, :n_C] * self.output_scale    # relative correction
        rel_rho = raw[:, n_C] * self.output_scale

        cell_C = self.C_flat_init * (1.0 + rel_C * self.cloak_mask[:, None])
        cell_rho = self.rho_init * (1.0 + rel_rho * self.cloak_mask)

        return (cell_C, cell_rho)


def make_neural_reparam(
    cell_decomp: CellDecomposition,
    params_init: tuple[jnp.ndarray, jnp.ndarray],
    hidden_size: int = 256,
    n_layers: int = 4,
    n_fourier: int = 32,
    seed: int = 42,
    output_scale: float = 0.1,
) -> tuple[list[dict], NeuralReparam]:
    """Create a NeuralReparam and initialize the MLP weights.

    Returns
    -------
    theta : MLP parameters (list of {W, b})
    reparam : NeuralReparam instance with decode() method
    """
    cell_C_init, cell_rho_init = params_init
    n_cells, n_C_params = cell_C_init.shape
    n_out = n_C_params + 1  # C_flat components + 1 rho

    # Normalize cell centers to [0, 1] for better conditioning
    centers = jnp.array(cell_decomp.cell_centers)
    lo = centers.min(axis=0)
    hi = centers.max(axis=0)
    centers_norm = (centers - lo) / (hi - lo + 1e-10)

    features = fourier_features(centers_norm, n_fourier)
    n_features = features.shape[1]

    mask = jnp.array(cell_decomp.cloak_mask)

    # MLP: features → hidden → ... → hidden → n_out
    layer_sizes = [n_features] + [hidden_size] * (n_layers - 1) + [n_out]
    key = jax.random.PRNGKey(seed)
    theta = init_mlp(key, layer_sizes)

    # Scale down the last layer so initial MLP output ≈ 0 (start near init params)
    theta[-1]["W"] = theta[-1]["W"] * 0.01
    theta[-1]["b"] = theta[-1]["b"] * 0.0

    reparam = NeuralReparam(
        cell_features=features,
        C_flat_init=cell_C_init,
        rho_init=cell_rho_init,
        cloak_mask=mask,
        output_scale=output_scale,
    )

    return theta, reparam


# ── Optimization loop ────────────────────────────────────────────────


@dataclass
class NeuralOptimizationResult:
    """Result of neural-reparameterized optimization."""
    theta: list[dict]
    best_theta: list[dict]  # weights at lowest total loss
    opt_state: AdamState     # final Adam state (for warm restart)
    params: tuple[jnp.ndarray, jnp.ndarray]  # final decoded (cell_C, cell_rho)
    loss_history: list[float] = field(default_factory=list)
    cloak_history: list[float] = field(default_factory=list)
    l2_history: list[float] = field(default_factory=list)


def run_optimization_neural(
    fwd_pred,
    params_init: tuple[jnp.ndarray, jnp.ndarray],
    u_ref_boundary: jnp.ndarray,
    boundary_indices: jnp.ndarray,
    reparam: NeuralReparam,
    theta_init: list[dict],
    n_iters: int = 100,
    lr: float = 1e-3,
    lr_end: float | None = None,
    lr_schedule: str = "linear",
    lambda_l2: float = 1e-4,
    plot_callback=None,
    plot_every: int = 1,
    step_callback=None,
    opt_state_init: AdamState | None = None,
) -> NeuralOptimizationResult:
    """Run optimization over MLP weights (neural reparameterization).

    Parameters
    ----------
    fwd_pred : callable
        Differentiable forward prediction from ``ad_wrapper``.
    params_init : (cell_C_flat, cell_rho) — original initial material values
    reparam : NeuralReparam with decode()
    theta_init : initial MLP weights
    opt_state_init : if provided, resume Adam from this state (warm restart)
    step_callback : optional callable(step, total, cloak, l2, neighbor, params)
        Same signature as raw optimization for compatibility.
    """
    theta = jax.tree.map(jnp.copy, theta_init)
    opt_state = opt_state_init if opt_state_init is not None else adam_init(theta)
    loss_history: list[float] = []
    cloak_history: list[float] = []
    l2_history: list[float] = []

    best_loss = float("inf")
    best_theta = jax.tree.map(jnp.copy, theta)

    boundary_indices_jnp = jnp.array(boundary_indices)

    def loss_fn(theta):
        params = reparam.decode(theta)
        sol_list = fwd_pred(params)
        u_cloak = sol_list[0]
        L_cloak = cloaking_loss(u_cloak, u_ref_boundary, boundary_indices_jnp)
        L_l2 = l2_regularization(params, params_init)
        return L_cloak + lambda_l2 * L_l2, L_cloak

    loss_and_grad = jax.value_and_grad(
        lambda t: loss_fn(t)[0],
        has_aux=False,
    )

    # Separate function to get cloak loss (no extra cost — we decompose from total)
    def _get_components(theta):
        params = reparam.decode(theta)
        L_l2 = float(l2_regularization(params, params_init))
        return L_l2

    for step in range(n_iters):
        # Learning rate schedule
        t_frac = step / max(n_iters - 1, 1)
        if lr_end is None:
            cur_lr = lr
        elif lr_schedule == "cosine":
            cur_lr = lr_end + 0.5 * (lr - lr_end) * (1 + math.cos(math.pi * t_frac))
        else:
            cur_lr = lr + (lr_end - lr) * t_frac

        loss_val, grads = loss_and_grad(theta)
        loss_val_float = float(loss_val)
        loss_history.append(loss_val_float)

        L_l2 = _get_components(theta)
        L_cloak = loss_val_float - lambda_l2 * L_l2
        cloak_history.append(L_cloak)
        l2_history.append(L_l2)

        grad_norm = float(jnp.sqrt(sum(
            jnp.sum(l["W"]**2) + jnp.sum(l["b"]**2) for l in grads
        )))
        print(
            f"  Step {step:4d} | total = {loss_val_float:.4e}"
            f"  cloak_pct = {np.sqrt(max(L_cloak, 0)) * 100:.2f}"
            f"  L2 = {L_l2:.4e}"
            f"  lr={cur_lr:.2e}"
            f"  |grad|={grad_norm:.4e}"
        )

        if loss_val_float < best_loss:
            best_loss = loss_val_float
            best_theta = jax.tree.map(jnp.copy, theta)

        if step_callback is not None:
            params = reparam.decode(theta)
            step_callback(step, loss_val_float, L_cloak, L_l2, 0.0, params)

        if plot_callback is not None and step % plot_every == 0:
            params = reparam.decode(theta)
            sol_list = fwd_pred(params)
            plot_callback(step, np.asarray(sol_list[0]))

        updates, opt_state = adam_update(grads, opt_state, lr=cur_lr)
        theta = jax.tree.map(lambda p, u: p + u, theta, updates)

    # Final state
    final_params = reparam.decode(theta)

    if plot_callback is not None:
        sol_list = fwd_pred(final_params)
        plot_callback(n_iters, np.asarray(sol_list[0]))

    return NeuralOptimizationResult(
        theta=theta,
        best_theta=best_theta,
        opt_state=opt_state,
        params=final_params,
        loss_history=loss_history,
        cloak_history=cloak_history,
        l2_history=l2_history,
    )
