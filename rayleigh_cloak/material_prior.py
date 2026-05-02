"""Dataset-based material prior for the cloaking optimisation.

Loads a Gaussian-mixture .npz produced by ``dataset.cellular_chiral.fit_gmm``
and provides a JAX-traceable flat-top penalty:

    penalty(C, rho) = mean over cloak cells of  max(0, threshold - log p(λ, μ, ρ))

The penalty is zero whenever the cell sits comfortably inside the dataset's
density support (log p ≥ threshold) and grows linearly as the cell drifts
into the tails. See ``fit_gmm.py`` for how ``threshold`` is set (a quantile
of the dataset's own log p, default the 25th percentile so the borders are
already penalised).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class GMMPrior:
    """Pre-loaded GMM, ready for use in a JAX loss.

    All arrays live on the JAX device. The Cholesky factor of the *precision*
    (i.e. inverse covariance) is stored directly — it's what sklearn already
    computes during fit, and it lets us evaluate log-prob without an inverse.

    Attributes
    ----------
    weights              : (K,)              mixture weights
    means                : (K, 3)            in standardised (λ, μ, ρ) space
    precisions_cholesky  : (K, 3, 3)         Cholesky of inverse covariance
    feature_mean         : (3,)              standardisation mean (raw → std)
    feature_std          : (3,)              standardisation std
    threshold            : scalar            flat-top threshold τ
    """
    weights: jnp.ndarray
    means: jnp.ndarray
    precisions_cholesky: jnp.ndarray
    feature_mean: jnp.ndarray
    feature_std: jnp.ndarray
    threshold: jnp.ndarray


def load_gmm_prior(path: str | Path, dtype=jnp.float32) -> GMMPrior:
    """Load a .npz produced by ``fit_gmm.py`` into a :class:`GMMPrior`."""
    data = np.load(str(path), allow_pickle=True)
    cov_type = str(data["covariance_type"])
    if cov_type != "full":
        raise NotImplementedError(
            f"GMMPrior currently supports covariance_type='full' only; "
            f"got {cov_type!r}. Refit with `--covariance-type=full`."
        )
    return GMMPrior(
        weights=jnp.asarray(data["weights"], dtype=dtype),
        means=jnp.asarray(data["means"], dtype=dtype),
        precisions_cholesky=jnp.asarray(data["precisions_cholesky"], dtype=dtype),
        feature_mean=jnp.asarray(data["feature_mean"], dtype=dtype),
        feature_std=jnp.asarray(data["feature_std"], dtype=dtype),
        threshold=jnp.asarray(float(data["threshold"]), dtype=dtype),
    )


# ── log-prob and penalty ────────────────────────────────────────────


def _gmm_log_prob_standardised(x_std: jnp.ndarray, prior: GMMPrior) -> jnp.ndarray:
    """log p(x) for x already in standardised space.

    Uses sklearn's convention where ``precisions_cholesky`` is L such that
    Σ⁻¹ = L L^T, i.e. y = (x - μ) @ L gives the Mahalanobis vector with
    ‖y‖² = (x - μ)^T Σ⁻¹ (x - μ).

    Parameters
    ----------
    x_std : (..., 3) standardised features
    prior : GMMPrior
    """
    d = prior.means.shape[-1]
    diff = x_std[..., None, :] - prior.means              # (..., K, d)
    y = jnp.einsum("...kd,kde->...ke", diff, prior.precisions_cholesky)  # (..., K, d)
    mahal_sq = jnp.sum(y * y, axis=-1)                    # (..., K)
    log_det_pc = jnp.sum(
        jnp.log(jnp.diagonal(prior.precisions_cholesky, axis1=-2, axis2=-1)),
        axis=-1,
    )                                                     # (K,)
    log_per_k = -0.5 * d * jnp.log(2.0 * jnp.pi) + log_det_pc - 0.5 * mahal_sq
    log_w = jnp.log(prior.weights)
    return jax.scipy.special.logsumexp(log_w + log_per_k, axis=-1)  # (...)


def _flat_to_lame(cell_C_flat: jnp.ndarray, n_C_params: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Extract (λ, μ) per cell from the cell_C_flat array.

    For ``n_C_params=2`` the flat layout is already [λ, μ]. For other
    parameterisations we go through the (2,2,2,2) tensor representation
    and read λ = C[0,0,1,1], μ = C[0,1,0,1] — the same indices used by
    ``materials.C_to_flat2``.
    """
    if n_C_params == 2:
        return cell_C_flat[..., 0], cell_C_flat[..., 1]

    # Lazy import to keep this module light when only the GMM bits are used.
    from rayleigh_cloak.materials import _get_converters

    _, from_flat = _get_converters(n_C_params)
    cell_C_full = jax.vmap(from_flat)(cell_C_flat)        # (n_cells, 2,2,2,2)
    lam = cell_C_full[..., 0, 0, 1, 1]
    mu = cell_C_full[..., 0, 1, 0, 1]
    return lam, mu


def gmm_flat_top_penalty(
    cell_C_flat: jnp.ndarray,
    cell_rho: jnp.ndarray,
    cloak_mask: jnp.ndarray,
    prior: GMMPrior,
    n_C_params: int,
) -> jnp.ndarray:
    """Mean of ``max(0, τ - log p(λ, μ, ρ))`` over cloak cells.

    Parameters
    ----------
    cell_C_flat : (n_cells, n_C_params)  per-cell stiffness in flat form
    cell_rho    : (n_cells,)             per-cell density
    cloak_mask  : (n_cells,) bool        which cells are inside the cloak
    prior       : GMMPrior
    n_C_params  : layout of cell_C_flat

    Returns
    -------
    scalar JAX value — the regularisation term to be multiplied by ``weight``.
    """
    lam, mu = _flat_to_lame(cell_C_flat, n_C_params)
    feat = jnp.stack([lam, mu, cell_rho], axis=-1)        # (n_cells, 3)
    feat_std = (feat - prior.feature_mean) / prior.feature_std
    log_p = _gmm_log_prob_standardised(feat_std, prior)   # (n_cells,)
    pen = jnp.maximum(0.0, prior.threshold - log_p)       # flat-top
    mask = cloak_mask.astype(pen.dtype)
    # Mean over cloak cells (avoid division by zero on the empty-mask edge).
    denom = jnp.maximum(jnp.sum(mask), 1.0)
    return jnp.sum(pen * mask) / denom
