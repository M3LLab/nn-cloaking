"""Cloaking loss metrics.

Provides :func:`compute_cloaking_loss` which measures how well the cloaked
field matches the reference field on all physical boundaries and across the
full physical domain outside the cloak.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rayleigh_cloak.optimize import (
    get_outside_cloak_indices,
    get_right_boundary_indices,
    get_all_physical_boundary_indices,
)


@dataclass
class CloakingLoss:
    dist_boundary: float  # distortion % on all four physical boundaries
    dist_right: float     # distortion % on right physical boundary only
    dist_outside: float   # distortion % over all physical nodes outside cloak
    n_boundary: int       # number of nodes on all physical boundaries
    n_right: int          # number of nodes on right boundary
    n_outside: int        # number of nodes outside cloak


def _relative_l2(u_cloak: np.ndarray, u_ref: np.ndarray) -> float:
    """Relative L2 displacement difference: ||u_cloak - u_ref||^2 / ||u_ref||^2."""
    diff = u_cloak - u_ref
    ref_norm_sq = float(np.sum(u_ref ** 2)) + 1e-30
    return float(np.sum(diff ** 2) / ref_norm_sq)


def _distortion_pct(u_cloak: np.ndarray, u_ref: np.ndarray) -> float:
    """100 * ||u_cloak - u_ref|| / ||u_ref||."""
    return 100.0 * np.sqrt(_relative_l2(u_cloak, u_ref))


def compute_cloaking_loss(
    cloak_result,
    ref_result,
    geometry,
    tol: float = 1e-3,
) -> CloakingLoss:
    """Compute cloaking distortion metrics.

    Parameters
    ----------
    cloak_result:
        ``SolutionResult`` from the cloaked simulation.  Must have
        ``.mesh``, ``.u``, ``.params``, and ``.kept_nodes`` attributes.
    ref_result:
        ``SolutionResult`` from the reference (no-cloak) simulation on the
        shared full mesh.  Must have ``.u``.
    geometry:
        Cloak geometry object exposing ``in_cloak()`` / ``in_defect()``.
    tol:
        Spatial tolerance passed to the boundary/region index selectors.
    """
    params = cloak_result.params
    kept_nodes = cloak_result.kept_nodes
    pts = np.asarray(cloak_result.mesh.points)

    # All four physical boundaries
    bnd_idx = get_all_physical_boundary_indices(
        pts, params.x_off, params.y_off, params.W, params.H, tol=tol,
    )
    u_ref_bnd = ref_result.u[kept_nodes[bnd_idx]]
    u_cloak_bnd = cloak_result.u[bnd_idx]
    dist_boundary = _distortion_pct(u_cloak_bnd, u_ref_bnd)

    # Right physical boundary only
    x_right = params.x_off + params.W
    right_idx = get_right_boundary_indices(pts, x_right, tol=tol)
    u_ref_right = ref_result.u[kept_nodes[right_idx]]
    u_cloak_right = cloak_result.u[right_idx]
    dist_right = _distortion_pct(u_cloak_right, u_ref_right)

    # All physical-domain nodes outside cloak
    outside_idx = get_outside_cloak_indices(
        pts, geometry,
        params.x_off, params.y_off, params.W, params.H,
        tol=tol,
    )
    u_ref_outside = ref_result.u[kept_nodes[outside_idx]]
    u_cloak_outside = cloak_result.u[outside_idx]
    dist_outside = _distortion_pct(u_cloak_outside, u_ref_outside)

    return CloakingLoss(
        dist_boundary=dist_boundary,
        dist_right=dist_right,
        dist_outside=dist_outside,
        n_boundary=len(bnd_idx),
        n_right=len(right_idx),
        n_outside=len(outside_idx),
    )
