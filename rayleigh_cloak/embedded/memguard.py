"""Memory guard: refuse to mesh / assemble / factorise when RAM would run out.

A failed run killed by the OOM killer (exit 137) loses all work of that
process, so every substantial step first estimates its peak memory and
compares it with ``MemAvailable`` from /proc/meminfo.

Estimates (TRI6, 4 real DOFs per node; calibrated on this code, see
``BYTES_PER_CELL``):
  * meshing           ~ 2 kB per element (gmsh + meshio copies)
  * assembly + solve  ~ BYTES_PER_CELL per element: jax-fem element Jacobians
    (24x24 doubles) + I/J index arrays, the CSR copy, and the PARDISO factor.
"""

from __future__ import annotations

import gc
import os
import resource

import numpy as np

# Peak resident bytes per TRI6 element for one assemble+solve (calibrated:
# see docs/circular_cloak_step1.md, section 6).
BYTES_PER_CELL = {"TRI6": 75_000, "TRI3": 15_000}   # measured ~71 kB/TRI6 cell
MESH_BYTES_PER_CELL = 2_000
SAFETY = 1.25                 # margin on top of the estimate
RESERVE_BYTES = 1.5 * 2**30   # always leave this much for the OS / other jobs


class InsufficientMemory(RuntimeError):
    pass


def available_bytes() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))


def peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _gib(x: float) -> str:
    return f"{x / 2**30:.2f} GiB"


def check(need_bytes: float, what: str, verbose: bool = True) -> None:
    """Raise :class:`InsufficientMemory` if ``need_bytes`` (x SAFETY) plus the
    reserve exceeds the currently available RAM."""
    gc.collect()
    avail = available_bytes()
    need = need_bytes * SAFETY
    if verbose:
        print(f"  [mem] {what}: need ~{_gib(need)} (incl. x{SAFETY} margin), "
              f"available {_gib(avail)}")
    if need + RESERVE_BYTES > avail:
        raise InsufficientMemory(
            f"{what}: estimated {_gib(need)} + reserve {_gib(RESERVE_BYTES)} exceeds "
            f"available {_gib(avail)}. Coarsen the mesh (mesh.scale / "
            f"elems_per_wavelength) or free memory.")


def estimate_cells(cfg, sizes: dict, hole_radius: float) -> int:
    """Rough element count from the size field (equilateral-triangle packing)."""
    c = cfg.cloak
    X0, X1, Y0, Y1 = cfg.outer_box
    k = 2.31  # triangles per h^2 of area
    area_tot = (X1 - X0) * (Y1 - Y0)
    area_disk_b = np.pi * c.b ** 2
    n = k * (area_tot - area_disk_b) / sizes["h_bg"] ** 2
    n += k * area_disk_b / sizes["h_cloak"] ** 2
    # graded band at the inner circle: h grows linearly from h_in to h_cloak
    h0, h1, G = sizes["h_inner"], sizes["h_cloak"], sizes["inner_grading"]
    if h1 > h0:
        n += k * 2 * np.pi * hole_radius * G * (1 / (h0 * h1))
    # source refinement disk
    rs = 4 * cfg.source.sigma
    n += k * np.pi * rs ** 2 / sizes["h_src"] ** 2
    return int(n)


def check_mesh(cfg, sizes: dict, hole_radius: float) -> int:
    n = estimate_cells(cfg, sizes, hole_radius)
    check(n * MESH_BYTES_PER_CELL, f"meshing (~{n} elements est.)")
    # also check that the subsequent solve will fit, before spending time meshing
    check(n * BYTES_PER_CELL.get(cfg.mesh.ele_type, 60_000),
          f"solve on that mesh (~{n} elements est.)")
    return n


def check_solve(n_cells: int, ele_type: str, what: str = "assembly + solve") -> None:
    check(n_cells * BYTES_PER_CELL.get(ele_type, 60_000), f"{what} ({n_cells} elements)")
