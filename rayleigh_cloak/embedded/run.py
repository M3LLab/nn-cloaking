"""Run the comparison cases for one configuration.

Cases (all with identical source, frequency, domain and absorbing layers):

A  homogeneous background, no hole            (full mesh, incl. the disk r < a)
B  traction-free hole r < a, no cloak          (same mesh with the disk removed)
C  hole + regularised ideal BGM cloak a<r<b    (same submesh as B)
D  virtual problem: traction-free hole r < eps (own mesh). By construction of
   the regularised map, C must reproduce D exactly outside r = b and
   D(Xi^{-1}(x)) inside the cloak, so C-vs-D isolates implementation and
   discretisation error, while D-vs-A is the intrinsic near-cloak error of
   the eps regularisation.

A, B and C share every node outside r = a, so their exterior discretisation
errors are identical; metrics are nevertheless computed on a mesh-independent
grid of sampling points so that different meshes can be compared too.
"""

from __future__ import annotations

import json
import os
import time

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from rayleigh_cloak.embedded.config import EmbeddedCloakConfig
from rayleigh_cloak.embedded import memguard
from rayleigh_cloak.embedded.linsolve import solve_linear
from rayleigh_cloak.embedded.mesh import TaggedMesh, build_mesh
from rayleigh_cloak.embedded.problem import build_problem, make_geometry
from rayleigh_cloak.embedded.sampling import (
    PointEvaluator, annulus_grid, exterior_grid, rel_l2, ring_points,
)

ALL_CASES = ("A", "B", "C", "D")


class CaseResult:
    def __init__(self, name, mesh: TaggedMesh, u: np.ndarray, info: dict):
        self.name, self.mesh, self.u, self.info = name, mesh, u, info

    def save(self, path: str) -> None:
        np.savez_compressed(path, points=self.mesh.points, cells=self.mesh.cells,
                            region=self.mesh.region, u=self.u,
                            ele_type=self.mesh.ele_type,
                            info=json.dumps(self.info))

    @staticmethod
    def load(path: str) -> "CaseResult":
        z = np.load(path, allow_pickle=False)
        mesh = TaggedMesh(z["points"], z["cells"], z["region"], str(z["ele_type"]), {})
        return CaseResult(os.path.basename(path)[5:6], mesh, z["u"], json.loads(str(z["info"])))

    def evaluator(self, xy):
        return PointEvaluator(self.mesh.points, self.mesh.cells, xy)


def _solve(name: str, mesh: TaggedMesh, cfg: EmbeddedCloakConfig, cloak_on: bool) -> CaseResult:
    memguard.check_solve(mesh.n_cells, mesh.ele_type, f"case {name} assembly + solve")
    t0 = time.time()
    problem = build_problem(mesh, cfg, cloak_on=cloak_on)
    t_setup = time.time() - t0
    u4, info = solve_linear(problem)
    u = u4[:, :2] + 1j * u4[:, 2:]
    info.update(case=name, n_nodes=mesh.n_nodes, n_cells=mesh.n_cells, t_setup=t_setup,
                source=problem.source_stats, mesh_sizes=mesh.sizes)
    if cloak_on:
        info["cloak"] = problem.cloak_stats
    info["peak_rss_gib"] = memguard.peak_rss_bytes() / 2**30
    print(f"  case {name}: {mesh.n_nodes} nodes, {info['n_dofs']} dofs, "
          f"setup {t_setup:.1f}s asm {info['t_assemble']:.1f}s solve {info['t_solve']:.1f}s, "
          f"res {info['rel_residual']:.1e}, src-norm ratio {problem.source_stats['ratio']:.5f}, "
          f"peak RSS {info['peak_rss_gib']:.1f} GiB")
    del problem
    return CaseResult(name, mesh, u, info)


def run_cases(cfg: EmbeddedCloakConfig, cases=ALL_CASES, out_dir: str | None = None,
              tag: str = "") -> dict[str, CaseResult]:
    """Solve the requested cases; save ``case_X.npz`` + ``config.json`` in out_dir."""
    out_dir = out_dir or cfg.output_dir
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        f.write(cfg.model_dump_json(indent=2))
    print(f"=== run {tag or out_dir}: eps={cfg.cloak.eps} map={cfg.cloak.radial_map} "
          f"scale={cfg.mesh.scale} L_abs={cfg.absorbing.thickness} xi_max={cfg.absorbing.xi_max}")
    res: dict[str, CaseResult] = {}
    need_full = any(c in cases for c in "ABC")
    if need_full:
        full = build_mesh(cfg, msh_path=os.path.join(out_dir, "mesh_full.msh"))
        sub, _ = full.without_disk()
        if "A" in cases:
            res["A"] = _solve("A", full, cfg, cloak_on=False)
        if "B" in cases:
            res["B"] = _solve("B", sub, cfg, cloak_on=False)
        if "C" in cases:
            res["C"] = _solve("C", sub, cfg, cloak_on=True)
    if "D" in cases:
        if cfg.cloak.eps <= 0:
            print("  case D skipped (eps = 0 has no virtual hole)")
        else:
            meshD = build_mesh(cfg, hole_radius=cfg.cloak.eps, hole_is_eps=True,
                               msh_path=os.path.join(out_dir, "mesh_virtual.msh"))
            subD, _ = meshD.without_disk()
            res["D"] = _solve("D", subD, cfg, cloak_on=False)
    for k, r in res.items():
        r.save(os.path.join(out_dir, f"case_{k}.npz"))
    return res


def load_cases(out_dir: str) -> dict[str, CaseResult]:
    out = {}
    for k in ALL_CASES:
        p = os.path.join(out_dir, f"case_{k}.npz")
        if os.path.exists(p):
            out[k] = CaseResult.load(p)
    return out


def compute_metrics(cfg: EmbeddedCloakConfig, res: dict[str, CaseResult],
                    ref: CaseResult | None = None) -> dict:
    """Normalised complex-displacement L2 errors on the common sampling sets.

    ``ref`` overrides case A as the reference (used for cross-mesh studies).
    """
    ref = ref if ref is not None else res.get("A")
    P = exterior_grid(cfg)
    out = {"n_exterior_points": int(len(P)), "exterior": {}, "rings": {}}
    vals = {k: r.evaluator(P)(r.u) for k, r in res.items()}
    u_ref = ref.evaluator(P)(ref.u) if ref is not None else None
    if u_ref is not None:
        out["ref_rms"] = float(np.sqrt(np.nanmean(np.sum(np.abs(u_ref) ** 2, axis=1))))
        for k, v in vals.items():
            if res.get(k) is not ref:
                out["exterior"][f"{k}_vs_ref"] = rel_l2(v, u_ref)
    if "A" in vals:
        from rayleigh_cloak.embedded.analytic import green_gaussian
        out["exterior"]["A_vs_analytic"] = rel_l2(vals["A"], green_gaussian(cfg, P))
    if "C" in vals and "D" in vals:
        out["exterior"]["C_vs_D"] = rel_l2(vals["C"], vals["D"])
    for rho in cfg.metrics.ring_radii_over_b:
        Pr = ring_points(cfg, rho * cfg.cloak.b)
        ring = {}
        u_rr = ref.evaluator(Pr)(ref.u) if ref is not None else None
        for k, r in res.items():
            if u_rr is not None and r is not ref:
                ring[f"{k}_vs_ref"] = rel_l2(r.evaluator(Pr)(r.u), u_rr)
        out["rings"][f"{rho:g}b"] = ring
    # interior mapping check: u_C(x) vs u_D(Xi^{-1}(x)) inside the annulus
    if "C" in res and "D" in res:
        geo = make_geometry(cfg)
        h = min(cfg.metrics.grid_spacing, (cfg.cloak.b - cfg.cloak.a) / 40)
        Pin = annulus_grid(cfg, h, inset=1e-3 * cfg.cloak.b)
        Xv = np.asarray(jax.vmap(geo.virtual_point)(jax.numpy.asarray(Pin)))
        uC = res["C"].evaluator(Pin)(res["C"].u)
        uD = res["D"].evaluator(Xv)(res["D"].u)
        out["interior_C_vs_D_mapped"] = rel_l2(uC, uD)
        # same but excluding the eps boundary layer next to r = a (outer 90 %)
        r = np.hypot(Pin[:, 0] - cfg.cloak.xc, Pin[:, 1] - cfg.cloak.yc)
        far = r > cfg.cloak.a + 0.1 * (cfg.cloak.b - cfg.cloak.a)
        out["interior_C_vs_D_mapped_outer90"] = rel_l2(uC[far], uD[far])
    return out
