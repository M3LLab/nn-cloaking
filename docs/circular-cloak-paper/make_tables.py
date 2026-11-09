"""Write the paper's result tables from the saved run data, and print the numbers quoted in the text.

    .venv/bin/python docs/circular-cloak-paper/make_tables.py
"""
import glob, json, math, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "output")
T = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables")
J = lambda *p: json.load(open(os.path.join(OUT, *p)))

def sci(x, n=2):
    if x == 0: return "0"
    m, e = f"{x:.{n}e}".split("e")
    return f"${m}\\times 10^{{{int(e)}}}$"
def num(x, n=3): return f"{x:.{n}f}"
def emit(name, cols, head, rows):
    s = ["\\begin{tabular}{" + cols + "}", "\\toprule", head + " \\\\", "\\midrule"]
    s += [" & ".join(r) + " \\\\" for r in rows]
    s += ["\\bottomrule", "\\end{tabular}", ""]
    open(os.path.join(T, name + ".tex"), "w").write("\n".join(s))

V = "circular_step1/vertical"
b = J(V, "base.json")
emit("base", "lcc", "Comparison & $e$ (magnitude) & $e^{c}$ (complex)", [
  ["B: hole without cloak vs.\\ A", sci(b["mag_B_vs_ref"]), sci(b["B_vs_ref"])],
  ["C: regularised cloak vs.\\ A", sci(b["mag_C_vs_ref"]), sci(b["C_vs_ref"])],
  ["\\quad C vs.\\ A on the ring $r=1.25\\,b$", sci(b["ring1.25b_mag_C_vs_ref"]), sci(b["ring1.25b_C_vs_ref"])],
  ["\\quad C vs.\\ A on the ring $r=2\\,b$", sci(b["ring2b_mag_C_vs_ref"]), sci(b["ring2b_C_vs_ref"])],
  ["D: $\\varepsilon$-hole vs.\\ A", sci(b["mag_D_vs_ref"]), sci(b["D_vs_ref"])],
  ["C vs.\\ D, exterior", sci(b["mag_C_vs_D"]), sci(b["C_vs_D"])],
  ["$\\mathbf{u}_C(\\mathbf{x})$ vs.\\ $\\mathbf{u}_D(\\Xi^{-1}(\\mathbf{x}))$, annulus", sci(b["mag_interior"]), sci(b["interior"])],
  ["A vs.\\ closed form, Eq.~\\eqref{eq:green}", sci(b["mag_A_vs_analytic"]), sci(b["A_vs_analytic"])]])

a = 0.5
E = J(V, "eps.json")
emit("eps", "ccccccc", "$\\varepsilon/a$ & $h(r{=}a)$ [mm] & C vs.\\ A & D vs.\\ A & C vs.\\ D & interior & C vs.\\ A, $e^{c}$",
  [[f"{r['eps']/a:g}", f"{r['h_inner']*1e3:.1f}", sci(r["mag_C_vs_ref"]), sci(r["mag_D_vs_ref"]), sci(r["mag_C_vs_D"]), sci(r["mag_interior"]), sci(r["C_vs_ref"])] for r in E])

M = J(V, "mesh.json")
ls = 1.5
emit("mesh", "ccccccc", "$\\lambda_S/h_{\\mathrm{bg}}$ & Elements & A vs.\\ closed form & A vs.\\ finest & C vs.\\ A & D vs.\\ A & C vs.\\ D",
  [[f"{ls/r['h_bg']:.1f}", f"{r['n_cells_A']:,}".replace(",", "\\,"), sci(r["mag_A_vs_analytic"]), ("ref." if r["A_vs_finest"] == 0 else sci(r["A_vs_finest"])), sci(r["mag_C_vs_ref"]), sci(r["mag_D_vs_ref"]), sci(r["mag_C_vs_D"])] for r in M])

def layer(v):
    d = json.loads(v)
    if d.get("absorbing.kind") == "damping":
        return f"damping, $L={d['absorbing.thickness']:g}$\\,m, $\\xi_{{\\max}}={d['absorbing.xi_max']:g}$, $p={d['absorbing.power']}$"
    L = d.get("absorbing.thickness", 2.6); R = d.get("absorbing.reflection", 1e-6); p = d.get("absorbing.power", 2)
    return f"PML, $L={L:g}$\\,m, $R=10^{{{round(math.log10(R))}}}$, $p={p}$"
Ab = J(V, "absorbing.json")
rows = [["PML, $L=2.6$\\,m, $R=10^{-6}$, $p=2$ (base)", f"{b['n_cells_A']:,}".replace(",", "\\,"), sci(b["mag_A_vs_analytic"]), sci(b["A_vs_analytic"]), sci(b["mag_C_vs_ref"])]]
rows += [[layer(r["variant"]), f"{r['n_cells_A']:,}".replace(",", "\\,"), sci(r["mag_A_vs_analytic"]), sci(r["A_vs_analytic"]), sci(r["mag_C_vs_ref"])] for r in Ab]
emit("layers", "lcccc", "Layer & Elements & A vs.\\ closed form, $e$ & A vs.\\ closed form, $e^{c}$ & C vs.\\ A, $e$", rows)

mp = J(V, "map.json")[0]
emit("map", "lccc", "Radial map & C vs.\\ A & D vs.\\ A & C vs.\\ D", [
  ["linear", sci(b["mag_C_vs_ref"]), sci(b["mag_D_vs_ref"]), sci(b["mag_C_vs_D"])],
  ["quadratic", sci(mp["mag_C_vs_ref"]), sci(mp["mag_D_vs_ref"]), sci(mp["mag_C_vs_D"])]])

LAB = {"vertical": "vertical", "horizontal": "horizontal", "both": "$(1,1)/\\sqrt{2}$"}
rows = []
for root, src in (("circular_step1", "$(-3,-3)$"), ("circular_step1_offdiag", "$(-3.5,-2)$")):
    for f in LAB:
        q = J(root, f, "base.json")
        rows.append([src, LAB[f], num(q["mag_B_vs_ref"]), sci(q["mag_C_vs_ref"]), sci(q["mag_D_vs_ref"]), sci(q["mag_C_vs_D"]), f"{q['mag_B_vs_ref']/q['mag_C_vs_ref']:.0f}"])
emit("forces", "cccccccc"[:7], "Source & Force & B vs.\\ A & C vs.\\ A & D vs.\\ A & C vs.\\ D & B/C", rows)

R = sorted((json.load(open(p)) for p in glob.glob(os.path.join(OUT, "circular_frequency_sweep", "f_*.json"))), key=lambda r: r["f_star"])
pick = [0.1, 0.4, 0.5, 1.0, 1.5, 2.0, 2.5]
rows = []
for r in R:
    if r["f_star"] not in pick: continue
    s = r["sources"]["off_diagonal"]["forces"]
    rows.append([f"{r['f_star']:g}", f"{r['f_star']*200:.0f}"] + sum(([num(s[f]["ratio_beyond_B"]), num(s[f]["mag_B_vs_A"]), sci(s[f]["mag_C_vs_A"], 1)] for f in ("vertical", "horizontal")), []))
emit("sweep", "cccccccc", "$f^*$ & $f$ [Hz] & $\\eta_B$ (V) & B vs.\\ A (V) & C vs.\\ A (V) & $\\eta_B$ (H) & B vs.\\ A (H) & C vs.\\ A (H)", rows)

D = J("circular_frequency_sweep", "ratio_diagnostic.json")
def row(lab, k, fmt): return [lab] + [fmt(D[f][F][k]) for f in ("0.4", "1") for F in ("vertical", "both")]
s3 = lambda x: f"${x:+.3f}$"; p3 = lambda x: f"{x:.3f}"
emit("diagnostic", "lcccc", " & $f^*=0.4$, V & $f^*=0.4$, both & $f^*=1$, V & $f^*=1$, both", [
  row("$\\eta_B$ beyond the cloak", "ratio", p3), row("\\quad near the axis, $|d_\\perp|<2$\\,m", "ratio_near_axis", p3), row("\\quad to the sides, $|d_\\perp|\\geq 2$\\,m", "ratio_sides", p3),
  row("P share of the energy of $\\mathbf{u}_A$", "p_share", lambda x: f"{x:.2f}"), row("$\\langle\\|\\mathbf{u}_A\\|\\rangle$ [$10^{-7}$\\,m]", "ref_mean", lambda x: f"{x*1e7:.2f}"),
  row("scattered energy $\\langle\\|\\mathbf{u}_{\\mathrm{sc}}\\|^2\\rangle/\\langle\\|\\mathbf{u}_A\\|^2\\rangle$", "scattered_energy", s3),
  row("interference $2\\langle\\mathrm{Re}\\,\\mathbf{u}_A^*\\!\\cdot\\mathbf{u}_{\\mathrm{sc}}\\rangle/\\langle\\|\\mathbf{u}_A\\|^2\\rangle$", "interference", s3)])

# ── numbers quoted in the prose ──
cs = [(q["mag_C_vs_A"], r["f_star"]) for r in R for v in r["sources"].values() for q in v["forces"].values()]
bs = [q["mag_B_vs_A"] for r in R for v in r["sources"].values() for q in v["forces"].values()]
eta = [q["ratio_beyond_B"] for r in R for v in r["sources"].values() for q in v["forces"].values()]
dev = [abs(q["ratio_beyond_C"] - 1) for r in R for v in r["sources"].values() for q in v["forces"].values()]
an = [q["mag_A_vs_analytic"] for r in R for v in r["sources"].values() for q in v["forces"].values()]
anc = [q["complex_A_vs_analytic"] for r in R for v in r["sources"].values() for q in v["forces"].values()]
print("sweep C min/max", min(cs), max(cs)); print("sweep B min/max", min(bs), max(bs)); print("eta_B min/max", min(eta), max(eta))
print("eta_C max dev", max(dev)); print("A vs analytic max mag/complex", max(an), max(anc))
print("points", len(cs))
print("eps slopes D:", [round(math.log(E[i]["mag_D_vs_ref"] / E[i+1]["mag_D_vs_ref"]) / math.log(E[i]["eps"] / E[i+1]["eps"]), 2) for i in range(len(E)-1)])
print("mesh orders A vs analytic:", [round(math.log(M[i]["mag_A_vs_analytic"] / M[i+1]["mag_A_vs_analytic"]) / math.log(M[i]["h_bg"] / M[i+1]["h_bg"]), 2) for i in range(len(M)-1)])
print("mesh orders complex:", [round(math.log(M[i]["A_vs_analytic"] / M[i+1]["A_vs_analytic"]) / math.log(M[i]["h_bg"] / M[i+1]["h_bg"]), 2) for i in range(len(M)-1)])
print("n ext", J(V, "base", "metrics.json")["n_exterior_points"], "cloak stats", J(V, "base", "metrics.json")["info"]["C"]["cloak"], "dofs", b["n_dofs_A"], "peak", b["peak_rss_gib"])
print("base sizes", J(V, "base", "metrics.json")["info"]["A"]["mesh_sizes"])

# ── Step 2: pixelated and optimised cloaks ─────────────────────────────
S2 = os.path.join(ROOT, "docs", "figures", "circular_step2")
J2 = lambda *p: json.load(open(os.path.join(S2, *p)))
STEP1 = [[-3.0, -3.0], [-3.5, -2.0]]
base2 = J2("baseline.json")
hole = J2("hole_reference.json")


def st(rows, f, group="test", sources=None):
    q = [r["e"] for r in rows if abs(r["f_star"] - f) < 1e-9 and r["group"] == group and (sources is None or r["source"] in sources)]
    return (float(np.median(q)), float(max(q))) if q else (float("nan"), float("nan"))


import numpy as np  # noqa: E402
grids = [("50x10", 500), ("36x7", 252), ("24x5", 120), ("15x4", 60)]
worst = lambda m, mat, g: max(r["e"] for r in base2 if r["map"] == m and r["material"] == mat and r["grid"] == g)
rows = []
for m in ("linear", "quadratic"):
    rows.append([m, "hole without cloak", "--", sci(worst(m, "hole (B)", "-"))])
    rows.append([m, "ideal, continuous", "--", sci(worst(m, "ideal continuous (C)", "-"))])
    rows.append([m, "symmetrised, continuous", "--", sci(worst(m, "symmetrised continuous", "-"))])
    for g, n in grids:
        rows.append([m, "ideal pixels, centre / average", str(n), sci(worst(m, "ideal pixels, centre", g)) + " / " + sci(worst(m, "ideal pixels, average", g))])
        rows.append([m, "symmetrised pixels, average", str(n), sci(worst(m, "symmetrised pixels, average", g))])
emit("s2_baseline", "lllc", "Map & Material & Pixels & worst $e$", rows)

RUNS = [("f1_500", "1", "500", "quadratic"), ("f1_500_linear", "1", "500", "linear"), ("f1_252", "1", "252", "quadratic"),
        ("f1_120", "1", "120", "quadratic"), ("f1_60", "1", "60", "quadratic"),
        ("band_500", "0.5, 1, 1.5", "500", "quadratic"), ("pool_500", "pool [0.5, 1.5]", "500", "quadratic")]
rows = []
for r, fs, n, m in RUNS:
    R = J2("runs", r, "evaluation.json")
    H = [json.loads(l) for l in open(os.path.join(S2, "runs", r, "history.jsonl"))]
    tm, tw = st(R, 1.0)
    trm, _ = st(R, 1.0, "train")
    sw = st(R, 1.0, sources=STEP1)[1]
    rows.append([r.replace("_", "\\_"), fs, n, m, f"{H[-1]['step']}", num(trm, 3), num(tm, 3), num(tw, 3), num(sw, 3)])
emit("s2_runs", "lllllcccc", "Run & Training $f^*$ & Pixels & Map & Steps & train, median & test, median & test, worst & Step 1 sources, worst", rows)

fs = [0.5, 0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.4, 1.5, 1.75, 2.0]
E = {r: J2("runs", r, "evaluation.json") for r in ("f1_500", "band_500", "pool_500")}
def c(R, f):
    m_, w_ = st(R, f)
    return "--" if np.isnan(m_) else f"{m_:.3f} / {w_:.3f}"
rows = [[f"{f:g}", c(hole, f), c(E["f1_500"], f), c(E["band_500"], f), c(E["pool_500"], f)] for f in fs]
emit("s2_frequency", "ccccc", "$f^*$ & hole, no cloak & single frequency & three frequencies & frequency pool", rows)

P = J2("pilot_evaluation.json")
PILOT_SRC = {"off_diagonal": "$(-3.5,-2)$, trained", "on_diagonal": "$(-3,-3)$, not trained"}
rows = [[PILOT_SRC[r["source"]], r["force"], sci(r["e"]), sci(r["e_c"])] for r in P]
emit("s2_pilot", "llcc", "Source & Force & $e$ & $e^c$", rows)

print("--- step 2 numbers")
for r, *_ in RUNS:
    R = J2("runs", r, "evaluation.json")
    H = [json.loads(l) for l in open(os.path.join(S2, "runs", r, "history.jsonl"))]
    print(r, "test f1", st(R, 1.0), "step1 src", st(R, 1.0, sources=STEP1), "s/step", round(np.mean([h["seconds"] for h in H[-50:]]), 1),
          "val first/last", round(H[0]["val_rms_ec"], 4), round(H[-1]["val_rms_ec"], 4))
print("hole", {f: st(hole, f) for f in fs if not np.isnan(st(hole, f)[0])})
pm = [st(E["pool_500"], f) for f in fs[:9]]
hm = [st(hole, f) for f in (0.5, 0.75, 1.0, 1.25, 1.5)]
print("pool band median range", min(x[0] for x in pm), max(x[0] for x in pm), "worst range", min(x[1] for x in pm), max(x[1] for x in pm))
print("ratios hole/pool median at 0.5,.75,1,1.25,1.5", [round(st(hole, f)[0] / st(E['pool_500'], f)[0], 2) for f in (0.5, 0.75, 1.0, 1.25, 1.5)],
      "worst", [round(st(hole, f)[1] / st(E['pool_500'], f)[1], 2) for f in (0.5, 0.75, 1.0, 1.25, 1.5)])
print("pilot", [(r["source"], r["force"], round(r["e"], 4)) for r in P])
import sys; sys.path.insert(0, ROOT)
import jax  # noqa: E402
from rayleigh_cloak.embedded.neural_field import polar_features  # noqa: E402
from rayleigh_cloak.embedded.pixels import PolarPixelGrid  # noqa: E402
nf = polar_features(PolarPixelGrid(0.5, 1.5, 0, 0, 50, 10), 8, 4).shape[1]
print("features", nf, "parameters", nf * 128 + 128 + 2 * (128 * 128 + 128) + 128 * 7 + 7)
for r in ("f1_500", "pool_500"):
    d = np.load(os.path.join(S2, "runs", r, "best_design.npz"))
    C = d["C"]
    V = np.stack([C[:, 0, 0, 0, 0], C[:, 0, 0, 1, 1], C[:, 0, 0, 0, 1], C[:, 1, 1, 0, 0], C[:, 1, 1, 1, 1], C[:, 1, 1, 0, 1], C[:, 0, 1, 0, 0], C[:, 0, 1, 1, 1], C[:, 0, 1, 0, 1]], 1).reshape(-1, 3, 3)
    ev = np.linalg.eigvalsh(V) / 144e6
    print(r, "Voigt eig/mu min max", ev.min().round(3), ev.max().round(1), "rho/rho0", (d["rho"] / 1600).min().round(3), (d["rho"] / 1600).max().round(3), "best step", int(d["step"]))
