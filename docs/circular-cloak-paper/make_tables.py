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
