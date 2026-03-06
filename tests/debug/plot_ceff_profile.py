"""Plot C_eff tensor components along a vertical line through the cloak.

Samples OFF-axis (x = x_c + c/4) so that F_21 != 0 and all Cosserat
components are non-trivial.  Uses the augmented Voigt 4x4 representation:

    indices:  0 -> (1,1),  1 -> (2,2),  2 -> (1,2),  3 -> (2,1)

The 4x4 matrix M_{IJ} = C_{ijkl} where I->(i,j), J->(k,l).
With major symmetry M = M^T, the upper triangle has 10 unique entries.
"""
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from boundaries import (
    C_eff, C0, C_to_voigt4,
    x_c, y_top, a, b, c,
)

_a, _b, _c = float(a), float(b), float(c)

# Sample OFF-axis so F21 != 0  →  all Cosserat components visible
x_line = float(x_c) + _c / 4.0

N = 300
depths = np.linspace(_a, 1.5 * _b, N)
ys = float(y_top) - depths
pts = jnp.array([[x_line, y] for y in ys])

# Evaluate C_eff and convert to augmented Voigt 4x4
C_all = jax.vmap(C_eff)(pts)                     # (N, 2,2,2,2)
M_all = jax.vmap(C_to_voigt4)(C_all)             # (N, 4, 4)
M0    = np.array(C_to_voigt4(C0))                # (4, 4)

# Augmented Voigt index labels
voigt_labels = ["11", "22", "12", "21"]

# Upper triangle of the symmetric 4x4 → 10 unique entries
components = {}
for I in range(4):
    for J in range(I, 4):
        label = f"$M_{{{voigt_labels[I]},{voigt_labels[J]}}}$"
        components[label] = (I, J)


# ── Plot ──
fig, ax = plt.subplots(figsize=(10, 6))

for label, (I, J) in components.items():
    vals = np.array(M_all[:, I, J])
    # Skip if identically zero everywhere
    if np.max(np.abs(vals)) < 1e-3:
        continue

    ax.plot(depths, vals / 1e9, label=label, linewidth=1.5)
    # Dashed horizontal line for reference value
    ref_val = M0[I, J] / 1e9
    if abs(ref_val) > 1e-6:
        ax.axhline(ref_val, color=ax.get_lines()[-1].get_color(),
                   ls=':', lw=0.6, alpha=0.5)

ax.axvline(_a, color='grey', ls='--', lw=2.0, label='inner (a)')
ax.axvline(_b, color='grey', ls=':',  lw=2.0, label='outer (b)')

ax.set_xlabel("Depth from free surface [m]")
ax.set_ylabel(r"$C_{ijkl}$ [GPa]")
ax.set_title(
    f"Augmented-Voigt C_eff components  "
    f"(x = x_c + c/4,  depth a → b)"
)
ax.legend(fontsize=8, ncol=2, loc="best")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("output/ceff_profile.png", dpi=150)
plt.show()
print("Saved → output/ceff_profile.png")
