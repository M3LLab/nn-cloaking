# Step 1: comparison of force directions

Exterior normalised L2 error of the total displacement magnitude ‖u‖ on the common sampling grid (`mag_`); the last rows give the complex-displacement error for reference. Each direction is a separate solve.

## Base case

| metric | vertical | horizontal | both |
|---|---|---|---|
| mag_B_vs_ref | 1.581e-01 | 1.581e-01 | 1.044e-01 |
| mag_C_vs_ref | 6.681e-05 | 6.665e-05 | 3.356e-05 |
| mag_D_vs_ref | 7.667e-05 | 7.668e-05 | 3.766e-05 |
| mag_C_vs_D | 1.183e-05 | 1.191e-05 | 5.124e-06 |
| mag_interior | 4.648e-05 | 4.629e-05 | 2.078e-05 |
| mag_A_vs_analytic | 5.000e-05 | 5.061e-05 | 5.083e-05 |
| ring1.25b_mag_C_vs_ref | 1.012e-04 | 1.007e-04 | 6.176e-05 |
| ring2b_mag_C_vs_ref | 7.677e-05 | 7.664e-05 | 4.224e-05 |
| B_vs_ref | 3.116e-01 | 3.116e-01 | 2.217e-01 |
| C_vs_ref | 1.301e-04 | 1.300e-04 | 7.608e-05 |
| C_vs_D | 2.610e-05 | 2.603e-05 | 1.178e-05 |
| A_vs_analytic | 2.489e-04 | 2.456e-04 | 2.378e-04 |

## C vs A against eps (‖u‖ error)

| eps | vertical | horizontal | both |
|---|---|---|---|
| 1.000e-01 | 3.118e-02 | 3.118e-02 | 1.538e-02 |
| 5.000e-02 | 7.872e-03 | 7.872e-03 | 3.859e-03 |
| 2.500e-02 | 1.884e-03 | 1.884e-03 | 9.321e-04 |
| 1.000e-02 | 2.826e-04 | 2.823e-04 | 1.413e-04 |
| 5.000e-03 | 6.681e-05 | 6.665e-05 | 3.356e-05 |
| 2.500e-03 | 1.619e-05 | 1.628e-05 | 8.024e-06 |

## Mirror check

The problem is symmetric about y = x, so in the continuum the horizontal field is the vertical field reflected (x <-> y, u_x <-> u_y). The directions are therefore not independent tests at this source position; the numbers below measure the mesh's asymmetry. Complex-displacement error, since this compares the fields themselves.

| case | horizontal vs mirrored vertical |
|---|---|
| A | 2.643e-04 |
| B | 2.635e-04 |
| C | 2.642e-04 |
| D | 2.644e-04 |

## Superposition check

The problem is linear, so the `both` solve should equal (u_V + u_H) / sqrt(2). Complex-displacement error: magnitudes do not superpose.

| case | both vs (u_V + u_H) / sqrt(2) |
|---|---|
| A | 1.483e-14 |
| B | 1.517e-14 |
| C | 9.028e-14 |
| D | 1.399e-14 |
