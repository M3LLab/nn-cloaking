# Step 1: comparison of force directions

Exterior normalised L2 error of the total displacement magnitude ‖u‖ on the common sampling grid (`mag_`); the last rows give the complex-displacement error for reference. Each direction is a separate solve.

## Base case

| metric | vertical | horizontal | both |
|---|---|---|---|
| mag_B_vs_ref | 1.876e-01 | 1.293e-01 | 1.110e-01 |
| mag_C_vs_ref | 7.946e-05 | 5.358e-05 | 4.021e-05 |
| mag_D_vs_ref | 9.125e-05 | 6.131e-05 | 4.546e-05 |
| mag_C_vs_D | 1.402e-05 | 8.989e-06 | 6.242e-06 |
| mag_interior | 3.561e-05 | 4.507e-05 | 3.852e-05 |
| mag_A_vs_analytic | 4.904e-05 | 4.968e-05 | 5.073e-05 |
| ring1.25b_mag_C_vs_ref | 9.574e-05 | 9.434e-05 | 7.163e-05 |
| ring2b_mag_C_vs_ref | 8.397e-05 | 6.108e-05 | 4.578e-05 |
| B_vs_ref | 3.583e-01 | 2.706e-01 | 2.360e-01 |
| C_vs_ref | 1.524e-04 | 1.064e-04 | 8.512e-05 |
| C_vs_D | 3.241e-05 | 2.077e-05 | 1.452e-05 |
| A_vs_analytic | 2.536e-04 | 2.395e-04 | 2.387e-04 |

## Superposition check

The problem is linear, so the `both` solve should equal (u_V + u_H) / sqrt(2). Complex-displacement error: magnitudes do not superpose.

| case | both vs (u_V + u_H) / sqrt(2) |
|---|---|
| A | 1.586e-14 |
| B | 1.863e-14 |
| C | 5.705e-14 |
| D | 1.481e-14 |
