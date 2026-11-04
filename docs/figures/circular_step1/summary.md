# Step 1 — continuous regularised ideal cloak: results

Normalised complex-displacement L2 error on the common exterior sampling grid (spacing 0.05 m, r > 1.02 b, |x − x_s| > 0.75 m, physical box only).  `X_vs_ref` = ‖u_X − u_A‖ / ‖u_A‖.

## Base case

| metric | value |
|---|---|
| eps | 5.000e-03 |
| B_vs_ref | 3.116e-01 |
| C_vs_ref | 1.330e-04 |
| D_vs_ref | 1.509e-04 |
| C_vs_D | 3.759e-05 |
| A_vs_analytic | 9.475e-04 |
| interior | 1.653e-04 |
| ring1.25b_B_vs_ref | 4.406e-01 |
| ring1.25b_C_vs_ref | 1.890e-04 |
| ring1.25b_D_vs_ref | 2.136e-04 |
| ring2b_B_vs_ref | 3.280e-01 |
| ring2b_C_vs_ref | 1.408e-04 |
| ring2b_D_vs_ref | 1.590e-04 |
| n_cells_A | 88470 |
| n_dofs_A | 710580 |
| h_bg | 1.250e-01 |
| h_inner | 6.689e-03 |
| peak_rss_gib | 6.649e+00 |

## eps study

| eps | B_vs_ref | C_vs_ref | D_vs_ref | C_vs_D | interior | h_inner | n_cells_A |
|---|---|---|---|---|---|---|---|
| 1.000e-01 | 3.116e-01 | 6.265e-02 | 6.229e-02 | 3.859e-04 | 9.947e-04 | 2.000e-02 | 80564 |
| 5.000e-02 | 3.116e-01 | 1.565e-02 | 1.564e-02 | 2.023e-05 | 1.794e-04 | 2.000e-02 | 80564 |
| 2.500e-02 | 3.116e-01 | 3.728e-03 | 3.825e-03 | 1.122e-04 | 3.989e-04 | 2.000e-02 | 80564 |
| 1.000e-02 | 3.116e-01 | 5.457e-04 | 6.053e-04 | 7.580e-05 | 2.985e-04 | 1.342e-02 | 82490 |
| 5.000e-03 | 3.116e-01 | 1.330e-04 | 1.509e-04 | 3.759e-05 | 1.653e-04 | 6.689e-03 | 88470 |
| 2.500e-03 | 3.116e-01 | 4.012e-05 | 3.763e-05 | 3.223e-05 | 1.460e-04 | 3.339e-03 | 100390 |

## mesh study

| scale | h_bg | n_cells_A | n_dofs_A | A_vs_analytic | B_vs_ref | C_vs_ref | D_vs_ref | C_vs_D | A_vs_finest | C_vs_finest | peak_rss_gib |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2.000e+00 | 2.500e-01 | 23128 | 186436 | 6.923e-03 | 3.116e-01 | 4.121e-04 | 1.480e-04 | 4.048e-04 | 6.782e-03 | 6.910e-03 | 2.294e+00 |
| 1.400e+00 | 1.750e-01 | 46016 | 370148 | 2.170e-03 | 3.117e-01 | 1.494e-04 | 1.488e-04 | 1.052e-04 | 1.962e-03 | 1.994e-03 | 3.831e+00 |
| 1.000e+00 | 1.250e-01 | 88470 | 710580 | 9.475e-04 | 3.116e-01 | 1.330e-04 | 1.509e-04 | 3.759e-05 | 5.800e-04 | 5.854e-04 | 6.833e+00 |
| 7.500e-01 | 9.375e-02 | 154820 | 1242324 | 7.428e-04 | 3.116e-01 | 1.365e-04 | 1.516e-04 | 2.019e-05 | 0.000e+00 | 0.000e+00 | 1.138e+01 |

## absorbing study

| variant | A_vs_analytic | A_vs_baseA | B_vs_ref | C_vs_ref | n_cells_A |
|---|---|---|---|---|---|
| {"absorbing.thickness": 4.0, "absorbing.xi_max": 4.0, "absorbing.power": 2} | 3.527e-02 | 3.519e-02 | 3.096e-01 | 1.321e-04 | 64386 |
| {"absorbing.thickness": 4.0} | 3.630e-03 | 3.037e-03 | 3.113e-01 | 1.329e-04 | 64386 |
| {"absorbing.thickness": 8.0} | 7.068e-04 | 4.246e-04 | 3.116e-01 | 1.330e-04 | 116578 |
| {"absorbing.xi_max": 0.75} | 2.399e-03 | 2.327e-03 | 3.120e-01 | 1.331e-04 | 88470 |
| {"absorbing.xi_max": 3.0} | 1.725e-03 | 9.571e-04 | 3.117e-01 | 1.330e-04 | 88470 |

## map study

| variant | C_vs_ref | D_vs_ref | C_vs_D | interior |
|---|---|---|---|---|
| {"cloak.radial_map": "quadratic"} | 1.225e-04 | 1.509e-04 | 3.932e-05 | 1.423e-04 |
