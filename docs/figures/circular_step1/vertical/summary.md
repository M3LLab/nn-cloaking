# Step 1 — continuous regularised ideal cloak: results

Normalised L2 error of the total displacement magnitude ‖u‖ = sqrt(|u_x|² + |u_y|²) on the common exterior sampling grid (spacing 0.05 m, r > 1.02 b, |x − x_s| > 0.75 m, physical box only). `mag_X_vs_ref` = ‖ ‖u_X‖ − ‖u_A‖ ‖ / ‖ ‖u_A‖ ‖. Columns without `mag_` are the complex-displacement error ‖u_X − u_A‖ / ‖u_A‖, which also sees phase and direction.

## Base case

| metric | value |
|---|---|
| eps | 5.000e-03 |
| B_vs_ref | 3.116e-01 |
| C_vs_ref | 1.301e-04 |
| D_vs_ref | 1.512e-04 |
| C_vs_D | 2.610e-05 |
| A_vs_analytic | 2.489e-04 |
| interior | 1.246e-04 |
| mag_interior | 4.648e-05 |
| mag_B_vs_ref | 1.581e-01 |
| mag_C_vs_ref | 6.681e-05 |
| mag_D_vs_ref | 7.667e-05 |
| mag_A_vs_analytic | 5.000e-05 |
| mag_C_vs_D | 1.183e-05 |
| ring1.25b_B_vs_ref | 4.406e-01 |
| ring1.25b_mag_B_vs_ref | 2.300e-01 |
| ring1.25b_C_vs_ref | 1.846e-04 |
| ring1.25b_mag_C_vs_ref | 1.012e-04 |
| ring1.25b_D_vs_ref | 2.139e-04 |
| ring1.25b_mag_D_vs_ref | 1.145e-04 |
| ring2b_B_vs_ref | 3.280e-01 |
| ring2b_mag_B_vs_ref | 1.537e-01 |
| ring2b_C_vs_ref | 1.375e-04 |
| ring2b_mag_C_vs_ref | 7.677e-05 |
| ring2b_D_vs_ref | 1.593e-04 |
| ring2b_mag_D_vs_ref | 8.714e-05 |
| n_cells_A | 82738 |
| n_dofs_A | 664516 |
| h_bg | 9.375e-02 |
| h_inner | 6.689e-03 |
| peak_rss_gib | 6.131e+00 |

## eps study

| eps | mag_B_vs_ref | mag_C_vs_ref | mag_D_vs_ref | mag_C_vs_D | mag_interior | C_vs_ref | h_inner | n_cells_A |
|---|---|---|---|---|---|---|---|---|
| 1.000e-01 | 1.581e-01 | 3.118e-02 | 3.107e-02 | 1.212e-04 | 3.381e-04 | 6.265e-02 | 2.000e-02 | 72534 |
| 5.000e-02 | 1.581e-01 | 7.872e-03 | 7.875e-03 | 5.338e-06 | 3.672e-05 | 1.564e-02 | 2.000e-02 | 72534 |
| 2.500e-02 | 1.581e-01 | 1.884e-03 | 1.938e-03 | 5.779e-05 | 1.855e-04 | 3.718e-03 | 2.000e-02 | 72534 |
| 1.000e-02 | 1.581e-01 | 2.826e-04 | 3.072e-04 | 2.742e-05 | 1.003e-04 | 5.547e-04 | 1.342e-02 | 75098 |
| 5.000e-03 | 1.581e-01 | 6.681e-05 | 7.667e-05 | 1.183e-05 | 4.648e-05 | 1.301e-04 | 6.689e-03 | 82738 |
| 2.500e-03 | 1.581e-01 | 1.619e-05 | 1.914e-05 | 5.052e-06 | 2.121e-05 | 3.140e-05 | 3.339e-03 | 98246 |

## mesh study

| scale | h_bg | n_cells_A | n_dofs_A | mag_A_vs_analytic | mag_B_vs_ref | mag_C_vs_ref | mag_D_vs_ref | mag_C_vs_D | A_vs_finest | C_vs_finest | C_vs_ref | peak_rss_gib |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2.000e+00 | 1.875e-01 | 21562 | 173812 | 4.838e-04 | 1.582e-01 | 7.508e-05 | 7.458e-05 | 4.759e-05 | 4.778e-04 | 4.780e-04 | 1.640e-04 | 2.203e+00 |
| 1.400e+00 | 1.312e-01 | 43022 | 346036 | 1.520e-04 | 1.581e-01 | 6.579e-05 | 7.617e-05 | 1.840e-05 | 1.469e-04 | 1.481e-04 | 1.272e-04 | 3.526e+00 |
| 1.000e+00 | 9.375e-02 | 82738 | 664516 | 5.000e-05 | 1.581e-01 | 6.681e-05 | 7.667e-05 | 1.183e-05 | 4.700e-05 | 4.694e-05 | 1.301e-04 | 5.977e+00 |
| 7.500e-01 | 7.031e-02 | 144878 | 1162500 | 1.923e-05 | 1.581e-01 | 6.831e-05 | 7.694e-05 | 9.596e-06 | 0.000e+00 | 0.000e+00 | 1.338e-04 | 1.020e+01 |

## absorbing study

| variant | mag_A_vs_analytic | A_vs_baseA | mag_B_vs_ref | mag_C_vs_ref | C_vs_ref | n_cells_A |
|---|---|---|---|---|---|---|
| {"absorbing.thickness": 1.5, "absorbing.reflection": 0.0001} | 1.084e-04 | 9.741e-05 | 1.581e-01 | 6.711e-05 | 1.309e-04 | 66080 |
| {"absorbing.thickness": 4.0} | 5.013e-05 | 1.426e-05 | 1.581e-01 | 6.711e-05 | 1.309e-04 | 107256 |
| {"absorbing.reflection": 0.0001} | 6.598e-05 | 4.387e-05 | 1.581e-01 | 6.681e-05 | 1.301e-04 | 82738 |
| {"absorbing.reflection": 1e-08} | 4.988e-05 | 1.034e-06 | 1.581e-01 | 6.681e-05 | 1.301e-04 | 82738 |
| {"absorbing.power": 3} | 5.006e-05 | 1.017e-06 | 1.581e-01 | 6.681e-05 | 1.301e-04 | 82738 |
| {"absorbing.kind": "damping", "absorbing.thickness": 4.0, "absorbing.xi_max": 4.0, "absorbing.power": 2} | 1.795e-02 | 1.795e-02 | 1.567e-01 | 6.644e-05 | 1.300e-04 | 107256 |
| {"absorbing.kind": "damping", "absorbing.thickness": 6.0, "absorbing.xi_max": 1.5, "absorbing.power": 3} | 4.426e-04 | 4.435e-04 | 1.581e-01 | 6.711e-05 | 1.309e-04 | 149054 |

## map study

| variant | mag_C_vs_ref | mag_D_vs_ref | mag_C_vs_D | mag_interior | C_vs_ref |
|---|---|---|---|---|---|
| {"cloak.radial_map": "quadratic"} | 6.383e-05 | 7.667e-05 | 1.474e-05 | 4.304e-05 | 1.244e-04 |
