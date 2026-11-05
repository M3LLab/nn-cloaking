# Step 1 — continuous regularised ideal cloak: results

Normalised L2 error of the total displacement magnitude ‖u‖ = sqrt(|u_x|² + |u_y|²) on the common exterior sampling grid (spacing 0.05 m, r > 1.02 b, |x − x_s| > 0.75 m, physical box only). `mag_X_vs_ref` = ‖ ‖u_X‖ − ‖u_A‖ ‖ / ‖ ‖u_A‖ ‖. Columns without `mag_` are the complex-displacement error ‖u_X − u_A‖ / ‖u_A‖, which also sees phase and direction.

## Base case

| metric | value |
|---|---|
| eps | 5.000e-03 |
| B_vs_ref | 2.217e-01 |
| C_vs_ref | 7.608e-05 |
| D_vs_ref | 8.585e-05 |
| C_vs_D | 1.178e-05 |
| A_vs_analytic | 2.378e-04 |
| interior | 8.619e-05 |
| mag_interior | 2.078e-05 |
| mag_B_vs_ref | 1.044e-01 |
| mag_C_vs_ref | 3.356e-05 |
| mag_D_vs_ref | 3.766e-05 |
| mag_A_vs_analytic | 5.083e-05 |
| mag_C_vs_D | 5.124e-06 |
| ring1.25b_B_vs_ref | 4.722e-01 |
| ring1.25b_mag_B_vs_ref | 2.426e-01 |
| ring1.25b_C_vs_ref | 1.638e-04 |
| ring1.25b_mag_C_vs_ref | 6.176e-05 |
| ring1.25b_D_vs_ref | 1.854e-04 |
| ring1.25b_mag_D_vs_ref | 6.805e-05 |
| ring2b_B_vs_ref | 2.639e-01 |
| ring2b_mag_B_vs_ref | 1.232e-01 |
| ring2b_C_vs_ref | 9.222e-05 |
| ring2b_mag_C_vs_ref | 4.224e-05 |
| ring2b_D_vs_ref | 1.044e-04 |
| ring2b_mag_D_vs_ref | 4.783e-05 |
| n_cells_A | 82738 |
| n_dofs_A | 664516 |
| h_bg | 9.375e-02 |
| h_inner | 6.689e-03 |
| peak_rss_gib | 6.023e+00 |

## eps study

| eps | mag_B_vs_ref | mag_C_vs_ref | mag_D_vs_ref | mag_C_vs_D | mag_interior | C_vs_ref | h_inner | n_cells_A |
|---|---|---|---|---|---|---|---|---|
| 1.000e-01 | 1.044e-01 | 1.538e-02 | 1.534e-02 | 5.472e-05 | 2.347e-04 | 3.559e-02 | 2.000e-02 | 72534 |
| 5.000e-02 | 1.044e-01 | 3.859e-03 | 3.857e-03 | 1.595e-06 | 2.098e-05 | 8.843e-03 | 2.000e-02 | 72534 |
| 2.500e-02 | 1.044e-01 | 9.321e-04 | 9.505e-04 | 2.336e-05 | 9.047e-05 | 2.125e-03 | 2.000e-02 | 72534 |
| 1.000e-02 | 1.044e-01 | 1.413e-04 | 1.508e-04 | 1.194e-05 | 4.654e-05 | 3.210e-04 | 1.342e-02 | 75098 |
| 5.000e-03 | 1.044e-01 | 3.356e-05 | 3.766e-05 | 5.124e-06 | 2.078e-05 | 7.608e-05 | 6.689e-03 | 82738 |
| 2.500e-03 | 1.044e-01 | 8.024e-06 | 9.397e-06 | 1.829e-06 | 9.906e-06 | 1.820e-05 | 3.339e-03 | 98246 |

## mesh study

| scale | h_bg | n_cells_A | n_dofs_A | mag_A_vs_analytic | mag_B_vs_ref | mag_C_vs_ref | mag_D_vs_ref | mag_C_vs_D | A_vs_finest | C_vs_finest | C_vs_ref | peak_rss_gib |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2.000e+00 | 1.875e-01 | 21562 | 173812 | 4.751e-04 | 1.044e-01 | 3.017e-05 | 3.678e-05 | 1.140e-05 | 4.691e-04 | 4.687e-04 | 6.970e-05 | 2.234e+00 |
| 1.400e+00 | 1.312e-01 | 43022 | 346036 | 1.523e-04 | 1.044e-01 | 3.241e-05 | 3.740e-05 | 6.606e-06 | 1.471e-04 | 1.471e-04 | 7.361e-05 | 3.390e+00 |
| 1.000e+00 | 9.375e-02 | 82738 | 664516 | 5.083e-05 | 1.044e-01 | 3.356e-05 | 3.766e-05 | 5.124e-06 | 4.837e-05 | 4.837e-05 | 7.608e-05 | 5.987e+00 |
| 7.500e-01 | 7.031e-02 | 144878 | 1162500 | 1.983e-05 | 1.044e-01 | 3.446e-05 | 3.776e-05 | 4.143e-06 | 0.000e+00 | 0.000e+00 | 7.816e-05 | 1.005e+01 |

## absorbing study

| variant | mag_A_vs_analytic | A_vs_baseA | mag_B_vs_ref | mag_C_vs_ref | C_vs_ref | n_cells_A |
|---|---|---|---|---|---|---|
| {"absorbing.thickness": 1.5, "absorbing.reflection": 0.0001} | 7.830e-05 | 6.214e-05 | 1.044e-01 | 3.373e-05 | 7.647e-05 | 66080 |
| {"absorbing.thickness": 4.0} | 5.095e-05 | 1.436e-05 | 1.044e-01 | 3.373e-05 | 7.647e-05 | 107256 |
| {"absorbing.reflection": 0.0001} | 5.573e-05 | 2.825e-05 | 1.044e-01 | 3.356e-05 | 7.608e-05 | 82738 |
| {"absorbing.reflection": 1e-08} | 5.073e-05 | 7.365e-07 | 1.044e-01 | 3.356e-05 | 7.608e-05 | 82738 |
| {"absorbing.power": 3} | 5.094e-05 | 1.068e-06 | 1.044e-01 | 3.356e-05 | 7.608e-05 | 82738 |
| {"absorbing.kind": "damping", "absorbing.thickness": 4.0, "absorbing.xi_max": 4.0, "absorbing.power": 2} | 1.920e-02 | 1.920e-02 | 1.032e-01 | 3.332e-05 | 7.596e-05 | 107256 |
| {"absorbing.kind": "damping", "absorbing.thickness": 6.0, "absorbing.xi_max": 1.5, "absorbing.power": 3} | 4.903e-04 | 4.904e-04 | 1.044e-01 | 3.373e-05 | 7.647e-05 | 149054 |

## map study

| variant | mag_C_vs_ref | mag_D_vs_ref | mag_C_vs_D | mag_interior | C_vs_ref |
|---|---|---|---|---|---|
| {"cloak.radial_map": "quadratic"} | 3.247e-05 | 3.766e-05 | 6.426e-06 | 2.342e-05 | 7.355e-05 |
