# Step 1 — continuous regularised ideal cloak: results

Normalised L2 error of the total displacement magnitude ‖u‖ = sqrt(|u_x|² + |u_y|²) on the common exterior sampling grid (spacing 0.05 m, r > 1.02 b, |x − x_s| > 0.75 m, physical box only). `mag_X_vs_ref` = ‖ ‖u_X‖ − ‖u_A‖ ‖ / ‖ ‖u_A‖ ‖. Columns without `mag_` are the complex-displacement error ‖u_X − u_A‖ / ‖u_A‖, which also sees phase and direction.

## Base case

| metric | value |
|---|---|
| eps | 5.000e-03 |
| B_vs_ref | 3.116e-01 |
| C_vs_ref | 1.300e-04 |
| D_vs_ref | 1.511e-04 |
| C_vs_D | 2.603e-05 |
| A_vs_analytic | 2.456e-04 |
| interior | 1.247e-04 |
| mag_interior | 4.629e-05 |
| mag_B_vs_ref | 1.581e-01 |
| mag_C_vs_ref | 6.665e-05 |
| mag_D_vs_ref | 7.668e-05 |
| mag_A_vs_analytic | 5.061e-05 |
| mag_C_vs_D | 1.191e-05 |
| ring1.25b_B_vs_ref | 4.406e-01 |
| ring1.25b_mag_B_vs_ref | 2.300e-01 |
| ring1.25b_C_vs_ref | 1.845e-04 |
| ring1.25b_mag_C_vs_ref | 1.007e-04 |
| ring1.25b_D_vs_ref | 2.138e-04 |
| ring1.25b_mag_D_vs_ref | 1.145e-04 |
| ring2b_B_vs_ref | 3.280e-01 |
| ring2b_mag_B_vs_ref | 1.537e-01 |
| ring2b_C_vs_ref | 1.374e-04 |
| ring2b_mag_C_vs_ref | 7.664e-05 |
| ring2b_D_vs_ref | 1.592e-04 |
| ring2b_mag_D_vs_ref | 8.714e-05 |
| n_cells_A | 82738 |
| n_dofs_A | 664516 |
| h_bg | 9.375e-02 |
| h_inner | 6.689e-03 |
| peak_rss_gib | 5.996e+00 |

## eps study

| eps | mag_B_vs_ref | mag_C_vs_ref | mag_D_vs_ref | mag_C_vs_D | mag_interior | C_vs_ref | h_inner | n_cells_A |
|---|---|---|---|---|---|---|---|---|
| 1.000e-01 | 1.581e-01 | 3.118e-02 | 3.106e-02 | 1.225e-04 | 3.361e-04 | 6.265e-02 | 2.000e-02 | 72534 |
| 5.000e-02 | 1.581e-01 | 7.872e-03 | 7.874e-03 | 4.935e-06 | 3.571e-05 | 1.564e-02 | 2.000e-02 | 72534 |
| 2.500e-02 | 1.581e-01 | 1.884e-03 | 1.938e-03 | 5.787e-05 | 1.855e-04 | 3.718e-03 | 2.000e-02 | 72534 |
| 1.000e-02 | 1.581e-01 | 2.823e-04 | 3.072e-04 | 2.770e-05 | 1.001e-04 | 5.541e-04 | 1.342e-02 | 75098 |
| 5.000e-03 | 1.581e-01 | 6.665e-05 | 7.668e-05 | 1.191e-05 | 4.629e-05 | 1.300e-04 | 6.689e-03 | 82738 |
| 2.500e-03 | 1.581e-01 | 1.628e-05 | 1.914e-05 | 5.082e-06 | 2.152e-05 | 3.141e-05 | 3.339e-03 | 98246 |

## mesh study

| scale | h_bg | n_cells_A | n_dofs_A | mag_A_vs_analytic | mag_B_vs_ref | mag_C_vs_ref | mag_D_vs_ref | mag_C_vs_D | A_vs_finest | C_vs_finest | C_vs_ref | peak_rss_gib |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2.000e+00 | 1.875e-01 | 21562 | 173812 | 4.565e-04 | 1.582e-01 | 7.775e-05 | 7.460e-05 | 5.080e-05 | 4.496e-04 | 4.461e-04 | 1.655e-04 | 2.154e+00 |
| 1.400e+00 | 1.312e-01 | 43022 | 346036 | 1.461e-04 | 1.581e-01 | 6.548e-05 | 7.615e-05 | 1.874e-05 | 1.400e-04 | 1.410e-04 | 1.268e-04 | 3.469e+00 |
| 1.000e+00 | 9.375e-02 | 82738 | 664516 | 5.061e-05 | 1.581e-01 | 6.665e-05 | 7.668e-05 | 1.191e-05 | 4.704e-05 | 4.707e-05 | 1.300e-04 | 5.982e+00 |
| 7.500e-01 | 7.031e-02 | 144878 | 1162500 | 1.956e-05 | 1.581e-01 | 6.838e-05 | 7.694e-05 | 9.533e-06 | 0.000e+00 | 0.000e+00 | 1.339e-04 | 9.987e+00 |

## absorbing study

| variant | mag_A_vs_analytic | A_vs_baseA | mag_B_vs_ref | mag_C_vs_ref | C_vs_ref | n_cells_A |
|---|---|---|---|---|---|---|
| {"absorbing.thickness": 1.5, "absorbing.reflection": 0.0001} | 1.077e-04 | 9.693e-05 | 1.581e-01 | 6.715e-05 | 1.307e-04 | 66080 |
| {"absorbing.thickness": 4.0} | 5.024e-05 | 1.295e-05 | 1.581e-01 | 6.715e-05 | 1.307e-04 | 107256 |
| {"absorbing.reflection": 0.0001} | 6.745e-05 | 4.389e-05 | 1.581e-01 | 6.665e-05 | 1.300e-04 | 82738 |
| {"absorbing.reflection": 1e-08} | 5.040e-05 | 1.075e-06 | 1.581e-01 | 6.665e-05 | 1.300e-04 | 82738 |
| {"absorbing.power": 3} | 5.095e-05 | 1.202e-06 | 1.581e-01 | 6.665e-05 | 1.300e-04 | 82738 |
| {"absorbing.kind": "damping", "absorbing.thickness": 4.0, "absorbing.xi_max": 4.0, "absorbing.power": 2} | 1.795e-02 | 1.795e-02 | 1.567e-01 | 6.649e-05 | 1.298e-04 | 107256 |
| {"absorbing.kind": "damping", "absorbing.thickness": 6.0, "absorbing.xi_max": 1.5, "absorbing.power": 3} | 4.451e-04 | 4.434e-04 | 1.581e-01 | 6.715e-05 | 1.307e-04 | 149054 |

## map study

| variant | mag_C_vs_ref | mag_D_vs_ref | mag_C_vs_D | mag_interior | C_vs_ref |
|---|---|---|---|---|---|
| {"cloak.radial_map": "quadratic"} | 6.377e-05 | 7.668e-05 | 1.476e-05 | 4.327e-05 | 1.244e-04 |
