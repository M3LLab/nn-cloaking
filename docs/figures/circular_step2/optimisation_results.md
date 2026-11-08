# Step 2 optimisation results

‖u‖ error on the four **test** sources (median / worst), never used for training or selection. All runs: 8 training sources, 4 validation sources, quadratic map unless stated, 600 steps (pool: 1500, warm start from band).

| design | f*=0.5 | f*=0.6 | f*=0.75 | f*=0.9 | f*=1 | f*=1.1 | f*=1.25 | f*=1.4 | f*=1.5 | f*=1.75 | f*=2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bare hole | 0.138 / 0.222 | – | 0.155 / 0.278 | – | 0.158 / 0.263 | – | 0.165 / 0.266 | – | 0.168 / 0.253 | – | 0.168 / 0.255 |
| f1_500 | 0.106 / 0.189 | – | – | – | 0.042 / 0.053 | – | – | – | 0.214 / 0.353 | – | 0.235 / 0.380 |
| f1_500_linear | 0.135 / 0.229 | – | – | – | 0.049 / 0.075 | – | – | – | 0.233 / 0.365 | – | 0.265 / 0.447 |
| f1_252 | 0.133 / 0.244 | – | – | – | 0.057 / 0.072 | – | – | – | 0.238 / 0.405 | – | 0.228 / 0.382 |
| f1_120 | 0.132 / 0.243 | – | – | – | 0.065 / 0.083 | – | – | – | 0.235 / 0.346 | – | 0.228 / 0.350 |
| f1_60 | 0.147 / 0.281 | – | – | – | 0.099 / 0.131 | – | – | – | 0.219 / 0.363 | – | 0.245 / 0.341 |
| band_500 | 0.074 / 0.120 | – | 0.126 / 0.239 | – | 0.086 / 0.170 | – | 0.141 / 0.236 | – | 0.106 / 0.211 | – | 0.221 / 0.343 |
| pool_500 | 0.073 / 0.119 | 0.084 / 0.133 | 0.083 / 0.156 | 0.075 / 0.129 | 0.074 / 0.162 | 0.083 / 0.193 | 0.098 / 0.198 | 0.111 / 0.172 | 0.119 / 0.187 | 0.174 / 0.298 | – |

Runs: f1_* trained at f* = 1 (number = pixels; linear = linear map); band_500 at f* = 0.5, 1, 1.5; pool_500 draws 2 of 11 frequencies in [0.5, 1.5] per step. Missing entries were not evaluated (f* = 2 for pool_500 was refused by the memory check).
