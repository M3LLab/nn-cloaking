# Surrogate

Neural surrogate mapping cell-wise microstructure `(C_eff, ρ_eff)` + frequency `f*` to scalar transmission loss, trained on FEM samples from `dataset_gen/`.

## Files

- `dataset.py` — `SurrogateDataset` (reads HDF5), `SurrogateSample` / `SurrogateBatch` (pydantic + jaxtyping), `collate_surrogate`, `make_dataloader`. Reshapes the flat cell axis to `(n_x, n_y)`; with `cloak_only=True` non-cloak cells are zeroed.
- `model.py` — `ForwardFEM_CNN`: ConvNeXt-Small trunk over `(C, ρ)` channels → latent `z`; sinusoidal `FourierFeatures(f)`; MLP `SpectrumDecoder` on `concat(z, ff(f))`. Two modes:
  - `forward_at(batch, f=None)` → `(B,)` one frequency per sample.
  - `forward_spectrum(batch, f_grid)` → `(B, F)` shared grid.
- `train.py` — `TrainConfig` (pydantic) + `train()`. AdamW + MSE, writes `config.json`, `history.json`, `best.pt`, `last.pt`, periodic `epoch-NNNN.pt` to `out_dir`. Supports `resume_from`.
- `loss.py` — MSE helper (minimax / mean-spectrum variants stubbed).

## HDF5 input

Produced by `dataset_gen/`:

| dataset | shape | meaning |
|---|---|---|
| `cell_C_flat` | `(N, n_cells, n_C_params)` | stiffness params per cell |
| `cell_rho` | `(N, n_cells)` | density per cell |
| `f_star` | `(N,)` | normalised frequency |
| `loss` | `(N,)` | transmission loss (target) |
| `sample_type` | `(N,)` | `"init"` / `"random*"` / `"smooth*"` / `"opt*"` |
| `cloak_mask` | `(n_cells,)` | shared boolean cloak-cell mask |

Cell index layout: `idx = ix * n_y + iy`.

## Usage

```bash
conda activate jax-fem-env
python -m surrogate.train output/surrogate_dataset.h5
```

Programmatic:

```python
from surrogate.train import TrainConfig, train
model, history = train(TrainConfig(data_path="output/surrogate_dataset.h5"))
```
