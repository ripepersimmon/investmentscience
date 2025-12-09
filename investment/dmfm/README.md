# Deep Multi-Factor Model (DMFM)

Implementation of the model from *"Factor Investing with a Deep Multi-Factor Model"*
(arXiv:2210.12462) with no dependencies on the rest of the workspace.

## Components
- Stock context encoder: BatchNorm + MLP → hidden context `C`.
- Industry influence: masked GAT on industry graph → `H_I`; industry neutrality `C_bar_I = C - H_I`.
- Universe influence: masked GAT on universe graph → `H_U`; universe neutrality `C_bar_U = C_bar_I - H_U`.
- Deep factors: per-horizon linear head on `[C, C_bar_I, C_bar_U]` with LeakyReLU.
- Factor attention: per-horizon attention over original features to reconstruct each deep factor.
- Loss: averages over horizons of `(attn_mse) - (factor_return) - (icir)`.

## Data expected
Per rebalancing date *t* you need:
- `features`: tensor `(N, F)` of original factors per stock.
- `industry_mask`: boolean `(N, N)` with `True` where two stocks share an industry (include self-loops).
- `universe_mask` (optional): boolean `(N, N)`; defaults to fully connected with self-loops.
- `forward_returns[k]`: tensor `(N,)` of realized returns at horizon `k` (matching the heads, e.g. 3/5/10/15/20 days).

Wrap these arrays into `DMFMDataset` (see `dmfm/data.py`).

## Quick synthetic check
```bash
python -m dmfm.train --device cpu --epochs 2 --log-interval 1
```
This uses random data to confirm the pipeline runs end-to-end.

## Integrating real data
1) Build tensors for each date: features `(N,F)`, industry_mask `(N,N)`, forward returns per horizon.
2) Create `DMFMDataset(features, industry_masks, forward_returns, universe_masks=None)`.
3) Feed the dataset to a DataLoader with the provided `collate` function in `dmfm/train.py`.
4) Adjust hyperparameters in `dmfm/config.py` or via CLI flags in `dmfm/train.py`.

## Notes
- Graph attention is masked scaled dot-product; it stays faithful to the paper's neutralization logic.
- The loss aligns deep factors with attention reconstructions, rewards higher factor returns, and higher IC/ICIR.
- No workspace files are read; everything here is self-contained and based on the paper text.
