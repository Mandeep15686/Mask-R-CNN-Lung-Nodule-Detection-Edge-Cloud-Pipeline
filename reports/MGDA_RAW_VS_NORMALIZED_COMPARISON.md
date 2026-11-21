# Raw MGDA vs Unit-Normalized MGDA — Controlled Comparison

## Protocol

- Seed: `20261121`; identical initial state SHA256: `77fba3fea25953626ee0280d45b6e2ec8fdba98fbc22a6f355fbb9b414e4bbba`.
- GPU: NVIDIA GeForce RTX 4050 Laptop GPU with CUDA 12.4.
- Shared cache manifest SHA256: `6ba9850e5d0288e3115e168438c0dbdec2b3b7c0d401d0159f41294e34db6639`.
- Five objectives; both runs use `mask_head`, the same 89/89 UID-disjoint train/validation split, batch size 4, LR 0.005, SGD, and the same scheduler.
- Both runs used 1,070 training batches and 935 validation batches per epoch. The saved per-epoch `train_sample_order_sha256` values are checked to match.
- Raw MGDA is the existing `solve_mgda(..., gradient_normalization='raw')` control. Unit MGDA normalizes each objective gradient before building the Gram matrix, then applies `lambda_i / ||g_i||` to original losses, so the final backward reproduces the optimized normalized-gradient combination.
- Determinism was enabled where supported. CUDA detection/ROI operations may remain nondeterministic; this is a paired run, not a statistical significance test.

## Results

| Metric | Raw MGDA | Unit-normalized MGDA |
|---|---:|---:|
| Best validation Dice | 0.778054 | 0.780696 |
| Best-Dice checkpoint epoch | 8 | 8 |
| HD95 at best-Dice checkpoint | 3.10908 | 5.16017 |
| Boundary F1 at best-Dice checkpoint | 0.853248 | 0.870507 |
| Final validation Dice | 0.766786 | 0.764827 |
| Final HD95 | 5.98214 | 7.45478 |
| Final Boundary F1 | 0.859445 | 0.833508 |
| Minimum HD95 across epochs | 3.10908 | 4.15887 |
| Maximum Boundary F1 across epochs | 0.866694 | 0.870507 |
| Recorded wall-clock duration (s; includes outliers) | 2390.43 | 6521.94 |
| Mean recorded epoch duration (s; includes outliers) | 239.043 | 652.194 |
| Longest recorded epoch (s) | 339.989 | 4353.28 |
| Mean BCE simplex weight | 4.21172e-06 | 0.0915352 |
| Mean Dice simplex weight | 0.35027 | 0.318271 |
| Mean Boundary simplex weight | 0.587084 | 0.104494 |
| Mean Focal simplex weight | 4.71501e-05 | 0.131698 |
| Mean HD-distance simplex weight | 0.0625948 | 0.354002 |
| Mean BCE original gradient norm | 130.079 | 54.0614 |
| Mean Dice original gradient norm | 1.80914 | 4.56243 |
| Mean Boundary original gradient norm | 0.601177 | 1.12757 |
| Mean Focal original gradient norm | 58.7515 | 19.8477 |
| Mean HD-distance original gradient norm | 28.0313 | 39.318 |

## Interpretation

Raw-gradient mean weights were BCE `4.21172e-06` and Focal `4.71501e-05`; unit-normalized mean weights were BCE `0.0915352` and Focal `0.131698`.
Negative pairwise-cosine fraction: raw `0.000`, normalized `0.000`.
Final validation deltas (normalized minus raw): Dice `-0.00195917`, HD95 `+1.47264`, Boundary F1 `-0.0259366`.

Relative changes at each profile's best-Dice checkpoint (normalized vs raw):
- Dice: `+0.340%` (higher is better; raw `0.778054`, normalized `0.780696`).
- HD95: `+65.971%` (lower is better; raw `3.10908`, normalized `5.16017`).
- Boundary F1: `+2.023%` (higher is better; raw `0.853248`, normalized `0.870507`).
- Recorded elapsed time was `+172.8%` higher for unit-normalized MGDA (raw `2390.43s`, normalized `6521.94s`).
- Runtime is not a reliable throughput comparison here: normalized epoch 6 took `4353.28s`, versus `540.83s` for its next-longest epoch. Treat the total-time difference as an observed run condition, not as a stable estimate of normalization overhead.
These are observations from one paired seed. They do not establish statistical significance or independent test performance. Nodule masks are generated approximations, so segmentation metrics measure agreement with the cache masks.

## Artifacts

- `MGDA-raw-maskhead-10ep`: `D:\lung-nodule\experiments\MGDA-raw-maskhead-10ep`
- `MGDA-normalized-maskhead-10ep`: `D:\lung-nodule\experiments\MGDA-normalized-maskhead-10ep`
- Shared initialization: `D:\lung-nodule\experiments\MGDA-raw-vs-normalized-initial\model_state.pth`
- Plots: `reports/mgda_raw_vs_normalized/`
