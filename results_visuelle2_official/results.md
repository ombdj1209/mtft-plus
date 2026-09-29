# VISUELLE 2.0, official short-observation protocol

WAPE in %, MAE in units, mean ± sd over seeds. Validation = most recent 10 % of training pairs.

## 2-1

| Model | seeds | WAPE | MAE | new-product WAPE | seen-product WAPE | Cov80 |
|---|---|---|---|---|---|---|
| Naive (our re-run) | – | 101.93 | 1.128 | | | |
| SES (our re-run) | – | 97.86 | 1.082 | | | |
| TFT | 5 | 68.07 ± 0.46 | 0.75 ± 0.01 | 67.83 ± 0.50 | 68.58 ± 0.40 | 0.87 |
| MTFT-Picnic | 5 | 68.54 ± 0.92 | 0.76 ± 0.01 | 68.33 ± 0.93 | 69.00 ± 0.90 | 0.87 |
| MTFT-Picnic + siblings | 5 | 65.25 ± 0.14 | 0.72 ± 0.00 | 65.04 ± 0.15 | 65.70 ± 0.19 | 0.87 |
| MTFT+ v3 (ours) | 5 | 65.73 ± 0.69 | 0.73 ± 0.01 | 65.60 ± 0.69 | 66.03 ± 0.69 | 0.87 |
| Chronos-2 + covariates + cross-learning | 1 | 97.35 | 1.08 | 95.98 | 100.28 | 0.80 |
| Chronos-2 + covariates | 1 | 96.11 | 1.06 | 94.69 | 99.15 | 0.79 |
| Chronos-2 + cross-learning | 1 | 95.93 | 1.06 | 94.22 | 99.57 | 0.79 |
| Chronos-2 group (multivariate) | 1 | 95.97 | 1.06 | 94.29 | 99.56 | 0.79 |
| Chronos-2 | 1 | 96.46 | 1.07 | 94.83 | 99.95 | 0.76 |
| Chronos-2 fine-tuned + covariates | 5 | 80.95 ± 0.42 | 0.90 ± 0.00 | 80.05 ± 0.43 | 82.87 ± 0.41 | 0.82 |
| Chronos-2 fine-tuned group | 5 | 90.03 ± 0.03 | 1.00 ± 0.00 | 88.51 ± 0.04 | 93.29 ± 0.02 | 0.79 |
| Chronos-2 fine-tuned | 5 | 90.19 ± 0.03 | 1.00 ± 0.00 | 88.68 ± 0.03 | 93.41 ± 0.02 | 0.80 |
| CrossAttnRNN (official code) | 5 | 87.76 ± 0.21 | 0.97 ± 0.00 | 86.35 ± 0.20 | 90.78 ± 0.24 | – |
| CrossAttnRNN w/ image (official code) | 5 | 86.93 ± 0.37 | 0.96 ± 0.00 | 85.50 ± 0.31 | 89.99 ± 0.62 | – |
| MTFT-Picnic + siblings + LA-pool | 5 | 65.06 ± 0.39 | 0.72 ± 0.00 | 64.86 ± 0.38 | 65.47 ± 0.41 | 0.86 |
| MTFT+ v4 (ours) | 5 | 65.90 ± 0.44 | 0.73 ± 0.00 | 65.70 ± 0.38 | 66.33 ± 0.59 | 0.85 |
| v4 w/o lifecycle attention | 5 | 66.32 ± 0.42 | 0.73 ± 0.00 | 66.11 ± 0.45 | 66.76 ± 0.34 | 0.86 |

Paired product-cluster bootstrap, MTFT+ v4 (ours) − comparator (WAPE points, 95 % CI), and per-seed wins:

| Comparator | ΔWAPE | wins |
|---|---|---|
| TFT | **-2.17 [-2.35, -1.99]** | 5/5 |
| MTFT-Picnic | **-2.64 [-2.88, -2.42]** | 5/5 |
| MTFT-Picnic + siblings | **+0.65 [+0.54, +0.76]** | 0/5 |
| MTFT+ v3 (ours) | **+0.17 [+0.05, +0.27]** | 2/5 |
| Chronos-2 + covariates + cross-learning | **-31.45 [-32.51, -30.37]** | 5/5 |
| Chronos-2 + covariates | **-30.21 [-31.20, -29.22]** | 5/5 |
| Chronos-2 + cross-learning | **-30.02 [-31.04, -29.00]** | 5/5 |
| Chronos-2 group (multivariate) | **-30.07 [-31.09, -29.04]** | 5/5 |
| Chronos-2 | **-30.56 [-31.57, -29.55]** | 5/5 |
| Chronos-2 fine-tuned + covariates | **-15.04 [-15.51, -14.59]** | 5/5 |
| Chronos-2 fine-tuned group | **-24.13 [-24.90, -23.40]** | 5/5 |
| Chronos-2 fine-tuned | **-24.28 [-25.04, -23.56]** | 5/5 |
| CrossAttnRNN (official code) | **-21.86 [-22.63, -21.09]** | 5/5 |
| CrossAttnRNN w/ image (official code) | **-21.03 [-21.82, -20.26]** | 5/5 |
| MTFT-Picnic + siblings + LA-pool | **+0.85 [+0.74, +0.95]** | 0/5 |
| v4 w/o lifecycle attention | **-0.42 [-0.52, -0.32]** | 4/5 |

Controls, A − B (WAPE points, 95 % CI; all test pairs / new products only), per-seed wins of A:

| A | B | ΔWAPE all | ΔWAPE new products | wins |
|---|---|---|---|---|
| MTFT-Picnic + siblings | MTFT-Picnic | **-3.29 [-3.52, -3.08]** | **-3.29 [-3.59, -3.02]** | 5/5 |
| MTFT-Picnic | TFT | **+0.48 [+0.30, +0.66]** | **+0.50 [+0.28, +0.77]** | 1/5 |
| MTFT+ v3 (ours) | MTFT-Picnic + siblings | **+0.48 [+0.37, +0.59]** | **+0.55 [+0.41, +0.68]** | 1/5 |

Published (Skenderi et al. 2022; in the released training code the validation loader is built from the test split):

| Method | WAPE | MAE |
|---|---|---|
| Naive | 101.922 | 1.13 |
| SES | 97.85 | 1.08 |
| kNN | 87.11 | 0.94 |
| kNN + image | 88.97 | 0.96 |
| CrossAttnRNN | 23.2 | 0.26 |
| CrossAttnRNN w/ images | 23.7 | 0.26 |

## 2-10

| Model | seeds | WAPE | MAE | new-product WAPE | seen-product WAPE | Cov80 |
|---|---|---|---|---|---|---|
| Naive (our re-run) | – | 118.24 | 1.308 | | | |
| SES (our re-run) | – | 111.34 | 1.232 | | | |
| TFT | 5 | 67.61 ± 0.26 | 0.75 ± 0.00 | 67.54 ± 0.24 | 67.75 ± 0.35 | 0.84 |
| MTFT-Picnic | 5 | 67.68 ± 0.55 | 0.75 ± 0.01 | 68.30 ± 0.65 | 66.36 ± 0.43 | 0.84 |
| MTFT-Picnic + siblings | 5 | 66.84 ± 0.06 | 0.74 ± 0.00 | 67.32 ± 0.15 | 65.81 ± 0.19 | 0.84 |
| MTFT+ v3 (ours) | 5 | 65.66 ± 0.27 | 0.73 ± 0.00 | 65.72 ± 0.33 | 65.53 ± 0.29 | 0.85 |
| Chronos-2 + covariates + cross-learning | 1 | 115.43 | 1.28 | 114.95 | 116.45 | 0.78 |
| Chronos-2 + covariates | 1 | 114.24 | 1.26 | 113.61 | 115.57 | 0.78 |
| Chronos-2 + cross-learning | 1 | 111.96 | 1.24 | 111.05 | 113.91 | 0.78 |
| Chronos-2 group (multivariate) | 1 | 112.71 | 1.25 | 111.88 | 114.50 | 0.78 |
| Chronos-2 | 1 | 111.09 | 1.23 | 110.10 | 113.22 | 0.77 |
| Chronos-2 fine-tuned + covariates | 5 | 93.69 ± 0.06 | 1.04 ± 0.00 | 92.82 ± 0.06 | 95.56 ± 0.08 | 0.81 |
| Chronos-2 fine-tuned group | 5 | 96.27 ± 0.06 | 1.06 ± 0.00 | 95.18 ± 0.05 | 98.59 ± 0.06 | 0.80 |
| Chronos-2 fine-tuned | 5 | 94.97 ± 0.05 | 1.05 ± 0.00 | 93.79 ± 0.04 | 97.49 ± 0.06 | 0.81 |
| CrossAttnRNN (official code) | 5 | 87.68 ± 0.28 | 0.97 ± 0.00 | 86.58 ± 0.32 | 90.03 ± 0.23 | – |
| CrossAttnRNN w/ image (official code) | 5 | 87.10 ± 0.78 | 0.96 ± 0.01 | 85.77 ± 0.79 | 89.94 ± 0.81 | – |
| MTFT-Picnic + siblings + LA-pool | 5 | 66.30 ± 0.38 | 0.73 ± 0.00 | 66.29 ± 0.44 | 66.32 ± 0.65 | 0.84 |
| MTFT+ v4 (ours) | 5 | 65.27 ± 0.35 | 0.72 ± 0.00 | 65.11 ± 0.42 | 65.63 ± 0.42 | 0.85 |
| v4 w/o lifecycle attention | 5 | 66.15 ± 0.34 | 0.73 ± 0.00 | 66.44 ± 0.49 | 65.55 ± 0.33 | 0.85 |

Paired product-cluster bootstrap, MTFT+ v4 (ours) − comparator (WAPE points, 95 % CI), and per-seed wins:

| Comparator | ΔWAPE | wins |
|---|---|---|
| TFT | **-2.33 [-2.79, -1.89]** | 5/5 |
| MTFT-Picnic | **-2.41 [-2.89, -1.94]** | 5/5 |
| MTFT-Picnic + siblings | **-1.56 [-2.02, -1.09]** | 5/5 |
| MTFT+ v3 (ours) | **-0.39 [-0.75, -0.02]** | 4/5 |
| Chronos-2 + covariates + cross-learning | **-50.16 [-52.24, -48.12]** | 5/5 |
| Chronos-2 + covariates | **-48.96 [-50.92, -47.05]** | 5/5 |
| Chronos-2 + cross-learning | **-46.69 [-48.47, -44.91]** | 5/5 |
| Chronos-2 group (multivariate) | **-47.44 [-49.22, -45.67]** | 5/5 |
| Chronos-2 | **-45.82 [-47.54, -44.11]** | 5/5 |
| Chronos-2 fine-tuned + covariates | **-28.42 [-29.50, -27.36]** | 5/5 |
| Chronos-2 fine-tuned group | **-30.99 [-32.03, -29.95]** | 5/5 |
| Chronos-2 fine-tuned | **-29.70 [-30.71, -28.68]** | 5/5 |
| CrossAttnRNN (official code) | **-22.40 [-23.14, -21.67]** | 5/5 |
| CrossAttnRNN w/ image (official code) | **-21.83 [-22.52, -21.16]** | 5/5 |
| MTFT-Picnic + siblings + LA-pool | **-1.03 [-1.39, -0.65]** | 5/5 |
| v4 w/o lifecycle attention | **-0.88 [-1.27, -0.50]** | 5/5 |

Controls, A − B (WAPE points, 95 % CI; all test pairs / new products only), per-seed wins of A:

| A | B | ΔWAPE all | ΔWAPE new products | wins |
|---|---|---|---|---|
| MTFT-Picnic + siblings | MTFT-Picnic | **-0.85 [-1.24, -0.46]** | **-0.99 [-1.56, -0.39]** | 5/5 |
| MTFT-Picnic | TFT | +0.08 [-0.34, +0.52] | **+0.76 [+0.23, +1.27]** | 3/5 |
| MTFT+ v3 (ours) | MTFT-Picnic + siblings | **-1.17 [-1.58, -0.74]** | **-1.59 [-2.17, -1.00]** | 5/5 |

Published (Skenderi et al. 2022; in the released training code the validation loader is built from the test split):

| Method | WAPE | MAE |
|---|---|---|
| Naive | 118.176 | 1.31 |
| SES | 111.265 | 1.23 |
| kNN | 91.13 | 0.98 |
| kNN + image | 97.97 | 1.06 |
| CrossAttnRNN | 35.13 | 0.39 |
| CrossAttnRNN w/ image | 32.25 | 0.36 |

## demand

| Model | seeds | WAPE | MAE | new-product WAPE | seen-product WAPE | Cov80 |
|---|---|---|---|---|---|---|
| train mean curve (our re-run) | – | 89.38 | 1.050 | | | |
| train median curve (our re-run) | – | 77.32 | 0.909 | | | |
| TFT | 5 | 65.00 ± 0.34 | 0.76 ± 0.00 | 65.07 ± 0.36 | 64.86 ± 0.51 | 0.85 |
| MTFT-Picnic | 5 | 64.46 ± 0.26 | 0.76 ± 0.00 | 65.14 ± 0.30 | 63.03 ± 0.48 | 0.84 |
| MTFT-Picnic + siblings | 5 | 64.01 ± 0.30 | 0.75 ± 0.00 | 64.90 ± 0.39 | 62.16 ± 0.25 | 0.85 |
| MTFT+ v3 (ours) | 5 | 63.07 ± 0.43 | 0.74 ± 0.01 | 63.73 ± 0.58 | 61.68 ± 0.17 | 0.85 |
| Chronos-2 + covariates + cross-learning | 1 | 100.57 | 1.18 | 100.50 | 100.71 | 0.81 |
| Chronos-2 + covariates | 1 | 100.57 | 1.18 | 100.50 | 100.71 | 0.80 |
| Chronos-2 + cross-learning | 1 | 100.57 | 1.18 | 100.50 | 100.71 | 0.67 |
| Chronos-2 group (multivariate) | 1 | 100.57 | 1.18 | 100.50 | 100.71 | 0.61 |
| Chronos-2 | 1 | 100.57 | 1.18 | 100.50 | 100.71 | 0.67 |
| Chronos-2 fine-tuned + covariates | 5 | 79.42 ± 0.10 | 0.93 ± 0.00 | 79.39 ± 0.12 | 79.49 ± 0.13 | 0.87 |
| Chronos-2 fine-tuned group | 5 | 91.47 ± 0.39 | 1.07 ± 0.00 | 90.56 ± 0.41 | 93.37 ± 0.34 | 0.88 |
| Chronos-2 fine-tuned | 5 | 80.61 ± 0.18 | 0.95 ± 0.00 | 80.25 ± 0.20 | 81.36 ± 0.15 | 0.92 |
| CrossAttnRNN w/ image (official code) | 1 | 83.07 | 0.98 | 82.80 | 83.65 | – |
| MTFT-Picnic + siblings + LA-pool | 5 | 64.15 ± 0.69 | 0.75 ± 0.01 | 65.17 ± 0.99 | 62.01 ± 0.19 | 0.84 |
| MTFT+ v4 (ours) | 5 | 63.22 ± 0.37 | 0.74 ± 0.00 | 63.97 ± 0.49 | 61.64 ± 0.45 | 0.85 |
| v4 w/o lifecycle attention | 5 | 63.46 ± 0.28 | 0.75 ± 0.00 | 64.28 ± 0.37 | 61.74 ± 0.23 | 0.84 |

Paired product-cluster bootstrap, MTFT+ v4 (ours) − comparator (WAPE points, 95 % CI), and per-seed wins:

| Comparator | ΔWAPE | wins |
|---|---|---|
| TFT | **-1.79 [-2.25, -1.34]** | 5/5 |
| MTFT-Picnic | **-1.24 [-1.73, -0.79]** | 5/5 |
| MTFT-Picnic + siblings | **-0.80 [-1.29, -0.35]** | 5/5 |
| MTFT+ v3 (ours) | +0.15 [-0.13, +0.42] | 2/5 |
| Chronos-2 + covariates + cross-learning | **-37.36 [-37.72, -36.99]** | 5/5 |
| Chronos-2 + covariates | **-37.35 [-37.72, -36.99]** | 5/5 |
| Chronos-2 + cross-learning | **-37.36 [-37.73, -36.99]** | 5/5 |
| Chronos-2 group (multivariate) | **-37.36 [-37.73, -36.99]** | 5/5 |
| Chronos-2 | **-37.36 [-37.73, -36.99]** | 5/5 |
| Chronos-2 fine-tuned + covariates | **-16.21 [-16.72, -15.74]** | 5/5 |
| Chronos-2 fine-tuned group | **-28.25 [-28.67, -27.84]** | 5/5 |
| Chronos-2 fine-tuned | **-17.39 [-17.88, -16.93]** | 5/5 |
| CrossAttnRNN w/ image (official code) | **-20.21 [-21.58, -18.94]** | 1/1 |
| MTFT-Picnic + siblings + LA-pool | **-0.93 [-1.41, -0.48]** | 5/5 |
| v4 w/o lifecycle attention | -0.24 [-0.57, +0.07] | 4/5 |

Controls, A − B (WAPE points, 95 % CI; all test pairs / new products only), per-seed wins of A:

| A | B | ΔWAPE all | ΔWAPE new products | wins |
|---|---|---|---|---|
| MTFT-Picnic + siblings | MTFT-Picnic | -0.44 [-0.92, +0.04] | -0.24 [-0.91, +0.48] | 4/5 |
| MTFT-Picnic | TFT | **-0.55 [-1.06, -0.03]** | +0.06 [-0.57, +0.68] | 5/5 |
| MTFT+ v3 (ours) | MTFT-Picnic + siblings | **-0.95 [-1.41, -0.54]** | **-1.17 [-1.80, -0.54]** | 5/5 |

Published (Skenderi et al. 2022; in the released training code the validation loader is built from the test split):

| Method | WAPE | MAE |
|---|---|---|
| CrossAttnRNN w/ image | 83.33 | 0.97 |
