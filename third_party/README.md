# Third-party code

This directory vendors the official code of the two published baselines. It is used to re-run them under a common protocol (`examples/visuelle2_published.py`). Both copies keep their original licence files and copyright notices.

| Directory | Source | Licence |
|---|---|---|
| `visuelle2.0-code-main/` | <https://github.com/HumaticsLAB/visuelle2.0-code> (Skenderi et al., CVPRW 2022) | **CC BY-NC-SA 4.0** ([`LICENSE.txt`](visuelle2.0-code-main/LICENSE.txt)): attribution, non-commercial use only, share-alike |
| `GTM-Transformer-main/` | <https://github.com/HumaticsLAB/GTM-Transformer> (Skenderi et al., *Journal of Forecasting* 2024) | **MIT** ([`LICENSE`](GTM-Transformer-main/LICENSE)), © 2021 HumaticsLAB |

The MIT licence of this repository does **not** apply to this directory.

## Changes to the vendored files

Only one vendored file is edited:

- `visuelle2.0-code-main/dataset.py`, line 69: `.copy()` is added so the sales array is writable under pandas ≥ 3 (copy-on-write). The values are unchanged. The line is marked `# PATCH`. This modified file is distributed under CC BY-NC-SA 4.0, like the original.

All other compatibility changes are applied at run time from `examples/visuelle2_published.py`, without editing the vendored files:

1. `fairseq.optim.adafactor.Adafactor` is replaced by `transformers.optimization.Adafactor` (same algorithm and arguments).
2. Lightning 1.6 `validation_epoch_end` is replaced by Lightning 2 `on_validation_epoch_end` (same metrics).
3. `torch.load(weights_only=False)` is used for the official dataset caches (the PyTorch ≥ 2.6 default changed).
4. The demand model's image encoder is built with `embedding_dim` (512); the released 300-d default fails at the first batch.
5. Demand images are resized to 256 × 256. The released 299 × 299 gives a 10 × 10 ResNet grid, which is incompatible with `view(-1, 64, 2048)`.
6. Frozen image trunks are evaluated once and cached. For the demand model, only the part up to ResNet-101 layer2 is cached, and layer3–4 are fine-tuned as released. Consequence: the frozen layers' batch-norm uses fixed ImageNet statistics, not the released training-mode behaviour.
7. GTM-Transformer receives a store embedding so that it forecasts per shop on VISUELLE 2.0. The released code targets VISUELLE 1.0 at product level.

These changes are documented in the pre-registration amendments (`results/v4_preregistration.md`, (e) and (h)) and in `results/visuelle2/published_baselines.md`.
