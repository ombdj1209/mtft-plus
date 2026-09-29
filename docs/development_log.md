# MTFT+ development log

This log records the state of the project, its research rules and the decisions taken, in the order they were made. Read it with `docs/method_v1_v3.md` and the reports under `results/`.

## 0. Owner and goal

- The owner is building this as a public portfolio and research project. It builds on Picnic's paper "Multimodal Temporal Fusion Transformers are Good Product Demand Forecasters" (Sukel, Rudinac & Worring, IEEE MultiMedia 31(2), 2024).
- Goal: show on **real public data** whether MTFT+ v3, especially cross-location sibling learning, beats the MTFT design, and turn that into a research paper (arXiv report → workshop → journal).
- The owner prefers autonomous diagnosis and fixing, complete files rather than partial diffs, and no unnecessary explanation.

## 1. Non-negotiable research rules

1. **Honesty over wins.** Report negative and null results plainly. Never reword claims to look better than the numbers. Earlier versions failed and are documented as such; keep doing that.
2. **No test-set tuning.** Hyperparameters, early stopping, calibration and reconciliation weights use validation data only.
3. **Equal treatment.** Every model in a comparison gets the same seeds, training budget, data split and tuning effort.
4. **Controls are mandatory.** Any new ingredient added to "ours" must also be tested added to the baseline (as `MTFT-Picnic + siblings` did). Attribute gains to the ingredient, not the architecture, when the control says so.
5. **Seeds.** Minimum 5 seeds for any claim on real data. Report per-seed win counts alongside the article-cluster bootstrap, because the bootstrap does not capture training-run variance.
6. **Naming.** "MTFT-Picnic" is our re-implementation of the paper's design, never "Picnic's model". Do not claim to beat Picnic's production system.
7. **Frozen synthetic generator.** Do not modify `generate_synthetic_retail` in `mtft_plus/data.py`; the synthetic results depend on it being fixed.

## 2. Where things stand

### Repo map
```
mtft_plus/features.py        SigLIP/OpenCLIP extractor + disk cache; MultimodalTokenizer (feature/modality dropout); SigLIP alignment loss
mtft_plus/model.py           MTFT backbone; fusion modes none / static_scalar / static_pooled / cross_attention
mtft_plus/layers.py          GRN, GLU, vectorised VSN (GroupedGRN), interpretable MHA, monotone quantile head
mtft_plus/losses.py          QuantileLoss (pinball), newsvendor_quantile
mtft_plus/data.py            frozen synthetic generator; build_analogues / build_category_analogues; DemandWindowDataset (analogues=, siblings=)
mtft_plus/reconciliation.py  Hierarchy (national / FC / article-national / article x FC), MinTReconciler (low-rank, dense A), SeasonalRidgeForecaster, swanson_mean
mtft_plus/calibration.py     LogConformalCalibrator (split-conformal in log1p space)
mtft_plus/panel.py           load_panel (real long-format CSVs, freq-aware) and panel_to_frames
mtft_plus/lightning_module.py  training module (AdamW + OneCycle; quantile + alignment loss)
mtft_plus/metrics.py         WAPE, MASE, pinball, wQL, coverage
benchmark.py                 multi-seed benchmark; MAIN_MODELS, V3_MODELS, ABLATIONS; resumable via results/partials/<seed>__<slug>.json; --panel for real data
report.py                    mean ± sd, paired article-cluster bootstrap (--ours selects the model); --readme splices tables between RESULTS markers
examples/prepare_panel.py    CSVs + images -> (SigLIP) EmbeddingBank -> pickled panel
tests/test_components.py     15 tests, all passing
results/                     v3 results; results/v1, results/v2 archived write-ups and tables
```

### Version history (synthetic benchmark, 5 FCs x 1,000 articles x 540 days)

| Version | Idea | Outcome | Lesson |
|---|---|---|---|
| v1 (1 seed) | SigLIP + decoder cross-attention vs 10-d static unimodal features | Tie overall; worse on cold start (0.4469 vs 0.4251) | 768-d embeddings overfit ~900 products; reconcile means, not medians; calibrate intervals |
| v2 (3 seeds) | + tokenizer dropout, analogue retrieval, conformal calibration | 0.4113 vs 0.4126 MTFT-Picnic, lost on seed 23; cross-attention ablation showed no gain | Fusion is not the bottleneck; cross-series signal is unused |
| v3 (3 seeds) | Drop cross-attention; add cross-FC sibling channels | 0.4072 vs 0.4126 WAPE, won 3/3; cold 0.4145 vs 0.4244 | ~70 % of the gain (84 % on cold start) comes from siblings alone (control 0.4088); multimodal stack adds a further −0.0016 [−0.0022, −0.0009], not significant on cold start |

Reconciliation (v3): MinT national WAPE 0.0316 vs 0.0492 bottom-up. Bottom-up is better for article-national supplier quotas (0.2071 vs 0.2311). MinT + conformal coverage: 0.84 national, 0.80 article-national.

### Key design decisions and why
- **Log1p targets + pinball loss.** Quantiles are equivariant under monotone transforms, so back-transforming is exact.
- **Monotone quantile head.** Non-crossing by construction.
- **MinT reconciles Swanson means (0.3·q10 + 0.4·q50 + 0.3·q90), not medians.** Summing medians biased national totals low (v1 bug). Quantiles are shifted by each node's mean adjustment.
- **Analogues exclude the article itself.** Training windows then match the cold-start setting. Features are past-only; there is a test for this.
- **Sibling channel** = mean log1p demand of the same article in the other locations, encoder steps only, plus a 28-step recent level as a static real. There is a test for correctness and past-only use.
- **Cross-attention is dropped from v3 by default.** It is still available as `fusion="cross_attention"`.

## 3. Known limitations and technical debt (fix before real-data runs)

1. **CPU hard-coded.** `benchmark.py` uses `pl.Trainer(accelerator="cpu", ...)`. Add `--accelerator/--devices/--precision` flags. The owner's GPU is an RTX 2070 (Turing): use `precision="16-mixed"`, not bf16. Check VRAM with `nvidia-smi` (likely 8 GB).
2. **Dense reconciliation will not scale.** `Hierarchy.aggregation_matrix` and `MinTReconciler` use a dense A of shape (n_agg × n_bottom). VISUELLE 2.0 (~5,355 products × 110 shops ≈ 589k bottom series) needs a sparse A (`torch.sparse` or `scipy.sparse`). Rewrite `CWC^T`, `WC^T`, `C y` and bottom-up with sparse ops, and add a sparse-vs-dense equivalence test. Also consider dropping sparse-zero bottom series (product-shop pairs never stocked) from the hierarchy.
3. **Daily assumptions are hard-coded in several places.** Parametrise them by a `season` / `steps_per_week` setting before running weekly data:
   - `SeasonalRidgeForecaster`: anchor lag `7*(h//7+1)`, 28-step level, `min_history=35`, train origins starting at 35.
   - `seasonal_naive_scale(season=7)` for MASE (weekly data: season 52 if history allows, else 1).
   - Analogue and sibling static level uses the last 28 steps.
   - Day-of-week categorical and `doy` features (harmless on weekly data, but dow will be constant).
   - Defaults `encoder_length=56`, `horizon=14`, `cold_start_days=21`, `n_*_origins=4`.
4. **Memory.** Panel cubes are dense (A, F, T) float32. At VISUELLE scale that is roughly 0.4 GB per array, and there are several arrays. It fits in 16 GB, but `SeedContext` builds `cov_b` (m, T, 6) float64 and aggregated copies; move those to float32 and/or chunk.
5. **Bootstrap scope.** `report.py` resamples articles, not training runs. Keep per-seed win counts in every claim.
6. **Conformal and MinT share the validation window.** They estimate different parameters, but the samples are not independent; state this in the paper.
7. **The second v2 ablation ("ours w/o retrieval (v1 + reg.)") never finished.** Not needed for v3; optional.
8. **Order-quota CSVs** use a heuristic newsvendor service level from shelf life (not available in VISUELLE). Treat them as a demo only.
9. **Unimodal baseline features.** On real data, supply ResNet-152 / DistilBERT features as `unimodal_image.npy` / `unimodal_text.npy` so MTFT-Picnic is faithful. Otherwise it compresses SigLIP embeddings; say so if unavoidable.

## 4. Next tasks (in order, with acceptance criteria)

1. **Environment.** CUDA PyTorch installed, `pip install -e ".[multimodal,dev]"`, and `pytest -q tests` shows 15 passed.
2. **GPU flags** (§3.1). `python benchmark.py --quick --seeds 7 --out /tmp/q` runs on the GPU.
3. **VISUELLE 2.0 converter.** Data is in `data/visuelle2` (downloaded manually from humaticslab.github.io/forecasting/visuelle).
   - Inspect the real files first; do not assume column names.
   - Write `examples/convert_visuelle2.py` producing our schema: `sales.csv` (product, shop, week, units, discount), `catalog.csv` (category, text built from tags/attributes, image_path, launch_date), `locations.csv` (shop, country), and `weather.csv` if available.
   - Print row counts, date range, zero share and number of launches as sanity checks.
4. **Sparse reconciliation and weekly parametrisation** (§3.2–3.4). Add new tests; all tests pass.
5. **Embeddings.** Use `examples/prepare_panel.py` with `google/siglip-base-patch16-224`, cached. Optionally compare SigLIP 2 later.
6. **Smoke run on a subset** (e.g. 500 products, 1 seed) to check runtime and memory, then the full run: 5 seeds × {TFT, MTFT-Picnic, MTFT-Picnic + siblings, MTFT+ v3}. Then `python report.py --out results_visuelle2 --ours "MTFT+ v3 (ours)"`.
7. **Official VISUELLE protocol.** Reproduce the dataset paper's new-product forecasting split and metrics. Report our models there next to published baselines (e.g. GTM-Transformer), using their reported numbers with citations, or re-running their code if available.
8. **Chronos-2 zero-shot baseline** (`chronos-forecasting` package), with and without covariates, on the same test windows.
9. **Write-up.** `results/visuelle2/README.md` with tables, per-seed wins, bootstrap intervals, and what transferred and what did not. Update the main README only with numbers that exist. Summarise the key results for the owner.

## 5. Hypotheses to test on VISUELLE (state results either way)

- H1: Cross-store sibling learning improves the MTFT design (control: MTFT-Picnic + siblings vs MTFT-Picnic).
- H2: MTFT+ v3 beats MTFT-Picnic + siblings (the multimodal increment).
- H3: Sibling benefit is smaller for cold-start items, because all shops launch a product simultaneously and siblings also lack history.
- H4: With a much larger catalogue (5k+ products), aligned embeddings and analogues help cold start more than on the synthetic benchmark.
- H5: MinT improves shop and national totals; bottom-up may remain best for product-level totals.

## 6. Publication plan

- **Now:** arXiv technical report framed as a controlled study, v1 → v3 including failures.
- **After VISUELLE results:** workshop paper (time-series / retail forecasting workshops at KDD, NeurIPS or ICML).
- **Journal / main venue** (e.g. International Journal of Forecasting): needs 2+ real datasets, 5–10 seeds, and strong external baselines (Chronos-2, TimesFM, GTM-Transformer). Position sibling learning relative to global models, hierarchical forecasting and Chronos-2 group attention; the novelty is careful application and measurement.

## 6b. Progress log (2026-09-25)

**Machine.** RTX 2060 8 GB (Turing, sm_75; `nvidia-smi` shows driver 591.86, CUDA 13.1), Windows 11, Python 3.11. Everything is installed in the project `.venv` (`.\.venv\Scripts\python.exe`); the global Python still has CPU torch 2.7.1. CUDA wheel: `torch==2.14.0+cu130` from download.pytorch.org/whl/cu130. The newest cu128 wheel is 2.11. pip kept failing on this network with `SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC`; `curl -C - --retry-all-errors` into a local .whl and then `pip install <file>` works.

**Done**
- Task 1 (env): `.venv` + `pip install -e ".[multimodal,dev]"`; tests: 19 passed (15 original + 4 new).
- Task 2 (GPU): `benchmark.py --accelerator auto|gpu|cpu --devices N --precision 32-true|16-mixed|bf16-mixed`. Defaults: `auto` (GPU if available) and `32-true`. Inference runs in fp32 on the same device, and the best-epoch state is kept on the CPU. The partial JSON records device, precision and GPU name. `--quick --seeds 7` on the GPU takes 29 s for all 9 models.
  - **Precision measured, and HANDOFF §3.1's 16-mixed advice reversed.** Full-budget synthetic seed 7, MTFT-Picnic / v3:

    | Setting | WAPE | Train time |
    |---|---|---|
    | GPU fp32 | 0.4109 / 0.4040 | 30 s / 44 s |
    | GPU 16-mixed | 0.4096 / 0.4032 | 37 s / 52 s |
    | Archived CPU fp32 | 0.4109 / 0.4032 | 526 s / 722 s |

  - fp16 is slower because the model is tiny and batches are built in numpy on one core. At `--quick` (40 steps), fp16 cost about 0.05 WAPE: GradScaler skips early steps. GPU fp32 is reproducible run to run (±1e-4) but not bit-identical to CPU. Do not mix settings within a comparison.
- Task 4 (sparse + weekly):
  - `Hierarchy.aggregation_matrix(sparse=True)`. `MinTReconciler` accepts sparse A and never forms W Cᵀ (applied as a factored operator). Test: sparse = dense to 1e-9. At VISUELLE size (5,355 × 110 = 594,516 nodes) fit takes 1.4 s and 1.3 GB RSS; dense A alone would be 26 GB.
  - Frequency settings live in `SyntheticConfig` (`freq, season, level_window, min_history, steps_per_year`). The generator does not read them, and daily defaults reproduce v1–v3. `load_panel` fills them from `freq_defaults(freq)`: weekly = season 1, level 4, min history 5, 52 steps/yr. They are used by the ridge (anchor, level, phase dummies, origin start and stride), MASE scales, the analogue/sibling level statics and the age feature.
  - `SeedContext` no longer materialises the (m, T, 6) float64 `cov_b` or the full `Yall`; the `w_b` loop is vectorised.
  - Weekly-grid bugs fixed in `load_panel`:
    - the grid now starts at the period containing the first sale (previously the first Monday *after* it);
    - launch dates snap down to their period (previously a mid-week launch rounded *up*, masking the first real sales week);
    - weather is averaged per period (previously only rows dated on a Monday were kept);
    - a period containing a holiday is flagged;
    - promo and discount are averaged per cell (previously last row wins).
  - `prepare_panel.py` gets freq-aware window defaults and `--season/--level-window/--min-history`. `benchmark.py --no-quotas` added; `--write-quotas` could never be switched off.
  - Equivalence check: the old and new code, `--quick --seeds 7`, TFT + v3 on CPU, agree on every metric in the partial JSONs to a relative 8e-9. That is float noise, so synthetic results are unchanged.

## 6c. Progress log (2026-09-26): VISUELLE 2.0

The data is in `visuelle2/` in the project root; the owner downloaded it, and a second copy sits in `Downloads\visuelle2`. The encoder models are in `.hf_models/`: SigLIP base, ResNet-152, DistilBERT and Chronos-2. They were fetched with a Range-resume downloader because the HF hub client stalls on this network.

**What the data actually is (inspected, task 3)**
- **`sales.csv` rows.** Each row is (product, shop, release_date, restock, 12 weekly sales). Release dates are all Mondays, and the 12 weeks are counted from each pair's own release, so sales are observed for 12 weeks per pair only. 5,355 products × 110 shops (ids 0–125) = 106,850 pairs.
- **Launches are staggered across shops.** A product has a median of 4 distinct release dates, and 71.8 % of pairs launch after the product's first shop. HANDOFF §5 H3's premise ("all shops launch simultaneously") is false for VISUELLE.
- **Data quirks.**
  - 3,386 negative cells, probably returns; they are clipped to 0 per cell.
  - One corrupt image, `AI19/04442.png`, byte-identical in both copies; it is treated as a missing image modality.
  - 11 shops map to weather localities with no rows; their weather is set to missing, i.e. the mean after z-scoring.
- **`customer_data.csv` is loyalty-card transactions only.** It covers about 85 % of units and 65 % of cells match exactly, so it is not usable as ground truth. But 16 % of its transactions fall after week 12: products keep selling, so cells after the 12-week window are *unobserved*, not zero.
- **Official split.** `stfore_train/test.csv` = `sales.csv` / 53 exactly, the norm scalar. Test is 10,684 pairs of 749 products, and 246 of those products are never in train.

**New code**
- **`examples/convert_visuelle2.py`.** Writes `sales.csv`, `series.csv`, `catalog.csv` (text = "color fabric category"), `locations.csv`, `weather.csv` (daily) and `holidays.csv` (Italian). `series.csv` holds the per-pair launch and window end. `--end` defaults to 2019-12-30, the last release week, because later weeks have no new launches and aggregates there would be artefacts. Use `--end 2020-03-16` for the official protocol, `data/visuelle2_converted_full`.
- **`load_panel(series=...)`.**
  - Adds `launch_loc` (A, F) and a `live` (A, F, T) mask (launched and observed). These drive targets, metrics, MASE history, sibling and analogue means, `build_index` and the aggregate masks. Unobserved cells are zeroed.
  - The sibling channel is the mean over *live* other shops. It reduces exactly to v3's definition when all locations launch together, and the synthetic results are unchanged to 8e-9.
  - Cells that are not live are forecast as 0. That is known in advance: launch plan + fixed 12-week window.
  - A new `new-in-shop` subset: series new to their shop, for products already sold elsewhere.
- **Reconciliation on VISUELLE.** The pre-specified MinT-shrink breaks: national WAPE ≈ 3.0. The covariance comes from 16 validation residuals of series that are mostly dead by the test period, because life cycles are 12 weeks.
  - Added MinT-WLS (diagonal) as a labelled post-hoc variant.
  - Added a validation-only selection rule (fit on validation origins 1–2, score origins 3–4). It picks WLS in 20/20 runs.
  - The no-history prior is extended to every node live in test but not in validation.
  - Bottom-up beats both on VISUELLE.
- **Coverage fixes.** `metrics.coverage` gets a 1e-6 tolerance at the bounds: zero-inflated data otherwise counts y = 0 as uncovered when a float-noise shift leaves lo = 3e-8. Bottom-up at the bottom level is now exactly the base forecast. On synthetic data this changes only "bottom-up, article × FC" Cov80, 0.78 → 0.82, which now equals base; the archived v3 table had the same artefact, 0.79 vs base 0.80.
- **`examples/unimodal_features.py`.** ResNet-152 (2048-d, pooled) and DistilBERT (768-d, mean-pooled) features, so MTFT-Picnic uses separate encoders, as §3.9 requires. The preprocessing is implemented from `preprocessor_config.json` because transformers 5 needs torchvision.
- **`examples/visuelle2_official.py`.** Official protocol for SO-fore 2-1, 2-10 and demand, with restock cleaning as in `dataset.py:frame_series`.
  - Own-history window = 2, as in the protocol.
  - Validation = the most recent 10 % of *train* pairs. In the released code the validation loader is built from the test split, `train_dl.py: val_dataloaders=testloader`.
  - Test pairs are removed from the panel that training sees.
  - Naive/SES re-runs reproduce the paper: 2-1 101.93 / 97.86 vs published 101.92 / 97.85; 2-10 118.24 / 111.34 vs 118.18 / 111.27.
  - SES is computed in closed form; a test pins it to statsmodels.
- **`examples/chronos2_baseline.py`.** Zero-shot Chronos-2 on the same windows as our models, in three variants: target only, + covariates (siblings, discount), and + cross-learning per product group.
- **Other changes.** `report.py --extra-pairs "A|B,..."` adds control tests; per-seed wins are shown in all bootstrap tables; zero-shot baselines are paired with every seed; I/O is UTF-8 (Windows cp932 console).

**Pre-registered choices (validation only)**
- Weekly calendar panel: encoder 12, horizon 4, 4 validation + 4 test origins, so test = 2019-09-16 to 2019-12-30. Cold start = first launch ≥ 3 weeks before the test start.
- Budget: 16 epochs × 200 batches, chosen over 8 × 100 by validation loss (seed 1, TFT and v3). Applied to all models in both protocols.

**Operational lesson.** Two training processes at once on this Windows (WDDM) GPU run about 9× slower each: 979 s vs 106 s. Run jobs sequentially.

Results: `results_visuelle2/` (calendar), `results_visuelle2_official/` (official), write-up in `results/visuelle2/README.md`.

**Status of HANDOFF §4.**
- Tasks 1–6 and 8 (zero-shot) are done. Task 7 is done for Naive and SES, which are re-run and match the paper; the published deep models are cited with caveats. Task 9 is the write-up.
- Headline: siblings (H1) are significant everywhere except demand. The multimodal increment (H2) is task-dependent. H3's premise is false for VISUELLE. MinT loses to bottom-up (H5 rejected).

**MTFT+ v4 (2026-09-26/27): done.** Pre-registration: `results/v4_preregistration.md`, three dated amendments. Report: `results/visuelle2/v4_README.md`.
- New: `mtft_plus/lifecycle.py` (causal lifecycle-aligned siblings and analogues), `LifecycleAttention` in `model.py`, models in `benchmark.py` (`V4_MODELS`, `V4_ABLATIONS`), fine-tuned Chronos-2 in `examples/chronos2_baseline.py` (`finetune-calendar`, `finetune-official`), product-level bootstrap in `report.py`, forecasts saved as `.npz`.
- **Criterion 1 (v4 beats the same-information control M1) fails.** v4 is worse on the calendar benchmark (+0.0034) and on 2-1 (+0.85), and better on 2-10 (−1.03) and demand (−0.93), all 5/5 seeds.
- **Criterion 2 (v4 beats the best fine-tuned Chronos-2) passes everywhere:** calendar −0.116, 2-1 −15.0, 2-10 −28.4, demand −16.2 WAPE points, all 5/5.
- **LA information (M1 − M0) helps:** calendar −0.0037 at shop level and −0.0145 at product level.
- Bottom-level WAPE sits on the Poisson noise floor of each model's own mean forecast.

**Proposed next experiments** (from the v3 study; items 1–2 are now implemented as v4; each new ingredient is also added to the MTFT-Picnic control):
1. *Lifecycle-aligned siblings.* Feed the sales that shops which launched the product earlier made at each future product age, as known-future inputs. Use only calendar time before the origin.
2. *Lifecycle-aligned analogues.* Similar products' week-1-to-12 curves across all shops, instead of the same-shop calendar window.
3. Attention over individual sibling shops instead of a flat mean.
4. A count likelihood (zero-inflated NB) for weekly counts with a mean of about 1.3 and 37 % zeros.
5. Google Trends covariates (shipped with the dataset).
6. Capacity tuning on validation, with equal effort for every model.

**Stronger baselines for a paper:** fine-tuned Chronos-2 (the reviewer's first question), GTM-Transformer and a recent multimodal fashion model under honest validation, and a second real dataset. Also save forecasts (not only metrics), so metric fixes never need re-training.

## 7. Command reference

```bash
pytest -q tests
python benchmark.py --quick --seeds 7 --out /tmp/q                   # smoke test
python benchmark.py --no-ablations                                   # synthetic, 3 seeds
python report.py --out results --ours "MTFT+ v3 (ours)" --readme README.md
python examples/prepare_panel.py --data-dir data/visuelle2_converted --freq W-MON \
       --encoder-length 12 --horizon 4 --cold-start-steps 3 --out panel_v2.pkl
python benchmark.py --panel panel_v2.pkl --seeds 1,2,3,4,5 --no-ablations \
       --models "TFT,MTFT-Picnic,MTFT-Picnic + siblings,MTFT+ v3 (ours)" --out results_visuelle2
```

(Encoder, horizon and cold-start values for weekly data are placeholders; choose them from the official protocol in task 7.)
