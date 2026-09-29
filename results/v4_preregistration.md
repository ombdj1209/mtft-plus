# MTFT+ v4: pre-registration (written 2026-09-26, before any v4 result exists)

This file fixes the design, comparisons and success criteria for v4. Its git/file history is the record that it came first. Results will be reported against it whether or not they support v4. Any later change to this plan is appended below as a dated amendment with its reason; nothing is edited in place.

## Motivation (from the VISUELLE 2.0 v3 study)

On real fast-fashion data each (product, shop) series lives 12 weeks, and launches are staggered across shops: a median of 4 release dates per product, and 71.8 % of pairs launch after the product's first shop. The v3 design aligns everything on calendar time:

- **Siblings** = the same product in other shops *at the same calendar week*.
- **Analogues** = similar products *in the same shop, at the same calendar week*. These are mostly not stocked there, or are at a different life stage.

The v3 multimodal stack did not beat the siblings-only control on the calendar benchmark: −0.0012 [−0.0031, +0.0006].

**Hypothesis.** Demand information should be borrowed **by product age**, not by calendar week. A shop that launched earlier has already revealed the product's early life curve, and similar products launched earlier reveal a full life curve.

## New information: lifecycle-aligned (LA) curves

The information is strictly causal: a value is used only if it happened before the forecast origin.

- **LA siblings.** For target series (a, f) at origin o and step t with target age k = t − launch(a, f): the same product's sales in shop g at age k, i.e. y(a, g, launch(a, g) + k), used only if that calendar time is at most o − 1 and inside g's observation window. Up to M = 32 other shops are used, those launched earliest before o; ties go by shop index.
- **LA analogues.** The K = 16 most similar products b (cosine similarity of concatenated SigLIP image + text embeddings, excluding a) that launched before o. Each contributes its mean log1p per-shop sales at age k, averaged over the shops whose age-k week is at most o − 1.
- **LA-pool.** The information above averaged into 4 known inputs per step:
  - the mean over the available sibling slots;
  - the share of sibling slots available;
  - the similarity-softmax-weighted analogue mean (temperature 0.05);
  - the weighted analogue coverage.

## Models (identical data, split, seeds, budget, d_model = 32, 16 epochs × 200 batches × 256)

| Name | Backbone and fusion | Inputs |
|---|---|---|
| M0 = MTFT-Picnic + siblings (existing) | static scalars (ResNet-152 / DistilBERT, PCA 10+10) | calendar siblings |
| **M1 = MTFT-Picnic + siblings + LA-pool** (primary control) | same as M0 | + LA-pool |
| A1 = v4 w/o lifecycle attention | aligned SigLIP tokenizer (static pooled) + age embedding | calendar siblings + LA-pool |
| **v4 = MTFT+ v4 (ours)** | A1 + **lifecycle attention**: per-step attention over the individual LA-sibling tokens (with learned shop embeddings) and the individual LA-analogue tokens (queries and keys include projected product embeddings, so the relevance of each analogue is learned from content) | same information as A1 and M1 |
| A2 = v4 w/o analogue attention | v4 without the analogue branch | same |
| A3 = v4 w/o sibling attention | v4 without the sibling branch | same |
| A4 = v4, unimodal retrieval | v4 with analogues retrieved and embedded using MTFT-Picnic's separate-encoder features | same information type |

The v3 same-shop calendar analogue channel is not used in v4, A1–A4 or M1. Existing rows (TFT, MTFT-Picnic, M0, v3) are re-used from `results_visuelle2` / `results_visuelle2_official`. That is valid only if the code change leaves them bit-identical, which will be verified by re-running one seed.

## Foundation-model baseline

**Chronos-2, fine-tuned** (full fine-tuning, `Chronos2Pipeline.fit`) on the same training windows, with validation-based checkpoint selection. Two variants:

- (i) target only;
- (ii) with covariates: discount, the calendar sibling mean, and the LA-pool channels (known future).

The better of the two **on validation loss** is the comparator. The learning rate is chosen from {1e-5, 1e-6} and the steps from {1000, 3000}, on validation only, with the same search for both variants. The zero-shot rows are kept for reference.

## Protocols

- **Calendar benchmark** (encoder 12, horizon 4, test 2019-09-16 to 2019-12-30).
- **Official VISUELLE 2.0 protocol:** SO-fore 2-1, 2-10 and demand, with an encoder of 2 weeks and validation = the most recent 10 % of training pairs.
- 5 seeds (1–5). Paired product-cluster bootstrap (95 %) plus per-seed wins.

## Primary comparisons and success criteria (fixed now)

1. **Architecture claim: v4 vs M1.** v4 has lower WAPE with a 95 % interval excluding 0, on at least 4/5 seeds, on the calendar benchmark **and** on at least 2 of the 3 official tasks.
2. **Foundation-model claim: v4 vs the best fine-tuned Chronos-2.** Same criterion.
3. **Secondary, reported regardless:**
   - the information effect (M1 − M0);
   - the backbone and alignment effect (A1 − M1);
   - the attention effect (v4 − A1);
   - the branch ablations (A2, A3);
   - the value of aligned retrieval (v4 − A4);
   - new-in-shop, cold-start and new-product subsets.

If a criterion fails, v4 is reported as not meeting it. No re-definition of metrics, subsets, splits or seeds after results are seen. Hyperparameters are not tuned per model beyond the equal procedure above.

## Exploratory (reported as exploratory, not as tests)

Zero-inflated negative-binomial head; Google Trends covariates; Chronos-2 as the temporal backbone inside v4.

## Amendments

**2026-09-26 (a), before any fine-tuning or v4 test result.** These details were unspecified above and are now fixed for Chronos-2 fine-tuning:

- The (learning rate, steps) grid {1e-5, 1e-6} × {1000, 3000} runs on the **first seed** for each variant and each protocol/task, selected by validation wQL. The chosen configuration is then fitted for seeds 1–5, so that every fine-tuned model has 5 seeds, like the trained models.
- Each seed samples 200,000 training windows from the same training index the other models use. Batch 256; 1,000–3,000 steps is 0.26–0.77 M window-samples, against 0.82 M for our models.
- Inside `fit`, the checkpoint is selected every 100 steps on the validation windows (the library's `load_best_model_at_end`).
- Past-only covariates (the calendar sibling mean) are NaN over the horizon. Known covariates: discount and the 4 LA-pool channels.

**2026-09-26 (b), before the 5-seed v4 runs.**

*Disclosure.* A one-seed pipeline check (seed 1, calendar benchmark) printed test metrics.

| Model | val loss | test WAPE |
|---|---|---|
| M0 | 0.3161 | 0.6462 |
| M1 | 0.3133 | 0.6448 |
| v4 | 0.3170 | 0.6455 |

M0 and v3 reproduced their stored results bit-for-bit, as required. These numbers are part of the record; seed 1 stays in the 5-seed runs.

*Secondary analyses added now* (declared before the 5-seed results; the primary criteria above are unchanged):

1. **Noise floor.** For each live test cell, the expected WAPE of the best point forecast if y ~ Poisson(λ̂), with λ̂ the model's own mean forecast. Poisson is the least dispersed count model, so this floor is optimistic: the true floor is at least as high. We report achieved − floor.
2. **Product level.** Article-national WAPE / wQL of the bottom-up forecasts, for all models, with the same product-cluster bootstrap. This is the level at which buying and allocation decisions are made, and where count noise averages out.

**2026-09-26 (c), before any full Chronos-2 fine-tuning run (compute).** A 200-step timing run shows about 0.3–0.4 s per step on this GPU, so the grid in (a) would need 14+ hours. Because `fit` keeps the best validation checkpoint every 100 steps, a 3,000-step run subsumes the 1,000-step setting, up to the learning-rate schedule. The grid is therefore **lr ∈ {1e-5, 1e-6} × 3,000 steps**, still chosen on validation wQL on the first seed and then fitted for seeds 1–5.

*Disclosure.* Pipeline checks on SO-fore 2-10, seed 1, printed test WAPE:

| Check | Target only | + covariates |
|---|---|---|
| 20 steps, 2,000 windows | 97.44 | 98.96 |
| 200 steps, 20,000 windows (val wQL, target only) | 0.6190 | |

These are not results; they are recorded here because they were seen.

*Development rule from here on.* Any revision of v4 is decided on validation data only, on seed 1, and recorded here as a named version (v4.1, …) before its test results are computed. Every version tested is reported, including the pre-registered v4.

**2026-09-27 (d), before running any of the tasks below.** These tasks follow a review of the v4 report. All results will be reported, including unfavourable ones. Earlier results stay in the record; a corrected result is reported next to the original, never in place of it.

1. **Chronos-2 fairness.**
   - (a) *Diagnose* the calendar zero-shot WAPE of 1.025, which is above the all-zeros forecast (1.0). Checks:
     - pre-launch NaN/zero handling in the context (NaN-padded vs trimmed to the live history);
     - output scale (units vs log);
     - median vs mean;
     - time alignment of predictions (score against actuals shifted by −1 / +1 step).
     If a bug is found it is fixed and zero-shot is re-run; the old numbers are kept.
   - (b) *Group attention*, with group = all shops of the same product:
     - (i) `cross_learning=True` with one batch per (product, origin);
     - (ii) a multivariate target holding the live shops of the product in the context window (at most 64 variates; the target shops first, then the other shops by most recent launch). Zero-shot and fine-tuned, target only: the library requires one covariate schema per input, which a varying shop set cannot meet.
     Fine-tuning uses the grid of amendment (c) on seed 1, the chosen setting on seeds 1–5, and the same windows and validation-only selection.
   - (c) *Trivial baselines* on the calendar benchmark:
     - all-zeros;
     - last value (the last observed own week; 0 if none);
     - LA-sibling mean as the forecast (expm1 of the pooled LA-sibling value at each horizon step; falls back to the last value when no sibling is available at that age).
   - (d) Correct the statement that Chronos-2 is univariate. Re-score criterion 2 against the best fine-tuned Chronos-2 variant, **selected on validation loss** among {fine-tuned, fine-tuned + covariates, fine-tuned group} for each protocol/task.
2. **Published baselines.** Official CrossAttnRNN code (github.com/HumaticsLAB/visuelle2.0-code) on SO-fore 2-1, 2-10 and demand, and GTM-Transformer (github.com/HumaticsLAB/GTM-Transformer):
   - Vendored under `third_party/` with minimal compatibility patches, each listed in the report (Lightning 2 API; `fairseq` Adafactor replaced by the `transformers` Adafactor with the same arguments).
   - Checkpoint selection on the validation split our models use (the 10 % most recent training pairs), 5 seeds.
   - GTM-Transformer is a new-product model with no own-history input, so it is run on the demand task. A store embedding is added, as in the official CrossAttnRNN demand model, so that it can forecast per shop. It is reported as not applicable to SO-fore.
   - Compute rule: if one run of the image variant is estimated at > 4 h, the frozen ResNet trunk is cached once. The deviation is then labelled, and the no-image variants remain exact.
   - One diagnostic run per task reproduces the official selection (test set as validation, seed 21), only to check whether the published numbers are reproducible. It is labelled a diagnostic, not a result.
   - The WAPE definition is confirmed by re-computing our Naive run with their `calc_error_metrics`.
3. **MinT-shrink failure.** A diagnosis only, no tuning, on the stored validation and test base forecasts of M1 seed 1 (re-run, bit-reproducible):
   - the condition number of C W Cᵀ;
   - λ̂;
   - prior variances of no-history nodes;
   - the magnitude and sign of the MinT adjustments per level;
   - the share of bottom forecasts clamped at 0.
   Bottom-up stays the recommended method if it stays best.
4. **Noise floor.** It is reworded as a *consistency check*: a Poisson floor computed from each model's own mean forecast, not a true floor.
5. **v4.1 (exploratory, development rule).** History-aware gating: the lifecycle-attention variables are multiplied by g = σ(a·n_obs + b), with n_obs the number of observed own-history steps in the encoder window. The pooled LA inputs are unchanged.
   - *Disclosure:* the idea is motivated by v4's *test* results (attention helps only with short or absent history).
   - It is decided on **validation loss, seed 1**: calendar + 3 official tasks. It is adopted if its validation loss is lower than v4's in at least 3 of those 4 settings.
   - The decision is recorded here before any v4.1 test result. It is reported as exploratory; a confirmatory claim needs the second dataset.
6. **Second dataset: M5** (Walmart; Corporación Favorita as fallback). It is pre-registered in a separate file, `results/m5_preregistration.md`, before any M5 model run.
7. **Write-up.** Update `results/visuelle2/v4_README.md` and write `paper/draft.md`.

**2026-09-27 (e), before any full run of the published baselines (compute).** One-epoch timing runs gave 48–134 s per epoch for CrossAttnRNN and 54 s for GTM-Transformer. The official schedules (30 and 200 epochs) with 5 seeds would therefore need more than 30 GPU-hours.

- *Rule:* checkpoint selection is unchanged (best validation score, on our validation split). Training stops after **5 validation checks without improvement**, up to the official maximum (30 epochs CrossAttnRNN, validated every epoch; 200 epochs GTM-Transformer, validated every 5 epochs).
- *Configurations:* those the paper reports.
  - CrossAttnRNN with and without image on 2-1 and 2-10;
  - CrossAttnRNN with image on demand;
  - GTM-Transformer (+ store) on demand.
  Each has 5 seeds.
- *Diagnostics:* the official-selection runs (test set as validation, seed 21) keep the exact official schedule, with no early stopping.
- *Disclosure (smoke tests, 1–5 epochs, seed 1, not results):*

  | Smoke test | Epochs | Test WAPE |
  |---|---|---|
  | CrossAttnRNN w/ image, 2-10 | 1 | 110.54 |
  | CrossAttnRNN, 2-1 | 1 | 96.98 |
  | CrossAttnRNN w/ image, demand | 1 | 192.98 |
  | GTM-Transformer, demand | 5 | 85.49 |

  The official `calc_error_metrics` reproduced our WAPE exactly in every run.
- *Further patches found while testing* (the released code fails without them):
  - (i) pandas 3 copy-on-write: `dataset.py` needs a writable copy of the sales array;
  - (ii) PyTorch ≥ 2.6 `torch.load(weights_only=True)`: the official dataset cache needs the old default;
  - (iii) the released demand model builds its image encoder with a 300-d output while its attention layers use `embedding_dim` = 512; the encoder is built with `embedding_dim`.
- *Incident:* the Chronos-2 group fine-tuning on the calendar benchmark crashed with a CUDA error, caused by a concurrent GPU smoke test. It is re-run from the start; no partial result was kept. From now on, GPU jobs run strictly one at a time through a queue.

**2026-09-27 (f), v4.1 decision (amendment (d) 5), recorded before any v4.1 test metric was read.** `examples/v41_decision.py` compares only the `best_val` field (validation quantile loss, seed 1) of v4.1 and v4:

| Setting | v4.1 | v4 | Lower |
|---|---|---|---|
| Calendar | 0.3167 | 0.3162 | v4 |
| Official 2-1 | 0.2919 | 0.2935 | v4.1 |
| Official 2-10 | 0.2907 | 0.2893 | v4 |
| Official demand | 0.2877 | 0.2892 | v4.1 |

v4.1 is lower in **2/4** settings, and the rule requires ≥ 3/4, so **v4.1 is not adopted**.

- It is not run with 5 seeds and is not included in M5.
- Its seed-1 test numbers exist, because the runs evaluate on test at the end. They will be reported once, labelled exploratory and not adopted, with no further change to the gate.

**2026-09-27 (g), requested by the project owner. Written after amendment (f): the v4.1 decision was already applied and recorded there. The owner asked for this amendment to come before the decision, and that order was not met.** The v4.1 rule was applied as pre-registered in (d) 5 and is not changed; its outcome (not adopted) stands.

1. **What the gate can do on each protocol.**
   - On the official tasks n_obs is constant within each task: the encoder is L = 2 steps, and the gate input `launched` (age ≥ 0) gives n_obs / L = 1 for every 2-1 and 2-10 window and 0 for every demand window (`examples/visuelle2_official.py`, `TASKS`; `mtft_plus/data.py`, `launched = (age >= 0)`). There, g = σ(α · n_obs/L + β) reduces to one learned scalar per task.
   - Any official-task difference between v4.1 and v4 must therefore be attributed to rescaling the lifecycle-attention variables, not to history awareness. This applies to the seed-1 test differences reported in `results/visuelle2/v41_exploratory.md`.
   - The history-aware mechanism is testable only where n_obs varies across windows: on the calendar benchmark, where the v4.1 validation loss was higher than v4's (0.3167 vs 0.3162), and on M5.
2. **The M5 pre-registration names the tested model.** This is recorded in `results/m5_preregistration.md`, amendment (b), after the v4.1 decision and before any M5 model is trained or evaluated at full budget. The tested model is **v4**, because v4.1 was not adopted.
3. **M5 adds RMSSE / WRMSSE as a secondary metric**, specified in `results/m5_preregistration.md`, amendment (b).

**2026-09-27 (h), requested by the project owner, before any full-budget run of the published baselines. Supersedes the stopping rule of (e).**

- **Released configurations.**
  - `third_party/visuelle2.0-code-main/train_dl.py`: `--epochs` default 30 (line 170), `max_epochs=args.epochs` (line 141), validation every epoch (line 142), no early stopping; `ModelCheckpoint` keeps the best `val_wWAPE`. The released `launchers.sh` uses the defaults.
  - `third_party/GTM-Transformer-main/train.py`: `--epochs` default 200 (line 113), `check_val_every_n_epoch=5` (line 95), no early stopping; `ModelCheckpoint` keeps the best `val_mae`.
- **What our (e) runs did.** They stopped after 5 validation checks without improvement: after 6–15 of 30 epochs (CrossAttnRNN) and 75 of 200 (GTM, seed 1). This is shorter than the released configurations train.
- **Rule from now on.** Every published-baseline run uses the released epoch budget (30 / 200) with no early stopping, and the best checkpoint is still chosen on our validation split (the 10 % most recent training pairs), never on test. Flag: `examples/visuelle2_published.py --full-budget`.
  - This applies to all six configurations and all 5 seeds, including those already completed with early stopping: CrossAttnRNN without / with image on 2-1 and 2-10, CrossAttnRNN with image on demand, and GTM-Transformer (+ store) on demand.
  - These full-budget runs are the **primary** published-baseline rows.
- **Secondary row.** The early-stopped runs are kept unchanged and reported as a disclosed secondary row, labelled "[early stopping, secondary]", from `results_visuelle2_official/published_early_stopped/` (moved there by `examples/published_secondary.py`).
  - At the time of this amendment they cover: 2-1 without image, seeds 1–2; 2-10 without image, seed 1; 2-10 with image, seeds 1–5; demand with image, seeds 1–5; GTM demand, seeds 1–5 (being run).
  - 2-1 without image seeds 3–5, 2-10 without image seeds 2–5 and all of 2-1 with image were lost to a memory bug in our wrapper. It stacked one copy of the cached image features per data row; it is now fixed, with no effect on values.
- **Disclosure.** The early-stopped test results listed above were seen before this amendment. The change was requested for fidelity to the released configurations. It leaves no free choice: the budget is fixed by the released defaults, and selection stays validation-only.
- **Diagnostics unchanged.** The official-selection diagnostics (test split as validation, seed 21) already used the exact released schedule.
- **Compute:** about 29 GPU-hours, queued after the M5 runs.

**2026-09-27 (h, addendum), before any full-budget or diagnostic run.**

- **Released-selection diagnostics checked; no fix needed.** They already use the released budget with no early stopping (`early = not (official_selection or full_budget)` in `examples/visuelle2_published.py`). They also match the released defaults on:
  - epochs: 30 CrossAttnRNN, 200 GTM;
  - validation every 1 / 5 epochs;
  - batch size 128 and seed 21;
  - the test split as the validation loader, with the best monitored checkpoint kept (`val_wWAPE` / `val_mae`).
- **Remaining, disclosed differences from the released setup:**
  - (i) The frozen image trunks are precomputed. Their outputs are identical, but this rules out any random image augmentation in the released loaders.
  - (ii) The compatibility patches listed in (e).
  - (iii) For GTM-Transformer there is no released VISUELLE 2.0 configuration: the released code targets VISUELLE 1.0 and forecasts per product. Our GTM adds a store embedding, so its diagnostic cannot reproduce a published VISUELLE 2.0 number. It only shows the effect of test-set selection.
- **Queue order, set by the owner:**
  1. full-budget runs, lost-seed configurations first;
  2. the four released-selection diagnostics;
  3. a hold until the published-baseline summary has gone to the owner;
  4. M5, then WRMSSE.
- **The move of the early-stopped results to the secondary folder runs first, not last as the owner asked.** This is a mechanical dependency: the full-budget runs write the same result file names, and the script skips existing files. It would otherwise skip every seed that already has an early-stopped result. The move changes no values.
- **The incomplete early-stopped secondary row is not re-run** (owner's decision): 2-1 without image, 2 seeds; 2-10 without image, 1 seed; 2-1 with image, none.

**2026-09-27 (h, addendum 2), owner's decision (option b), before the full-budget demand CrossAttnRNN run and before any full-budget or diagnostic run.**

1. **256 × 256 input for the demand CrossAttnRNN: a necessary patch.**
   - The released loader resizes all images to 299 × 299 (`visuelle2.0-code-main/dataset.py` line 51). ResNet-101 then gives a 10 × 10 feature grid, which is incompatible with the released `x.view(-1, 64, 2048)` (`models/CrossAttnRNNDemand.py` line 89), since 64 positions need 8 × 8. At 256 × 256 the grid is 8 × 8.
   - Both the early-stopped and the full-budget runs use this patch. It was stated in the header of `examples/visuelle2_published.py` but not in an amendment until now.
2. **Frozen batch-norm difference, all cached trunks** (Inception-v3 up to Mixed_7a; ResNet-101, either whole or up to layer2; ResNet-50 for GTM).
   - The released code keeps the frozen trunks inside a model in training mode. Their batch-norm layers therefore normalise with batch statistics during training and update their running averages; `requires_grad=False` does not stop either.
   - Cached features are computed once in inference mode, with the fixed ImageNet running statistics.
   - This difference cannot be removed without dropping the cache, which would mean recomputing the trunks for every row in every epoch. It is disclosed for every published-baseline row.
3. **The early-stopped (secondary) demand CrossAttnRNN row used a fully frozen, cached ResNet-101.** The released code fine-tunes layer3 and layer4. That row is therefore not the released demand model, and it is labelled accordingly.
4. **Option b, implemented** (`examples/visuelle2_published.py`: `CachedResNetL2`, cache kind `resnet101_l2`). Full-budget and released-selection demand CrossAttnRNN runs:
   - cache ResNet-101 `children()[:6]` (up to layer2, 512 × 32 × 32, fp16);
   - train layer3 and layer4 from the ImageNet weights, as in the released `CrossAttnRNNDemand.py` lines 74–82;
   - then apply the released reshape, fc and dropout.
   - Layer3 uses gradient checkpointing per block, because the fp32 activations at batch size 128 exceed the 8 GB GPU. Batch-norm momentum is 0 during the recomputation, so running averages update once per step, as without checkpointing.
   - Batch size (128), fp32 precision, optimiser and schedule are unchanged. Features are now stored once per image (`Bank`); older caches load bit-identically.
   - **CPU tests** (`examples/test_demand_encoder.py`, output `results/visuelle2/diagnostics/demand_encoder_test.txt`):
     - (A) against the released `ImageEncoder` on 8 real images: identical with fp32 layer2 features; max |difference| 3.4e-4 with the fp16 cache (outputs up to 2.9);
     - (B) checkpointed against plain layer3 in training mode: identical outputs, gradients and running statistics;
     - (C) one released `training_step`: gradients reach layer3 and layer4 (42.1 M trainable encoder parameters).
   - **Compute (estimated, not yet measured on the GPU):** about 4 h per seed, 20–21 GPU-hours for 5 seeds, plus about 4 h for the released-selection diagnostic. The first epoch of the queued job will give the measured time.
   - GPU memory with checkpointing is estimated at about 5 GB; that is also unverified until the job runs.
5. **Optional, after M5 only:** one uncached seed of the demand CrossAttnRNN, with trunks recomputed from images in training mode as released, to measure the combined caching / batch-norm effect. It is not queued yet.

**2026-09-27 (h, addendum 3), owner's rule for out-of-memory failures, fixed before the demand CrossAttnRNN full-budget run.**

- **Scope.** This applies only to demand CrossAttnRNN runs with the fine-tuned layer3–4 encoder: the full-budget runs and the released-selection diagnostic.
- **Rule.** If a seed fails with CUDA out-of-memory, it restarts from scratch, with the same seed:
  1. with gradient checkpointing on layer4 as well as layer3 (exact, like layer3);
  2. if that still fails, with training micro-batches of 64 and 2-step gradient accumulation (effective batch 128; Lightning `accumulate_grad_batches=2`).
- **Consequence of level 2.** The batch-norm layers of the trainable layer3–4 then normalise with batch statistics from **64-sample micro-batches**, not 128. This departs from the released configuration and would be disclosed with the results.
- Validation and test inference stay at batch size 128, in inference mode.
- Later seeds of the same job start at the level that fitted.
- **Recording and automation.** The level used is recorded per result (`spec.memory_path`, `memory_level`) and reported in the published-baseline summary. It is implemented in `run_with_fallback` in `examples/visuelle2_published.py`, and the queue continues without intervention.
- **Tests** (`examples/test_demand_encoder.py`, output in `results/visuelle2/diagnostics/demand_encoder_test.txt`):
  - (D) layer3 + layer4 checkpointing is identical to plain training in outputs, gradients and running statistics;
  - (E) the fallback goes through levels 0 → 1 → 2, never retries other models, and re-raises errors other than out-of-memory.
- **Checkpoint folders.** A run's checkpoint folder is now cleared at its start and after the best weights are loaded, so a retry cannot pick up a checkpoint from the failed attempt.

**2026-09-29 (i), compute restriction, decided by the owner before any affected result exists.**

All experiments run on a single RTX 2060 SUPER (8 GB). The released demand CrossAttnRNN fine-tunes ResNet-101 layer3–4 in fp32. It measured about 16 minutes per epoch, or about 8 hours per seed, and the full published-baseline plan would have needed about 58 more GPU-hours. Because of this compute restriction, the published-baseline programme is cut to:

| Item | Kept | Dropped |
|---|---|---|
| CrossAttnRNN w/ image, demand (released layer3–4 fine-tuning, full budget) | seeds 1–2 | seeds 3–5 |
| GTM-Transformer (+ store), demand, full budget | seed 1 | seeds 2–5 |
| Released-selection diagnostics | 2-1 and 2-10 | demand and GTM |

- **Unchanged:** 2-1 and 2-10 (with and without image) already have 5 full-budget seeds each. The early-stopped secondary rows are kept as they are.
- **Why these cuts cannot change a conclusion.**
  - The re-run–published gap on the short-observation tasks is 55–64 WAPE points, against a seed-to-seed spread of 0.3–1.5 points.
  - The demand result already reproduces the published number (83.07 vs 83.33, seed 1).
  - The two kept diagnostics address the only open question: whether the released checkpoint selection explains the gap.
- **Reporting.**
  - Rows with fewer than 5 seeds show per-seed values and no interval, following Agarwal et al. (2021) for small run counts.
  - The NeurIPS checklist accepts a compute justification for this.
- **Timing and mechanics.**
  - Seed 1 of the demand job was already saved (83.07). Seed 2 was running when the cut was decided; its test value had not been seen.
  - A watcher (`stop_demand_after_seed2.ps1`) ends the job once the seed-2 result file exists.
  - The queue order is otherwise unchanged.
