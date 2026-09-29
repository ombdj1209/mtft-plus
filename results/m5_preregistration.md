# M5 (Walmart) second dataset: pre-registration (written 2026-09-27, before any M5 model run)

This extends [`v4_preregistration.md`](v4_preregistration.md) (amendment (d), task 6) to a second real dataset. It was written after inspecting the raw files but before building the panel or running any model. Later changes are appended as dated amendments; nothing is edited in place.

## Data (inspected)

M5 Forecasting (Walmart), public mirror of the competition data (Nixtla `m5-forecasts`, `m5.zip`):

- 3,049 items in 10 stores (CA ×4, TX ×3, WI ×3) = 30,490 item × store series.
- Daily sales from 2011-01-29: `sales_train_evaluation.csv` (days 1–1941) and the official evaluation horizon `sales_test_evaluation.csv` (days 1942–1969).
- Weekly prices per item × store, and a calendar with events and SNAP days.
- 68 % zero days; mean 1.13 units per series-day.
- **Launches are staggered.** 1,170 items are first sold after day 365. Across the 10 stores their launches spread over a median of 89 days (p90 627), and 81 % of their store-series launch after the item's first store. There are no images and no product text beyond the category / department codes.
- **Few young series at the test point.** Only 16 series are first sold in the last 140 days before the end of the data. Grocery items do not end after a fixed life, so lifecycle alignment can only matter for young series.

## Pre-registered expectation

Because almost no series is young at the official test origin, **the LA information is expected to have little effect on overall M5 accuracy.** A positive effect is expected only in the *young-series* subset (defined below). A null overall result on M5 is therefore an expected outcome, not a failure of the hypothesis. It will be reported as such.

## Panel and protocol

- **Grid:** daily, days 1–1969.
- **Launch of a series:** the first day of the first price week of that item × store. Availability is known to the retailer in advance; using the first *sale* would leak the launch-day demand. A series is live from its launch onwards; there is no observation-window end.
- **Windows:** encoder L = 56, horizon H = 28 (the M5 horizon).
- **Split:** validation = days 1914–1941 (the official M5 "validation" horizon), test = days 1942–1969 (the official evaluation horizon); one origin each.
- **Training windows:** origins every 7th day from day 56 to the last origin whose horizon ends before validation. A stride of 1 would give 55 M windows; the stride applies equally to all models.
- **Cold start:** items first launched within 28 days before the test start or later (held out of training, as on VISUELLE).
- **Young series:** series with age < 56 days at the test origin.
- **Covariates:**
  - *discount* = 1 − price / the item × store's maximum price up to that week, known in advance (price plan);
  - *promo* = discount > 5 %;
  - *holiday* = any calendar event on that day;
  - *pre-holiday* as in the pipeline.
  - SNAP is omitted: it does not fit the fixed known-covariate slots. The omission applies to all models. Weather is absent (0).
- **LA features:** as on VISUELLE, with life-cycle window W = 56 days. Siblings: up to 9 other stores. Analogues: K = 16 items, retrieved by cosine similarity of static keys [department one-hot, category one-hot, standardised log mean price]. There are no image or text embeddings.
- **Multimodal inputs:** M5 has none. The aligned / unimodal embedding inputs are zero with the modality masks off, so MTFT-Picnic reduces to its TFT backbone with categorical statics, and the v3 / v4 tokenizers receive no content. This is stated in every table.

## Models (5 seeds, 1–5; identical budget 16 × 200 × 256, d_model 32)

- TFT, MTFT-Picnic, M0 = MTFT-Picnic + siblings, M1 = M0 + LA-pool, MTFT+ v3, MTFT+ v4.
- MTFT+ v4.1 only if adopted under amendment (d) 5 before M5 is run. It is then run in addition to v4, not instead of it.
- Chronos-2:
  - zero-shot with context 56 (the models' window) and with context 512 (Chronos-2's strength, reported for fairness);
  - zero-shot group attention over the 10 stores (multivariate, context 56);
  - fine-tuned (target only; + covariates; group), context 56, on the same training windows. Grid lr ∈ {1e-5, 1e-6} × 3,000 steps on seed 1 (validation wQL), then 5 seeds;
  - the best fine-tuned variant on validation is the comparator.
- Trivial: all-zeros, last value, seasonal naive (same weekday last week), LA-sibling mean.

## Metrics and criteria (as on VISUELLE)

- WAPE, wQL and Cov80 at item × store level. Product level (item totals) and store level via bottom-up and the pre-specified reconciliations. Paired item-cluster bootstrap plus per-seed wins.
- Subsets: cold-start items, young series.
- **Criterion 1 (architecture):** v4 (and v4.1 if run) beats M1, significantly and on ≥ 4/5 seeds.
- **Criterion 2 (foundation model):** v4 beats the best fine-tuned Chronos-2, significantly and on ≥ 4/5 seeds.
- **Secondary:** M1 − M0 (the LA information) overall and on young series.

## Amendments

**2026-09-27 (a), after building the panel, before any M5 model run.** With price-plan launches, the official test origin (day 1941) has **0 young series, 0 cold-start items and 0 new-in-shop series**. The LA information therefore cannot act there. The launch dates alone (no sales outcomes, no model results) show yearly launch bursts. The latest large one is days 1120–1147 (Feb 2014), with 613 series.

A **second, launch-rich split** is added:

- The panel is truncated at day 1175. Validation = days 1120–1147, test = days 1148–1175 (2014-03-22 to 2014-04-18).
- At that origin there are 943 young series (age < 56 days) on 216 items, 315 series launching inside the test horizon, and 83 cold-start items.
- Everything else is as above: the same models, seeds, budget and criteria.

**Roles of the two splits:**

- *Official split* (days 1942–1969): primary for criterion 2 (Chronos-2) and the no-harm check of the LA information.
- *Launch-rich split:* primary for the LA hypothesis (M1 − M0 on young series and cold-start items) and for criterion 1.

Both are reported in full.

**2026-09-27 (b), after the v4.1 decision (`v4_preregistration.md` (f)), before any full-budget M5 model run.**

1. **Tested model.** v4.1 was not adopted, so the M5 comparison tests **MTFT+ v4**. The model list is unchanged: TFT, MTFT-Picnic, M0, M1, v3, v4 and the Chronos-2 variants. The "v4.1 if adopted" clause is void.

2. **Secondary metric: RMSSE / WRMSSE** (`examples/m5_wrmsse.py`), computed from the saved test forecasts, so no model is re-run for it.
   - It follows the M5 competition: 12 aggregation levels weighted 1/12, and dollar-sales weights over the 28 days before the test origin.
   - Scale = mean squared one-step difference of each series' own history, from its first non-zero sale to the origin.
   - Forecasts are aggregated bottom-up. Two point forecasts are scored: the Swanson mean of the quantiles (primary for this squared-error metric) and the median (the point forecast used for WAPE).
   - Series with an undefined or zero scale are excluded and counted.
   - WRMSSE gives zero weight to series without sales in the last 28 days, so cold starts do not enter it. The bottom-level RMSSE of young series is reported separately.
   - **Implementation check** (no model involved): the M5 benchmark forecasts computed from our panel score WRMSSE 1.7520 (Naive) and 0.8470 (seasonal Naive) on the evaluation horizon, identical to the published 1.752 and 0.847.
   - Criteria 1 and 2 remain defined on WAPE. WRMSSE is reported for every model and does not decide any criterion.

3. **Launch-stagger statistics** (launch = first price week; computed from the panels, no sales outcomes involved).

   | | Items | Distinct launch dates per item, median (mean) | Items with > 1 launch date | Series launching after the item's first store | Launch spread, days, median (p90) |
   |---|---|---|---|---|---|
   | Official panel, all items | 3,049 | 4 (3.94) | 88.0 % | 46.9 % | 70 (966) |
   | … on sale from day 1 (left-censored) | 1,457 | 3 (2.98) | 77.3 % | 25.0 % | 42 (1,329) |
   | … first launched later | 1,592 | 5 (4.81) | 97.7 % | 66.9 % | 98 (777) |
   | Launch-rich panel (days 1–1176), all items stocked by day 1176 | 2,850 | 3 (3.60) | 86.5 % | 43.7 % | 49 (630) |
   | … first launched later | 1,393 | 4 (4.39) | 96.5 % | 65.2 % | 56 (475) |

   - *Correction to the "Data" section above.* The figure there, "81 % of their store-series launch after the item's first store", was computed from first **sales**. With the pre-registered price-week launches it is **66.9 %** (items first launched after day 365: 1,146 items, 66.9 %; median spread 84 days). The data section is not edited; this amendment supersedes it.
   - Launch dates of items on sale from day 1 are left-censored: their stagger is understated.

   **Implications for testing age alignment.**
   - (i) Staggered launches exist on M5, as on VISUELLE (median 4–5 launch dates per later-launched item; about two thirds of their series launch after the first store). Age-aligned siblings therefore carry information in principle.
   - (ii) The LA window is W = 56 days, so LA information exists only for targets younger than 56 days.
     - At the official origin, no series is young (median age 1,787 days): LA inputs are empty at test time, and **the official split can only test that LA inputs do no harm**.
     - On the launch-rich split, 943 series are young. Of these, 600 have at least one earlier-launched sibling store, but the earliest sibling leads by a median of only 14 days, and only 208 by ≥ 28 days. Same-age sibling values are therefore mostly available for the first weeks of life only, and for part of the 28-day horizon they are not yet observed.
     - The remaining 343 young series and the 83 cold-start items rely on analogues. On M5 these are retrieved from department / category / price keys, not from content.
   - (iii) A test of age alignment on M5 is therefore **weak by construction**. It has a small effective sample (943 young series of 27,581) and short sibling leads. A null result on M5 would not contradict the VISUELLE result. A positive result on the young subset would be the confirmatory evidence the owner asked for. Both outcomes will be reported as such.

4. **Disclosure of M5 test numbers computed before this amendment.**
   - (i) The trivial baselines on the launch-rich split: all-zeros 1.000, last value 0.899, LA-sibling mean 0.899 and seasonal naive 0.900 WAPE. These were computed and reported to the owner.
   - (ii) A 2-batch CPU smoke test of all six models on the launch-rich split (`--epochs 1 --batches-per-epoch 2`). Its test numbers are meaningless (WAPE 1.0–8.1) and were not used for any decision.
   - No model, setting or criterion was changed after either.

**2026-09-29 (c), before any M5 model run.**

1. **Compute restriction.**
   - No Chronos-2 fine-tuning on the launch-rich split. Its zero-shot rows (univariate, + covariates, + cross-learning, context 512) are kept, but the zero-shot multivariate-group row there is dropped with the group job.
   - The official split, which is primary for criterion 2, keeps the full Chronos-2 programme: zero-shot, fine-tuned target-only, + covariates and multivariate group.
   - All trained models (TFT, MTFT-Picnic, M0, M1, v3, v4) run on both splits with 5 seeds, as registered.
   - On the launch-rich split, criterion 2 is not scored; only the zero-shot Chronos-2 rows are reported there.
2. **Grocery subset (secondary analysis).**
   - M5's FOODS category (departments FOODS_1–3) is public grocery data, the closest public match to the grocery setting of the MTFT paper (Sukel et al., 2024).
   - Every M5 metric, and the paired bootstrap of the pre-registered comparisons, is also reported on the FOODS item × store series. The comparisons are M1 − M0, v4 − M1, v4 − Chronos-2 and **M0 − MTFT-Picnic, the sibling effect**.
   - This is computed after the fact from the saved test forecasts; no model is re-run.
   - It is secondary and decides no criterion.
   - No public grocery dataset we know of provides product images or text together with multi-store sales. The multimodal parts of the models are therefore tested on VISUELLE 2.0 (fashion) only. On M5 they receive no content and are stated as such.
