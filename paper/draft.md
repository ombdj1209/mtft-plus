# Align by Age, Not by Calendar: Lifecycle-Aligned Cross-Series Learning for Short-Lifecycle Retail Forecasting

**Om Prakash Bhardwaj** · p-bharadwaj@jmc-ltd.co.jp · omprakashbdj1209@gmail.com

*Workshop draft. Numbers marked [pending] are filled from the final result files; every number in this draft is traceable to a file listed in Appendix A.*

## Abstract

*Provisional. The M5 sentences are completed when the M5 runs finish.*

Short-lifecycle retail series have few or no observations of their own. Staggered launches mean that other shops have already revealed the same product's early life curve, and earlier similar products have revealed whole curves. Global forecasters and foundation models pool information across series at the same *calendar* time, however, and so miss this signal.

We propose lifecycle-aligned (LA) cross-series information: the same product's sales in earlier-launched shops, and similar earlier products' curves, both indexed by *product age* and computed strictly causally. In a pre-registered study on VISUELLE 2.0 (5,355 fast-fashion products, 110 shops), five seeds per model, we separate the information from the architecture that uses it.

- **Pooled LA information** added to a multimodal TFT lowers shop-level WAPE on the calendar benchmark (−0.0037 [−0.0053, −0.0022]; −0.0145 at product level) and on the two short-observation tasks. It does not help pure new-product demand.
- **Attention over the individual age-aligned tokens** beats this pooled control only where a series' own history is short or absent (−1.03 and −0.93 WAPE points). It is worse where own history exists, so the pre-registered architecture criterion fails.
- **Fully fine-tuned Chronos-2**, given the same covariates and the product's shops as a group, remains 12–30 WAPE points behind.
- **The published baselines**, re-run with their official code and validation-only checkpoint selection, score 87–88 WAPE on the short-observation tasks, far from their published 23–35.

[M5 sentence pending.]

## 1 Introduction

Fast-fashion products live for a few weeks. A product is launched in one shop, then in others over the following weeks, sells for one short life cycle and is replaced. For such a (product, shop) series the forecaster has two to twelve observations, often none. Global deep forecasters (DeepAR, TFT) pool information across series through shared weights. Recent foundation models (Chronos-2) pool it through in-context group attention. Both, however, align the pooled series by *calendar time*. When launches are staggered, the other shops of the same product are at a different age in the same calendar week, and similar products are mostly not stocked at all.

We ask a narrow question: **does it help to borrow demand information by product age instead of calendar week?** A shop that launched a product three weeks earlier has already revealed the first three weeks of its life curve. A similar product launched last season has revealed a full curve. Both are available causally at the forecast origin if they are re-indexed by age.

Contributions:

1. **Lifecycle-aligned (LA) information.** We give two strictly causal, age-aligned sources: LA siblings (the same product in earlier-launched shops, at the same age) and LA analogues (similar earlier products, at the same age). We also give a four-channel pooled summary (LA-pool) that any sequence model can take as known inputs.
2. **A controlled, pre-registered study** on VISUELLE 2.0, on both a calendar benchmark and the official protocol, with M5 (Walmart) as a second dataset. The design isolates the information from the architecture: the pooled control (M1) receives the same LA information as the attention model (v4).
3. **A negative architectural result and its boundary.** Attention over the individual age-aligned tokens does not beat the pooled control where own history exists. It helps only when the series' own history is short (2 weeks) or absent.
   - A history-aware gate (v4.1, exploratory) was not adopted under its pre-registered validation rule.
   - [M5 sentence pending.]
4. **Fair foundation-model and published baselines.** These include Chronos-2 zero-shot, with cross-learning and multivariate groups, and fully fine-tuned with covariates and groups. They also include the official CrossAttnRNN and GTM-Transformer code re-run with validation-only checkpoint selection.

## 2 Related work

**Forecasting by analogy.** Judgemental forecasting has long used analogous past cases for new situations: structured analogies (Green & Armstrong, 2007), and analogies for the sales of new products (Goodwin, Dyussekeneva & Meeran, 2013). The classic finding is that analogies help when they are selected by similarity and adjusted for differences. LA analogues are an automated, causal version of this: similar products are retrieved by content embeddings, and their curves are indexed by age so that "week 3 of the analogue" informs "week 3 of the target".

**New-product forecasting from similar items.** Diffusion models (Bass, 1969) describe a new product's life curve with a few parameters, often set from comparable products. Data-driven approaches cluster or retrieve comparable products and learn from their curves (Baardman et al., 2018, "Leveraging comparables for new product sales forecasting"). They also combine product attributes and images with sales histories in neural models (Singh et al., 2019; Ekambaram et al., 2020). Most of these methods align the comparables' curves at launch, as we do. They do not use the same product's staggered launches across shops, which is the strongest signal in our data.

**VISUELLE and GTM-Transformer.** VISUELLE (Skenderi et al., 2024) and VISUELLE 2.0 (Skenderi et al., 2022) are public fast-fashion benchmarks with images, tags and shop-level sales. GTM-Transformer (Skenderi et al., 2024) forecasts new-product demand from the image, text and Google Trends. The VISUELLE 2.0 paper introduces the short-observation tasks and a cross-attention RNN baseline. We re-run both official codebases.
- In the released training scripts, the data loader passed to Lightning for validation, which also drives checkpoint monitoring, is built from the test split (Appendix C).
- For a comparison under a common protocol, we instead select checkpoints on a validation split drawn from the training pairs, as for all our models, and report runs with the released selection separately (§4.4).

**Global and hierarchical forecasting.** Global models share parameters across series and outperform local models on large retail collections (Salinas et al., 2020; Montero-Manso & Hyndman, 2021). TFT (Lim et al., 2021) adds static and known-future inputs. MTFT (Sukel, Rudinac & Worring, 2024) adds image and text embeddings as static inputs and is our backbone; MTFT-Picnic is our re-implementation of that design, not Picnic's system. Cross-series information can also enter through reconciliation: Optimal combination (Hyndman et al., 2011) and MinT (Wickramasuriya, Athanasopoulos & Hyndman, 2019) combine forecasts across a hierarchy. We evaluate MinT and bottom-up on the article × shop hierarchy.

**Foundation models and cross-learning.** Chronos (Ansari et al., 2024) and Chronos-2 (Ansari et al., 2025) are pre-trained forecasters. Chronos-2 supports covariates and *group attention*: series in one group attend to each other in context ("cross-learning"), and multivariate targets share one group. This is the closest foundation-model analogue of sibling learning. We therefore give Chronos-2 the same product's shops as a group, both zero-shot and after full fine-tuning.

## 3 Method

### 3.1 Setting

A panel has products a, shops f and time steps t. The series (a, f) launches at ℓ(a, f) and is *live* (stocked and observed) on a known set of steps. At origin o, a model forecasts y(a, f, o … o+H−1) from its encoder window of L steps. The **age** of the target at step t is k = t − ℓ(a, f). Every quantity below is computed from sales at calendar times ≤ o − 1 only.

### 3.2 Lifecycle-aligned siblings and analogues

- **LA siblings.** For target (a, f) and horizon or encoder step t with age k, the sibling in shop g ≠ f is y(a, g, ℓ(a, g) + k). It is used only if ℓ(a, g) + k ≤ o − 1 and the cell is live. Up to M = 32 shops are used, the earliest launched first.
- **LA analogues.** The K = 16 most similar products b ≠ a that launched before o, by cosine similarity of product embeddings (SigLIP image + text on VISUELLE; department, category and log price on M5). Each analogue contributes its mean log1p per-shop sales at age k, over the shops whose age-k step is known before o.

Both are precomputed for all origins from event times with one cumulative sum: S[o, b, k] and N[o, b, k] are the sum and count of age-k log sales known before o.

### 3.3 Models

All models share the MTFT backbone, data, splits and budget.

| Name | Backbone | Cross-series inputs |
|---|---|---|
| M0 = MTFT-Picnic + siblings | static image/text scalars (PCA of ResNet-152 / DistilBERT) | calendar siblings: mean log demand of the product in other live shops, same calendar week |
| **M1 = M0 + LA-pool** | as M0 | + 4 pooled LA channels per step: sibling mean, sibling availability, similarity-weighted analogue mean, analogue coverage |
| v3 | SigLIP tokenizer, static pooled | calendar siblings + calendar same-shop analogues |
| **v4** | SigLIP tokenizer + age embedding | calendar siblings + LA-pool + *lifecycle attention* |
| v4.1 (exploratory) | v4 | lifecycle attention gated by own-history length |

**Lifecycle attention (v4).** At every step the decoder state attends over two token sets, each with a learned null token:
- the individual LA-sibling values, with learned shop embeddings and launch lag;
- the individual LA-analogue values, with projected product embeddings, similarity and support.

The attended summaries enter as additional per-step variables of the variable-selection network. M1 gets the same information averaged, so M1 versus v4 isolates the architecture.

**History-aware gate (v4.1).** The lifecycle-attention variables are multiplied by g = σ(α · n_obs / L + β), where n_obs is the number of observed own-history steps in the encoder window, and α and β are learned (initialised at 0, so g = 0.5). The idea was motivated by the v4 test results (§5.2): attention helps where own history is short. It is therefore reported as exploratory. It was adopted or rejected on validation loss, seed 1, before any v4.1 test number was seen (pre-registration amendment (d) 5).

## 4 Experimental protocol

### 4.1 Pre-registration

The v4 design, comparisons and success criteria were written before any v4 result. Every later change was appended as a dated amendment before the results it concerns (Appendix B). The two primary criteria:
1. v4 beats M1 (the architecture claim).
2. v4 beats the best fine-tuned Chronos-2, where "best" is chosen on validation (the foundation-model claim).

Each needs a 95 % paired product-cluster bootstrap interval excluding 0 and at least 4/5 seed wins, on the calendar benchmark and on at least 2 of the 3 official tasks.

### 4.2 Data

- **VISUELLE 2.0.** 5,355 products in 110 shops, with 12 observed weeks per (product, shop) pair and staggered launches: a median of 4 release dates per product, and 71.8 % of pairs launch after the product's first shop.
  - *Calendar benchmark:* weekly grid, encoder 12, horizon 4, four test origins (2019-09-16 to 2019-12-30), cold-start products held out of training.
  - *Official protocol:* SO-fore 2-1, SO-fore 2-10 and new-product demand, official split and WAPE. Our Naive and SES re-runs match the published numbers to within 0.1 WAPE points.
- **M5 (Walmart).** 3,049 items in 10 stores, daily. A series launches on the first day of its first price week. Encoder 56, horizon 28. Two splits:
  - the official evaluation horizon, where no series is young;
  - a pre-registered launch-rich split (test 2014-03-22 to 2014-04-18), with 943 series younger than 56 days and 83 cold-start items.

### 4.3 Training and evaluation

- 5 seeds, 16 epochs × 200 batches × 256, d_model 32, early stopping on validation.
- Quantile loss at 0.1 / 0.5 / 0.9.
- Metrics: WAPE, wQL and 80 % coverage at the bottom level; national, shop and product totals via bottom-up and MinT.
- Paired product-cluster bootstrap; per-seed wins, since seed variance can be of the same size as method differences (Bouthillier et al., 2021; Agarwal et al., 2021).

### 4.4 Baselines

- **Trivial:** all-zeros, last value, LA-sibling mean used directly, and seasonal naive (M5).
- **Chronos-2:**
  - zero-shot, univariate, with covariates, with cross-learning over the product's shops, and with the product's shops as a multivariate group;
  - fully fine-tuned, target only, + covariates (discount, calendar siblings, LA-pool) and multivariate group. Learning rate from {1e-5, 1e-6} × 3,000 steps, with checkpoints chosen on validation.
- **Published models:** CrossAttnRNN (with and without image) and GTM-Transformer, from the official code.
  - Patches were needed for current library versions, and GTM gets a store embedding for the shop-level demand task.
  - Checkpoints are chosen on a validation split (the 10 % most recent training pairs), the same rule as for every other model.
  - Runs with the released configuration, where the validation loader is built from the test split (Appendix C), are reported separately as diagnostics of reproducibility, not as comparisons.

## 5 Results

All numbers are means over 5 seeds. Intervals are paired product-cluster bootstrap 95 % intervals (2,000 resamples), stratified by seed. "Wins" counts the seeds on which the first model has the lower WAPE. Negative differences favour the first model.

### 5.1 Lifecycle-aligned information helps where there is history to align

**Table 1.** Calendar benchmark: shop-week level, 4 test origins, 185,952 live test cells. Cold start = 691 held-out products. New-in-shop = 629 series of products already sold elsewhere.

| Model | WAPE | wQL | cold-start WAPE | new-in-shop WAPE |
|---|---|---|---|---|
| All-zeros / last value / LA-sibling mean | 1.0000 / 1.0427 / 1.0058 | | 1.0000 / 1.0078 / 0.9753 | |
| TFT | 0.6547 | 0.4227 | 0.6457 | 0.6926 |
| MTFT-Picnic | 0.6515 | 0.4197 | 0.6420 | 0.6954 |
| M0 = MTFT-Picnic + calendar siblings | 0.6446 | 0.4145 | 0.6349 | 0.6816 |
| MTFT+ v3 (calendar siblings + analogues) | 0.6434 | 0.4140 | 0.6344 | 0.6792 |
| **M1 = M0 + LA-pool** | **0.6409** | **0.4114** | **0.6317** | 0.6843 |
| v4 w/o lifecycle attention (A1) | 0.6426 | 0.4134 | 0.6339 | 0.6839 |
| v4 | 0.6443 | 0.4142 | 0.6354 | 0.6844 |
| Chronos-2, zero-shot (best of 5 variants) | 1.0252 | 0.7118 | 0.9904 | 1.1657 |
| Chronos-2, fine-tuned + covariates | 0.7599 | 0.5345 | 0.7475 | 0.8275 |

**Pooled LA information versus calendar siblings (M1 − M0):**

| Setting | ΔWAPE | Wins |
|---|---|---|
| Calendar benchmark, shop-week level | **−0.0037 [−0.0053, −0.0022]** | 4/5 |
| Calendar benchmark, product totals across shops | **−0.0145 [−0.0181, −0.0109]** (about 4.5 % relative) | |
| SO-fore 2-1 | **−0.20** | 4/5 |
| SO-fore 2-10 | **−0.54** | 5/5 |
| New-product demand | +0.13, not significant | 3/5 |

The pattern follows the information available. In the demand task the target has no own history, and at its launch week the product has often not been sold in another shop, so there is little to align.

**Size of the effect.** Shop-week WAPE differences are small because the targets are small counts (mean 1.14 units per live cell). As a consistency check, not a floor, every model's bottom-level WAPE is within ±0.01 of the WAPE that Poisson noise around *its own* mean forecast would give (v4, seed 1: 0.6455 vs 0.6450). Differences between good models are 3–4× larger at the product level.

### 5.2 Lifecycle attention: the pre-registered architecture criterion fails

**Table 2.** Official VISUELLE 2.0 protocol, WAPE %. Own history is 2 weeks (2-1, 2-10) or none (demand).

| Model | 2-1 | 2-10 | demand |
|---|---|---|---|
| Naive / SES (our re-run of the paper's baselines) | 101.93 / 97.86 | 118.24 / 111.34 | – |
| M0 = MTFT-Picnic + siblings | 65.25 | 66.84 | 64.01 |
| MTFT+ v3 | 65.73 | 65.66 | 63.07 |
| M1 = M0 + LA-pool | **65.06** | 66.30 | 64.15 |
| v4 w/o lifecycle attention (A1) | 66.32 | 66.15 | 63.46 |
| v4 | 65.90 | **65.27** | 63.22 |

**Criterion 1 (v4 − M1)** required a significant win on at least 4/5 seeds on the calendar benchmark *and* on at least 2 of 3 official tasks. It fails on the calendar benchmark, so it fails overall:

| Setting | v4 − M1 | Wins |
|---|---|---|
| Calendar benchmark | **+0.0034 [+0.0019, +0.0050]** | 1/5 |
| SO-fore 2-1 | **+0.85** | 0/5 |
| SO-fore 2-10 | **−1.03 [−1.39, −0.65]** | 5/5 |
| Demand | **−0.93 [−1.41, −0.48]** | 5/5 |

On new products only, v4 − M1 is +0.84, −1.18 and −1.20 on 2-1, 2-10 and demand.

**Mechanism.** Attention over individual age-aligned tokens helps when the series' own history is short (2-10, two weeks) or absent (demand). It hurts when two recent weeks (2-1) or up to twelve weeks (calendar encoder) already pin the level.

**Ablations on the calendar benchmark.**
- Removing the analogue branch helps: v4 − A2 = +0.0023 [+0.0012, +0.0034].
- The sibling branch helps: v4 − A3 = −0.0015 [−0.0026, −0.0002].
- Aligned (SigLIP) and separate-encoder retrieval do not differ: +0.0006 [−0.0007, +0.0020].

**Exploratory v4.1 gate.** A history-aware gate on the attention variables was motivated by these test results. It lost to v4 on its pre-registered validation rule (lower validation loss in 2/4 settings; ≥ 3 required) and was not adopted. Its seed-1 test numbers are reported in the supplementary files. On the official tasks the number of observed steps is constant within each task, so there the gate reduces to one learned scalar per task and cannot test history awareness.

### 5.3 Chronos-2, zero-shot and fully fine-tuned

**Criterion 2 passes everywhere.** The comparator is the fine-tuned Chronos-2 variant with the lowest validation loss among target-only, + covariates and multivariate group:

| Setting | Comparator | v4 − comparator | Wins |
|---|---|---|---|
| Calendar | + covariates | **−0.116 [−0.120, −0.112]** | 5/5 |
| SO-fore 2-1 | + covariates | **−15.04 [−15.51, −14.59]** | 5/5 |
| SO-fore 2-10 | target only | **−29.70 [−30.71, −28.68]** | 5/5 |
| Demand | target only | **−17.39 [−17.88, −16.93]** | 5/5 |

**Table 3.** Chronos-2 variants, WAPE (calendar) and WAPE % (official):

| Chronos-2 variant | Calendar | 2-1 | 2-10 | demand |
|---|---|---|---|---|
| Zero-shot, univariate | 1.0252 | 96.11 | 111.09 | 100.57 |
| Zero-shot + cross-learning over the product's shops | 1.0292 | 95.93 | 111.96 | 100.57 |
| Zero-shot, product's shops as one multivariate group | 1.0292 | 95.97 | 112.71 | 100.57 |
| Fine-tuned, target only | 0.8559 | 90.19 | 94.97 | 80.61 |
| Fine-tuned + covariates (discount, calendar siblings, LA-pool) | 0.7599 | 80.95 | 93.69 | 79.42 |
| Fine-tuned, multivariate group | 0.8971 | 90.03 | 96.27 | 91.47 |

**Why zero-shot is above the all-zeros WAPE (1.025 vs 1.000).** We found no input error:
- NaN-padded and trimmed contexts give identical results.
- Outputs are in units, and median and mean agree.
- Synthetic patterns come out on the correct steps.

The excess comes from the life cycle:
- 26.5 % of live units fall in windows with no own history, which are forecast as 0.
- Windows with 1–3 weeks of history are better than zeros (0.886).
- Later windows over-forecast the decline: 1.114 at 4–7 weeks and 1.321 at 8–12 weeks, where forecasts are 1.18–1.34 × the actuals.

**What helps Chronos-2.** Fine-tuning helps most, and covariates add to it. Grouping by product helps neither zero-shot nor after fine-tuning. Chronos-2's group attention relates series at the same calendar time, whereas the useful sibling information lies at the same age.

### 5.4 Published baselines re-run with the official code

**Table 4.** Test WAPE %. Primary rows use the released epoch budget and checkpoints chosen on our validation split. The early-stopped runs are a secondary row in the supplementary files.

| Configuration | Published | Re-run (5 seeds) | Best fine-tuned Chronos-2 | v4 |
|---|---|---|---|---|
| CrossAttnRNN, 2-1 | 23.20 | 87.76 | 80.95 | 65.90 |
| CrossAttnRNN w/ image, 2-1 | 23.70 | 86.93 | 80.95 | 65.90 |
| CrossAttnRNN, 2-10 | 35.13 | 87.68 | 94.97 | 65.27 |
| CrossAttnRNN w/ image, 2-10 | 32.25 | 87.10 | 94.97 | 65.27 |
| CrossAttnRNN w/ image, demand (released layer3–4 fine-tuning) | 83.33 | 83.07 (seed 1; seeds 2–5 [pending]) | 80.61 | 63.22 |
| GTM-Transformer + store, demand | – | [pending] | 80.61 | 63.22 |

- **Metric.** The official metric function reproduces our WAPE to rounding, and our Naive and SES re-runs match the paper to within 0.1. The gap is therefore not a difference in the metric.
- **Training budget.** Training the full released budget instead of early stopping changes results by at most about 2 points.
- **Released selection.** Runs with the released checkpoint selection (test split as validation loader, seed 21) are [pending]. We make no claim about how the published numbers were obtained (Appendix C).

### 5.5 Reconciliation: bottom-up is best

For product totals, shop totals and the national total, summing the bottom-level forecasts is better than any reconciliation we tried. For M1, seed 1, the WAPE at each level is:

| Level | Bottom-up | Ridge base | MinT-WLS | MinT-shrink (pre-specified) |
|---|---|---|---|---|
| National | 0.100 | 0.143 | 0.104 | 3.115 |
| Shop | 0.164 | 0.304 | 0.176 | 3.137 |
| Product | 0.310 | 0.585 | 0.381 | 0.609 |

**Why MinT-shrink fails.**
- It estimates the covariance of 594,516 nodes from 16 residual vectors.
- Of the 22,249 series live in test, only 6,650 were live in validation.
- The resulting system is ill-conditioned: cond(C W Cᵀ) = 2.75 × 10¹³.
- Its adjustments offset each other, with the negative mass equal to 94 % of the positive mass. Clamping the negative bottom forecasts to zero then breaks coherence: the national forecast goes from 3,426 to 54,519 units, against 13,248 actual.

The shrinkage estimator suits long-lived series whose residual correlations persist. With 12-week life cycles they do not.

### 5.6 M5 (Walmart): second dataset

[pending: the pre-registered M5 study (models as above, 5 seeds; official split and launch-rich split; WAPE primary, WRMSSE secondary) is running. The pre-registration already records the expectation that LA information has little overall effect on M5. Only 943 of 27,581 test series are young on the launch-rich split, and the earliest sibling leads by a median of 14 days.]

## 6 Discussion and limitations

**Scope of the lifecycle-alignment claim.** The evidence for LA information comes from the pooled control, M1 − M0. It supports the claim only in settings where the target has its own history or where siblings have history:
- the calendar benchmark: −0.0037 WAPE [−0.0053, −0.0022], 4/5 seeds; −0.0145 at product level;
- SO-fore 2-1: −0.20, 4/5;
- SO-fore 2-10: −0.54, 5/5.

It is **not** supported on the new-product demand task: +0.13, not significant, 3/5 seeds. There the target has no own history, and at a pair's launch week the product has often not been sold in any other shop yet. We therefore do not claim that age alignment helps pure cold-start forecasting. The gains of v4's attention on the demand task (−0.93 vs M1) are an architecture effect on top of information that, pooled, does not help.

**What transfers and what does not.**
- *Transfers:* the age-aligned *information*, pooled into four channels any sequence model can take. It is cheap, causal and architecture-agnostic.
- *Does not transfer as a general improvement:* a more expressive attention mechanism over the same information. We report the failed architecture criterion as pre-registered.

The boundary we observe, attention helps only with short or absent own history, is a finding about where token-level cross-series attention is worth its variance. It is not a proof that it cannot work elsewhere.

**Foundation models.** Chronos-2 was given every advantage we could: covariates, cross-learning, multivariate groups, full fine-tuning and validation-based selection. It is still 12–30 points behind a small task-specific model trained on the same windows. We attribute the gap to three things Chronos-2 does not have here:
- product content (images and text);
- age alignment of related series (its group attention works at the same calendar time);
- enough context: 2–12 points per series.

We do not claim this holds for long-lived series; M5 addresses that.

**Published baselines.** We could not approach the published SO-fore numbers with the released code under a common validation protocol. We report the discrepancy, the released checkpoint-selection setting (Appendix C) and a diagnostic run with that setting. We draw no conclusion about the original experiments.

**Limitations.**
- **One fashion dataset for the main claims**, pending the M5 study. The M5 test of age alignment is weak by construction (§5.6).
- **Small effects.** Bottom-level effects are small in absolute terms because the targets are small counts. Product-level differences are larger and are reported alongside.
- **Bootstrap scope.** The bootstrap resamples products, not training runs; per-seed wins are reported with every interval.
- **Reproducibility.** v4 is not bit-reproducible on GPU (non-deterministic embedding backward); the 5-seed spread covers it.
- **Deviations from the pre-registration**, all recorded as dated amendments (Appendix B):
  - the reduced fine-tuning grid for Chronos-2;
  - one seed-1 pipeline check that showed test numbers before the 5-seed runs;
  - a corrected criterion-2 comparator on two tasks, which increased the margins;
  - the full-budget re-runs of the published baselines, requested after the early-stopped runs.
- **Published-baseline fidelity.** The released models' frozen image trunks were cached. Their batch-norm layers therefore use fixed ImageNet statistics, whereas in the released code they run in training mode. GTM-Transformer needed a store embedding to forecast per shop.
- **Compute restriction.** All experiments ran on one RTX 2060 SUPER (8 GB); our models are small (d_model 32). The released demand CrossAttnRNN fine-tunes ResNet-101 in fp32 at about 8 GPU-hours per seed. Because of this restriction, some published-baseline rows have fewer seeds (Table 4); their per-seed values are shown without intervals:
  - demand CrossAttnRNN, 2 seeds;
  - GTM-Transformer at full budget, 1 seed;
  - released-selection diagnostics on 2-1 and 2-10 only.

  These cuts cannot change the conclusions. The re-run–published gap (55–64 points) is about 50× the seed spread, and the demand configuration reproduces its published number (83.07 vs 83.33). On M5, Chronos-2 is fine-tuned on the official split only (amendment M5 (c)).
- **Grocery.** The multimodal components are a proof of concept on public fashion data. No public grocery dataset we know of provides product images or text together with multi-store sales. The cross-series findings are tested on grocery through M5's FOODS subset, a pre-specified secondary analysis.

## Appendix A: where every number comes from

| Content | File |
|---|---|
| Pre-registration and all amendments | `results/v4_preregistration.md`, `results/m5_preregistration.md` |
| Calendar benchmark tables and bootstrap | `results_visuelle2/results.md` (per-run files in `results_visuelle2/partials/`) |
| Official protocol tables and bootstrap | `results_visuelle2_official/results.md` (per-run files in `results_visuelle2_official/partials/`) |
| Scored report against the pre-registration | `results/visuelle2/v4_README.md` |
| v3 study (calendar siblings, reconciliation) | `results/visuelle2/README.md` |
| Chronos-2 zero-shot diagnosis | `results/visuelle2/diagnostics/chronos_diagnostics.txt` |
| MinT diagnosis | `results/visuelle2/diagnostics/mint_diagnostics.txt` |
| Published baselines | `results/visuelle2/published_baselines.md`; early-stopped runs in `results_visuelle2_official/published_early_stopped/`; released-selection runs in `results_visuelle2_official/diagnostics/` |
| Demand encoder tests | `results/visuelle2/diagnostics/demand_encoder_test.txt` |
| v4.1 (exploratory) | `results/visuelle2/v41_exploratory.md`, `results_visuelle2_v41/` |
| M5 | `results_m5/`, `results_m5_launch/` [pending] |

## Appendix B: amendments to the pre-registration (all dated, each recorded before the results it concerns unless stated)

- **(a)** Chronos-2 fine-tuning details.
- **(b)** Disclosure of a seed-1 pipeline check; secondary analyses (product level, noise consistency check); the development rule for later changes.
- **(c)** Chronos-2 grid reduced to lr ∈ {1e-5, 1e-6} × 3,000 steps, for compute.
- **(d)** Fairness tasks: the Chronos-2 diagnosis, cross-learning and groups, trivial baselines, the published baselines, the MinT analysis, the noise wording, the v4.1 rule and the M5 pre-registration.
- **(e)** Early-stopping rule for the published baselines (superseded by (h)); smoke-test disclosure; compatibility patches.
- **(f)** v4.1 decision on validation loss: not adopted.
- **(g)** On the official tasks the v4.1 gate is a per-task scalar. *Written after (f), at the owner's request.*
- **(h)** Full released epoch budget for all published baselines; early-stopped runs become a secondary row. *Requested after the early-stopped results were seen; the budget is fixed by the released defaults.*
  - Addendum 1: the order of the diagnostics.
  - Addendum 2: the released layer3–4 fine-tuning for the demand model; the 256-px patch; the batch-norm difference.
  - Addendum 3: the out-of-memory fallback rule.
- **M5 (a)** Launch-rich split added, because the official M5 origin has no young series.
- **M5 (b)** Tested model = v4; WRMSSE as a secondary metric; launch-stagger statistics.

## Appendix C: checkpoint selection in the released code

We cite the released files as vendored in `third_party/`. These two files are unmodified; our compatibility patches are applied from `examples/visuelle2_published.py` and in `dataset.py`.

- **VISUELLE 2.0 (CrossAttnRNN)**, `third_party/visuelle2.0-code-main/train_dl.py`:
  - lines 28–29 read `stfore_test.csv` into `test_df`;
  - line 65 builds the dataset from it (`sales_df=test_df`), and lines 85–87 build `testloader` from that dataset;
  - lines 126–131 define a `ModelCheckpoint` with `monitor="val_wWAPE"`, `mode="min"`, `save_top_k=2`;
  - line 148 calls `trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=testloader)`.
- **GTM-Transformer**, `third_party/GTM-Transformer-main/train.py`:
  - line 24 reads `test.csv` into `test_df`, and line 36 builds `test_loader` from it;
  - lines 81–87 define a `ModelCheckpoint` with `monitor='val_mae'`, `save_top_k=1`;
  - line 95 sets `check_val_every_n_epoch=5`;
  - lines 99–100 call `trainer.fit(..., val_dataloaders=test_loader)`.

In both scripts, the metric monitored for checkpointing is therefore computed on the test split. We make no claim about how the published numbers were produced. For comparability, our re-runs use a validation split from the training pairs for both checkpointing and early stopping. The configuration as released is reported separately.

## References

- Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A. C., & Bellemare, M. G. (2021). Deep reinforcement learning at the edge of the statistical precipice. *Advances in Neural Information Processing Systems, 34*, 29304–29320.
- Ansari, A. F., Stella, L., Turkmen, C., Zhang, X., Mercado, P., Shen, H., Shchur, O., Rangapuram, S. S., Pineda Arango, S., Kapoor, S., Zschiegner, J., Maddix, D. C., Wang, H., Mahoney, M. W., Torkkola, K., Wilson, A. G., Bohlke-Schneider, M., & Wang, Y. (2024). Chronos: Learning the language of time series. *Transactions on Machine Learning Research*. arXiv:2403.07815.
- Ansari, A. F., Shchur, O., Küken, J., Auer, A., Han, B., Mercado, P., Rangapuram, S. S., Shen, H., Stella, L., Zhang, X., Goswami, M., Kapoor, S., Maddix, D. C., Guerron, P., Hu, T., Yin, J., Erickson, N., Desai, P. M., Wang, H., Rangwala, H., Karypis, G., Wang, Y., & Bohlke-Schneider, M. (2025). Chronos-2: From univariate to universal forecasting. arXiv:2510.15821.
- Baardman, L., Levin, I., Perakis, G., & Singhvi, D. (2018). Leveraging comparables for new product sales forecasting. *Production and Operations Management, 27*(12), 2340–2343. https://doi.org/10.1111/poms.12963
- Bass, F. M. (1969). A new product growth for model consumer durables. *Management Science, 15*(5), 215–227. https://doi.org/10.1287/mnsc.15.5.215
- Bouthillier, X., Delaunay, P., Bronzi, M., Trofimov, A., Nichyporuk, B., Szeto, J., Mohammadi Sepahvand, N., Raff, E., Madan, K., Voleti, V., Ebrahimi Kahou, S., Michalski, V., Arbel, T., Pal, C., Varoquaux, G., & Vincent, P. (2021). Accounting for variance in machine learning benchmarks. *Proceedings of Machine Learning and Systems, 3*. arXiv:2103.03098.
- Ekambaram, V., Manglik, K., Mukherjee, S., Sajja, S. S. K., Dwivedi, S., & Raykar, V. (2020). Attention based multi-modal new product sales time-series forecasting. In *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining* (pp. 3110–3118). https://doi.org/10.1145/3394486.3403362
- Goodwin, P., Dyussekeneva, K., & Meeran, S. (2013). The use of analogies in forecasting the annual sales of new electronics products. *IMA Journal of Management Mathematics, 24*(4), 407–422. https://doi.org/10.1093/imaman/dpr025
- Green, K. C., & Armstrong, J. S. (2007). Structured analogies for forecasting. *International Journal of Forecasting, 23*(3), 365–376. https://doi.org/10.1016/j.ijforecast.2007.05.005
- Hyndman, R. J., Ahmed, R. A., Athanasopoulos, G., & Shang, H. L. (2011). Optimal combination forecasts for hierarchical time series. *Computational Statistics & Data Analysis, 55*(9), 2579–2589. https://doi.org/10.1016/j.csda.2011.03.006
- Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal fusion transformers for interpretable multi-horizon time series forecasting. *International Journal of Forecasting, 37*(4), 1748–1764. https://doi.org/10.1016/j.ijforecast.2021.03.012
- Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). M5 accuracy competition: Results, findings, and conclusions. *International Journal of Forecasting, 38*(4), 1346–1364. https://doi.org/10.1016/j.ijforecast.2021.11.013
- Montero-Manso, P., & Hyndman, R. J. (2021). Principles and algorithms for forecasting groups of time series: Locality and globality. *International Journal of Forecasting, 37*(4), 1632–1653. https://doi.org/10.1016/j.ijforecast.2021.03.004
- Salinas, D., Flunkert, V., Gasthaus, J., & Januschowski, T. (2020). DeepAR: Probabilistic forecasting with autoregressive recurrent networks. *International Journal of Forecasting, 36*(3), 1181–1191. https://doi.org/10.1016/j.ijforecast.2019.07.001
- Singh, P. K., Gupta, Y., Jha, N., & Rajan, A. (2019). Fashion retail: Forecasting demand for new items. KDD AI4Fashion Workshop. arXiv:1907.01960.
- Skenderi, G., Joppi, C., Denitto, M., & Cristani, M. (2024). Well googled is half done: Multimodal forecasting of new fashion product sales with image-based Google Trends. *Journal of Forecasting, 43*(6), 1982–1997. https://doi.org/10.1002/for.3104
- Skenderi, G., Joppi, C., Denitto, M., Scarpa, B., & Cristani, M. (2022). The multi-modal universe of fast-fashion: The Visuelle 2.0 benchmark. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)* (pp. 2240–2245). https://doi.org/10.1109/CVPRW56347.2022.00245
- Sukel, M., Rudinac, S., & Worring, M. (2024). Multimodal temporal fusion transformers are good product demand forecasters. *IEEE MultiMedia, 31*(2), 48–60. https://doi.org/10.1109/MMUL.2024.3373827
- Wickramasuriya, S. L., Athanasopoulos, G., & Hyndman, R. J. (2019). Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization. *Journal of the American Statistical Association, 114*(526), 804–819. https://doi.org/10.1080/01621459.2018.1448825

*References checked against Crossref, arXiv, OpenReview and the publishers' pages on 2026-09-29. The page numbers for Agarwal et al. (2021) come from secondary indexes only.*
