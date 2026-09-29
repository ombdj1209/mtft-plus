# MTFT+ — Retrieval-augmented, contrastive multimodal TFT with coherent probabilistic supply quotas

An open, reproducible extension of the Multimodal Temporal Fusion Transformer (Sukel, Rudinac & Worring, *IEEE MultiMedia* 31(2), 2024), built as a controlled study of *which* multimodal ideas actually improve grocery demand forecasting.

**Status: research prototype; results are on a synthetic benchmark.** They show the pipeline is correct and quantify effects under controlled conditions. They are not evidence about Picnic's data, and "MTFT-Picnic" below is our re-implementation of the paper's design (unimodal encoders compressed to 10 dims per modality, used as static covariates), not Picnic's code.

```
mtft_plus/
  features.py          SigLIP / OpenCLIP extractor + disk cache, regularised multimodal tokenizer, SigLIP alignment loss
  model.py             MultimodalCrossAttention, CrossAttentionTemporalFusionDecoder, MultimodalTemporalFusionTransformer
  layers.py            GRN, GLU, vectorised VSN, interpretable MHA, monotone quantile head
  losses.py            QuantileLoss, newsvendor_quantile
  data.py              frozen synthetic generator, analogue retrieval (embedding / category), window dataset
  reconciliation.py    Hierarchy, MinT-shrink (low-rank, O(T²n) shrinkage), seasonal ridge base model, Swanson mean
  calibration.py       split-conformal quantile calibration in log1p space
  panel.py             real-data adapter (long-format CSVs, daily or weekly)
  lightning_module.py  Lightning training (AdamW + OneCycle, quantile + alignment objective)
  metrics.py           WAPE, MASE, pinball, wQL, coverage
benchmark.py           multi-seed, equal-compute benchmark with controls and ablations (resumable)
report.py              mean ± sd and paired article-cluster bootstrap; updates this README
examples/prepare_panel.py   real CSVs + images -> SigLIP embeddings -> panel for `benchmark.py --panel`
tests/                 14 tests: loss, causality, MinT vs dense closed form, shrinkage vs brute force,
                       analogue features use only the past, conformal coverage, panel round-trip, tiny SigLIP
results/               v2 results (results.md, summary.json, partials/), results/v1/ archived v1 run
```

```bash
pip install -e ".[multimodal,dev]"
pytest -q tests
python benchmark.py              # 3 seeds x 5 models + 2 ablations, ~3 h on 1 CPU core (resumable)
python report.py --readme README.md
```

## What changed in v2, and why

v1 (archived in `results/v1/`) found that aligned SigLIP embeddings with cross-attention *tied* a 10-dim static baseline overall and were *worse* on cold-start SKUs, while MinT reconciliation was the robust gain. v2 addresses each diagnosed failure without touching the data generator:

| v1 finding | v2 change |
|---|---|
| 768-d embeddings overfit ~900 training products (cold-start worse) | feature + modality dropout in the tokenizer (§1.6) |
| cold-start SKUs have no history for the model to anchor on | retrieval-augmented analogue memory over the SigLIP space (§1.5) |
| national 80 % intervals covered 45–59 % | per-level split-conformal calibration before MinT (§1.7) |
| single seed; ties within noise | 3 data seeds, paired article-cluster bootstrap, category-analogue control, ablations |
| synthetic only | real-data adapter + `examples/prepare_panel.py` + `benchmark.py --panel` |

---

## 1. Mathematical formalism

**Notation.** Articles $i=1..N$, fulfilment centres $f=1..F$, bottom series $b=(i,f)$, encoder length $L$, horizon $H$, quantile set $\mathcal Q=\{0.1,0.5,0.9\}$.

### 1.1 Unified contrastive embeddings

A frozen SigLIP pair $(f_v, f_t)$ maps image $x^v_i$ and text $x^t_i$ into one space:
$z^v_i = f_v(x^v_i)/\lVert f_v(x^v_i)\rVert,\; z^t_i = f_t(x^t_i)/\lVert f_t(x^t_i)\rVert \in \mathbb R^{D}$ ($D=768$ for `siglip-base-patch16-224`), pretrained with

$$\mathcal L_{\text{SigLIP}} = -\frac1n\sum_{i,j}\log\sigma\!\big(s_{ij}\,(t\,\langle z^v_i,z^t_j\rangle + b)\big),\qquad s_{ij}=\begin{cases}+1 & i=j\\-1 & i\neq j\end{cases}$$

The trainable tokenizer builds three typed tokens (plus optional patch/word tokens):

$$e^v_i = P_v(z^v_i)+\tau_v,\quad e^t_i = P_t(z^t_i)+\tau_t,\quad e^{vt}_i = P_{vt}\!\big(\sqrt D\,z^v_i\odot z^t_i\big)+\tau_{vt}$$

Because $\langle z^v, z^t\rangle=\mathbf 1^\top(z^v\odot z^t)$, a linear read-out of $e^{vt}$ is a *learned, concept-weighted cross-modal agreement* (e.g. "organic" in the description agreeing with visual freshness in the photo). With independently trained encoders (DistilBERT / ResNet-152) the coordinates of $z^v$ and $z^t$ are unrelated and this token carries no signal — this is the formal reason the shared space matters.

A Perceiver resampler with learned latents $\Lambda\in\mathbb R^{m\times d}$ compresses the $n_{\text{tok}}$ tokens $E_i$ into $m$ product tokens:
$M_i = \mathrm{GRN}\big(\mathrm{LN}(\Lambda + \mathrm{MHA}(\Lambda, E_i, E_i))\big)\in\mathbb R^{m\times d}.$

To keep the fine-tuned projections aligned, the objective adds the sigmoid loss on $(e^v, e^t)$ over *unique articles* in the batch (an article appears once per FC; duplicates would be false negatives):
$\mathcal L_{\text{align}} = -\frac1n\sum_{i,j}\log\sigma\big(s_{ij}(t'\langle \hat e^v_i,\hat e^t_j\rangle+b')\big).$

### 1.2 Dynamic cross-attention temporal fusion

The TFT backbone (VSNs, static contexts $c_s,c_e,c_h,c_c$, LSTM encoder–decoder, static enrichment, causal interpretable self-attention) yields decoder states $h_\tau\in\mathbb R^d,\ \tau=1..H$, which depend on known future covariates $x_\tau$ (promotion, discount depth, weather, holidays). Stacking $H_{\text{dec}}=[h_1..h_H]$:

$$Q=\mathrm{LN}(H_{\text{dec}})W_Q,\qquad K=M W_K,\qquad V=M W_V$$

$$A^{(j)}=\mathrm{softmax}\!\Big(\frac{Q^{(j)}K^{(j)\top}}{\sqrt{d_k}}\Big)\in\mathbb R^{H\times m},\qquad C=\big[A^{(1)}V^{(1)}\,\|\,\cdots\,\|\,A^{(h)}V^{(h)}\big]W_O$$

$$h'_\tau=\mathrm{LN}\big(h_\tau+\mathrm{GLU}(C_\tau)\big),\qquad \mathrm{GLU}(u)=\sigma(W_1u+b_1)\odot(W_2u+b_2)$$

followed by the position-wise GRN, gated skip to the LSTM output, and the quantile head.

*Static concatenation vs cross-attention.* Under static concatenation, product information enters only through time-invariant contexts $c_\bullet(z_i)$; its contribution at step $\tau$ is modulated only indirectly. Under cross-attention the weights $A_\tau = A(h_\tau(x_\tau), M_i)$ are an explicit function of the step's covariates, so which product evidence is read (e.g. packaging/brand tokens under a 35 % discount) varies per step. Per-step modality attribution is exposed as $\Pi_\tau=\bar A_\tau R_i\in\Delta^{n_{\text{tok}}}$ with $R_i$ the resampler attention (`model.return_mm_attribution = True`). Cost is $O(H\,m\,d)$ per series, independent of patch count.

### 1.3 Probabilistic multi-horizon loss

With $u=\tilde y-\hat y_q$ and $\tilde y=\log(1+y)$:

$$\rho_q(u)=\max\big(q\,u,(q-1)\,u\big),\qquad \mathcal L_Q=\frac{1}{|\Omega|}\sum_{(b,t,\tau)\in\Omega}\;\sum_{q\in\mathcal Q} w_q\,\rho_q\big(\tilde y_{b,t+\tau}-\hat y_{q,b,t,\tau}\big)$$

$\Omega$ masks pre-launch cells. Quantiles are equivariant under monotone maps, $Q_q(g(Y))=g(Q_q(Y))$, so $\hat y_q^{\text{units}}=\exp(\hat y_q)-1$ is exact. Non-crossing by construction: $\hat y_{0.5}=a_2,\ \hat y_{0.1}=a_2-\mathrm{softplus}(a_1),\ \hat y_{0.9}=a_2+\mathrm{softplus}(a_3)$. Total objective $\mathcal L=\mathcal L_Q+\beta\,\mathcal L_{\text{align}}$ ($\beta=0.05$).

Asymmetric inventory risk maps to a service quantile via the newsvendor critical ratio $q^\*=c_u/(c_u+c_o)$: short-shelf-life SKUs (high write-off cost $c_o$) order near $q_{50}$, ambient SKUs near $q_{90}$ (`newsvendor_quantile`, used for `national_order_quotas.csv`). $w_q$ lets you up-weight the tail that drives the order.

### 1.4 Hierarchical reconciliation (MinT-shrink)

Nodes $y=[y_{\text{agg}};b]$, $y_{\text{agg}}=Ab$, $S=[A;I_m]$ with national total, $F$ FC totals, $N$ article-national totals (supplier order quotas) and $m=NF$ bottoms: $n=1+F+N+NF=6{,}006$.

$$\tilde y = SG\hat y,\qquad G=(S^\top W^{-1}S)^{-1}S^\top W^{-1}\;\;\Longleftrightarrow\;\;\tilde y=\hat y-WC^\top(CWC^\top)^{-1}C\hat y,\quad C=[I_{n_a},-A]$$

$$\hat W=\hat\lambda\hat D+(1-\hat\lambda)\hat\Sigma,\quad \hat\Sigma=\frac1T\sum_t e_te_t^\top,\quad \hat D=\mathrm{diag}(\hat\Sigma),\quad \hat\lambda=\frac{\sum_{i\neq j}\widehat{\mathrm{Var}}(\hat r_{ij})}{\sum_{i\neq j}\hat r_{ij}^2}$$

Implementation details that make this production-safe at $n\approx 6\text{k}$ with $T=56$ residual vectors:

* $\hat\lambda$ (Schäfer–Strimmer, identical to `hts::shrink.estim`) is computed in $O(T^2n)$ via $\sum_{i\ne j}\sum_k x_{ki}^2x_{kj}^2=\sum_k[(\sum_i x_{ki}^2)^2-\sum_i x_{ki}^4]$ and $\lVert X^\top X\rVert_F=\lVert XX^\top\rVert_F$.
* $\hat W=\mathrm{diag}(\delta)+U^\top U$ is never materialised: only $WC^\top\ (n\times n_a)$ and a Cholesky of $CWC^\top\ (n_a\times n_a)$.
* Nodes without history (cold-start SKUs) get a prior variance $\hat D_{ii}=\kappa\,\hat\mu_i$, $\kappa$ = median variance-to-mean ratio of nodes with history.
* MinT is an expectation-level projection, so **means** are reconciled ($\hat\mu\approx0.3q_{10}+0.4q_{50}+0.3q_{90}$, Swanson); summing bottom medians is biased low for right-skewed counts. Each node's quantiles are shifted by its own mean adjustment, clipped at 0 and re-sorted; bottom means are clipped and re-aggregated so coherence holds exactly (max error ≈ 1e-11 in the benchmark).
* Aggregate base forecasts come from a direct multi-horizon seasonal ridge per aggregate node (`SeasonalRidgeForecaster`), falling back to bottom-up where a node has < 35 days of history.

### 1.5 Retrieval-augmented analogue memory (new in v2)

Cold-start SKUs have no demand history, but similar products do. Let $k_i=[z^v_i;z^t_i]$ be the frozen SigLIP key and $\mathcal P$ the set of training (warm) articles. For every article, including training articles, retrieve

$$\mathcal N(i)=\operatorname{top\text{-}k}_{j\in\mathcal P,\;j\neq i}\cos(k_i,k_j),\qquad \omega_{ij}=\frac{\exp(\cos(k_i,k_j)/T)}{\sum_{l\in\mathcal N(i)}\exp(\cos(k_i,k_l)/T)}$$

($k=10$, $T=0.05$). Excluding $i$ itself makes training windows identical in form to cold-start windows at test time. For bottom series $(i,f)$ and encoder step $t<t_0$ the analogue channel is

$$a_{i,f,t}=\frac{\sum_{j\in\mathcal N(i)}\omega_{ij}\,\mathbb 1[t\ge \ell_j]\,\log(1+y_{j,f,t})}{\sum_{j\in\mathcal N(i)}\omega_{ij}\,\mathbb 1[t\ge \ell_j]}$$

with $\ell_j$ the launch step, plus an availability flag and a static analogue level $\bar a_{i,f}=\frac1{28}\sum_{t=t_0-28}^{t_0-1}a_{i,f,t}$. Only data before the forecast origin $t_0$ is touched; `tests/test_analogue_features_use_only_past` corrupts every value from $t_0$ onward and asserts the features are unchanged.

### 1.6 Regularised multimodal tokenizer (new in v2)

v1 showed 768-d aligned embeddings overfitting ~900 training products (cold-start WAPE worse than a 10-d bottleneck). During training the tokenizer applies feature dropout $\tilde z=\mathrm{Dropout}_{p=0.2}(z)$ to both modalities and, with probability 0.2 per row, drops exactly one modality (never both) through the key-padding mask. This also makes the model robust to SKUs with a missing photo or description.

### 1.7 Split-conformal quantile calibration (new in v2)

v1's national 80 % intervals covered only 45–59 %. For each hierarchy level and each non-median quantile level $\tau$, on the validation origins:

$$\delta_\tau=\hat Q^{\text{conf}}_\tau\big(\log(1+y)-\log(1+\hat y_\tau)\big),\qquad \hat y'_\tau=\exp\big(\log(1+\hat y_\tau)+\delta_\tau\big)-1$$

with $\hat Q^{\text{conf}}_\tau$ the $\lceil (n+1)\tau\rceil$-th (upper) or $\lfloor (n+1)\tau\rfloor$-th (lower) order statistic. The log1p scale lets one shift serve nodes of very different size. Medians are untouched, so WAPE and MASE are unchanged. Calibrated base quantiles are then reconciled with MinT as in §1.4. Caveat: the validation window is also used to fit $\hat W$; the two uses estimate different parameters, but they are not independent samples.
---

## 2. Benchmark protocol

**Data.** The v1 generator, frozen: 5 FCs (NL-AMS, NL-UTR, DE-DUS, DE-KOL, FR-PAR) × 1,000 articles × 540 days, negative-binomial demand with FC-level weather, national/local promotions with post-promo dips, country holidays, category seasonality, 20 % late launches and 10 % cold-start SKUs (launched 0–21 days before the test period, excluded from training). Product semantics are observable only through simulated embeddings: *aligned* (shared image–text concept space) or *unimodal* (independent per-modality spaces, same information and nuisance). Data seeds 7, 11 and 23 regenerate the whole world.

**Models (equal budget: d=32, 4 heads, 8 epochs × 100 batches × 256 windows, AdamW + OneCycle, best validation weights).**

| Model | Product features | Fusion | Analogues |
|---|---|---|---|
| TFT | none | – | – |
| MTFT-Picnic | unimodal, PCA-10 per modality | static scalars | – |
| TFT + category analogues (control) | none | – | 10 random same-category products |
| MTFT-Picnic + RA | unimodal, PCA-10 | static scalars | top-10 in PCA-10 space |
| **MTFT+ v2 (ours)** | aligned SigLIP, dropout-regularised | cross-attention + alignment loss | top-10 in SigLIP space |
| ablation: ours w/o cross-attention | as ours | static pooled | as ours |
| ablation: ours w/o retrieval | as ours | cross-attention | – |

The category control is the critical one: if it matches ours, the gain comes from category membership, not from images and text.

**Evaluation.** Encoder 56 d, horizon 14 d; 4 validation origins (early stopping, MinT residuals, conformal calibration), 4 non-overlapping test origins. WAPE and MASE on the median; wQL = mean over $q$ of $2\sum\rho_q/\sum|y|$; Cov80 = coverage of $[q_{10},q_{90}]$. Significance: paired bootstrap over articles (2,000 resamples, stratified by seed); a difference counts only if its 95 % interval excludes zero.

## 3. Results

Tables below are generated by `report.py` from `results/partials/`.

<!--RESULTS:BEGIN-->
Data seeds: [7, 11, 23]. Mean ± sd across seeds (sd omitted for single-seed rows).

### Article × FC level (test)

| Model | seeds | val loss | WAPE | MASE | wQL | Cov80 | cold-start WAPE | cold-start wQL |
|---|---|---|---|---|---|---|---|---|
| TFT | 3 | 0.4112 ± 0.0044 | 0.4163 ± 0.0050 | 0.7854 ± 0.0375 | 0.2646 ± 0.0033 | 0.791 ± 0.0032 | 0.4252 ± 0.0113 | 0.2699 ± 0.0076 |
| MTFT-Picnic | 3 | 0.4080 ± 0.0037 | 0.4126 ± 0.0041 | 0.7755 ± 0.0370 | 0.2620 ± 0.0020 | 0.794 ± 0.0039 | 0.4244 ± 0.0144 | 0.2692 ± 0.0091 |
| TFT + category analogues | 3 | 0.4108 ± 0.0046 | 0.4165 ± 0.0063 | 0.7860 ± 0.0383 | 0.2652 ± 0.0034 | 0.793 ± 0.0061 | 0.4233 ± 0.0110 | 0.2690 ± 0.0071 |
| MTFT-Picnic + RA | 3 | 0.4082 ± 0.0041 | 0.4151 ± 0.0058 | 0.7767 ± 0.0391 | 0.2636 ± 0.0034 | 0.793 ± 0.0028 | 0.4260 ± 0.0141 | 0.2706 ± 0.0090 |
| MTFT+ v2 (ours) | 3 | 0.4082 ± 0.0036 | 0.4113 ± 0.0046 | 0.7745 ± 0.0351 | 0.2612 ± 0.0025 | 0.792 ± 0.0032 | 0.4208 ± 0.0090 | 0.2670 ± 0.0059 |
| ours w/o cross-attention | 1 | 0.4099 | 0.4054 | 0.7984 | 0.2572 | 0.792 | 0.4192 | 0.2647 |

### Paired article-cluster bootstrap: ours − comparator (negative = ours better), 95 % CI

| Comparator | seeds | ΔWAPE all | ΔwQL all | ΔWAPE cold-start | ΔwQL cold-start |
|---|---|---|---|---|---|
| TFT | 3 | **-0.0050 [-0.0060, -0.0040]** | **-0.0033 [-0.0040, -0.0027]** | **-0.0042 [-0.0074, -0.0009]** | **-0.0028 [-0.0049, -0.0004]** |
| MTFT-Picnic | 3 | **-0.0015 [-0.0026, -0.0004]** | **-0.0008 [-0.0018, -0.0001]** | -0.0032 [-0.0074, +0.0009] | -0.0020 [-0.0048, +0.0007] |
| TFT + category analogues | 3 | **-0.0050 [-0.0057, -0.0043]** | **-0.0038 [-0.0044, -0.0033]** | -0.0023 [-0.0048, +0.0003] | **-0.0019 [-0.0035, -0.0001]** |
| MTFT-Picnic + RA | 3 | **-0.0045 [-0.0065, -0.0028]** | **-0.0029 [-0.0044, -0.0016]** | -0.0055 [-0.0135, +0.0005] | -0.0038 [-0.0101, +0.0008] |
| ours w/o cross-attention | 1 | **+0.0015 [+0.0007, +0.0024]** | **+0.0014 [+0.0009, +0.0020]** | +0.0006 [-0.0023, +0.0038] | +0.0011 [-0.0016, +0.0040] |

Bold = 95 % interval excludes zero.

### Hierarchy: WAPE / wQL / Cov80 by level (MinT + conformal unless stated)

| Model | Method | national | fulfilment centre | article-national | article × FC |
|---|---|---|---|---|---|
| TFT | MinT + conformal | 0.0314 / 0.0187 / 0.86 | 0.0369 / 0.0223 / 0.83 | 0.2374 / 0.1645 / 0.79 | 0.4470 / 0.2844 / 0.77 |
| MTFT-Picnic | MinT + conformal | 0.0327 / 0.0192 / 0.85 | 0.0376 / 0.0226 / 0.82 | 0.2324 / 0.1611 / 0.80 | 0.4398 / 0.2793 / 0.77 |
| TFT + category analogues | MinT + conformal | 0.0313 / 0.0186 / 0.86 | 0.0366 / 0.0222 / 0.84 | 0.2374 / 0.1645 / 0.79 | 0.4467 / 0.2844 / 0.77 |
| MTFT-Picnic + RA | MinT + conformal | 0.0332 / 0.0193 / 0.87 | 0.0381 / 0.0227 / 0.82 | 0.2332 / 0.1617 / 0.80 | 0.4406 / 0.2804 / 0.77 |
| MTFT+ v2 (ours) | bottom-up | 0.0605 / 0.0404 / 0.40 | 0.0622 / 0.0398 / 0.57 | 0.2120 / 0.1401 / 0.77 | 0.4113 / 0.2612 / 0.79 |
| MTFT+ v2 (ours) | base | 0.0535 / 0.0362 / 0.46 | 0.0626 / 0.0412 / 0.53 | 0.2779 / 0.1973 / 0.71 | 0.4113 / 0.2612 / 0.79 |
| MTFT+ v2 (ours) | MinT | 0.0306 / 0.0184 / 0.82 | 0.0362 / 0.0224 / 0.83 | 0.2322 / 0.1604 / 0.75 | 0.4395 / 0.2799 / 0.76 |
| MTFT+ v2 (ours) | MinT + conformal | 0.0306 / 0.0183 / 0.86 | 0.0362 / 0.0219 / 0.84 | 0.2322 / 0.1611 / 0.80 | 0.4395 / 0.2792 / 0.77 |
| ours w/o cross-attention | MinT + conformal | 0.0374 / 0.0211 / 0.71 | 0.0431 / 0.0260 / 0.70 | 0.2307 / 0.1592 / 0.78 | 0.4306 / 0.2732 / 0.77 |

MinT λ̂ range 0.917–0.950; max coherence error 7.1e-10.

### Modality attribution (ours): image / text / fused

| seed | promo days | non-promo days |
|---|---|---|
| 7 | 0.304 / 0.284 / 0.411 | 0.303 / 0.284 / 0.413 |
| 11 | 0.280 / 0.281 / 0.439 | 0.280 / 0.280 / 0.441 |
| 23 | 0.277 / 0.316 / 0.407 | 0.276 / 0.315 / 0.409 |

<!--RESULTS:END-->

### 3.1 Findings (3 data seeds, equal compute)

1. **MTFT+ v2 is the best model on average, but its edge over the MTFT-style baseline is small and not seed-robust.** Mean article × FC WAPE 0.4113 vs 0.4126 for MTFT-Picnic (−0.3 % relative), wQL 0.2612 vs 0.2620. The article-cluster bootstrap pooled over seeds gives ΔWAPE −0.0015 [−0.0026, −0.0004]. That interval captures sampling of articles, **not** training-run variance, and v2 lost to MTFT-Picnic on seed 23 (0.4108 vs 0.4096; best on that seed was MTFT-Picnic + RA, 0.4087). With 3 seeds, a seed-level paired test has no power. Treat this as "at least as good, possibly marginally better", not as a win.
2. **Robust gains over tabular and category-only baselines.** v2 beats plain TFT and the TFT + category-analogue control on all three seeds (≈ −1.2 % WAPE; ΔWAPE −0.0050 [−0.0060, −0.0040] and −0.0050 [−0.0057, −0.0043]).
3. **Cold start: the v1 regression is fixed, but images and text are not proven to beat category membership.** v2 has the lowest mean cold-start WAPE (0.4208 vs 0.4233 category control, 0.4244 MTFT-Picnic, 0.4252 TFT). Only the difference to TFT is significant (−0.0042 [−0.0074, −0.0009]); against the category control the interval is [−0.0048, +0.0003]. In v1 the SigLIP models were *worse* than TFT on cold start (0.447 vs 0.431).
4. **Retrieval alone is not the mechanism.** Adding analogues to the MTFT-style baseline hurt on two seeds and helped on one (mean 0.4151 vs 0.4126). The gain appears only in combination with the aligned, regularised tokenizer. **Ablation (seed 7, single seed): removing cross-attention does not hurt.** Static pooled fusion with the same embeddings, regularisation and retrieval reached WAPE 0.4054 and cold-start WAPE 0.4192, versus 0.4070 and 0.4198 for full v2. On this benchmark, dynamic cross-attention fusion is not the source of the improvement; aligned embeddings plus regularisation plus retrieval are. This matches v1's finding that promotion-dependent attention does not emerge when product × covariate interactions are static. The second ablation (retrieval removed) is in the table if it completed; otherwise run `python benchmark.py --seeds 7`.
5. **Reconciliation trade-off, stated plainly.** MinT-shrink roughly halves national WAPE relative to bottom-up (0.0306 vs 0.0605 for v2) and beats the independent base forecast at national and FC level, with exact coherence (error ≤ 7e-10). For **article-national totals (the supplier order quotas)**, however, plain bottom-up is more accurate (WAPE 0.2120 vs 0.2322 for MinT), and MinT costs bottom-level accuracy (0.4113 → 0.4395). The right production choice depends on which level the decision uses: MinT for FC and national capacity planning, bottom-up for article-level supplier orders.
6. **Calibration fixed.** Split-conformal plus MinT lifts national 80 % coverage to 0.86 (v1: 0.46–0.59), FC 0.84, article-national 0.80, bottom 0.77, while leaving medians, and hence WAPE and MASE, unchanged.

**Bottom line.** On this benchmark, the multimodal ideas give a reliable gain over tabular models and a marginal, not-yet-robust gain over the published MTFT design; the operationally largest effects come from reconciliation and calibration. Whether v2 beats the MTFT design on real data is open. The pipeline is ready to test it on VISUELLE 2.0 (§4).


## 4. Running on real data

```bash
# my_data/: sales.csv, catalog.csv (text, image_path), locations.csv [, weather.csv, holidays.csv]
python examples/prepare_panel.py --data-dir my_data --out panel.pkl                 # SigLIP encoding + cache
python examples/prepare_panel.py --data-dir my_data --freq W-MON --encoder-length 12 --horizon 4 \
       --cold-start-steps 3 --out panel.pkl                                        # weekly data
python benchmark.py --panel panel.pkl --seeds 1,2,3 --out results_real
python report.py --out results_real
```

Supplying `unimodal_image.npy` / `unimodal_text.npy` (e.g. ResNet-152 and DistilBERT features, rows in catalogue order) makes the MTFT-style baseline faithful to separate encoders; without them it compresses the aligned embeddings instead, and any write-up should say so.

**Recommended public benchmark: VISUELLE 2.0** (Skenderi et al., CVPRW 2022; humaticslab.github.io/forecasting/visuelle). It has 5,355 fast-fashion products across 110 shops (2016–2019), with product images, textual tags, shop-level sales, discounts and weather: the ingredients this pipeline needs, and a recognised cold-start task. It is fashion, not grocery; claims should be scoped accordingly.

## 5. What can and cannot be claimed

* Can: every component is tested; MinT-shrink yields exactly coherent supply quotas at 6k nodes; the effect sizes in §3 hold under this generator with the stated uncertainty.
* Cannot: that MTFT+ beats Picnic's production model or the published MTFT on Picnic data. That needs the same real data, the authors' baseline or a faithful re-implementation, and multiple seeds.
* The generator encodes the mechanisms the models exploit (e.g. category-level co-movement, image–text concept sharing). Synthetic wins are a necessary check, not a sufficient one.

## 6. Abstract

Multimodal Temporal Fusion Transformers (MTFT) improve grocery demand forecasts by adding product images and descriptions, typically as independently encoded, low-dimensional static covariates. We study which extensions of this design actually help, in a controlled, equal-compute benchmark with frozen data generation, three data seeds, category-only controls and paired bootstrap inference. MTFT+ combines frozen SigLIP image–text embeddings with a dropout-regularised Perceiver tokenizer, retrieval of demand analogues in the shared embedding space, and decoder cross-attention to product tokens; a single-seed ablation indicates the cross-attention component itself contributes nothing on this benchmark. It trains non-crossing q10/q50/q90 forecasts and reconciles 6,006 article × fulfilment-centre series into coherent supply quotas with a low-rank MinT-shrink projection, followed by split-conformal calibration. MTFT+ reliably outperforms tabular and category-analogue baselines (−1.2 % WAPE on all seeds) and removes the cold-start degradation seen when high-dimensional aligned embeddings are used without regularisation. Its advantage over a compact 10-dimensional static MTFT-style baseline, however, is small (−0.3 % WAPE) and not consistent across seeds. Reconciliation halves national-level error relative to bottom-up aggregation but is less accurate than bottom-up for article-level supplier quotas, and conformal calibration raises national interval coverage from below 0.6 to 0.86. We conclude that the benefit of richer multimodal fusion over compact static features is marginal at a catalogue of ~1,000 SKUs, and release the pipeline, tests and a real-data adapter so that the question can be settled on public multimodal retail data such as VISUELLE 2.0.


## Citation

Sukel, M., Rudinac, S., & Worring, M. (2024). Multimodal Temporal Fusion Transformers are Good Product Demand Forecasters. *IEEE MultiMedia*, 31(2), 48–60. doi:10.1109/MMUL.2024.3373827

Skenderi, G., Joppi, C., Denitto, M., Scarpa, B., & Cristani, M. (2022). The multi-modal universe of fast-fashion: the Visuelle 2.0 benchmark. *CVPR Workshops*.
