# MTFT+ — Contrastive Multimodal Cross-Attention TFT with MinT Reconciliation

Production PyTorch / Lightning pipeline extending the Multimodal Temporal Fusion Transformer of Sukel, Rudinac & Worring (IEEE MultiMedia 31(2), 2024; Picnic Technologies / University of Amsterdam).

```
mtft_plus/
  features.py            MultimodalFeatureExtractor (SigLIP / OpenCLIP, disk cache), MultimodalTokenizer, siglip_alignment_loss
  model.py               MultimodalCrossAttention, CrossAttentionTemporalFusionDecoder, MultimodalTemporalFusionTransformer
  losses.py              QuantileLoss, newsvendor_quantile
  layers.py              GRN, GLU, VSN, interpretable MHA, monotone quantile head
  reconciliation.py      Hierarchy, MinTReconciler (shrinkage, low-rank), SeasonalRidgeForecaster, swanson_mean
  lightning_module.py    MTFTLightningModule (AdamW + OneCycle, quantile + alignment objective)
  data.py                synthetic 5-FC x 1,000-article generator, vectorised window dataset
  metrics.py             WAPE, MASE, pinball, wQL, coverage
benchmark.py             end-to-end benchmark (4 models x 3 reconciliation methods x 4 hierarchy levels)
tests/                   unit tests (loss, causality, MinT vs dense closed form, λ vs brute force, tiny SigLIP extractor)
```

```bash
pip install -r requirements.txt
pytest -q tests                      # 11 tests
python benchmark.py                  # paper-scale synthetic run  (~38 min on 1 CPU core; add --resume after interruptions)
python benchmark.py --quick          # smoke test (~2 min)
```

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

---

## 2. Benchmark

**Data** (`mtft_plus/data.py`). 5 FCs (NL-AMS, NL-UTR, DE-DUS, DE-KOL, FR-PAR) × 1,000 articles × 540 days, 12 categories with shelf lives 3–365 d, negative-binomial demand driven by FC-level weather (country-correlated AR(1)), national/local promotion campaigns with post-promo dips, country holidays, category seasonality, launches (20 % late launches, 10 % cold-start SKUs launched 0–21 days before the test period and excluded from training). Latent product semantics (category, organic, visual freshness, premium packaging, brand strength, seasonal look) drive base level, promo elasticity and weather/season response and are observable only through simulated embeddings:

* *aligned* (SigLIP-like): image and text evidence written onto one shared concept dictionary plus modality gap, nuisance "style" subspaces and noise;
* *unimodal* (ResNet / DistilBERT-like): the same evidence on independent per-modality dictionaries, identical nuisance and noise, then PCA → 10 dims per modality (the Picnic-style bottleneck).

**Protocol.** Encoder 56 d, horizon 14 d; train on origins before day 428; 4 validation origins (early stopping, MinT residuals); 4 non-overlapping test origins (days 484–539). Same backbone, optimiser and budget for all models (d=32, 4 heads, 8 epochs × 100 batches × 256 windows, AdamW + OneCycle, best-val weights). WAPE and MASE on the median; pinball in units; wQL = mean over $q$ of $2\sum\rho_q/\sum|y|$; Cov80 = empirical coverage of $[q_{10},q_{90}]$. MASE uses the in-sample seasonal-naive (m=7) scale and is undefined for cold-start series.

### 2.1 Results — article × FC level (test, 20,000 series-origins)

| Model | Val pinball (log) | WAPE | MASE | Pinball | wQL | Cov80 | Cold-start WAPE | Cold-start wQL |
|---|---|---|---|---|---|---|---|---|
| TFT | 0.4134 | 0.4161 | 0.8162 | 2.247 | 0.2640 | 0.787 | 0.4314 | 0.2726 |
| MTFT-Picnic (10-d, static) | 0.4099 | 0.4119 | 0.8038 | 2.231 | 0.2622 | 0.794 | 0.4251 | 0.2690 |
| MTFT-SigLIP (static pooled) | 0.4105 | 0.4214 | 0.8078 | 2.289 | 0.2690 | 0.788 | 0.4502 | 0.2911 |
| MTFT-SigLIP-XAttn (ours) | 0.4091 | 0.4164 | 0.8018 | 2.261 | 0.2657 | 0.791 | 0.4469 | 0.2880 |

### 2.2 Results — reconciliation (WAPE / wQL by level)

| Model | Method | national | fulfilment centre | article national | article x fc |
|---|---|---|---|---|---|
| TFT | base (incoherent) | 0.0465 / 0.0316 | 0.0557 / 0.0367 | 0.2699 / 0.1902 | 0.4161 / 0.2640 |
| TFT | bottom-up | 0.0779 / 0.0558 | 0.0789 / 0.0532 | 0.2242 / 0.1478 | 0.4161 / 0.2640 |
| TFT | MinT-shrink | 0.0379 / 0.0235 | 0.0435 / 0.0272 | 0.2362 / 0.1628 | 0.4381 / 0.2797 |
| MTFT-Picnic (10-d, static) | base (incoherent) | 0.0465 / 0.0316 | 0.0557 / 0.0367 | 0.2694 / 0.1899 | 0.4119 / 0.2622 |
| MTFT-Picnic (10-d, static) | bottom-up | 0.0912 / 0.0685 | 0.0922 / 0.0645 | 0.2246 / 0.1480 | 0.4119 / 0.2622 |
| MTFT-Picnic (10-d, static) | MinT-shrink | 0.0417 / 0.0259 | 0.0464 / 0.0290 | 0.2328 / 0.1606 | 0.4311 / 0.2758 |
| MTFT-SigLIP (static pooled) | base (incoherent) | 0.0465 / 0.0316 | 0.0557 / 0.0367 | 0.2716 / 0.1912 | 0.4214 / 0.2690 |
| MTFT-SigLIP (static pooled) | bottom-up | 0.1175 / 0.0948 | 0.1187 / 0.0888 | 0.2410 / 0.1594 | 0.4214 / 0.2690 |
| MTFT-SigLIP (static pooled) | MinT-shrink | 0.0447 / 0.0275 | 0.0494 / 0.0308 | 0.2349 / 0.1620 | 0.4330 / 0.2791 |
| MTFT-SigLIP-XAttn (ours) | base (incoherent) | 0.0465 / 0.0316 | 0.0557 / 0.0367 | 0.2717 / 0.1911 | 0.4164 / 0.2657 |
| MTFT-SigLIP-XAttn (ours) | bottom-up | 0.0978 / 0.0751 | 0.0983 / 0.0696 | 0.2327 / 0.1541 | 0.4164 / 0.2657 |
| MTFT-SigLIP-XAttn (ours) | MinT-shrink | 0.0462 / 0.0285 | 0.0494 / 0.0310 | 0.2345 / 0.1618 | 0.4317 / 0.2776 |

MinT shrinkage intensity λ̂ ≈ 0.935, 0.947, 0.941, 0.949 (T=56 residual vectors for n=6,006 nodes); max coherence error ≤ 4.1e-10.

Cross-attention modality attribution (test, mean over decoder steps): promo days image 0.276 / text 0.310 / fused 0.414; non-promo days 0.275 / 0.309 / 0.415.

### 2.3 Findings

1. **Fusion strategy (single seed, equal compute): no reliable winner.** All four variants lie within ≈2 % of each other at the article × FC level. Cross-attention has the lowest validation pinball loss (0.4091) and the lowest MASE (0.8018), but the Picnic-style 10-d static baseline has the best WAPE (0.4119), pinball and wQL. These gaps are of the size expected from seed variance; a multi-seed run is required before any ranking is claimed.
2. **Cold start: the hypothesis was not confirmed here.** On the 100 unseen SKUs, both SigLIP variants are *worse* than the tabular TFT (WAPE 0.447–0.450 vs 0.431), while the 10-d PCA baseline is best (0.425). With only ~900 unique training products, 768-d inputs let the tokenizer memorise article identity; the PCA bottleneck acts as a regulariser. This is the opposite of the bottleneck argument and is the most important result of this run.
3. **No promotion-dependent weighting emerged.** Modality attribution is identical on promo and non-promo days (Δ < 0.002). In this generator promo elasticity is a *static* product attribute, which TFT static enrichment can already express; cross-attention has no structural advantage unless product–covariate interactions change within a horizon (e.g. competitor promos, weather × packaging), which the benchmark does not contain. The fused image⊙text token receives the most attention (0.41 vs 0.28 image / 0.31 text), consistent with the alignment argument, but it did not translate into accuracy.
4. **MinT is the robust gain.** Versus bottom-up, MinT-shrink cuts national WAPE by 51–62 % for every model (TFT: 0.078 → 0.038) and beats the independent base forecast at national and FC level for all four models (only narrowly for cross-attention at national level, 0.0462 vs 0.0465), while making every level coherent (error ≤ 4e-10). The cost is a small loss at the bottom level (TFT: WAPE 0.416 → 0.438) and at article-national level relative to bottom-up; this is the usual MinT trade-off with near-diagonal shrinkage (λ̂ ≈ 0.94).
5. **Aggregate intervals are too narrow.** Cov80 at national level is 0.45–0.59 for MinT (0.46 base). The ridge's in-sample residual quantiles under-state out-of-sample uncertainty; conformal calibration on the validation origins is the next fix.

**Validity.** Synthetic data encodes the mechanisms by construction and uses simulated, not real, SigLIP embeddings (Hugging Face was unreachable from the build environment; the real extractor is unit-tested against a randomly initialised `SiglipModel`). Results establish that the pipeline is correct and runnable at paper scale; they are not evidence about Picnic's data. Recommended next steps: 5 seeds with paired Diebold–Mariano tests; a generator variant with time-varying product × covariate interactions; embedding dropout / low-rank tokenizer for cold start; conformal aggregate intervals; real SigLIP features on a catalogue with ≥10k SKUs.

---

## 3. Abstract

Multimodal Temporal Fusion Transformers improve grocery demand forecasts by injecting product images and descriptions, but typically do so through independently trained encoders compressed to a few dimensions and appended as static covariates. We present MTFT+, which (i) encodes products with a frozen SigLIP vision-language model so image and text share one contrastive space, enabling an explicit image⊙text agreement token and a sigmoid alignment regulariser; (ii) replaces static concatenation with multi-head cross-attention in which each forecast step's decoder state queries a set of resampled product tokens; (iii) trains non-crossing q10/q50/q90 forecasts with a masked pinball objective tied to newsvendor service levels; and (iv) reconciles article × fulfilment-centre forecasts into national supply quotas with a memory-efficient MinT-shrink projection that handles cold-start nodes through prior variances and reconciles means rather than medians. On a controlled synthetic benchmark of 5 fulfilment centres and 1,000 articles (6,006 hierarchy nodes), the fusion variants are statistically indistinguishable at equal compute; cross-attention attains the lowest validation loss and MASE, but a 10-dimensional static baseline is more accurate on cold-start SKUs, indicating that high-dimensional aligned embeddings need stronger regularisation when the catalogue is small. MinT-shrink reduces national-level WAPE by 51–62 % relative to bottom-up aggregation and yields exactly coherent quotas. We release the pipeline, tests and benchmark to enable evaluation on real catalogues.

## 4. PR pitch

**Title:** MTFT+: aligned multimodal features, cross-attention fusion and MinT-coherent supply quotas

**What:** Drop-in extension of the MTFT forecaster: SigLIP feature extraction with a content-addressed cache; a cross-attention decoder (fusion mode is a config flag, so the current static-concat model is reproduced exactly as `static_scalar`); a monotone quantile head and pinball loss; and a MinT-shrink reconciliation layer that turns article × FC forecasts into coherent national order quotas (`national_order_quotas.csv`, service level from shelf life).

**Why merge now:** The reconciliation layer is independently valuable. On the benchmark it roughly halves national-level WAPE versus bottom-up and guarantees that FC plans sum to the supplier order. It runs in seconds at 6k nodes and scales as O(n·n_a) memory.

**What it does not yet show:** that cross-attention or aligned embeddings beat the current 10-d static features. Single-seed synthetic results are a tie, and 10-d features are better on cold start. The PR therefore ships fusion as an opt-in flag and proposes an A/B on real catalogue data (≥10k SKUs, 5 seeds) before changing the default.

**Risk:** Additive. Default fusion is unchanged, 11 unit tests cover loss, causality, MinT against the dense closed form and the extractor, and the benchmark is reproducible with one command (`python benchmark.py`, `--resume` supported).

