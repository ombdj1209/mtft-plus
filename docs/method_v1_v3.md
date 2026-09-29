# MTFT+ v3 — Multimodal, retrieval- and sibling-augmented TFT with coherent probabilistic supply quotas

An open, reproducible extension of the Multimodal Temporal Fusion Transformer (Sukel, Rudinac & Worring, *IEEE MultiMedia* 31(2), 2024), developed as a controlled study of which ideas actually improve grocery demand forecasting, and in what order we learned it.

**Status: research prototype; all results are on a synthetic benchmark.** "MTFT-Picnic" is our re-implementation of the paper's design (unimodal encoders compressed to 10 dims per modality, used as static covariates), not Picnic's code or data. The results validate the method under controlled conditions; a claim about real data requires the evaluation in §4.

**TL;DR (3 data seeds, equal compute, frozen generator).**

| Article × FC level, test | WAPE | Cold-start WAPE | vs MTFT-Picnic (95 % CI) | Seeds won |
|---|---|---|---|---|
| MTFT-Picnic (published design, re-implemented) | 0.4126 | 0.4244 | – | – |
| MTFT-Picnic + siblings (control) | 0.4088 | 0.4163 | −0.0038 | 3/3 |
| **MTFT+ v3 (ours)** | **0.4072** | **0.4145** | **−0.0055 [−0.0064, −0.0047]** | **3/3** |

* **v3 beats the MTFT design on every seed**: −1.3 % WAPE overall, −2.3 % on cold-start SKUs (−0.0097 [−0.0135, −0.0063]), and it beats every other baseline and control.
* **Most of that gain is cross-FC sibling learning**, not multimodality: giving the published design the same sibling channel recovers ≈ 70 % of it (≈ 84 % on cold start). This is the main practical finding, and it works in any architecture.
* **The multimodal stack still adds a small, consistent gain on top**: v3 vs MTFT-Picnic + siblings is −0.0016 [−0.0022, −0.0009] WAPE (−0.4 %), won on 3/3 seeds, but not significant on cold start ([−0.0048, +0.0002]).
* **Supply quotas**: MinT-shrink cuts national WAPE vs bottom-up (0.0316 vs 0.0492) with exact coherence and 84 % national interval coverage after conformal calibration. Bottom-up remains more accurate for article-level supplier orders (0.2071 vs 0.2311).
* How we got here, including two versions that did not work, is in [How we got here](#how-we-got-here-v1--v2--v3).


```
mtft_plus/
  features.py          SigLIP / OpenCLIP extractor + disk cache, regularised multimodal tokenizer, SigLIP alignment loss
  model.py             MTFT backbone with fusion modes (none / static_scalar / static_pooled / cross_attention)
  layers.py            GRN, GLU, vectorised VSN, interpretable MHA, monotone quantile head
  losses.py            QuantileLoss, newsvendor_quantile
  data.py              frozen synthetic generator, analogue retrieval, sibling channels, window dataset
  reconciliation.py    Hierarchy, MinT-shrink (low-rank, O(T²n) shrinkage), seasonal ridge base model, Swanson mean
  calibration.py       split-conformal quantile calibration in log1p space
  panel.py             real-data adapter (long-format CSVs, daily or weekly)
  lightning_module.py  Lightning training (AdamW + OneCycle, quantile + alignment objective)
  metrics.py           WAPE, MASE, pinball, wQL, coverage
benchmark.py           multi-seed, equal-compute benchmark: baselines, controls, v2, v3, ablations (resumable)
report.py              mean ± sd and paired article-cluster bootstrap; updates this README
examples/prepare_panel.py   real CSVs + images -> SigLIP embeddings -> panel for `benchmark.py --panel`
tests/                 19 tests (loss, causality, MinT vs dense closed form, sparse vs dense MinT, shrinkage vs brute
                       force, past-only analogue and sibling features, conformal coverage, panel round-trip, weekly
                       grid snapping, seasonal settings, tiny SigLIP extractor)
results/               v3 results (results.md, summary.json, partials/); results/v1/ and results/v2/ archived
```

```bash
pip install -e ".[multimodal,dev]"       # on Windows/Linux + NVIDIA: install a CUDA torch wheel first
pytest -q tests
python benchmark.py --no-ablations       # 3 seeds x 7 models, ~3.5 h on 1 CPU core (resumable)
python benchmark.py --no-ablations --accelerator gpu   # GPU, fp32 (default --accelerator auto, --precision 32-true)
python report.py --readme README.md
```

The v1–v3 numbers below were produced on CPU in fp32 (`--accelerator cpu`). On an RTX 2060 SUPER in fp32, seed 7 reproduces them closely (MTFT-Picnic WAPE 0.4109 vs 0.4109, MTFT+ v3 0.4040 vs 0.4032) and trains 16–17× faster, but GPU runs are not bit-identical to CPU runs. Compare models only within one device/precision setting. `--precision 16-mixed` gave no speed-up for this small model and matched fp32 within noise at the full budget.

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

### 1.2 Dynamic cross-attention temporal fusion (v1–v2; optional in v3)

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

### 1.5 Retrieval-augmented analogue memory (v2)

Cold-start SKUs have no demand history, but similar products do. Let $k_i=[z^v_i;z^t_i]$ be the frozen SigLIP key and $\mathcal P$ the set of training (warm) articles. For every article, including training articles, retrieve

$$\mathcal N(i)=\operatorname{top\text{-}k}_{j\in\mathcal P,\;j\neq i}\cos(k_i,k_j),\qquad \omega_{ij}=\frac{\exp(\cos(k_i,k_j)/T)}{\sum_{l\in\mathcal N(i)}\exp(\cos(k_i,k_l)/T)}$$

($k=10$, $T=0.05$). Excluding $i$ itself makes training windows identical in form to cold-start windows at test time. For bottom series $(i,f)$ and encoder step $t<t_0$ the analogue channel is

$$a_{i,f,t}=\frac{\sum_{j\in\mathcal N(i)}\omega_{ij}\,\mathbb 1[t\ge \ell_j]\,\log(1+y_{j,f,t})}{\sum_{j\in\mathcal N(i)}\omega_{ij}\,\mathbb 1[t\ge \ell_j]}$$

with $\ell_j$ the launch step, plus an availability flag and a static analogue level $\bar a_{i,f}=\frac1{28}\sum_{t=t_0-28}^{t_0-1}a_{i,f,t}$. Only data before the forecast origin $t_0$ is touched; `tests/test_analogue_features_use_only_past` corrupts every value from $t_0$ onward and asserts the features are unchanged.

### 1.6 Regularised multimodal tokenizer (v2)

v1 showed 768-d aligned embeddings overfitting ~900 training products (cold-start WAPE worse than a 10-d bottleneck). During training the tokenizer applies feature dropout $\tilde z=\mathrm{Dropout}_{p=0.2}(z)$ to both modalities and, with probability 0.2 per row, drops exactly one modality (never both) through the key-padding mask. This also makes the model robust to SKUs with a missing photo or description.

### 1.7 Split-conformal quantile calibration (v2)

v1's national 80 % intervals covered only 45–59 %. For each hierarchy level and each non-median quantile level $\tau$, on the validation origins:

$$\delta_\tau=\hat Q^{\text{conf}}_\tau\big(\log(1+y)-\log(1+\hat y_\tau)\big),\qquad \hat y'_\tau=\exp\big(\log(1+\hat y_\tau)+\delta_\tau\big)-1$$

with $\hat Q^{\text{conf}}_\tau$ the $\lceil (n+1)\tau\rceil$-th (upper) or $\lfloor (n+1)\tau\rfloor$-th (lower) order statistic. The log1p scale lets one shift serve nodes of very different size. Medians are untouched, so WAPE and MASE are unchanged. Calibrated base quantiles are then reconciled with MinT as in §1.4. Caveat: the validation window is also used to fit $\hat W$; the two uses estimate different parameters, but they are not independent samples.

### 1.8 Cross-FC sibling learning (new in v3)

The same article is sold in $F$ fulfilment centres, and its series share promotions, product appeal and launch timing. Instead of forecasting each $(i,f)$ in isolation, v3 feeds the model the article's demand in the other centres, following the cross-learning idea behind group attention in Chronos-2 (Ansari et al., 2025). For encoder step $t<t_0$:

$$s_{i,f,t}=\frac{1}{F-1}\sum_{g\neq f}\log\big(1+y_{i,g,t}\big),\qquad \bar s_{i,f}=\frac{1}{28}\sum_{t=t_0-28}^{t_0-1}s_{i,f,t}$$

$s_{i,f,\cdot}$ enters the encoder VSN as an observed channel and $\bar s_{i,f}$ as a static real. The series' own target is excluded, so it never predicts itself. `tests/test_sibling_channel_is_mean_of_other_fcs_and_past_only` checks the value against a direct computation and that corrupting every value from $t_0$ onward leaves the features unchanged. Averaging $F-1$ noisy sibling series gives a lower-variance estimate of the article's current level. On seed 7, a naive level estimate improves from MAE 0.218 to 0.204 in log space when siblings are blended in, which is the headroom v3 exploits.

**v3 model.** Frozen SigLIP embeddings → regularised tokenizer (§1.6) → mean-pooled product token as a static variable (cross-attention dropped: the v2 ablation showed no gain) + analogue channels (§1.5) + sibling channels (§1.8) → TFT backbone → monotone quantile head (§1.3) → split-conformal calibration (§1.7) → MinT-shrink (§1.4).

---

## 2. Benchmark protocol

**Data.** The v1 generator, frozen across v1–v3 so the benchmark could not be tuned towards any method: 5 FCs (NL-AMS, NL-UTR, DE-DUS, DE-KOL, FR-PAR) × 1,000 articles × 540 days, negative-binomial demand with FC-level weather, national/local promotions with post-promo dips, country holidays, category seasonality, 20 % late launches and 10 % cold-start SKUs (launched 0–21 days before the test period, excluded from training). Product semantics are observable only through simulated embeddings: *aligned* (shared image–text concept space) or *unimodal* (independent per-modality spaces, same information and nuisance). Data seeds 7, 11 and 23.

**Models (equal budget: d=32, 4 heads, 8 epochs × 100 batches × 256 windows, AdamW + OneCycle, best validation weights, same seeds).**

| Model | Product features | Fusion | Analogues | Siblings |
|---|---|---|---|---|
| TFT | none | – | – | – |
| MTFT-Picnic | unimodal, PCA-10 per modality | static scalars | – | – |
| TFT + category analogues (control) | none | – | 10 same-category products | – |
| MTFT-Picnic + RA (control) | unimodal, PCA-10 | static scalars | top-10 in PCA-10 space | – |
| MTFT-Picnic + siblings (control) | unimodal, PCA-10 | static scalars | – | ✓ |
| MTFT+ v2 | aligned SigLIP, regularised | cross-attention | top-10 in SigLIP space | – |
| **MTFT+ v3 (ours)** | aligned SigLIP, regularised | static pooled | top-10 in SigLIP space | ✓ |

Each control isolates one ingredient. *Category analogues* tests whether retrieval needs images and text at all. *MTFT-Picnic + siblings* is the decisive control for v3: it gives the published design the same sibling channel, so any v3 advantage over it cannot come from siblings alone.

**Evaluation.** Encoder 56 d, horizon 14 d; 4 validation origins (early stopping, MinT residuals, conformal calibration), 4 non-overlapping test origins. WAPE and MASE on the median; wQL = mean over $q$ of $2\sum\rho_q/\sum|y|$; Cov80 = coverage of $[q_{10},q_{90}]$. Significance: paired bootstrap over articles (2,000 resamples, stratified by seed), plus the per-seed win count, because the bootstrap does not capture training-run variance.

## 3. Results

Tables generated by `report.py` (bootstrap: v3 − comparator).

<!--RESULTS:BEGIN-->
Data seeds: [7, 11, 23]. Mean ± sd across seeds (sd omitted for single-seed rows).

### Article × FC level (test)

| Model | seeds | val loss | WAPE | MASE | wQL | Cov80 | cold-start WAPE | cold-start wQL |
|---|---|---|---|---|---|---|---|---|
| TFT | 3 | 0.4112 ± 0.0044 | 0.4163 ± 0.0050 | 0.7854 ± 0.0375 | 0.2646 ± 0.0033 | 0.791 ± 0.0032 | 0.4252 ± 0.0113 | 0.2699 ± 0.0076 |
| MTFT-Picnic | 3 | 0.4080 ± 0.0037 | 0.4126 ± 0.0041 | 0.7755 ± 0.0370 | 0.2620 ± 0.0020 | 0.794 ± 0.0039 | 0.4244 ± 0.0144 | 0.2692 ± 0.0091 |
| TFT + category analogues | 3 | 0.4108 ± 0.0046 | 0.4165 ± 0.0063 | 0.7860 ± 0.0383 | 0.2652 ± 0.0034 | 0.793 ± 0.0061 | 0.4233 ± 0.0110 | 0.2690 ± 0.0071 |
| MTFT-Picnic + RA | 3 | 0.4082 ± 0.0041 | 0.4151 ± 0.0058 | 0.7767 ± 0.0391 | 0.2636 ± 0.0034 | 0.793 ± 0.0028 | 0.4260 ± 0.0141 | 0.2706 ± 0.0090 |
| MTFT-Picnic + siblings | 3 | 0.4054 ± 0.0036 | 0.4088 ± 0.0043 | 0.7698 ± 0.0340 | 0.2593 ± 0.0024 | 0.794 ± 0.0022 | 0.4163 ± 0.0070 | 0.2636 ± 0.0045 |
| MTFT+ v2 (ours) | 3 | 0.4082 ± 0.0036 | 0.4113 ± 0.0046 | 0.7745 ± 0.0351 | 0.2612 ± 0.0025 | 0.792 ± 0.0032 | 0.4208 ± 0.0090 | 0.2670 ± 0.0059 |
| MTFT+ v3 (ours) | 3 | 0.4059 ± 0.0035 | 0.4072 ± 0.0047 | 0.7689 ± 0.0352 | 0.2582 ± 0.0026 | 0.795 ± 0.0039 | 0.4145 ± 0.0113 | 0.2624 ± 0.0070 |
| ours w/o cross-attention | 1 | 0.4099 | 0.4054 | 0.7984 | 0.2572 | 0.792 | 0.4192 | 0.2647 |

### Paired article-cluster bootstrap: ours − comparator (negative = ours better), 95 % CI

| Comparator | seeds | ΔWAPE all | ΔwQL all | ΔWAPE cold-start | ΔwQL cold-start |
|---|---|---|---|---|---|
| TFT | 3 | **-0.0090 [-0.0098, -0.0084]** | **-0.0064 [-0.0070, -0.0058]** | **-0.0107 [-0.0131, -0.0084]** | **-0.0074 [-0.0092, -0.0058]** |
| MTFT-Picnic | 3 | **-0.0055 [-0.0064, -0.0047]** | **-0.0039 [-0.0047, -0.0032]** | **-0.0097 [-0.0135, -0.0063]** | **-0.0066 [-0.0091, -0.0044]** |
| TFT + category analogues | 3 | **-0.0090 [-0.0099, -0.0082]** | **-0.0069 [-0.0076, -0.0063]** | **-0.0089 [-0.0115, -0.0062]** | **-0.0065 [-0.0081, -0.0051]** |
| MTFT-Picnic + RA | 3 | **-0.0086 [-0.0103, -0.0070]** | **-0.0060 [-0.0073, -0.0048]** | **-0.0121 [-0.0195, -0.0064]** | **-0.0085 [-0.0143, -0.0041]** |
| MTFT-Picnic + siblings | 3 | **-0.0016 [-0.0022, -0.0009]** | **-0.0011 [-0.0017, -0.0007]** | -0.0022 [-0.0048, +0.0002] | -0.0014 [-0.0032, +0.0001] |
| MTFT+ v2 (ours) | 3 | **-0.0041 [-0.0046, -0.0034]** | **-0.0031 [-0.0035, -0.0027]** | **-0.0065 [-0.0093, -0.0039]** | **-0.0046 [-0.0066, -0.0029]** |
| ours w/o cross-attention | 1 | **-0.0023 [-0.0035, -0.0009]** | **-0.0015 [-0.0022, -0.0007]** | **-0.0061 [-0.0110, -0.0017]** | **-0.0038 [-0.0068, -0.0011]** |

Bold = 95 % interval excludes zero.

### Hierarchy: WAPE / wQL / Cov80 by level (MinT + conformal unless stated)

| Model | Method | national | fulfilment centre | article-national | article × FC |
|---|---|---|---|---|---|
| TFT | MinT + conformal | 0.0314 / 0.0187 / 0.86 | 0.0369 / 0.0223 / 0.83 | 0.2374 / 0.1645 / 0.79 | 0.4470 / 0.2844 / 0.77 |
| MTFT-Picnic | MinT + conformal | 0.0327 / 0.0192 / 0.85 | 0.0376 / 0.0226 / 0.82 | 0.2324 / 0.1611 / 0.80 | 0.4398 / 0.2793 / 0.77 |
| TFT + category analogues | MinT + conformal | 0.0313 / 0.0186 / 0.86 | 0.0366 / 0.0222 / 0.84 | 0.2374 / 0.1645 / 0.79 | 0.4467 / 0.2844 / 0.77 |
| MTFT-Picnic + RA | MinT + conformal | 0.0332 / 0.0193 / 0.87 | 0.0381 / 0.0227 / 0.82 | 0.2332 / 0.1617 / 0.80 | 0.4406 / 0.2804 / 0.77 |
| MTFT-Picnic + siblings | MinT + conformal | 0.0316 / 0.0190 / 0.83 | 0.0366 / 0.0224 / 0.82 | 0.2315 / 0.1608 / 0.80 | 0.4345 / 0.2755 / 0.78 |
| MTFT+ v2 (ours) | MinT + conformal | 0.0306 / 0.0183 / 0.86 | 0.0362 / 0.0219 / 0.84 | 0.2322 / 0.1611 / 0.80 | 0.4395 / 0.2792 / 0.77 |
| MTFT+ v3 (ours) | bottom-up | 0.0492 / 0.0323 / 0.51 | 0.0516 / 0.0331 / 0.68 | 0.2071 / 0.1371 / 0.78 | 0.4072 / 0.2582 / 0.79 |
| MTFT+ v3 (ours) | base | 0.0535 / 0.0362 / 0.46 | 0.0626 / 0.0412 / 0.53 | 0.2773 / 0.1972 / 0.71 | 0.4072 / 0.2582 / 0.80 |
| MTFT+ v3 (ours) | MinT | 0.0316 / 0.0191 / 0.79 | 0.0367 / 0.0230 / 0.82 | 0.2311 / 0.1598 / 0.76 | 0.4344 / 0.2759 / 0.76 |
| MTFT+ v3 (ours) | MinT + conformal | 0.0316 / 0.0189 / 0.84 | 0.0367 / 0.0224 / 0.82 | 0.2311 / 0.1604 / 0.80 | 0.4344 / 0.2753 / 0.77 |
| ours w/o cross-attention | MinT + conformal | 0.0374 / 0.0211 / 0.71 | 0.0431 / 0.0260 / 0.70 | 0.2307 / 0.1592 / 0.78 | 0.4306 / 0.2732 / 0.77 |

MinT λ̂ range 0.917–0.961; max coherence error 7.1e-10.

### Modality attribution (ours): image / text / fused

| seed | promo days | non-promo days |
|---|---|---|

<!--RESULTS:END-->

### 3.1 Findings

1. **v3 beats the MTFT-style baseline robustly.** Mean WAPE 0.4072 vs 0.4126, lower on every seed (0.4032 vs 0.4109, 0.4123 vs 0.4173, 0.4063 vs 0.4096). Pooled bootstrap ΔWAPE −0.0055 [−0.0064, −0.0047], ΔwQL −0.0039 [−0.0047, −0.0032]; cold-start ΔWAPE −0.0097 [−0.0135, −0.0063]. It also beats TFT, the category-analogue control, MTFT-Picnic + RA and v2, with every interval excluding zero.
2. **Decomposition: sibling learning is the main driver.** Adding only the sibling channel to MTFT-Picnic moves WAPE 0.4126 → 0.4088 and cold-start 0.4244 → 0.4163, about 70 % and 84 % of v3's total improvement. Sharing information across an article's FCs is the single largest improvement we found to the MTFT design, and it transfers to the published architecture without other changes.
3. **The multimodal stack adds a small but consistent increment.** Against MTFT-Picnic + siblings, v3 wins on 3/3 seeds (Δ −0.0012, −0.0007, −0.0027), pooled ΔWAPE −0.0016 [−0.0022, −0.0009] (−0.4 %). The cold-start increment is not significant ([−0.0048, +0.0002]), and on seed 11 the control was better on cold start (0.4234 vs 0.4264). "Aligned embeddings help cold start beyond siblings" is not established.
4. **Earlier conclusions hold.** Decoder cross-attention was dropped because it gave no gain (v2 ablation), and retrieval alone does not help the unimodal baseline (MTFT-Picnic + RA 0.4151 vs 0.4126). Both the retrieval and the embedding gains are small relative to cross-series information.
5. **Reconciliation trade-off, unchanged.** For v3, MinT-shrink gives national WAPE 0.0316 vs 0.0492 bottom-up and 0.0535 base, FC 0.0367 vs 0.0516, with coherence error ≤ 7e-10. For article-national supplier quotas, bottom-up is more accurate (0.2071 vs 0.2311), and MinT costs bottom-level accuracy (0.4072 → 0.4344). Use MinT for FC and national capacity planning, bottom-up for article-level orders.
6. **Calibration.** MinT + conformal gives 80 %-interval coverage of 0.84 (national), 0.82 (FC), 0.80 (article-national) and 0.77 (bottom), without changing medians.

**Caveats.** The article-cluster bootstrap does not capture training-run variance; the per-seed win counts are reported for that reason, and 3 seeds is a minimum. The generator gives an article's FC series shared promotions and product effects, so sibling learning is expected to help here. Its size on real data is an empirical question (§4). The single-seed ablation rows in the table predate v3 and compare against v2-era configurations.


## How we got here: v1 → v2 → v3

Each version was a hypothesis, tested against the same frozen generator; each failure determined the next change. Full write-ups: `results/v1/README_v1.md`, `results/v2/README_v2.md`.

| | Hypothesis | What happened | Lesson carried forward |
|---|---|---|---|
| **v1** (1 seed) | Aligned SigLIP embeddings + decoder cross-attention beat 10-d static unimodal features | Tie overall (WAPE 0.4164 vs 0.4119 for MTFT-Picnic); **worse on cold start** (0.4469 vs 0.4251); no promotion-dependent attention emerged. MinT halved national error vs bottom-up (0.078 → 0.038), but national 80 % intervals covered only 46–59 % | 768-d embeddings overfit ~900 products; the 10-d bottleneck acts as a regulariser. Summing bottom *medians* biases totals low, so reconcile means. Intervals need calibration |
| **v2** (3 seeds) | Regularise the tokenizer, add retrieval of demand analogues, calibrate intervals | Best on average (0.4113 vs 0.4126) and v1's cold-start regression fixed, but the edge over MTFT-Picnic was **not seed-robust**: v2 lost on seed 23. Retrieval alone *hurt* the baseline; category-only analogues nearly matched it on cold start. **Ablation: removing cross-attention did not hurt** (0.4054 vs 0.4070). Conformal lifted national coverage to 0.86 | The fusion mechanism was not the bottleneck. The largest untapped signal was *across series*: each article's demand in the other FCs |
| **v3** (3 seeds) | Drop cross-attention; add cross-FC sibling learning (Chronos-2-style cross-learning), keeping aligned embeddings, regularisation and retrieval | See §3 | – |

Two methodological choices mattered as much as any model change. First, the **frozen generator**: nothing was tuned to favour a method. Second, the **controls** (category analogues, Picnic + RA, Picnic + siblings) and the **multi-seed rule**: v2's seed-7 result looked like a significant win and disappeared on seed 23.


## 4. Running on real data

```bash
# my_data/: sales.csv, catalog.csv (text, image_path), locations.csv [, weather.csv, holidays.csv]
python examples/prepare_panel.py --data-dir my_data --out panel.pkl                 # SigLIP encoding + cache
python examples/prepare_panel.py --data-dir my_data --freq W-MON --encoder-length 12 --horizon 4 \
       --cold-start-steps 3 --out panel.pkl                                        # weekly data
python benchmark.py --panel panel.pkl --seeds 1,2,3,4,5 --no-quotas --out results_real
python report.py --out results_real
```

Grid settings follow `--freq` (`mtft_plus.panel.freq_defaults`). Daily: season 7 (same-weekday anchor and MASE scale), 28-step level window, 35-step minimum history. Weekly: season 1 (last-value anchor, non-seasonal MASE), 4-step level window, 5-step minimum history. Override them with `--season / --level-window / --min-history`. Rows are snapped to the start of the grid period that contains them: units are summed, promo and discount averaged, weather averaged, and a period containing a holiday is flagged. Launch dates are snapped the same way. Reconciliation uses a sparse aggregation matrix and never forms W Cᵀ, so memory is O(n_agg² + T·n + nnz(A)).

Supplying `unimodal_image.npy` / `unimodal_text.npy` (e.g. ResNet-152 and DistilBERT features, rows in catalogue order) makes the MTFT-style baseline faithful to separate encoders; without them it compresses the aligned embeddings instead, and any write-up should say so.

**Recommended public benchmark: VISUELLE 2.0** (Skenderi et al., CVPRW 2022; humaticslab.github.io/forecasting/visuelle). It has 5,355 fast-fashion products across 110 shops (2016–2019), with product images, textual tags, shop-level sales, discounts and weather: the ingredients this pipeline needs, and a recognised cold-start task. It is fashion, not grocery; claims should be scoped accordingly.

### 4b. Results on VISUELLE 2.0

Full write-up: [results/visuelle2/README.md](results/visuelle2/README.md). 5 seeds per model; paired product-cluster bootstrap. Full tables are in the write-up.

| | Calendar benchmark (WAPE) | Official 2-1 (WAPE %) | Official 2-10 | Official demand |
|---|---|---|---|---|
| TFT | 0.6547 | 68.07 | 67.61 | 65.00 |
| MTFT-Picnic (ResNet-152 + DistilBERT, 10-d each) | 0.6515 | 68.54 | 67.68 | 64.46 |
| MTFT-Picnic + siblings | 0.6446 | **65.25** | 66.84 | 64.01 |
| MTFT+ v3 (ours) | **0.6434** | 65.73 | **65.66** | **63.07** |
| Chronos-2 zero-shot (best variant) | 1.0252 | 96.11 | 111.09 | 100.57 |

- **Siblings transfer to real data.** MTFT-Picnic + siblings beats MTFT-Picnic on every seed of the calendar benchmark and of SO-fore 2-1 and 2-10 (calendar −0.0069 [−0.0084, −0.0054]). They account for about 85 % of v3's calendar gain over MTFT-Picnic.
- **The aligned multimodal stack is task-dependent.**
  - Calendar benchmark: null (−0.0012 [−0.0031, +0.0006], 2/5 seeds).
  - SO-fore 2-1: it hurts (+0.48 points, 1/5).
  - SO-fore 2-10 and demand, where own history is short or absent: it helps (−1.17 and −0.95 points, 5/5).
- **Reconciliation.** MinT-shrink breaks on 12-week life cycles, and bottom-up is best at every level.
- **Published comparison.** Our Naive/SES re-runs reproduce the published baselines to within 0.1 point. We could not approach the published CrossAttnRNN numbers (23–35 % WAPE); in the released training code the validation loader is built from the test split, and we make no claim about how the published numbers were produced. See the write-up for caveats.

## 5. What can and cannot be claimed

* Can: every component is tested; the §3 effect sizes and per-seed wins hold under this generator with the stated uncertainty; MinT-shrink yields exactly coherent supply quotas at 6k nodes.
* Cannot: that MTFT+ beats Picnic's production system or the published MTFT on Picnic data. That needs the same real data, the authors' baseline or a faithful re-implementation, and multiple seeds.
* The generator encodes mechanisms the models exploit (FC-level co-movement of an article, category structure, image–text concept sharing). A synthetic win is a necessary check, not a sufficient one; §4 is the sufficient one.

## 6. Abstract

Multimodal Temporal Fusion Transformers (MTFT) forecast grocery demand from sales history, context and product images and text, encoding each modality independently into a low-dimensional static covariate. We ask which extensions of this design improve accuracy, using an equal-compute benchmark with a frozen data generator (5 fulfilment centres × 1,000 articles), three data seeds, targeted controls and paired bootstrap inference. Over three iterations, richer fusion alone (SigLIP embeddings with decoder cross-attention) failed to beat the compact MTFT design and degraded cold-start accuracy; regularisation and retrieval of demand analogues removed the degradation but gave no seed-robust gain; and an ablation showed cross-attention contributed nothing. The final model, MTFT+ v3, combines regularised SigLIP product embeddings and embedding-space analogue retrieval with cross-fulfilment-centre sibling learning: each series observes its article's demand in the other centres, an adaptation of the cross-learning principle behind Chronos-2. MTFT+ v3 improves WAPE by 1.3 % overall and 2.3 % on cold-start SKUs over the MTFT design, winning on every seed. A control that adds only the sibling channel to the MTFT design recovers about 70 % of the overall and 84 % of the cold-start gain, identifying cross-series learning as the dominant factor; the multimodal components add a further 0.4 %, consistent across seeds but not significant on cold start. A low-rank MinT-shrink reconciliation produces exactly coherent supply quotas across 6,006 series, halving national error relative to bottom-up aggregation, with conformal calibration bringing interval coverage to 0.80–0.84 at aggregate levels. We release the pipeline, tests and a real-data adapter for evaluation on public multimodal retail data such as VISUELLE 2.0.

## 7. PR pitch

**MTFT+: cross-FC sibling learning, aligned product embeddings and MinT-coherent supply quotas.** Three drop-in, config-gated changes to an MTFT-style forecaster, each justified by a controlled benchmark:

1. **Sibling channels** (≈ 1 % WAPE, ≈ 2 % cold start on its own, every seed). The article's demand in the other FCs as two extra inputs; works with the existing architecture.
2. **Aligned SigLIP embeddings + analogue retrieval** (a further 0.4 %, every seed; not significant on cold start). Optional, behind `fusion="static_pooled"`.
3. **MinT-shrink + conformal calibration** for FC and national plans that add up exactly, with calibrated intervals. Bottom-up remains the recommended source for article-level supplier orders.

Additive and reversible. Defaults reproduce the current model, 15 tests cover leakage, causality and reconciliation, and `python benchmark.py && python report.py` reproduces every number. The next step is validation on real data before any default changes.


## Citation

Sukel, M., Rudinac, S., & Worring, M. (2024). Multimodal Temporal Fusion Transformers are Good Product Demand Forecasters. *IEEE MultiMedia*, 31(2), 48–60. doi:10.1109/MMUL.2024.3373827

Skenderi, G., Joppi, C., Denitto, M., Scarpa, B., & Cristani, M. (2022). The multi-modal universe of fast-fashion: the Visuelle 2.0 benchmark. *CVPR Workshops*.

Ansari, A. F., et al. (2025). Chronos-2: From Univariate to Universal Forecasting. arXiv:2510.15821.

Wickramasuriya, S. L., Athanasopoulos, G., & Hyndman, R. J. (2019). Optimal forecast reconciliation for hierarchical and grouped time series through trace minimization. *JASA*, 114(526), 804–818.
