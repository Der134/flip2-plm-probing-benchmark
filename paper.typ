#set page(
  paper: "a4",
  margin: (x: 15mm, y: 17mm),
  columns: 2,
  numbering: "1",
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 8pt, fill: rgb("#555555"), font: "Calibri")
      #grid(
        columns: (1fr, 1fr),
        align(left)[ESM-2 Scaling, Uncertainty, and Fine-Tuning Inversion on FLIP2],
        align(right)[Technical Report (v1.0)],
      )
      #v(-4pt)
      #line(length: 100%, stroke: 0.4pt + rgb("#cccccc"))
    ]
  },
)

#set text(font: ("Cambria", "Georgia", "Times New Roman"), size: 9.2pt, fill: rgb("#1a1a1a"))
#set par(justify: true, leading: 0.56em, first-line-indent: 0pt, spacing: 0.65em)
#set heading(numbering: "1.1")

#show heading.where(level: 1): it => {
  v(0.7em, weak: true)
  set text(font: "Calibri", size: 10.8pt, weight: "bold", fill: rgb("#0f2b46"))
  block(below: 0.4em, sticky: true)[#it]
}

#show heading.where(level: 2): it => {
  v(0.5em, weak: true)
  set text(font: "Calibri", size: 9.6pt, weight: "bold", fill: rgb("#1f4e79"))
  block(below: 0.3em, sticky: true)[#it]
}

#show figure.caption: it => {
  set text(size: 8pt, fill: rgb("#333333"), font: "Calibri")
  it
}

#show table: set text(size: 7.5pt, font: "Calibri")
#show table: set par(justify: false)
#show raw: set text(font: "Consolas", size: 8pt)

#let accent = rgb("#0f2b46")
#let lightbg = rgb("#f4f7fa")

// Full-width Title & Abstract Block
#place(
  top + center,
  scope: "parent",
  float: true,
  [
    #align(center)[
      #text(font: "Calibri", size: 15.5pt, weight: "bold", fill: accent)[
        Scaling, Data Efficiency, Uncertainty, and Fine-Tuning Inversion of ESM-2 Representations on FLIP2 Distribution Shifts
      ]
      #v(0.45em)
      #text(font: "Calibri", size: 9.5pt, fill: rgb("#333333"))[
        *Der134* #h(1.2em) | #h(1.2em) Code & Data: #link("https://github.com/Der134/flip2-plm-probing-benchmark")[`github.com/Der134/flip2-plm-probing-benchmark`] #h(1.2em) | #h(1.2em) November 2026
      ]
    ]
    #v(0.55em)
    #block(
      width: 100%,
      fill: lightbg,
      stroke: (left: 2.5pt + accent),
      inset: (x: 10pt, y: 8pt),
      radius: 2pt,
    )[
      #text(font: "Calibri", size: 8.8pt, weight: "bold", fill: accent)[Abstract — ]
      #text(size: 8.6pt)[
        Protein engineering tasks are defined by severe distribution shifts—extrapolating from local mutant neighborhoods to distant combinatorial variants, higher-order epistasis, unseen mutated positions, or distinct homologous scaffolds. Using *FLIP2* @didi2026flip2, we conduct a controlled, protocol-faithful evaluation of *frozen ESM-2 representations across five scales (`8M`–`3B`)* against sparse one-hot baselines and *end-to-end ESM-2 `3B` LoRA fine-tuning* across *6 datasets and 13 official splits* (`317,976` unique sequences, zero subsampling).
        Our results establish four practical findings:
        *(1) Extrapolation vs. Interpolation Bifurcation:* Under combinatorial distance (`amylase/close_to_far`), higher-order epistasis (`hydro/three_to_many`), and cross-homolog transfer (`hydro/to_P0*`), one-hot Ridge collapses (`NaN`) or turns negatively correlated ($rho in [-0.20, -0.14]$), whereas frozen ESM-2 `650M`/`3B` scales up to $rho = 0.54$–$0.70$ ($+0.26$–$+0.36$ cross-homolog) and saves $10 times$ labels on extrapolation splits.
        *(2) Normalization Controls Scaling Inversion:* Per-feature z-scoring (`zscore_fit`) on 2,560-D `3B` embeddings induces a spurious *inverse scaling law* on `trpb/two_to_many` (`8M=0.421 > 3B=0.336`) by amplifying low-variance noise; row-wise $ell_2$ normalization restores `3B` to `#1` (`0.443`, MSE `1.310 -> 0.888`).
        *(3) Selective Prediction & Epistemic Blind Spots:* A 5#sym.times MLP Deep Ensemble reduces RMSE by up to $-65.6\%$ at $50\%$ coverage on `trpb/by_position` ($"std" tilde.op |"err"| thin rho = 0.625$), while exposing an epistemic blind spot under cross-homolog mean offset (`hydro/to_P06241`).
        *(4) Fine-Tuning Inversion:* End-to-end `3B` LoRA (3 seeds) improves in-distribution `amylase/one_to_many` (`0.697 -> 0.726`) *but degrades every tested OOD split by $-0.18$ to $-0.30$ Spearman* because within-train validation (`val_mse`) cannot detect OOD manifold distortion.
        Finally, we present *ScienceBench-Bio50*, a companion 50-task audit of `google-deepmind/science-skills` (`v1.4.0`), raising live multi-database pass rate from *12% (6/50) to 90% (45/50)* and filing two patch-backed upstream issues (`#15`, `#16`).
      ]
    ]
    #v(0.5em)
  ]
)

= Introduction

Pretrained protein language models (pLMs) such as ESM-2 @lin2023esm2 are routinely deployed either as *frozen residue-mean feature extractors* paired with lightweight linear/MLP probes or via *parameter-efficient fine-tuning* such as LoRA @hu2022lora. While frozen probing is orders of magnitude cheaper, practitioners often assume that end-to-end LoRA fine-tuning is strictly superior—and that standard per-feature z-score standardization is universally beneficial.

Using *FLIP2* @didi2026flip2, whose official splits explicitly separate local mutational neighborhoods from real protein-engineering distribution shifts, we test four concrete hypotheses across the full ESM-2 ladder (`8M`, `35M`, `150M`, `650M`, `3B`):
- *H1 (Scaling & Data Efficiency):* Do larger frozen pLMs deliver disproportionate gains specifically on extrapolation and cross-homolog splits, and how much labeling budget do they save?
- *H2 (Normalization Sensitivity):* Does embedding preprocessing (`raw` vs. `l2_norm` vs. `zscore_fit`) alter model scaling rankings across single-WT scans versus multi-homolog datasets?
- *H3 (Uncertainty Quantification):* Can a lightweight 5#sym.times MLP Deep Ensemble @lakshminarayanan2017deep reliably rank prediction errors (`error-coverage`), and when does ensemble variance fail?
- *H4 (Frozen vs. LoRA):* Under FLIP2's realistic protocol—where validation data comes from the training distribution—does `3B` LoRA improve or degrade out-of-distribution (OOD) generalization?

In addition, to prepare our next-phase clinical variant benchmark (ClinVar VUS + AlphaGenome), we evaluate the reliability of existing scientific agent wrappers in `google-deepmind/science-skills` (`v1.4.0`) on 50 live cross-database queries (*ScienceBench-Bio50*).

= FLIP2 Benchmark Protocol & Dataset Audit

We strictly follow the verified `J-SNACKKB/FLIP` specification @dallago2021flip @didi2026flip2 across all 6 datasets and 13 splits (`317,976` unique sequences; @tab-datasets):
1. *Partitions:* `fit = (set == "train") & (~validation)`; `val = (set == "train") & validation` (strictly hyperparameter selection / early stopping); `test = (set == "test")` (evaluated once).
2. *Target Scaling:* For regression, `StandardScaler` is fit on *all* `set == "train"` rows (`fit | val`). Official NDCG is evaluated via `ndcg_score((y_test - y_test.min())[None, :], pred[None, :])`.
3. *Collapse Detection:* Predictions with `std(pred) < 1e-4` are flagged `pred_constant = True` and reported as `const` (`NaN` Spearman) to prevent iterative solver noise from producing spurious correlations.

#figure(
  placement: top,
  scope: "parent",
  table(
    columns: (0.8fr, 2.1fr, 0.65fr, 0.65fr, 1.5fr),
    stroke: (x: none, y: 0.4pt + rgb("#cccccc")),
    inset: 4pt,
    fill: (_, row) => if row == 0 { rgb("#e8eef5") } else if calc.odd(row) { rgb("#fafbfd") } else { white },
    table.header(
      [*Dataset*], [*Evaluated Splits (`fit / val / test`)*], [*Unique*], [*Len (aa)*], [*Task & Audit Notes*]
    ),
    [`amylase`], [`one_to_many` (`2,574/644/488`), `close_to_far` (`1,426/356/1,924`)], [3,706], [425], [Regression (`zscore_fit`); local vs. distance shift],
    [`rhomax`], [`by_wild_type` (`584/116/184`)], [878], [227–365], [Regression (`l2_norm`#super[\*]); 884 rows / 878 unique],
    [`ired`], [`two_to_many` (`3,746/662/4,178`)], [8,586], [290], [Regression (`zscore_fit`); CSV column order differs],
    [`hydro`], [`three_to_many`, `low_to_high`, `to_P06241`, `to_P0A9X9`, `to_P01053`], [24,935], [57–65], [Regression (`l2_norm`); multi-homolog & cross-protein],
    [`nucb`], [`two_to_many` (`2,088/5,367/48,304`)], [55,759], [142], [4-bin Ordinal Cls; *`bin=3` absent from `fit`*],
    [`trpb`], [`one_to_many` (`304`), `two_to_many` (`8,633`), `by_position` (`31,557`)], [228,298], [389], [Regression (`zscore_fit`); full 228k cached up to `3B`],
  ),
  caption: [FLIP2 datasets audited and evaluated without subsampling. #super[\*]All three scalings (`raw`, `l2_norm`, `zscore_fit`) are ablated across every split.],
) <tab-datasets>

= Experimental Setup & Scalable Pipeline

All notebooks are reproducible on Google Colab (CPU, L4 24 GB, G4 96 GB Blackwell) with SHA-256 keyed atomic `.npz` embedding caches:
- *Frozen Embeddings (`notebook_02`):* Mean pooling over residue tokens excluding BOS, EOS, and PAD in `bfloat16` with PyTorch SDPA and longest-first token-budget batching. All `317,976` unique sequences—including all `228,298` `trpb` sequences up to `esm2_3b`—were extracted completely.
- *Exact Ridge & Data Efficiency (`notebook_03`):* Sparse CSR one-hot encoding (`20 * max_len`) uses `lsqr` on `trpb` (`>150M` entries) and exact dense solves elsewhere. For PLM embeddings, a single primal ($N > D$) or dual ($D >= N$) `np.linalg.eigh` eigendecomposition solves all 21 regularization values $alpha in [10^(-4), 10^6]$ simultaneously by `val_mse`, with zero-copy z-score weight folding for 3-seed #sym.times 6-fraction data efficiency curves.
- *5#sym.times Deep Ensemble UQ (`notebook_03`):* Five 2-layer MLPs (`H -> 256 -> ReLU -> Dropout(0.2) -> 1`) trained with seeds `[0..4]` and `val_mse` early stopping on `650M` and `3B`. Selective prediction evaluates test RMSE at retained fractions `{1.0, 0.9, 0.8, 0.7, 0.6, 0.5}` sorted by ascending ensemble `std`.
- *ESM-2 `3B` LoRA (`notebook_04`):* Native PyTorch LoRA ($r=8, alpha=16, "dropout"=0.05$) injected into `query`, `key`, `value` projections across all 36 layers (`0.15%` trainable backbone params) plus an identical 2-layer MLP head (`2560 -> 256 -> 1`), trained in `bfloat16` with `AdamW` ($"lr"_"LoRA"=2 times 10^(-4)$, $"lr"_"head"=10^(-3)$, cosine decay, 3 seeds `[42, 43, 44]`) and `val_mse` checkpoint selection.

= Results & Analysis

== Frozen-Embedding Scaling vs. One-Hot

#figure(
  placement: top,
  scope: "parent",
  image("paper_figures/fig1_model_scale_probing.png", width: 97%),
  caption: [Frozen-embedding Ridge probing across all 12 FLIP2 regression splits (`eval_subset=full`).],
) <fig-scale>

#figure(
  placement: top,
  scope: "parent",
  table(
    columns: (1.5fr, 0.65fr, 0.6fr, 0.6fr, 0.6fr, 0.65fr, 0.75fr, 1.35fr),
    stroke: (x: none, y: 0.4pt + rgb("#cccccc")),
    inset: 3.5pt,
    fill: (_, row) => if row == 0 { rgb("#e8eef5") } else if calc.odd(row) { rgb("#fafbfd") } else { white },
    table.header(
      [*Dataset / Split*], [*One-Hot*], [*8M*], [*35M*], [*150M*], [*650M*], [*3B (Pri.)*], [*Best Any-Scaling ($rho$)*]
    ),
    [`amylase / one_to_many`], [*0.685*], [0.426], [0.435], [0.576], [0.610], [0.656], [`one_hot` / `l2` (*0.691*)],
    [`amylase / close_to_far`], [`const`], [0.119], [0.011], [0.239], [0.438], [*0.538*], [`esm2_3b` / `zscore` (*0.538*)],
    [`rhomax / by_wild_type`], [*0.259*], [`const`], [`const`], [`const`], [`const`], [0.257], [`esm2_650m` / `raw` (*0.327*)],
    [`ired / two_to_many`], [0.193], [0.150], [0.142], [0.188], [0.186], [*0.194*], [`esm2_3b` / `l2` (*0.196*)],
    [`hydro / three_to_many`], [0.179], [0.615], [0.534], [0.685], [0.571], [*0.703*], [`esm2_3b` / `raw` (*0.705*)],
    [`hydro / low_to_high`], [0.146], [0.240], [0.297], [0.371], [0.396], [*0.404*], [`esm2_3b` / `l2` (*0.404*)],
    [`hydro / to_P06241`], [-0.140], [0.100], [0.059], [0.284], [*0.321*], [0.004], [`esm2_650m` / `l2` (*0.321*)],
    [`hydro / to_P0A9X9`], [-0.176], [-0.082], [0.056], [-0.047], [0.307], [*0.362*], [`esm2_3b` / `l2` (*0.362*)],
    [`hydro / to_P01053`], [-0.204], [-0.168], [-0.019], [-0.198], [*0.265*], [0.167], [`esm2_650m` / `raw` (*0.274*)],
    [`trpb / one_to_many`], [0.455], [0.367], [0.448], [0.437], [0.462], [*0.462*], [`esm2_3b` / `zscore` (*0.462*)],
    [`trpb / two_to_many`], [0.403], [*0.421*], [0.415], [0.384], [0.390], [0.336], [`esm2_3b` / `l2` (*0.443*)],
    [`trpb / by_position`], [0.250], [*0.299*], [0.266], [0.273], [0.272], [0.250], [`esm2_8m` / `zscore` (*0.299*)],
  ),
  caption: [Test Spearman $rho$ across all 12 FLIP2 regression splits (`val_mse`-selected Ridge $alpha$).],
) <tab-ridge>

As shown in @fig-scale and @tab-ridge, representation value tracks *distribution-shift severity*. When mutations cluster around a single aligned wild-type (`amylase/one_to_many`, `trpb/one_to_many`), one-hot indicators remain highly competitive (`0.685`, `0.455`). Once test sequences move to distant combinatorial regimes (`amylase/close_to_far`, `hydro/three_to_many`) or unaligned homologs (`hydro/to_P0*`), one-hot collapses to a constant or yields negative correlation ($-0.14$ to $-0.20$), whereas frozen `650M`/`3B` embeddings reach $rho = 0.538$–$0.703$ within-family and $+0.274$–$+0.362$ across homologs.

== Normalization Ablation: Fixing 3B Scaling Inversion

#figure(
  placement: top,
  scope: "parent",
  image("paper_figures/fig6_scaling_ablation_mechanism.png", width: 97%),
  caption: [Three-way normalization ablation (`raw`, `l2_norm`, `zscore_fit`) diagnosing inverse scaling under narrow training support.],
) <fig-scaling>

Our full 3-way ablation (`ridge_probing_ablation.csv`, @fig-scaling) reveals that *per-feature z-scoring (`zscore_fit`) can invert pLM scaling laws*:
- On `trpb/two_to_many` (`8,633` shallow mutants evaluated on `217,507` higher-order variants), `zscore_fit` yields `8M (0.421) > 35M (0.415) > 150M (0.384) > 650M (0.390) > 3B (0.336)`—and `zscore_fit` 5#sym.times MLP drops `3B` to `0.310`.
- Switching to *row-wise `l2_norm`* restores monotonic scaling and raises *`esm2_3b` to `#1` overall ($rho = 0.443$, test MSE `1.310 -> 0.888`)*.
- *Mechanism:* In 2,560-D `3B` embeddings fit on narrow mutant neighborhoods, hundreds of dimensions vary minimally across `fit`. Dividing by near-zero training standard deviations inflates OOD perturbations into dominant variance; `l2_norm` preserves cosine geometry without amplifying stale dimensions. Similarly, on small variable-length `rhomax/by_wild_type` ($n_"fit"=584$), `raw` `650M` achieves $rho=0.327$ whereas `l2_norm` is over-regularized to `alpha=1e6`.

== Data Efficiency Curves

#figure(
  placement: top,
  scope: "parent",
  image("paper_figures/fig2_data_efficiency.png", width: 97%),
  caption: [Data efficiency curves across 3 random seeds and training fractions `{0.1, 0.2, 0.4, 0.6, 0.8, 1.0}`.],
) <fig-de>

Across 3 seeds (@fig-de), larger frozen pLMs deliver dramatic label efficiency specifically under distribution shift:
- On `amylase/close_to_far`, `esm2_3b` at *10% labels* ($n=142$, $rho=0.366 plus.minus 0.021$) outperforms `esm2_150m` at *100% labels* ($0.239$).
- On `hydro/three_to_many`, `esm2_3b` at *10% labels* ($n=116$, $rho=0.520 plus.minus 0.050$) nearly matches `esm2_35m` at 100% ($0.534$) and triples `one_hot` at 100% ($0.179$).
- On `hydro/to_P0A9X9`, `esm2_3b` at *10% labels* ($0.306 plus.minus 0.054$) matches `esm2_650m` at 100% ($0.308$).

== High-Activity Discovery on NucB

#figure(
  placement: top,
  scope: "parent",
  image("paper_figures/fig4_nucb_classification.png", width: 90%),
  caption: [NucB (`two_to_many`, $n_"test"=48,304$) where optimal class `bin=3` never appears in `fit`.],
) <fig-nucb>

In `nucb/two_to_many` ($n_"fit"=2,088$, $n_"test"=48,304$), labels are 4 ordinal activity bins `{0,1,2,3}` and `bin=3` is *completely absent from `fit`* (@fig-nucb, @tab-nucb). Frozen `esm2_3b` improves 4-bin Macro-F1 by *+51%* (`0.250 -> 0.377`), 4-bin accuracy (`0.422 -> 0.603` with `l2_norm`), high-activity (`bin >= 2`) AUROC (`0.685 -> 0.732`), and ordinal Spearman (`0.291 -> 0.458`).

#figure(
  table(
    columns: (1.1fr, 0.7fr, 0.75fr, 0.75fr, 0.75fr),
    stroke: (x: none, y: 0.4pt + rgb("#cccccc")),
    inset: 3.2pt,
    fill: (_, row) => if row == 0 { rgb("#e8eef5") } else if calc.odd(row) { rgb("#fafbfd") } else { white },
    table.header([*Model*], [*Acc (4b)*], [*Macro-F1*], [*AUROC (`>=2`)*], [*Ord. $rho$*]),
    [`one_hot`], [0.422], [0.250], [0.685], [0.291],
    [`esm2_8m`], [0.495], [0.332], [0.717], [0.176],
    [`esm2_35m`], [0.550], [0.291], [0.609], [0.148],
    [`esm2_150m`], [0.593], [0.350], [0.707], [0.424],
    [`esm2_650m`], [0.601], [0.366], [0.725], [0.452],
    [`esm2_3b`], [*0.603*], [*0.377*], [*0.732*], [*0.458*],
  ),
  caption: [NucB 4-bin classification & ordinal ranking (best scaling per metric).],
) <tab-nucb>

== Deep Ensemble UQ & Epistemic Blind Spots

#figure(
  placement: top,
  scope: "parent",
  image("paper_figures/fig3_uq_error_coverage.png", width: 96%),
  caption: [5#sym.times Deep Ensemble selective prediction curves across 24 `(split, model)` pairs (`650M`, `3B`).],
) <fig-uq>

Rejecting the 50% highest-uncertainty predictions lowers test RMSE in *21 of 24* cases (@fig-uq):
- *Strongest risk reduction:* `trpb/by_position` drops RMSE by *$-65.6\%$* (`650M`: `2.129 -> 0.732`, $"std" tilde.op |"err"| thin rho = +0.625$, Pearson $+0.768$) and *$-59.1\%$* (`3B`: `2.697 -> 1.103`, $rho=+0.585$); `trpb/two_to_many` drops *$-27.9\%$* / *$-30.7\%$* (`NDCG@10%` `0.939–0.944`); `hydro/three_to_many` drops *$-25.8\%$* (`3B`, $rho=+0.416$).
- *Epistemic OOD Blind Spot:* On `hydro/to_P06241` (`3B`) and `hydro/to_P01053` (`650M`), $"std" tilde.op |"err"|$ turns negative ($-0.16$), increasing RMSE at 50% coverage ($+10.8\%$, $+13.1\%$). When an unseen homolog shifts globally in mean activity, ensemble variance remains deceptively confident.

== Fine-Tuning Inversion: ESM-2 3B LoRA vs. Frozen Probing

#figure(
  placement: top,
  scope: "parent",
  image("paper_figures/fig5_lora_vs_frozen_3b.png", width: 97%),
  caption: [Fine-tuning inversion across 4 representative splits: LoRA helps locally (+0.03) but degrades all 3 OOD splits ($-0.18$ to $-0.30$).],
) <fig-lora>

#figure(
  placement: top,
  scope: "parent",
  table(
    columns: (1.35fr, 0.9fr, 0.6fr, 0.7fr, 0.75fr, 0.7fr, 1.0fr, 0.75fr),
    stroke: (x: none, y: 0.4pt + rgb("#cccccc")),
    inset: 3.5pt,
    fill: (_, row) => if row == 0 { rgb("#e8eef5") } else if calc.odd(row) { rgb("#fafbfd") } else { white },
    table.header(
      [*Split*], [*Regime*], [*1-Hot*], [*Frozen Ridge*], [*Frozen 5#sym.times MLP*], [*LoRA Val $rho$*], [*LoRA Test $rho$ (3s)*], [*$Delta$ vs. Frozen*]
    ),
    [`amylase / one_to_many`], [In-distribution], [0.685], [0.656], [0.697], [0.735], [*0.726 #sym.plus.minus 0.012*], [*+0.029*],
    [`amylase / close_to_far`], [Distance extrap.], [`const`], [0.538], [*0.589*], [0.427], [0.287 #sym.plus.minus 0.156], [*−0.302*],
    [`hydro / three_to_many`], [Epistasis extrap.], [0.179], [*0.703*], [0.680], [0.806], [0.524 #sym.plus.minus 0.013], [*−0.179*],
    [`hydro / to_P0A9X9`], [Cross-homolog], [-0.176], [*0.362*], [0.154], [0.773], [0.169 #sym.plus.minus 0.044], [*−0.193*],
  ),
  caption: [ESM-2 `3B` LoRA (`r=8, alpha=16`, 3 seeds) vs. frozen `3B` baselines. Test MSE on `hydro/three_to_many` drops (`1.045 -> 0.500`), showing LoRA calibrates scale while distorting rank geometry.],
) <tab-lora>

@tab-lora and @fig-lora reveal a sharp *fine-tuning inversion*:
1. On *in-distribution* `amylase/one_to_many`, `val` and `test` share the same neighborhood; LoRA transfers cleanly (`val=0.735 -> test=0.726`) and beats every frozen baseline.
2. On all three *OOD shifts*, FLIP2's within-train `val` split rewards aggressive local adaptation (`val` $rho = 0.806$ on `three_to_many`, `0.773` on `to_P0A9X9`), yet *test Spearman collapses by $-0.28$ to $-0.60$ relative to validation*. Fine-tuning overwrites pretrained evolutionary geometry that frozen linear probing preserves.

= Companion Audit: `ScienceBench-Bio50`

To stress-test scientific agent wrappers before scaling to clinical variant interpretation, we audited all *40 skills / 66 Python scripts* in `google-deepmind/science-skills` (`v1.4.0`) across *50 live multi-database tasks* (`ClinVar`, `gnomAD`, `Ensembl VEP`, `UniProt`, `AlphaFold DB`, `arXiv`, `EuropePMC`, `QuickGO`; @fig-bio50 and @tab-bio50).

#figure(
  placement: top,
  scope: "parent",
  image("paper_figures/fig7_sciencebench_bio50.png", width: 95%),
  caption: [`ScienceBench-Bio50`: Three surgical script fixes + `POLITE_HTTP_USER_AGENT` raise live pass rate from 12% to 90%.],
) <fig-bio50>

#figure(
  table(
    columns: (1.5fr, 0.75fr, 0.75fr),
    stroke: (x: none, y: 0.4pt + rgb("#cccccc")),
    inset: 3.2pt,
    fill: (_, row) => if row == 0 { rgb("#e8eef5") } else if calc.odd(row) { rgb("#fafbfd") } else { white },
    table.header([*Task Family (10 each)*], [*Upstream*], [*Patched*]),
    [`clinvar_coord` (VCV alleles)], [0 / 10], [*10 / 10*],
    [`gnomad_cli` (relative `--output`)], [0 / 10], [*10 / 10*],
    [`cross_db_chain` (ClinVar #sym.arrow VEP)], [0 / 10], [*9 / 10*],
    [`protein_struct` (UniProt #sym.arrow AFDB)], [0 / 10], [*9 / 10*],
    [`lit_and_go` (arXiv / EPMC / GO)], [6 / 10], [*7 / 10*],
    [*Total (50 live tasks)*], [*6 / 50 (12%)*], [*45 / 50 (90%)*],
  ),
  caption: [`ScienceBench-Bio50` pass rates before and after minimal fixes.],
) <tab-bio50>

- *ClinVar Silent Coordinate Corruption (Issue `#15`):* `clinvar_api.py`'s `_extract_allele_info()` broke on the first `<Gene>/<SequenceLocation>` in VCV XML rather than `<SimpleAllele>/<Location>/<SequenceLocation>`, returning 50–126 kb gene spans with `ref=null, alt=null` on *10/10* variants and breaking *10/10* downstream Ensembl VEP chains.
- *Relative `--output` CLI Crash (Issue `#16`):* Five call sites across `gnomad_database` (3 scripts) and `alphagenome_single_variant_analysis` called `os.makedirs(os.path.dirname(output_path), exist_ok=True)`, crashing *10/10* times on the relative filenames documented in `SKILL.md`.
- *arXiv & AlphaFold Friction:* `search_arxiv.py` emitted concatenated invalid JSON when `results > 1` (4/5 failures, open Issue `#13`), and `fetch_structure.py` hit HTTP 403 without `POLITE_HTTP_USER_AGENT` (10/10). Fixing these 3 script bugs + UA raised pass rate from *12% (6/50) to 90% (45/50)*.

= Practical Guidelines & Next Steps

1. *Default Protein Fitness Baseline:* Start with *L2-normalized frozen ESM-2 `650M`/`3B` Ridge* under combinatorial or cross-homolog shifts; reserve LoRA for in-distribution refinement or OOD-validated splits.
2. *Safe Screening with UQ:* Pair 5#sym.times MLP ensemble variance filtering with sequence-novelty/homolog checks to guard against global scaffold offset.
3. *Next Phase:* Extend this leakage-resistant probing, UQ, and patched multi-database verification pipeline to *ClinVar VUS* gene-held-out & temporal splits, benchmarking ESM-2 against *AlphaGenome* regulatory/splicing predictions.

#bibliography("refs.bib", style: "ieee")
