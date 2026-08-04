# Explainable Speech Emotion Recognition

Reference implementation for the paper *"Explainable Speech Emotion
Recognition"*. This repository contains the code used for all
experiments reported in the paper, including the reviewer-response
experiments (E1, E2, E4, E5) added during revision.

The framework quantifies **demographic bias in Speech Emotion
Recognition (SER)** by fitting a **Weighted Attribute Fairness (WAF)**
regression on per-class prediction loss, with support for pairwise
demographic interactions, SSL speech-feature controls, and a suite of
robustness / interpretability analyses (permutation importance,
KernelSHAP, coefficient trajectory, cluster-robust standard errors).

---

## Repository Structure

```
.
├── run_e1.py                Table III: linear WAF (main + interaction + speech)
├── run_e5.py                       E5 pipeline (legacy driver)
├── run_all.py                      End-to-end pipeline: E1 + E5 + IEMOCAP E4
├── train_ser.py                    HuBERT / WavLM SER fine-tuning
├── rerun_e4_waf.py                 IEMOCAP Sex-only WAF (E4)
├── example_script.py               Minimal training + evaluation example
├── smoke_ser.py                    Environment sanity check
├── requirements.txt                Full pinned dependencies
├── requirements_min.txt            Minimal dependencies
├── scripts/
│   └── download_cremad_audio.py    Batch downloader for CREMA-D wav files
└── src/
    ├── utils.py
    ├── databases/
    │   ├── constants.py            CREMA-D label vocabulary & demographic columns
    │   ├── cremad.py               CREMA-D loader
    │   ├── iemocap.py              IEMOCAP 4-class loader with speaker-holdout split
    │   └── synthetic/
    │       ├── constants.py        SEAB attribute mappings (Multi + XOR regimes)
    │       ├── generate_dataset.py Controlled synthetic-bias data generator
    │       ├── mutual_info.pyKraskov kNN MI (sklearn wrapper)
    │       └── run_e2.py           E2 correlation experiment (MI vs SP/EO/FPR/WAF)
    ├── ser_models/
    │   ├── main.py                 SER training entry point
    │   ├── trainer.py              Hugging Face `Trainer` wrapper
    │   ├── speech_processing.py
    │   ├── hubert.py               `HubertForSpeechClassification`
    │   ├── wavlm.py                `WavLMForSpeechClassification`
    │   ├── wav2vec2.py
    │   ├── classifier.py
    │   └── evaluator.py
    └── fairness/
        ├── constants.py
        ├── utils.py
        ├── example/
        │   └── waf_config.yaml     WAF hyperparameters (embedding_k=100,...)
        ├── trad_metrics/           Traditional fairness baselines (SP/EO/FPR)
        │   ├── main.py
        │   └── trad_metrics.py
        ├── waf/                Weighted Attribute Fairness core
        │   ├── config.py
        │   ├── dataset.py          Build WAF dataset (BCE-per-class target)
        │   ├── embeddings.py       Per-sample SSL embedding cache (.npy)
        │   ├── pca.py              Top-|PC1|-loading dimension selection
        │   ├── linear_waf.py       statsmodels OLS + Wald + BH + cluster-robust SE
        │   └── run_linear_waf.py   CLI for Table III
        └── analysis/               E5 interpretability analyses
            ├── k_trajectory.py     Coefficient trajectory vs number of speech dims
            └── shap_and_permutation.py
                                KernelSHAP + permutation importance
```

---

## Method Overview

* **WAF regression.** One OLS per emotion class \(c\). Target
  \(Y_{i,c} = \mathrm{BCE}(s_{i,c},\mathbb{1}[y_i=c])\) is the per-sample
  per-class binary cross-entropy loss of the fine-tuned SER model.
  Features: four demographic main effects (`AgeGroup`, `Ethnicity`,
  `Sex`, `Race`, binarised to \(\pm 1\)), six pairwise demographic
  interactions, and 100 speech features obtained by selecting the SSL
  embedding dimensions with the largest absolute PC1 loadings. Speech
  columns are z-normalised before fitting.
* **Statistical inference.** Wald p-values and standard errors are
  derived in closed form from a single OLS fit
  (`statsmodels.OLS.fit()`); no bootstrap is used. Multiple testing is
  corrected within each emotion by the Benjamini–Hochberg FDR
  procedure over the (main + interaction) coefficient block. A
  cluster-robust variant conditions on speaker ID
  (`cov_type='cluster'`).
* **Feature selection for speech control.** PCA is fit on the SSL
  embeddings; we extract the loadings of the first principal component
  and retain the top-\(k\) *original* embedding dimensions with the
  largest absolute contributions. This is a feature-selection
  procedure, not a PCA projection.
* **Synthetic validation (E2).** A controlled generator produces
  1600 samples under two regimes: an additive **Multi** regime (each
  emotion is biased toward one specific attribute) and a non-additive
  **XOR** regime (each emotion is triggered when exactly one of a
  specific attribute pair is privileged). Mutual information between
  each protected attribute and the per-class BCE loss is estimated with
  the Kraskov kNN estimator
  (`sklearn.feature_selection.mutual_info_regression`, \(k=3\),
  `random_state=1`). We report Pearson and Spearman correlations
  between MI and each fairness metric (SP, EO, FPR, WAF) with Fisher's
  \(z\)-transformed confidence intervals.
* **External interpretability validation (E5).** We fit a per-emotion
  ridge surrogate on the WAF feature matrix and compute (i)
  `sklearn.inspection.permutation_importance` with 20 repeats and
  (ii) KernelSHAP with100 background samples, 200 explained samples,
  `nsamples=100`. We also refit the WAF at
  \(k \in \{0,10,25,50,75,100\}\) to trace how demographic coefficients
  evolve as speech control is progressively added, and re-estimate
  cluster-robust confidence intervals on speaker ID for the final
  Table III model.

---

## Installation

```bash
conda create -n splexp python=3.11 -y
conda activate splexp
pip install -r requirements_min.txt
```

Key dependencies:

* `torch`, `transformers`, `datasets`
* `statsmodels` — OLS, Wald tests, cluster-robust standard errors
* `scikit-learn` — Ridge, PCA, permutation importance,
  `mutual_info_regression`
* `shap` — `KernelExplainer`
* `aif360` — SP / EO / FPR baselines

---

## Data

The code expects the following layout at run time (paths relative to
the repository root):

```
db/CREMA-D/CREMA-D audio + demographic csvs
db/IEMOCAP_full_release/             IEMOCAP release directory
```

* CREMA-D wav files can be downloaded via `scripts/download_cremad_audio.py`.
* IEMOCAP requires signing the licence agreement; see the official
  IEMOCAP distribution.

---

## Reproducing the Experiments

### E1 — Table III (linear WAF)

```bash
python run_e1.py --models hubert wavlm
```

Ablations:

```bash
python run_e1.py --models hubert --main_effects_only     # no interactions
python run_e1.py --models hubert --cluster_colActorID# cluster-robust SE
```

### E2 — Synthetic MI vs. metric correlations

```bash
python -m databases.synthetic.run_e2 --output_dir content/e2_synthetic
```

Produces the Multi and XOR regime tables and the correlation summary
used to compare WAF against SP / EO / FPR.

### E4 — IEMOCAP Sex-only WAF (speaker-holdout)

```bash
python train_ser.py --dataset iemocap --models hubert wavlm

python -m fairness.waf.run_linear_waf \
    --model_name hubert --dataset iemocap \
    --ser_model_ckpt content/ser_training_iemocap/models/hubert/final \
    --ser_result_dataset content/ser_training_iemocap/data/hubert \
    --waf_config src/fairness/example/waf_config.yaml \
    --xds_override Sex --cluster_col SpeakerID \
    --table_out content/waf_training/tables/iemocap_hubert_tableIII_sex.csv
```

### E5 — Interpretability suite

Coefficient trajectory (loads the cached WAF dataset produced by
`run_e1.py`, so \(k=100\) exactly reproduces the Table III coefficients):

```bash
python -m fairness.analysis.k_trajectory \
    --model_name hubert \
    --waf_dataset_dir content/waf_training/data/hubert/waf_dataset \
    --out_dir content/e5/hubert/trajectory
```

Permutation importance and KernelSHAP:

```bash
python -m fairness.analysis.shap_and_permutation \
    --model_name hubert \
    --waf_dataset_dir content/waf_training/data/hubert/waf_dataset \
    --out_dir content/e5/hubert/shap_perm
```

Cluster-robust confidence intervals (via the E1 CLI):

```bash
python -m fairness.waf.run_linear_waf \
    --model_name hubert \
    --ser_model_ckpt content/ser_training/models/hubert/final \
    --ser_result_dataset content/ser_training/data/hubert \
    --waf_config src/fairness/example/waf_config.yaml \
    --cluster_col ActorID \
    --table_out content/e5/hubert/hubert_tableIII_cluster.csv
```

### End-to-end pipeline

```bash
python run_all.py
```

Runs E1 on both backbones, then E5, then trains IEMOCAP SER, then
computes the IEMOCAP Sex-only WAF.

---

## Output Files

All results are written under `content/`:

* `content/ser_training/`, `content/ser_training_iemocap/` — SER model
  checkpoints and per-sample prediction score datasets.
* `content/waf_training/tables/{hubert,wavlm}_tableIII.csv` — Table III.
* `content/waf_training/tables/iemocap_{hubert,wavlm}_tableIII_sex.csv`
  — IEMOCAP Sex-only WAF (Table IV, Sex row).
* `content/e2_synthetic/e2_correlation_summary.csv` — Table V (E2).
* `content/e5/{hubert,wavlm}/trajectory/*.csv` — coefficient trajectory
  data and plots (E5(c)).
* `content/e5/{hubert,wavlm}/shap_perm/*.csv` — SHAP, permutation
  importance, and rank-agreement tables (E5(a,b)).
* `content/e5/{hubert,wavlm}/*_tableIII_cluster.csv` — speaker
  cluster-robust confidence intervals (E5(d)).

---

## Licence

Code is released under the MIT Licence. CREMA-D and IEMOCAP are
distributed under their own respective licences; please consult the
original providers before use.
