# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**Fraud Pattern Mining** — discover interpretable fraud patterns from 1.85M credit card transactions. The goal is not pure prediction but finding explainable rules that fraud analysts can act on. Three-phase pipeline: EDA → Association Rule Mining → ML models.

All executable code lives in `Fraud-Detection/notebooks/`. Source utilities are planned but not yet extracted to `Fraud-Detection/src/`.

## Setup

```bash
cd Fraud-Detection
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
jupyter notebook
```

Run notebooks in order: `phase1_eda.ipynb` → `phase2_pattern_mining.ipynb` → `phase3_detection_model.ipynb`.

Raw data (`fraudTrain.csv`, `fraudTest.csv`) goes in `Fraud-Detection/data/raw/`. These files are gitignored.

## Architecture

The pipeline flows strictly in sequence — each phase produces outputs consumed by the next:

**Phase 1 (`phase1_eda.ipynb`)** loads raw CSVs and engineers 8 features (Haversine distance, temporal buckets, age). Key finding: `distance_km` and `city_pop` are non-discriminative (AUC ≈ 0.5) and should be excluded from rule mining. Outputs 30+ figures to `outputs/figures/`.

**Phase 2 (`phase2_pattern_mining.ipynb`)** mines fraud patterns using FP-Growth on the fraud-only transaction set. Compares three baselines: simple threshold (`amt > $500`), Isolation Forest, and association rules. Adds three flag columns to the dataset: `flag_threshold`, `flag_iso_forest`, `flag_rules`. Saves `data/processed/transactions_phase2.csv`.

**Phase 3 (`phase3_detection_model.ipynb`)** trains four models on `transactions_phase2.csv`: RF Base, XGBoost Base, RF Rule-Augmented (adds Phase 2 flags as features), XGBoost Augmented. Uses 5-fold Stratified CV with `cross_val_predict` to avoid data leakage on the full 1.85M rows. Best model: XGBoost Base (AUC-PR: 0.894, F1: 0.781).

## Critical constraints

**Class imbalance (1:190 ratio):** Never use accuracy as a metric — use AUC-PR and F1. Handle imbalance via `class_weight='balanced'` (RF) or `scale_pos_weight=190` (XGBoost). SMOTE is not needed here.

**FP-Growth on full dataset is slow:** When testing changes, use a stratified sample first, then run on full data. Support thresholds must be very low (0.01%–0.1%) because the fraud base rate is only 0.52%.

**Threshold tuning:** Default 0.5 decision threshold is suboptimal for imbalanced data. Always sweep thresholds 0.1→0.9 to find the F1-maximizing cutoff.

**Reproducibility:** All notebooks must set `random_state` / `random_seed` before any stochastic operation.

## Key discriminative features

| Feature | AUC | Notes |
|---|---|---|
| `amt` | 0.834 | Fraud median $390 vs non-fraud $47; `very_high >$500` = 21.6% fraud rate |
| `is_night` (22–6h) | 0.782 | Night fraud rate 15× baseline |
| `category` | — | `shopping_net` (1.59%), `misc_net` (1.30%), `grocery_pos` (1.26%) |
| `distance_km` | 0.501 | Non-discriminative — exclude from rule mining |
| `city_pop` | 0.508 | Non-discriminative — exclude from rule mining |

Top association rule: `amt>$500 AND shopping_net AND night` → 63.7% fraud confidence, 122× lift.
