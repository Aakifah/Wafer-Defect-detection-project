# Wafer Defect Detection Engine

**Production-grade multi-label classification pipeline for integrated circuit quality assurance.**

<div align="center">

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11-EE4C2C.svg)](https://pytorch.org/)
[![MLflow](https://img.shields.io/badge/MLflow-3.12-0194E2.svg)](https://mlflow.org/)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED.svg)](https://onnxruntime.ai/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![CI/CD](https://img.shields.io/badge/GitHub-Actions-2088FF.svg)](https://github.com/features/actions)
[![Feast](https://img.shields.io/badge/Feast-0.47-5C4EE5.svg)](https://feast.dev/)
[![Ruff](https://img.shields.io/badge/Linter-Ruff-261230.svg)](https://docs.astral.sh/ruff/)
[![Mypy](https://img.shields.io/badge/TypeCheck-Mypy_Strict-2D9FD9.svg)](https://mypy-lang.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

## Overview

The Wafer Defect Detection Engine is a Convolutional Neural Network (CNN) architecture designed to detect and classify simultaneous hardware failure topologies in integrated circuit manufacturing. It processes 52×52 single-channel wafer map tensors and performs multi-label classification across eight defect topologies and one normal state.

Key highlights:

- **Progressive optimisation pipeline** — baseline CNN → Focal Loss → Bayesian hyperparameter tuning (Optuna) → ASVD compression → INT8 quantisation 
- **Dual drift detection** — Discrete-Time Markov Chains (DTMC) and Maximum Mean Discrepancy (MMD) continuously monitor production predictions against calibrated baselines
- **Automated retraining triggers** — drift thresholds gate automated model retraining without human intervention
- **Full MLOps observability** — metrics pushed to Prometheus Pushgateway, visualised in Grafana, models tracked in MLflow Registry with Champion/Competitor aliasing
- **Rigorous quality gates** — `mypy` strict mode, `ruff` linting, `pytest` unit/integration/property-based testing via `hypothesis`, all enforced in CI/CD

---

## Repository Structure and Data Flow

```
wafer-defect-detection/
├── data/
│   ├── raw/                        # Immutable source dataset (36,991 samples)
│   ├── processed/                  # Stratified splits (train/val/test — 70/15/15)
│   └── analysis_report.json        # Data quality and distribution report
│
├── src/
│   ├── ETL_data/                   # Cloud-native ETL pipeline
│   │   ├── upload_raw.py           #   Single-part binary upload to S3
│   │   ├── ingestion.py            #   Dedup → multi-label stratified split → S3 upload
│   │   ├── loader.py               #   S3-backed PyTorch Dataset / DataLoader factory
│   │   └── analysis.py             #   Data quality analysis (NaN, duplicates, class distribution)
│   │
│   ├── model/                      # Core model definitions
│   │   ├── architecture.py         #   CompleteCNN — 52×52 multi-label CNN with QuantStub
│   │   ├── loss.py                 #   Focal Loss for imbalanced multi-label classification
│   │   ├── compression.py          #   ASVD — adaptive SVD rank reduction for Linear layers
│   │   └── sssn/                   #   Spatial-Semantic Stacking Network ensemble
│   │       ├── mlsmote.py          #     Custom multi-label SMOTE oversampling
│   │       ├── level1_spatial.py   #     XGBoost ClassifierChain meta-learner
│   │       ├── level1_semantic.py  #     MLP probability calibration meta-learner
│   │       ├── level2_decider.py   #     Ridge classifier final decider
│   │       └── ensemble.py         #     MLflow PyFunc wrapper for unified SSSN pipeline
│   │
│   ├── training/                   # Progressive training pipeline
│   │   ├── utils.py                #   Shared: train_one_epoch, evaluate_model, save_results
│   │   ├── optimizer_benchmark.py  #   AdamW vs SGD+Nesterov benchmark sweep
│   │   ├── train_base.py           #   Baseline CNN with BCE loss
│   │   ├── train_focal.py          #   CNN with Focal Loss
│   │   ├── train_bayesian.py       #   Optuna Bayesian hyperparameter search
│   │   ├── train_asvd.py           #   ASVD-compressed CNN evaluation
│   │   ├── train_quantized.py      #   INT8 post-training static quantisation (PTQ)
│   │   ├── train_rf_standard.py    #   Standard Random Forest baseline
│   │   ├── train_rf_super.py       #   Bayesian-optimised ExtraTrees
│   │   ├── train_sssn.py           #   Spatial-Semantic Stacking Network (SSSN)
│   │   └── log_to_mlflow.py       #    Aggregate and log all metrics to MLflow
│   │
│   ├── inference/                  # Production inference engine
│   │   └── inference.py            #   Quantised ONNX inference + dual drift detection loop
│   │
│   ├── serving/                    # Batch serving and automated retraining
│   │   └── batch_inference.py      #   Champion vs Competitor eval, drift check, retrain trigger
│   │
│   ├── evaluation/                 # Post-training metrics and visualisation
│   │   ├── class_metrics.py        #   Per-class F1 and average precision
│   │   ├── visualize_metrics.py    #   Pipeline stage comparison charts
│   │   └── visualize_rf_comparisons.py  # CNN vs RF vs SSSN per-class mAP comparison
│   │
│   ├── telemetry/                  # Data drift detection
│   │   ├── calibrate_baseline.py   #   Generate DTMC baseline transition matrix
│   │   ├── calibrate_mmd.py       #    Generate MMD baseline penultimate embeddings
│   │   └── model/
│   │       ├── dtmc_drift.py       #   Discrete-Time Markov Chain drift detector
│   │       └── mmd_drift.py        #   Maximum Mean Discrepancy drift detector
│   │
│   ├── monitoring/                 # Prometheus / Grafana observability
│   │   ├── push_telemetry.py       #   Push DTMC / MMD drift metrics to Pushgateway
│   │   └── verify_observability.py #   Prometheus and Pushgateway health checks
│   │
│   └── benchmark/
│       └── matrices/               # Baseline and live drift detection state
│           ├── dtmc_baseline.pkl   #   DTMC reference transition matrix
│           ├── dtmc_live.pkl       #   Live DTMC transition matrix
│           ├── mmd_baseline.pkl    #   MMD reference penultimate embeddings
│           ├── mmd_live.pkl        #   Live MMD embeddings
│           └── telemetry_state_buffer.json  # Drift anomaly state
│
├── evaluation/                     # Saved model artifacts per pipeline stage
│   ├── comparison.csv              # Cross-stage metric comparison table
│   ├── optimizer_benchmark/        # AdamW vs SGD+Nesterov benchmark results
│   ├── base_cnn/                   # Baseline CompleteCNN model + metrics
│   ├── focal_cnn/                  # Focal Loss CNN model + metrics
│   ├── bayesian_cnn/               # Bayesian-optimised CNN model + metrics
│   ├── asvd_cnn/                   # ASVD-compressed CNN model + metrics
│   ├── quantized_cnn/              # INT8-quantised CNN model + metrics
│   ├── rf_standard/                # Standard Random Forest baseline + metrics
│   ├── rf_super/                   # Bayesian-optimised ExtraTrees + metrics
│   └── sssn_competitor/            # Spatial-Semantic Stacking Network (SSSN) + metrics
│
├── feature_store/                  # Feast feature store
│   ├── feature_store.yaml          #   Feast configuration (local, S3 registry, SQLite online store)
│   ├── features.py                 #   Entity and FeatureView definitions
│   ├── metadata_generator.py       #   Parquet metadata generation from S3 datasets
│   └── Dockerfile                  #   Feast service container build
│
├── infrastructure/
│   └── Dockerfile                  # Multi-stage production build (FastAPI + ONNX Runtime)
│
├── images/                         # Generated evaluation visualisations (SVG/PNG)
├── notebooks/
│   └── 01_analysis.ipynb           # Initial EDA and spatial distribution visualisation
│
├── tests/
│   ├── conftest.py                 #   Pytest hooks (disables hardware backends)
│   ├── unit/                       #   Component-level tests
│   │   ├── test_data_processing.py #     Data dedup, split, integrity
│   │   ├── test_loss.py            #     FocalLoss numeric stability
│   │   ├── test_training_scripts.py#     evaluate_model unit test
│   │   ├── test_optimizer_benchmark.py   Optimiser benchmark tests
│   │   ├── test_dtmc_telemetry.py  #     DTMC drift engine tests
│   │   └── test_mmd_telemetry.py   #     MMD drift engine tests
│   ├── integration/
│   │   └── test_dtmc_integration.py#   End-to-end DTMC pipeline validation
│   └── property/
│       └── test_tensor.py          #   Hypothesis property-based tensor dimension test
│
├── run_training_pipeline.py        # Master orchestrator — executes all stages sequentially
├── pyproject.toml                  # Project metadata, dependencies, Ruff / Mypy config
├── uv.lock                         # Deterministic package lock file
├── requirements.txt                # Pinned production dependencies
├── requirements-dev.txt            # Pinned development dependencies
├── .github/workflows/ci.yml          # CI/CD pipeline (4 stages)
└── optuna_study.db                 # Optuna hyperparameter search database
```

### Data Flow Diagram

```
┌────────────┐     ┌───────────────┐     ┌───────────────────┐
│  Raw .npz  │ ──► │  Ingest &     │ ──► │  S3 (MinIO)       │
│  (36,991)  │     │  Stratify     │     │  datasets/        │
└────────────┘     │  (70/15/15)   │     │  ├── raw/         │
                   └───────────────┘     │  └── processed/   │
                                         └──────┬────────────┘
                                                │
                   ┌────────────────────────────┘
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                               │
│                                                                    │
│  Optimiser Benchmark  ──►  RF Baselines  ──►  Bayesian CNN        │
│                                                   │               │
│  Base CNN  ──►  Focal CNN  ──►  ASVD  ──►  INT8 Quantise         │
│                                                   │               │
│  SSSN (competitor)  ◄────────────────────────────┘                │
│                                                                    │
│  MLflow Logging (Champion / Competitor aliasing)                  │
└──────────────────────┬───────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                    TELEMETRY & SERVING                             │
│                                                                    │
│  DTMC Baseline Calibration    MMD Baseline Calibration            │
│  Batch Inference (ONNX)  ──►  Dual Drift Detection               │
│                                                                    │
│  Pushgateway  ──►  Prometheus  ──►  Grafana                      │
│  Drift ──► Auto-Retrain                                          │
└──────────────────────────────────────────────────────────────────┘
```

---

## Codebase Flow

The pipeline is orchestrated by `run_training_pipeline.py`, which executes scripts sequentially via `subprocess`. Each stage reads from previous stage outputs stored in `evaluation/`, enabling incremental, reproducible execution.

### Stage 1 — Data Ingestion (`src/ETL_data/`)

| Script | Role |
|--------|------|
| `upload_raw.py` | Streams `Wafer_Map_Datasets.npz` to S3 as a single-part binary (avoids multipart ETag collisions) |
| `ingestion.py` | Downloads from S3, deduplicates exact (X, y) pairs, performs multi-label stratified split (70/15/15) using `iterative-stratification`, verifies zero cross-contamination, uploads partitions to S3 |
| `analysis.py` | Comprehensive quality audit: NaN/Inf checks, pixel value distributions, per-class frequency, label co-occurrence patterns, spatial statistics |

### Stage 2 — Training Pipeline (`src/training/`)

Executed in order by `run_training_pipeline.py`:

| # | Script | Purpose |
|---|--------|---------|
| 1 | `optimizer_benchmark.py` | Grid-sweeps AdamW vs SGD+Nesterov across learning rates, weight decays, and schedulers (cosine/step). Selects best configuration. |
| 2 | `train_rf_standard.py` | Standard `RandomForestClassifier` baseline (100 estimators). Flattens 52×52 → 2,704 features. |
| 3 | `train_rf_super.py` | Bayesian-optimised `ExtraTreesClassifier` via Optuna. Searches `n_estimators`, `max_depth`, `ccp_alpha` with `class_weight='balanced_subsample'`. |
| 4 | `train_bayesian.py` | Optuna hyperparameter search (35 trials) over learning rate, weight decay, and Focal gamma. Maximises mean Average Precision (mAP). Trains final golden model. |
| 5 | `train_base.py` | Baseline `CompleteCNN` with `BCEWithLogitsLoss`, AdamW, 35 epochs, early stopping. |
| 6 | `train_focal.py` | `CompleteCNN` with `FocalLoss` (gamma=2.0, class-balanced alpha) for minority class prioritisation. |
| 7 | `train_asvd.py` | Loads Bayesian model and applies Adaptive SVD compression (90% cumulative energy retention) to all Linear layers. |
| 8 | `train_quantized.py` | Post-Training Static Quantisation (PTQ) to INT8: fuses Conv+BatchNorm, calibrates activation ranges, converts, evaluates, exports TorchScript. |
| 9 | `train_sssn.py` | Spatial-Semantic Stacking Network — 5-phase ensemble: extract 512D features from frozen CNN → MLSMOTE oversampling → XGBoost ClassifierChain (Level 1A) + MLP (Level 1B) → Ridge Classifier (Level 2). |
| 10 | `log_to_mlflow.py` | Aggregates all `evaluation/*/metrics.json`, logs to MLflow Tracking, aliases `quantized_cnn` as Champion and `sssn_competitor` as Competitor. |

### Stage 3 — Telemetry Calibration (`src/telemetry/`)

| Script | Role |
|--------|------|
| `calibrate_baseline.py` | Generates DTMC baseline transition probability matrix from validation set predictions. |
| `calibrate_mmd.py` | Extracts penultimate-layer embeddings (~500 samples from training set) for MMD baseline distribution. |

### Stage 4 — Serving & Monitoring (`src/serving/`, `src/monitoring/`)

| Script | Role |
|--------|------|
| `batch_inference.py` | Loads Champion model from MLflow Registry, evaluates on holdout test set, runs dual DTMC + MMD drift detection, logs metrics to MLflow, triggers simulated retraining on drift. |
| `push_telemetry.py` | Reads `telemetry_state_buffer.json`, creates Prometheus Gauge metrics (anomaly count, DTMC/MMD scores, per-class mAP/F1), pushes to Pushgateway. |
| `verify_observability.py` | Health check: validates Pushgateway push + GET /metrics, queries Prometheus API with retry polling. |

---

## Core Competency

### Optimization

| Technique | Module |
|-----------|--------|
| **ASVD (Adaptive SVD)** | Rank-reduces Linear layer weight matrices via singular value decomposition, retaining configurable cumulative energy fraction |
| **INT8 Quantisation (PTQ)** | Post-Training Static Quantisation — Conv+BatchNorm fusion, activation range calibration, INT8 weight mapping |
| **Bayesian Optimisation (Optuna)** | Hyperparameter search over learning rate, weight decay, Focal gamma; n_estimators/max_depth for RF |
| **TorchScript Export** | Serialises quantised graph for production deployment, strips training backend |
| **ONNX Runtime (CPU Execution Provider)** | Low-latency inference with hardware-level vectorisation for CPU-bound edge deployment |

### MLOps & Services

| Tool | Purpose |
|------|---------|
| **MLflow** | Experiment tracking, model registry, Champion/Competitor aliasing, artifact lineage |
| **Feast** | Feature store — S3-backed registry, SQLite online store, file offline store |
| **Prometheus Pushgateway** | Metrics ingestion endpoint for batch telemetry (DTMC/MMD scores, per-class metrics) |
| **Grafana** | Real-time dashboards for drift monitoring and model performance |
| **Docker** | Multi-stage builds for production (FastAPI + ONNX) and Feast service |
| **GitHub Actions** | 4-stage pipeline: model registry → quality-gate → data-drift → monitoring |
| **S3 (MinIO)** | S3-compatible object storage for datasets, model artifacts, and Feast registry |
| **FastAPI** | REST API serving framework (production deployment target) |

### Algorithms

| Algorithm | Application |
|-----------|-------------|
| **CNN (CompleteCNN)** | 4-convolutional-layer backbone for 52×52 wafer map feature extraction |
| **Focal Loss** | Addresses extreme class imbalance (Scratch, Donut) by down-weighting easy negatives |
| **Random Forest** | Classic ensemble baseline — 100 estimators, bootstrap aggregation |
| **ExtraTrees** | Bayesian-optimised competitor RF with balanced subsample weighting |
| **XGBoost** | Gradient-boosted ClassifierChain for SSSN spatial meta-learning |
| **MLP (Multi-Layer Perceptron)** | Probability calibration in SSSN Level 1B semantic stream |
| **Ridge Classifier** | Stacking decider — combines Level 0 (CNN), Level 1A (XGBoost), Level 1B (MLP) |
| **Ensemble Learning (Stacking)** | 3-level stacked generalisation in SSSN with MLSMOTE minority oversampling |
| **Multi-Label SMOTE** | Custom k-NN interpolation oversampling for multi-label regime |
| **Discrete-Time Markov Chain (DTMC)** | Transition probability matrix comparison via Frobenius norm for categorical drift |
| **Maximum Mean Discrepancy (MMD)** | Two-sample kernel test on penultimate embeddings for distribution shift detection |

### Software Engineering

| Practice | Implementation |
|----------|----------------|
| **Static Typing** | `mypy` strict mode — all function signatures, return types, and generics checked |
| **Property-Based Testing** | `hypothesis` — parametric tensor dimension tests across batch sizes 1–64 |
| **Linting** | `ruff` — E, F, UP, I rulesets enforced at CI/CD quality gate |
| **Deterministic Packaging** | `uv` with lock file — cryptographic package resolution, CPU PyTorch index pinning |
| **CI/CD Quality Gates** | lint → typecheck → unit tests → integration tests → property tests (all blocking) |
| **Multi-Label Stratified Splitting** | `iterative-stratification` — preserves per-class proportions across train/val/test |

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker (for Feast and production deployment)
- S3-compatible object storage (e.g., MinIO)

### Installation

```bash
# Install uv package manager
pip install uv

# Sync dependencies (CPU PyTorch)
uv sync --frozen

# Install dev dependencies
uv sync --frozen --group dev
```

### Environment Configuration

Create a `.env` file:

```env
# S3 Object Storage (MinIO)
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
AWS_ENDPOINT_URL=http://localhost:9000

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# Pushgateway
PUSHGATEWAY_URL=http://localhost:9091

# Prometheus (optional)
PROMETHEUS_URL=http://localhost:9090
```

### Starting Local Services

```bash
# Start all infrastructure services (MinIO, MLflow, Prometheus, Pushgateway, Grafana)
docker compose up -d
```

### Running the Full Pipeline

```bash
PYTHONPATH=. uv run python run_training_pipeline.py
```

### Running Tests

```bash
uv run ruff check .        # Lint
uv run mypy src/           # Type check
PYTHONPATH=. uv run pytest tests/  # Unit + integration + property tests
```

---

## CI/CD Pipeline

```
model_registery ──► quality-gate ──► data-drift ──► monitoring
```

| Stage | Job | Description |
|-------|-----|-------------|
| **model_registery** | `tracking-models` | Aggregates all evaluation metrics and logs to MLflow |
| **quality-gate** | `quality-gate` | Runs ruff lint → mypy strict → full pytest suite |
| **data-drift** | `baseline-calibration` | Calibrates DTMC and MMD baselines from validation/training data |
| **data-drift** | `telemetry-evaluation` | Runs batch inference, computes class metrics, detects drift |
| **monitoring** | `telemetry-push` | Pushes drift and performance metrics to Prometheus Pushgateway |
| **monitoring** | `verify-observability` | Health-check — validates metrics reachability through Prometheus API |

All stages use deterministic `uv sync --frozen` with `ghcr.io/astral-sh/uv:python3.11-bookworm-slim`.

See `.github/workflows/ci.yml` for the full workflow definition.

---

## Monitoring Architecture

```
Batch Inference
      │
      ▼
 DTMC Detector  ──►  Frobenius norm (categorical drift)
 MMD Detector   ──►  RBF kernel distance (distribution shift)
      │
      ▼
 Pushgateway  ──►  Prometheus (pull, ~15s scrape interval)
      │
      ▼
 Grafana  ──►  Live dashboards (anomaly count, per-class mAP, drift thresholds)
```

Both DTMC and MMD detectors raise anomaly flags when computed distances exceed calibrated thresholds. Anomaly counts trigger automated retraining workflows in production.

---
