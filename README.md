# Retail Anomaly Detection: Data Tracking and Modeling During My Digital Consulting Internship


Retail store scan-quality anomaly detection pipeline — semi-supervised scoring,
interpretable factor analysis, and a deployable REST API.

> **Context**: Context: distributed retail outlets submit daily transaction logs to a centralized analytics platform.
> Data quality varies significantly due to inconsistent reporting, missing entries, and input errors.
> This system evaluates data reliability at the unit level and flags anomalous patterns using a three-step pipeline:
> LFM-based feature extraction → self-training pseudo-labeling → logistic regression classification.


> **Privacy**: all store identifiers in this repository have been masked (`STORE_NNNNN` format).
> In addition, the underlying data distribution has been modified and partially resampled to prevent re-identification.
> No personally identifiable information is present.

---

## Pipeline overview

```
11 indicator score columns  (3,757 stores)
            │
            ▼
┌─────────────────────────┐
│  Step 1 · LFM           │  Varimax factor analysis on all 3,757 rows
│  LFMFeatureExtractor    │  → 3 orthogonal factors (52.2% variance)
│  unsupervised           │  F1: Transaction behaviour
│                         │  F2: Inventory-sales relationship
└──────────┬──────────────┘  F3: Inventory deviation & activity
           │ F_full (3757 × 3)
           ▼
┌─────────────────────────┐
│  Step 2 · Self-Training │  Start: 318 labelled rows (62 flagged, 19.5%)
│  SelfTrainer            │  5 rounds · thresholds 0.85 / 0.15
│  semi-supervised        │  → 2,826 pseudo-labels added
│                         │  → 3,144 effective training rows
└──────────┬──────────────┘
           │ expanded labels
           ▼
┌─────────────────────────┐
│  Step 3 · Classifier    │  LogisticRegression (3 LFM factors + brand_wid)
│  FinalClassifier        │  CV AUC: 0.875 ± 0.086
│  supervised             │  → anomaly_proba for all 3,757 stores
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Step 4 · API           │  FastAPI · Docker · MLflow
│  /score  /generate      │  POST stores → anomaly probability
│  /report                │  GET  → city-level summary
└─────────────────────────┘
```

---

## Quickstart

### 1 · Clone and install

```bash
git clone https://github.com/yaniwu/Retail-Anomaly-Detection.git
cd Retail-Anomaly-Detection
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 2 · Run the pipeline

```bash
python scripts/pipeline.py --data data/raw/retail_processed.csv
```

Outputs written to `reports/`:

| File | Contents |
|---|---|
| `factor_loadings.csv` | Varimax loadings (11 indicators × 3 factors) |
| `self_training_log.csv` | Pseudo-labels added per round |
| `city_summary.csv` | City-level anomaly rates |
| `all_scores.csv` | Per-store F1/F2/F3 + anomaly probability |
| `models/final_clf.pkl` | Serialised final classifier |

### 3 · Generate PDF report

```bash
python scripts/generate_report.py
# → reports/pipeline_report.pdf  (5 pages)
```

### 4 · Open analysis notebook

```bash
jupyter lab notebooks/01_analysis.ipynb
```

The notebook justifies four design choices with experiments and visualisations:
§1 Why dimensionality reduction · §2 Why LFM over PCA/entropy · §3 Why 3 factors · §4 Why LR as base model

### 5 · One-command Docker demo

```bash
docker compose up --build
curl http://localhost:8000/health
```

---

## Project structure

```
Retail-Anomaly-Detection/
├── configs/
│   └── default.yaml              # All hyperparameters (LFM, self-training, classifier)
├── data/
│   ├── raw/                      # gitignored — retail_processed.csv, retail_raw.csv
│   └── synthetic/                # gitignored — CVAE output
├── notebooks/
│   ├── 01_analysis.ipynb         # Design decision analysis (4 sections)
│   └── 01_analysis.html          # Pre-rendered HTML export
├── retail_anomaly/
│   ├── features/
│   │   └── lfm.py                # LFMFeatureExtractor (fit/transform)
│   ├── scorer/
│   │   ├── lfm.py                # ImprovedLFM (original KDE-based scorer)
│   │   └── final_model.py        # FinalClassifier (LR wrapper + save/load)
│   ├── semi/
│   │   └── self_training.py      # SelfTrainer (pseudo-label loop)
│   ├── cvae/
│   │   ├── model.py              # RetailCVAE (PyTorch)
│   │   └── validation.py         # KS test helpers
│   ├── api/
│   │   └── app.py                # FastAPI service
│   └── utils/
│       ├── config.py             # YAML config loader
│       ├── data_loader.py        # load_pipeline_data
│       └── validation.py         # Input validation helpers
├── scripts/
│   ├── pipeline.py               # Main entry point (Steps 1–5)
│   ├── generate_report.py        # 5-page PDF report generator
│   ├── create_analysis_notebook.py  # Notebook generator (run once)
│   └── archive/
│       └── train_unsupervised.py # Original unsupervised script
├── reports/
│   ├── factor_loadings.csv       # committed
│   ├── model_comparison.csv      # committed
│   ├── shap_importance.csv       # committed
│   └── models/                   # gitignored (pkl/npy)
├── tests/
│   ├── test_lfm.py
│   ├── test_cvae.py
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml
└── pyproject.toml
```

---

## Design decisions

Full analysis with experiments in [`notebooks/01_analysis.ipynb`](notebooks/01_analysis.ipynb).

### Why LFM over PCA and entropy weighting?

| Method | AUC (CV, LR) | Interpretable factors |
|---|---|---|
| Entropy weighting | ~0.79 | No — single composite |
| PCA | ~0.85 | No — variance axes |
| **LFM (varimax)** | **~0.875** | Yes — named business factors |

LFM achieves the highest CV AUC and its varimax-rotated factors map to
named business concepts: transaction behaviour (F1), inventory-sales
relationship (F2), inventory deviation and activity (F3).

### Why 3 factors?

Parallel analysis (Horn 1965, n_iter=200, 95th percentile) recommends 3 factors.
The Kaiser criterion (λ > 1) agrees. Three factors explain 52.2% of standardised
variance — see `reports/factor_loadings.csv`.

### Why Self-Training with LR as base model?

With only 318 labelled rows (62 positive), LR outperforms LightGBM on CV AUC
due to its lower model complexity. LR's sigmoid output is also better calibrated
(lower ECE), producing more reliable probability estimates for the 0.85/0.15
pseudo-label thresholds. See §4 of the analysis notebook for calibration curves
and threshold sensitivity analysis.

### Key bug: eigenvector sign ambiguity

Factor regression scores have arbitrary eigenvector sign (−v is as valid as +v).
Without alignment, the composite score is inverted, producing AUC ≈ 0.28 instead
of 0.87. Fix: align each factor column so it correlates positively with the row
mean of the standardised input matrix before computing the composite.

---

## Results

| Model | CV AUC | Notes |
|---|---|---|
| LR · 3 LFM factors | **0.875 ± 0.086** | Best; pipeline default |
| RF · 11 scores | 0.868 ± 0.082 | |
| LR · 11 scores | 0.852 ± 0.066 | baseline |
| LightGBM · factors + scores | 0.849 ± 0.050 | |

Self-Training expanded the labelled set from **318 → 3,144 rows** over 5 rounds.

City anomaly rates (predicted): **A = 12.5%  ·  B = 31.5%  ·  C = 37.5%**

---

## Data

| File | Rows | Columns | Description |
|---|---|---|---|
| `retail_processed.csv` | 3,757 | 14 | 11 score columns + label (318 labelled) |
| `retail_raw.csv` | 3,757 | 24 | Raw behavioural indicators |
| `retail_labels.csv` | 3,757 | 3 | ID + city + label only |

All store identifiers are masked (`STORE_00001` – `STORE_03757`).

---

## Running tests

```bash
pytest                          # all tests
pytest tests/test_lfm.py -v     # scorer only
pytest tests/test_api.py -v     # API only
```

---

## Tech stack

| Layer | Library |
|---|---|
| Factor analysis | NumPy · SciPy (varimax via SVD) |
| Semi-supervised | scikit-learn (self-training loop) |
| Classifier | scikit-learn LogisticRegression |
| Generative model | PyTorch (CVAE) |
| API | FastAPI + Uvicorn |
| Experiment tracking | MLflow |
| Containerisation | Docker / docker-compose |
| Reporting | matplotlib · PdfPages |
| Testing | pytest + httpx |
