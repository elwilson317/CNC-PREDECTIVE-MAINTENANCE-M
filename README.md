# CNC Milling Machine Predictive Maintenance

**Development of an Intelligent Predictive Maintenance System for CNC Milling Machines Using Machine Learning**

A machine-learning prototype that analyses CNC milling-machine sensor readings to classify records as normal or defective, compare baseline classifiers, explore sequential modelling with an LSTM, and generate rule-based maintenance alerts. A Streamlit dashboard is included as a separate post-internship extension.

> **Status:** Research / demonstration prototype. It has not been validated as a safety-critical or production maintenance system. The rule thresholds are illustrative and should be checked against machine/OEM guidance and real operating data before any operational use.

## What is included

- `src/cnc_predictive_maintenance_full_pipeline.py` — original terminal-based Python pipeline: data loading, preprocessing, feature engineering, EDA, baseline model training/evaluation, feature importance, threshold-based alerts, and LSTM experiment.
- `dashboard/cnc_predictive_maintenance_dashboard.py` — Streamlit dashboard extension with overview, live-reading prediction, dataset exploration, model performance, and alert/report views.
- `requirements.txt` — Python dependencies for both components.
- `.gitignore` — excludes local datasets, generated artefacts, model bundles, and local environment files.

## Dataset

The source code expects an Excel workbook named `cncmilingdata2023-2026.xlsx` with a worksheet named `Sensor_Data`. The workbook is **not included in this repository**. Before running the project, obtain permission to use/share the dataset and place your permitted local copy in the repository root. Do not commit proprietary, confidential, or personally identifiable data.

Expected target: `Defect_Label` (`0 = Normal`, `1 = Defective`). The pipeline uses operating and sensor variables including spindle speed, feed rate, cutting depth, three-axis vibration, spindle/coolant temperature, coolant flow, power consumption, tool wear, acoustic emission, servo load, and surface roughness.

## Setup

Python 3.10 is a reasonable starting point. Create and activate a virtual environment from the repository root, then install dependencies:

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Put the permitted Excel dataset in the repository root, then run the original pipeline:

```bash
python src/cnc_predictive_maintenance_full_pipeline.py
```

The pipeline currently reads the dataset by filename from the current working directory. Run it from the repository root. It writes plots and model outputs to the working directory; generated outputs are ignored by Git by default.

To run the dashboard:

```bash
streamlit run dashboard/cnc_predictive_maintenance_dashboard.py
```

The dashboard also expects the workbook in the repository root. Some prediction paths can train a fallback model when a saved `model_bundle.joblib` is absent; results should be treated as demonstration output, not as a validated deployed model.

## Recorded baseline results

The following figures are the recorded results represented in the project dashboard; they are not recomputed when the dashboard opens.

| Model / configuration | Accuracy | Precision (defect) | Recall (defect) | F1-score (defect) | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest (baseline) | 97.18% | 97.56% | 75.00% | 84.81% | 87.88% |
| Decision Tree (baseline) | 86.74% | 42.34% | 72.50% | 53.46% | 84.93% |
| Logistic Regression (threshold 0.25) | 59.03% | 18.48% | 85.00% | 30.36% | 86.93% |

The dataset is imbalanced, so accuracy alone is not sufficient. Recall measures the share of defective records detected; precision measures how many records flagged as defective were actually defective. The lower logistic-regression threshold increased recall in the recorded experiment but also produced many false positives.

The LSTM experiment is included for comparison, but it did not demonstrate reliable performance in the recorded run. Do not describe it as a successful production-ready predictor.

## Important limitations and next steps

- Review preprocessing and rolling-window feature creation to prevent information leakage across train/test boundaries.
- Fit preprocessing only on training data and use a chronological split if records are time-ordered.
- Choose decision thresholds on a validation set, not the final test set.
- Validate the model on independent, representative machine data and report confusion matrices and precision-recall metrics.
- Verify all alert limits with machine-specific engineering/OEM requirements; current thresholds are code-defined examples.
- Test the pipeline end-to-end in a clean environment and document reproducible results before relying on them.
- The project estimates defect classification from dataset records; it does not establish remaining useful life or prove real-world downtime/cost savings.

## Suggested repository structure

```text
cnc-predictive-maintenance-ml/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── cnc_predictive_maintenance_full_pipeline.py
├── dashboard/
│   └── cnc_predictive_maintenance_dashboard.py
├── docs/
└── results/
```

## Disclaimer

For educational and research demonstration only. Predictions and recommendations must not replace qualified maintenance judgement, validated condition-monitoring systems, or manufacturer instructions.
