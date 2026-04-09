# 🧠 Stroke Risk Predictor

**ML-powered clinical screening tool with SHAP explainability**

A machine learning web application that predicts stroke risk from clinical features, trained on 5,110 patient records. Unlike typical classification projects that optimize for accuracy, this tool is designed around the asymmetric cost structure of medical screening — where missing an at-risk patient is far more dangerous than a false alarm.

🔗 **[Live Demo](https://your-app.vercel.app)** · 📄 **[Research Paper](./research_paper_outline.md)**

---

## Why This Project Is Different

Most stroke prediction projects on Kaggle report ~95% accuracy and call it a day. That metric is **meaningless** here — a model that predicts "no stroke" for every patient achieves 95.1% accuracy because only 4.87% of patients had strokes.

This project instead:

- **Optimizes for recall (sensitivity)** — catching 82% of at-risk patients at the cost of some false positives, which is the clinically appropriate tradeoff for a screening tool
- **Explains every prediction** using SHAP (SHapley Additive exPlanations), so a physician can see *why* a patient is flagged, not just *that* they are
- **Connects ML findings to neuroscience** — SHAP feature importance rankings align with established cerebrovascular pathophysiology
- **Deploys as an interactive tool** — not a static notebook, but a usable clinical interface

## Features

| Tab | Description |
|-----|-------------|
| **Risk Assessment** | Input patient data → get real-time risk score with per-feature SHAP contributions |
| **Model Comparison** | 6 ML algorithms compared on AUC, recall, precision, F1 |
| **Risk Factors** | Global SHAP importance + neuroscience interpretation of each factor |
| **Methodology** | Full 6-step pipeline explanation with key technical decisions |
| **About** | Project context, findings, limitations, author info |

## Results

| Model | AUC-ROC | Recall | Precision | F1 |
|-------|---------|--------|-----------|-----|
| **Logistic Regression** | **0.897** | **82.0%** | 16.3% | 0.272 |
| Random Forest | 0.865 | 42.0% | 31.3% | 0.359 |
| XGBoost | 0.847 | 58.0% | 16.8% | 0.260 |
| Gradient Boosting | 0.846 | 44.0% | 27.9% | 0.341 |
| SVM | 0.836 | 62.0% | 15.7% | 0.251 |
| KNN | 0.753 | 56.0% | 18.1% | 0.273 |

**Top 5 risk factors (SHAP):** Age (2.033), Avg glucose level (0.495), BMI (0.284), Heart disease (0.276), Hypertension (0.182)

## Tech Stack

**ML Pipeline:** Python, scikit-learn, XGBoost, SHAP, imbalanced-learn (SMOTE), pandas, matplotlib, seaborn

**Web App:** React 18, vanilla CSS (no framework dependencies)

**Deployment:** Vercel (free tier)

## Project Structure

```
stroke-risk-predictor/
├── public/
│   └── index.html
├── src/
│   ├── App.js              # Main React application (all 5 tabs)
│   └── index.js             # Entry point
├── notebooks/
│   └── stroke_ml_pipeline.py  # Full ML pipeline (run in Jupyter or standalone)
├── outputs/
│   ├── 01_eda_overview.png
│   ├── 02_roc_pr_curves.png
│   ├── 03_threshold_analysis.png
│   ├── 04_shap_global.png
│   ├── 05_shap_patient_1.png
│   ├── 05_shap_patient_2.png
│   └── model_results.json
├── research_paper_outline.md
├── package.json
└── README.md
```

## Quick Start

### Run the web app locally

```bash
git clone https://github.com/YOUR_USERNAME/stroke-risk-predictor.git
cd stroke-risk-predictor
npm install
npm start
```

Opens at `http://localhost:3000`

### Run the ML pipeline

```bash
# Install dependencies
pip install pandas numpy scikit-learn imbalanced-learn shap matplotlib seaborn xgboost

# Download the dataset from Kaggle:
# https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
# Place healthcare-dataset-stroke-data.csv in the project root

# Run the pipeline
python notebooks/stroke_ml_pipeline.py
```

### Deploy to Vercel

```bash
npm install -g vercel
vercel
```

## Methodology

1. **Data Collection** — 5,110 patient records, 11 clinical features (Kaggle/McKinsey)
2. **Preprocessing** — KNN imputation for BMI (201 missing), label encoding, StandardScaler
3. **Class Imbalance** — SMOTE on training set only + cost-sensitive balanced class weights
4. **Model Training** — 6 algorithms, 5-fold stratified cross-validation
5. **Explainability** — SHAP for global and individual patient-level feature attribution
6. **Threshold Optimization** — Lowered from 0.5 → 0.3 for clinical screening

## Neuroscience Connection

The SHAP analysis reveals risk factor rankings that align with established cerebrovascular pathophysiology:

- **Age (SHAP: 2.033)** — Stroke risk doubles each decade after 55 due to progressive atherosclerosis and cerebral small vessel disease
- **Glucose (0.495)** — Chronic hyperglycemia damages cerebral endothelial cells through AGEs, creating a pro-thrombotic state
- **BMI (0.284)** — Obesity promotes stroke via chronic inflammation, insulin resistance, and increased blood viscosity
- **Heart disease (0.276)** — Cardiac conditions, especially atrial fibrillation, cause cardioembolic strokes
- **Hypertension (0.182)** — The most modifiable risk factor; chronic pressure damages arterial walls and weakens cerebral vessels

## Limitations

- Cross-sectional dataset — no temporal/longitudinal modeling
- Missing features: atrial fibrillation, cholesterol, family history, medications
- SMOTE may introduce synthetic artifacts
- Single-source data limits generalizability
- Web app uses a simplified logistic approximation for real-time scoring

## License

MIT

## Disclaimer

This tool is for **educational and research purposes only**. It is not a medical device and should not be used for clinical decision-making without professional medical oversight.
