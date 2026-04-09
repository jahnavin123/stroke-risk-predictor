# Stroke Risk Prediction Using Machine Learning with Explainable AI: A Clinical Screening Approach

## Abstract

Stroke is a leading cause of mortality and long-term disability worldwide, yet many at-risk individuals go unidentified until a cerebrovascular event occurs. This study develops and evaluates a machine learning-based screening tool for stroke risk prediction using demographic and clinical features from 5,110 patient records. We address the critical challenge of severe class imbalance (19.5:1 ratio) through SMOTE oversampling and cost-sensitive learning, comparing six classification algorithms. Our best model (Logistic Regression with balanced class weights) achieves an AUC-ROC of 0.897 and recall of 82%, prioritizing sensitivity over specificity — a clinically appropriate tradeoff for screening applications where missing at-risk patients carries greater cost than false positives. We apply SHAP (SHapley Additive exPlanations) to provide individualized, interpretable risk factor contributions, bridging the gap between predictive accuracy and clinical transparency. Our analysis confirms that age, average glucose level, BMI, heart disease, and hypertension are the most influential stroke predictors, consistent with established neurovascular pathophysiology.

**Keywords:** stroke prediction, machine learning, SHAP, explainable AI, class imbalance, clinical decision support

---

## 1. Introduction

### 1.1 Background and Motivation
- Stroke statistics: leading cause of death globally (WHO), 1 in 4 adults over 25 will have a stroke in their lifetime
- The "golden hour" problem: early identification of at-risk individuals enables preventive intervention
- Gap: most clinical risk scores (e.g., CHA₂DS₂-VASc) are designed for specific populations; a general-purpose ML screening tool could expand access

### 1.2 Neuroscience of Stroke Risk
- Pathophysiology: ischemic vs. hemorrhagic stroke mechanisms
- How modifiable risk factors (hypertension, glucose, smoking) damage cerebral vasculature
- Age-related cerebral small vessel disease and its role in stroke susceptibility
- Connection between metabolic syndrome indicators (BMI, glucose) and cerebrovascular risk

### 1.3 Research Questions
1. Can machine learning models trained on readily available clinical features reliably identify patients at high stroke risk?
2. How should class imbalance be handled in a clinical screening context where false negatives are more costly than false positives?
3. Which features contribute most to stroke risk predictions, and do these align with established neuroscience literature?

---

## 2. Related Work

- Prior ML approaches to stroke prediction (Random Forest, SVM, Neural Networks)
- Comparison of reported accuracy metrics and their limitations
- The class imbalance problem in medical ML — why accuracy is misleading
- Explainable AI in healthcare: SHAP, LIME, and the regulatory push for interpretability (FDA guidance on ML/AI in medical devices)

---

## 3. Methods

### 3.1 Dataset Description
- Source: Kaggle Stroke Prediction Dataset (McKinsey & Company)
- 5,110 patients, 11 features, binary target
- Feature descriptions: demographic (age, gender, marital status, work type, residence), clinical (hypertension, heart disease, avg glucose level, BMI, smoking status)

### 3.2 Preprocessing Pipeline
- **Missing data handling**: KNN imputation for BMI (201 missing values, 3.93% of data) — chosen over mean/median to preserve feature relationships
- **Encoding**: Label encoding for categorical variables
- **Scaling**: StandardScaler for feature normalization
- **Train/test split**: 80/20 stratified split

### 3.3 Addressing Class Imbalance
- Problem: only 4.87% positive class (249/5,110)
- SMOTE (Synthetic Minority Oversampling Technique) applied to training set only
- Cost-sensitive learning via class_weight='balanced'
- Why NOT to use accuracy: a naive "predict no stroke always" model achieves 95.1% accuracy

### 3.4 Models Evaluated
- Logistic Regression (with balanced class weights)
- Random Forest (balanced, 200 trees)
- XGBoost (scale_pos_weight adjusted)
- Gradient Boosting (200 estimators)
- K-Nearest Neighbors
- Support Vector Machine (RBF kernel, balanced)

### 3.5 Evaluation Strategy
- 5-fold stratified cross-validation
- Primary metric: Recall (sensitivity) — clinical justification for prioritizing this
- Secondary metrics: AUC-ROC, Precision-Recall curves, F1, Brier score
- Threshold analysis: optimizing classification threshold for clinical screening

### 3.6 Explainability Framework
- SHAP (SHapley Additive exPlanations) for both global and local interpretability
- TreeExplainer for tree-based models, LinearExplainer for logistic regression
- Waterfall plots for individual patient risk explanations

---

## 4. Results

### 4.1 Exploratory Data Analysis
- Age distribution: stroke patients significantly older (mean ~68 vs ~40)
- Glucose: bimodal distribution in stroke group suggesting diabetic subpopulation
- Severe class imbalance visualization

### 4.2 Model Performance Comparison

| Model | AUC-ROC | Recall | Precision | F1 | Brier Score |
|-------|---------|--------|-----------|-----|-------------|
| **Logistic Regression** | **0.897** | **0.820** | 0.163 | 0.272 | 0.146 |
| Random Forest | 0.865 | 0.420 | 0.313 | 0.359 | 0.058 |
| XGBoost | 0.847 | 0.580 | 0.168 | 0.260 | 0.116 |
| Gradient Boosting | 0.846 | 0.440 | 0.279 | 0.341 | 0.060 |
| SVM | 0.836 | 0.620 | 0.157 | 0.251 | 0.094 |
| KNN | 0.753 | 0.560 | 0.181 | 0.273 | 0.109 |

### 4.3 Threshold Analysis
- Default threshold (0.5) vs. clinical threshold (0.3)
- Sensitivity-specificity tradeoff curves
- Clinical recommendation: lower threshold for screening to maximize patient safety

### 4.4 SHAP Feature Importance
- **Age** (SHAP: 2.033): dominant predictor, consistent with exponential stroke risk increase after age 55
- **Avg glucose level** (0.495): elevated glucose linked to endothelial dysfunction
- **BMI** (0.284): obesity as independent cardiovascular risk factor
- **Heart disease** (0.276): direct comorbidity relationship
- **Hypertension** (0.182): chronic vascular damage mechanism

### 4.5 Individual Patient Explanations
- Waterfall plot examples showing how each feature pushes prediction toward/away from stroke for specific patients
- Clinical utility: physicians can see *why* a patient is flagged, not just *that* they are

---

## 5. Discussion

### 5.1 Clinical Implications
- This model is designed as a **screening tool**, not a diagnostic tool
- High recall (82%) means catching 4 out of 5 at-risk patients in a primary care setting
- False positives are acceptable: they lead to additional testing, not harmful intervention
- Integration pathway: embed in electronic health records as an automated risk flag

### 5.2 Neuroscience Interpretation of SHAP Results
- Age as dominant feature: reflects cumulative cerebrovascular damage (atherosclerosis, amyloid angiopathy)
- Glucose-stroke connection: advanced glycation end products (AGEs) damage the blood-brain barrier
- The smoking paradox in this dataset: "Unknown" category may mask true associations
- Heart disease pathway: cardioembolic stroke mechanism (atrial fibrillation → clot → brain)

### 5.3 Limitations
- Dataset size (5,110 patients) limits generalizability
- No temporal data — cross-sectional snapshot, cannot model time-to-event
- Missing features: atrial fibrillation, cholesterol, family history, medication use
- Synthetic oversampling (SMOTE) may introduce artifacts
- Single-center bias possible (original data from McKinsey)

### 5.4 Ethical Considerations
- Model fairness across demographic groups (gender, age)
- Risk of over-reliance on algorithmic screening
- Importance of human-in-the-loop clinical decision making

---

## 6. Future Work

- Incorporate additional clinical features (ECG data, lipid panels, genetic markers)
- Longitudinal study design with survival analysis (Cox regression)
- Deep learning approaches (neural networks, attention mechanisms)
- Federated learning across multiple hospital systems
- Prospective validation in clinical trial setting
- Integration with wearable health monitoring devices

---

## 7. Conclusion

This study demonstrates that machine learning, combined with explainable AI, can serve as an effective stroke risk screening tool using readily available clinical data. By prioritizing recall over accuracy and providing transparent feature-level explanations through SHAP, this approach bridges the gap between predictive performance and clinical interpretability — a critical requirement for responsible deployment of AI in healthcare settings.

---

## References

[Include 15-25 references spanning: stroke epidemiology (WHO, AHA), ML in healthcare, SHAP methodology, SMOTE, class imbalance literature, neurovascular pathophysiology]

---

## Appendix

- A: Complete EDA visualizations
- B: Hyperparameter search results
- C: SHAP waterfall plots for additional patients
- D: Code repository link
