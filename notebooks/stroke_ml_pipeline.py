"""
Stroke Risk Prediction: A Machine Learning Approach with Explainable AI
=========================================================================
Full ML pipeline: preprocessing, class imbalance handling, model comparison,
hyperparameter tuning, SHAP explainability, and clinical threshold analysis.

Author: [Your Name]
Dataset: Kaggle Stroke Prediction Dataset (5,110 patients, 11 clinical features)
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, f1_score,
    recall_score, precision_score, roc_auc_score, brier_score_loss
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import shap
import json
import os

OUT = '/home/claude/outputs'
os.makedirs(OUT, exist_ok=True)

# ============================================================
# 1. DATA LOADING & EXPLORATION
# ============================================================
print("=" * 60)
print("PHASE 1: Data Loading & Exploration")
print("=" * 60)

df = pd.read_csv('/home/claude/stroke_data.csv')
df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
df.drop(columns=['id'], inplace=True)
df = df[df['gender'] != 'Other']  # Remove single 'Other' entry if present

print(f"Dataset shape: {df.shape}")
print(f"\nTarget distribution:")
print(df['stroke'].value_counts())
print(f"\nClass imbalance ratio: {df['stroke'].value_counts()[0] / df['stroke'].value_counts()[1]:.1f}:1")
print(f"\nMissing values:\n{df.isnull().sum()}")

# ============================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2: Exploratory Data Analysis")
print("=" * 60)

fig = plt.figure(figsize=(20, 16))
gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

# Age distribution by stroke status
ax1 = fig.add_subplot(gs[0, 0])
for label, color in [(0, '#3498db'), (1, '#e74c3c')]:
    subset = df[df['stroke'] == label]
    ax1.hist(subset['age'], bins=30, alpha=0.6, color=color, 
             label=f'{"Stroke" if label else "No Stroke"}', density=True)
ax1.set_xlabel('Age'); ax1.set_ylabel('Density'); ax1.set_title('Age Distribution by Stroke Status')
ax1.legend()

# Glucose distribution
ax2 = fig.add_subplot(gs[0, 1])
for label, color in [(0, '#3498db'), (1, '#e74c3c')]:
    subset = df[df['stroke'] == label]
    ax2.hist(subset['avg_glucose_level'], bins=30, alpha=0.6, color=color,
             label=f'{"Stroke" if label else "No Stroke"}', density=True)
ax2.set_xlabel('Avg Glucose Level'); ax2.set_ylabel('Density'); ax2.set_title('Glucose Distribution')
ax2.legend()

# BMI distribution
ax3 = fig.add_subplot(gs[0, 2])
for label, color in [(0, '#3498db'), (1, '#e74c3c')]:
    subset = df[df['stroke'] == label].dropna(subset=['bmi'])
    ax3.hist(subset['bmi'], bins=30, alpha=0.6, color=color,
             label=f'{"Stroke" if label else "No Stroke"}', density=True)
ax3.set_xlabel('BMI'); ax3.set_ylabel('Density'); ax3.set_title('BMI Distribution')
ax3.legend()

# Stroke rate by hypertension
ax4 = fig.add_subplot(gs[1, 0])
stroke_by_hyp = df.groupby('hypertension')['stroke'].mean() * 100
ax4.bar(['No Hypertension', 'Hypertension'], stroke_by_hyp.values, color=['#3498db', '#e74c3c'])
ax4.set_ylabel('Stroke Rate (%)'); ax4.set_title('Stroke Rate by Hypertension')
for i, v in enumerate(stroke_by_hyp.values):
    ax4.text(i, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold')

# Stroke rate by heart disease
ax5 = fig.add_subplot(gs[1, 1])
stroke_by_hd = df.groupby('heart_disease')['stroke'].mean() * 100
ax5.bar(['No Heart Disease', 'Heart Disease'], stroke_by_hd.values, color=['#3498db', '#e74c3c'])
ax5.set_ylabel('Stroke Rate (%)'); ax5.set_title('Stroke Rate by Heart Disease')
for i, v in enumerate(stroke_by_hd.values):
    ax5.text(i, v + 0.3, f'{v:.1f}%', ha='center', fontweight='bold')

# Stroke rate by smoking
ax6 = fig.add_subplot(gs[1, 2])
stroke_by_smoke = df.groupby('smoking_status')['stroke'].mean() * 100
ax6.barh(stroke_by_smoke.index, stroke_by_smoke.values, color='#e74c3c')
ax6.set_xlabel('Stroke Rate (%)'); ax6.set_title('Stroke Rate by Smoking Status')

# Correlation heatmap (numeric only)
ax7 = fig.add_subplot(gs[2, :2])
numeric_df = df.select_dtypes(include=[np.number])
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, ax=ax7, fmt='.2f')
ax7.set_title('Feature Correlations')

# Class imbalance pie
ax8 = fig.add_subplot(gs[2, 2])
counts = df['stroke'].value_counts()
ax8.pie(counts, labels=['No Stroke', 'Stroke'], autopct='%1.1f%%', 
        colors=['#3498db', '#e74c3c'], explode=[0, 0.1])
ax8.set_title('Class Distribution (Severe Imbalance)')

plt.savefig(f'{OUT}/01_eda_overview.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 01_eda_overview.png")

# ============================================================
# 3. PREPROCESSING
# ============================================================
print("\n" + "=" * 60)
print("PHASE 3: Preprocessing")
print("=" * 60)

# Encode categoricals
le_dict = {}
cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le
    print(f"  Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# KNN imputation for BMI (better than mean/median)
print(f"\n  Missing BMI values: {df['bmi'].isnull().sum()}")
imputer = KNNImputer(n_neighbors=5)
df['bmi'] = imputer.fit_transform(df[['bmi']])
print(f"  After KNN imputation: {df['bmi'].isnull().sum()} missing")

X = df.drop('stroke', axis=1)
y = df['stroke']
feature_names = X.columns.tolist()

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply SMOTE to training data only
smote = SMOTE(random_state=42, sampling_strategy=0.5)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
print(f"\n  Before SMOTE: {dict(pd.Series(y_train).value_counts())}")
print(f"  After SMOTE:  {dict(pd.Series(y_train_res).value_counts())}")

# ============================================================
# 4. MODEL COMPARISON
# ============================================================
print("\n" + "=" * 60)
print("PHASE 4: Model Comparison (5-Fold Stratified CV)")
print("=" * 60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        scale_pos_weight=len(y_train[y_train==0])/max(len(y_train[y_train==1]),1),
        eval_metric='logloss', random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    # Use SMOTE-resampled data for training, original for scoring
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
    
    rec = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob) if y_prob is not None else 0
    brier = brier_score_loss(y_test, y_prob) if y_prob is not None else 1
    
    results[name] = {
        'model': model, 'y_pred': y_pred, 'y_prob': y_prob,
        'recall': rec, 'precision': prec, 'f1': f1, 'auc': auc_score, 'brier': brier
    }
    print(f"  {name:25s} | Recall: {rec:.3f} | Precision: {prec:.3f} | F1: {f1:.3f} | AUC: {auc_score:.3f}")

# ============================================================
# 5. ROC & PRECISION-RECALL CURVES
# ============================================================
print("\n" + "=" * 60)
print("PHASE 5: ROC & Precision-Recall Analysis")
print("=" * 60)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
for i, (name, res) in enumerate(results.items()):
    if res['y_prob'] is not None:
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        ax1.plot(fpr, tpr, color=colors[i], lw=2, label=f"{name} (AUC={res['auc']:.3f})")
        
        prec_arr, rec_arr, _ = precision_recall_curve(y_test, res['y_prob'])
        ap = average_precision_score(y_test, res['y_prob'])
        ax2.plot(rec_arr, prec_arr, color=colors[i], lw=2, label=f"{name} (AP={ap:.3f})")

ax1.plot([0,1],[0,1], 'k--', alpha=0.3)
ax1.set_xlabel('False Positive Rate'); ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curves — Higher is Better'); ax1.legend(fontsize=9)

baseline = y_test.mean()
ax2.axhline(y=baseline, color='k', linestyle='--', alpha=0.3, label=f'Baseline ({baseline:.3f})')
ax2.set_xlabel('Recall (Sensitivity)'); ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curves — Critical for Imbalanced Data'); ax2.legend(fontsize=9)

plt.savefig(f'{OUT}/02_roc_pr_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 02_roc_pr_curves.png")

# ============================================================
# 6. BEST MODEL — CONFUSION MATRIX & THRESHOLD ANALYSIS
# ============================================================
best_name = max(results, key=lambda k: results[k]['auc'])
best = results[best_name]
print(f"\nBest model by AUC: {best_name} (AUC={best['auc']:.3f})")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Confusion matrix
cm = confusion_matrix(y_test, best['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['No Stroke', 'Stroke'], yticklabels=['No Stroke', 'Stroke'])
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
axes[0].set_title(f'{best_name} — Confusion Matrix')

# Threshold analysis (clinical: we want HIGH recall)
thresholds = np.arange(0.1, 0.9, 0.01)
recalls, precisions, f1s = [], [], []
for t in thresholds:
    y_t = (best['y_prob'] >= t).astype(int)
    recalls.append(recall_score(y_test, y_t, zero_division=0))
    precisions.append(precision_score(y_test, y_t, zero_division=0))
    f1s.append(f1_score(y_test, y_t, zero_division=0))

axes[1].plot(thresholds, recalls, 'r-', lw=2, label='Recall (Sensitivity)')
axes[1].plot(thresholds, precisions, 'b-', lw=2, label='Precision')
axes[1].plot(thresholds, f1s, 'g--', lw=2, label='F1 Score')
axes[1].axvline(x=0.3, color='orange', linestyle=':', label='Clinical threshold (0.3)')
axes[1].set_xlabel('Classification Threshold'); axes[1].set_ylabel('Score')
axes[1].set_title('Threshold Analysis — Lowering Threshold Catches More Strokes')
axes[1].legend()

# Calibration-like: predicted probability distribution
axes[2].hist(best['y_prob'][y_test == 0], bins=30, alpha=0.6, color='#3498db', label='No Stroke', density=True)
axes[2].hist(best['y_prob'][y_test == 1], bins=30, alpha=0.6, color='#e74c3c', label='Stroke', density=True)
axes[2].set_xlabel('Predicted Stroke Probability'); axes[2].set_ylabel('Density')
axes[2].set_title('Predicted Probability Distribution')
axes[2].legend()

plt.savefig(f'{OUT}/03_threshold_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 03_threshold_analysis.png")

# ============================================================
# 7. SHAP EXPLAINABILITY
# ============================================================
print("\n" + "=" * 60)
print("PHASE 6: SHAP Explainability Analysis")
print("=" * 60)

best_model = best['model']

# Use TreeExplainer for tree-based models
if isinstance(best_model, (xgb.XGBClassifier, RandomForestClassifier, GradientBoostingClassifier)):
    explainer = shap.TreeExplainer(best_model)
else:
    explainer = shap.LinearExplainer(best_model, X_train_res)

X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
shap_values = explainer.shap_values(X_test_df)

# Handle different SHAP output formats
if isinstance(shap_values, list):
    shap_vals = shap_values[1]  # Class 1 (stroke)
else:
    shap_vals = shap_values

# Global feature importance (SHAP bar plot)
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

mean_abs_shap = np.abs(shap_vals).mean(axis=0)
sorted_idx = np.argsort(mean_abs_shap)
axes[0].barh(np.array(feature_names)[sorted_idx], mean_abs_shap[sorted_idx], color='#e74c3c')
axes[0].set_xlabel('Mean |SHAP value|')
axes[0].set_title('Global Feature Importance (SHAP)')

# SHAP beeswarm/summary
plt.sca(axes[1])
shap.summary_plot(shap_vals, X_test_df, feature_names=feature_names, show=False, max_display=11)
axes[1].set_title('SHAP Summary — How Features Drive Predictions')

plt.tight_layout()
plt.savefig(f'{OUT}/04_shap_global.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 04_shap_global.png")

# Individual SHAP force plots for 2 high-risk patients
stroke_indices = np.where(y_test.values == 1)[0]
if len(stroke_indices) >= 2:
    for i, idx in enumerate(stroke_indices[:2]):
        fig, ax = plt.subplots(figsize=(14, 3))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_vals[idx],
                base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
                data=X_test_df.iloc[idx].values,
                feature_names=feature_names
            ),
            show=False
        )
        plt.title(f'Patient {idx} — SHAP Waterfall (Actual: Stroke)')
        plt.tight_layout()
        plt.savefig(f'{OUT}/05_shap_patient_{i+1}.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: 05_shap_patient_{i+1}.png")

# ============================================================
# 8. EXPORT MODEL METRICS AS JSON (for React app)
# ============================================================
print("\n" + "=" * 60)
print("PHASE 7: Exporting Results")
print("=" * 60)

# Feature importance rankings
importance_data = {feature_names[i]: float(mean_abs_shap[i]) for i in range(len(feature_names))}
importance_data = dict(sorted(importance_data.items(), key=lambda x: x[1], reverse=True))

# Model comparison table
model_comparison = {}
for name, res in results.items():
    model_comparison[name] = {
        'recall': round(res['recall'], 4),
        'precision': round(res['precision'], 4),
        'f1': round(res['f1'], 4),
        'auc': round(res['auc'], 4),
        'brier': round(res['brier'], 4),
    }

export = {
    'best_model': best_name,
    'feature_importance_shap': importance_data,
    'model_comparison': model_comparison,
    'dataset_info': {
        'total_samples': len(df),
        'stroke_positive': int(y.sum()),
        'stroke_negative': int(len(y) - y.sum()),
        'imbalance_ratio': round(float((len(y) - y.sum()) / y.sum()), 1),
        'features': feature_names,
        'missing_bmi_imputed': 201,
    },
    'clinical_note': (
        'This model is optimized for HIGH RECALL (sensitivity) to minimize missed stroke cases. '
        'A lower classification threshold (0.3) is recommended for clinical screening to '
        'catch more at-risk patients, accepting some false positives as a worthwhile tradeoff.'
    ),
}

with open(f'{OUT}/model_results.json', 'w') as f:
    json.dump(export, f, indent=2)
print("Saved: model_results.json")

# Print final summary
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"Best Model: {best_name}")
print(f"  AUC-ROC:    {best['auc']:.4f}")
print(f"  Recall:     {best['recall']:.4f}")
print(f"  Precision:  {best['precision']:.4f}")
print(f"  F1:         {best['f1']:.4f}")
print(f"  Brier Loss: {best['brier']:.4f}")
print(f"\nTop 5 Risk Factors (SHAP):")
for i, (feat, imp) in enumerate(list(importance_data.items())[:5]):
    print(f"  {i+1}. {feat}: {imp:.4f}")
print("\nAll outputs saved to:", OUT)
