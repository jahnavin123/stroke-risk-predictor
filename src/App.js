import React, { useState, useMemo } from "react";
const FEATURES = {
  age: { label: "Age", min: 1, max: 82, step: 1, default: 55, unit: "years" },
  avg_glucose_level: { label: "Avg glucose level", min: 55, max: 280, step: 1, default: 110, unit: "mg/dL" },
  bmi: { label: "BMI", min: 10, max: 60, step: 0.5, default: 28, unit: "kg/m2" },
};
const SELECTS = {
  hypertension: { label: "Hypertension", opts: [["0", "No"], ["1", "Yes"]] },
  heart_disease: { label: "Heart disease", opts: [["0", "No"], ["1", "Yes"]] },
  gender: { label: "Gender", opts: [["0", "Female"], ["1", "Male"]] },
  smoking_status: { label: "Smoking status", opts: [["2", "Never smoked"], ["1", "Formerly smoked"], ["3", "Smokes"], ["0", "Unknown"]] },
  ever_married: { label: "Ever married", opts: [["1", "Yes"], ["0", "No"]] },
};
const SHAP_WEIGHTS = {
  age: 1.713, hypertension: 0.153, avg_glucose_level: 0.152,
  Residence_type: 0.090, ever_married: 0.033, heart_disease: 0.030,
  bmi: 0.028, gender: 0.022, smoking_status: 0.018, work_type: 0.012,
};
const MODELS = [
  { name: "Logistic Regression", auc: 0.8376, recall: 0.8, precision: 0.1389, f1: 0.2367, best: true },
  { name: "Random Forest", auc: 0.7938, recall: 0.1, precision: 0.1852, f1: 0.1299 },
  { name: "Gradient Boosting", auc: 0.7918, recall: 0.14, precision: 0.2593, f1: 0.1818 },
  { name: "XGBoost", auc: 0.7908, recall: 0.48, precision: 0.1277, f1: 0.2017 },
  { name: "SVM", auc: 0.776, recall: 0.64, precision: 0.1168, f1: 0.1975 },
  { name: "KNN", auc: 0.6715, recall: 0.32, precision: 0.1032, f1: 0.1561 },
];
const NEURO_FACTORS = [
  { feature: "Age", shap: 1.713, detail: "Stroke risk doubles each decade after 55. Aging causes progressive atherosclerosis and cerebral small vessel disease." },
  { feature: "Hypertension", shap: 0.153, detail: "Chronic high blood pressure damages arterial walls through mechanical shear stress, promotes atherosclerotic plaque formation, and weakens small penetrating arteries in the brain." },
  { feature: "Avg glucose level", shap: 0.152, detail: "Chronic hyperglycemia damages endothelial cells lining cerebral blood vessels through advanced glycation end products (AGEs), creating a pro-thrombotic state." },
  { feature: "Heart disease", shap: 0.030, detail: "Cardiac conditions, especially atrial fibrillation, cause cardioembolic strokes. Irregular heart rhythm allows blood to pool and form clots that travel to the brain." },
  { feature: "BMI", shap: 0.028, detail: "Obesity promotes stroke through systemic inflammation, insulin resistance, dyslipidemia, and increased blood viscosity." },
];
const STEPS = [
  { title: "Data collection", desc: "5,110 patient records with 11 clinical features from the Kaggle Stroke Prediction Dataset" },
  { title: "Preprocessing", desc: "KNN imputation for 201 missing BMI values, label encoding for categoricals, StandardScaler normalization" },
  { title: "Class imbalance handling", desc: "SMOTE oversampling on training data only + cost-sensitive learning via balanced class weights" },
  { title: "Model training", desc: "6 algorithms compared. Primary metric: recall (sensitivity)" },
  { title: "Explainability", desc: "SHAP provides global feature importance and individual patient-level explanations" },
  { title: "Threshold optimization", desc: "Classification threshold lowered from 0.5 to 0.3 for clinical screening" },
];
function computeRisk(sliders, selects) {
  var c = {
    age: ((sliders.age - 40) / 42) * 1.713,
    avg_glucose_level: ((sliders.avg_glucose_level - 100) / 180) * 0.152,
    bmi: ((sliders.bmi - 28.9) / 31) * 0.028,
    heart_disease: Number(selects.heart_disease) * 0.030,
    hypertension: Number(selects.hypertension) * 0.153,
    gender: Number(selects.gender) * 0.022 * 0.3,
    ever_married: Number(selects.ever_married) * 0.033 * 0.2,
    smoking_status: selects.smoking_status === "3" ? 0.018 : selects.smoking_status === "1" ? 0.009 : 0,
  };
  var logit = -3.5;
  Object.values(c).forEach(function(v) { logit += v; });
  return { prob: 1 / (1 + Math.exp(-logit)), contributions: c };
}
function HBar(props) {
  var pct = props.max > 0 ? Math.min(Math.abs(props.value) / props.max * 100, 100) : 0;
  return React.createElement("div", { style: { width: Math.round(pct) + "%", minWidth: 3, height: 18, borderRadius: 3, background: props.color, transition: "width 0.4s ease" } });
}
function Metric(props) {
  return React.createElement("div", { style: { background: "#f8f7ff", borderRadius: 10, padding: "14px 16px", flex: "1 1 140px", minWidth: 130 } },
    React.createElement("div", { style: { fontSize: 10, color: "#888", textTransform: "uppercase", letterSpacing: 0.8 } }, props.label),
    React.createElement("div", { style: { fontSize: 20, fontWeight: 700, color: "#2d2a4a", marginTop: 4 } }, props.value),
    props.sub && React.createElement("div", { style: { fontSize: 11, color: "#aaa", marginTop: 2 } }, props.sub)
  );
}
export default function App() {
  var tabState = useState(0); var tab = tabState[0]; var setTab = tabState[1];
  var sliderState = useState({ age: 55, avg_glucose_level: 110, bmi: 28 }); var sliders = sliderState[0]; var setSliders = sliderState[1];
  var selectState = useState({ hypertension: "0", heart_disease: "0", gender: "0", smoking_status: "2", ever_married: "1" }); var selects = selectState[0]; var setSelects = selectState[1];
  var expandState = useState(null); var expandedFactor = expandState[0]; var setExpandedFactor = expandState[1];
  var result = useMemo(function() { return computeRisk(sliders, selects); }, [sliders, selects]);
  var prob = result.prob; var contributions = result.contributions;
  var pct = Math.round(prob * 100);
  var riskLevel = pct < 20 ? "Low" : pct < 50 ? "Moderate" : "High";
  var riskColor = pct < 20 ? "#1D9E75" : pct < 50 ? "#D4A017" : "#D94040";
  var sortedC = Object.entries(contributions).sort(function(a, b) { return Math.abs(b[1]) - Math.abs(a[1]); });
  var maxC = Math.max.apply(null, sortedC.map(function(x) { return Math.abs(x[1]); }).concat([0.01]));
  var globalSorted = Object.entries(SHAP_WEIGHTS).sort(function(a, b) { return b[1] - a[1]; });
  var maxG = globalSorted[0][1];
  var tabs = ["Risk assessment", "Models", "Risk factors", "Methodology", "About"];
  var ns = { page: { maxWidth: 760, margin: "0 auto", padding: "0 20px 40px", fontFamily: "'Segoe UI', system-ui, -apple-system, sans-serif", color: "#2d2a4a" } };
  var pill = function(active) { return { padding: "8px 16px", fontSize: 13, fontWeight: active ? 600 : 400, border: "none", borderRadius: 8, cursor: "pointer", fontFamily: "inherit", background: active ? "#534AB7" : "transparent", color: active ? "#fff" : "#888" }; };
  var card = { background: "#fff", borderRadius: 14, border: "1px solid #eae8f0", padding: "20px 24px", marginBottom: 16 };
  var note = function(bg, accent) { return { background: bg, borderLeft: "3px solid " + accent, borderRadius: "0 10px 10px 0", padding: "14px 18px", fontSize: 12, lineHeight: 1.7, color: "#555" }; };
  return React.createElement("div", { style: ns.page },
    React.createElement("div", { style: { textAlign: "center", padding: "36px 0 24px" } },
      React.createElement("div", { style: { display: "inline-block", background: "#f0eeff", padding: "4px 14px", borderRadius: 20, fontSize: 11, fontWeight: 600, letterSpacing: 1, color: "#534AB7", marginBottom: 10 } }, "MACHINE LEARNING + EXPLAINABLE AI"),
      React.createElement("h1", { style: { fontSize: 28, fontWeight: 700, margin: "8px 0 6px" } }, "Stroke risk predictor"),
      React.createElement("p", { style: { fontSize: 14, color: "#999", maxWidth: 500, margin: "0 auto" } }, "Clinical screening tool trained on 5,110 patients with SHAP-based interpretability")
    ),
    React.createElement("nav", { style: { display: "flex", gap: 0, background: "#fff", borderRadius: 12, border: "1px solid #e8e6f0", padding: 4, marginBottom: 28, flexWrap: "wrap" } },
      tabs.map(function(t, i) { return React.createElement("button", { key: i, onClick: function() { setTab(i); }, style: pill(tab === i) }, t); })
    ),
    tab === 0 && React.createElement("div", null,
      React.createElement("div", { style: card },
        React.createElement("div", { style: { fontSize: 17, fontWeight: 600, marginBottom: 12 } }, "Patient information"),
        React.createElement("div", { style: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 } },
          Object.entries(FEATURES).map(function(entry) { var k = entry[0]; var f = entry[1]; return React.createElement("div", { key: k },
            React.createElement("div", { style: { display: "flex", justifyContent: "space-between", marginBottom: 4 } },
              React.createElement("label", { style: { fontSize: 12, color: "#888" } }, f.label),
              React.createElement("span", { style: { fontSize: 12, fontWeight: 600, color: "#534AB7" } }, sliders[k] + " " + f.unit)
            ),
            React.createElement("input", { type: "range", min: f.min, max: f.max, step: f.step, value: sliders[k], onChange: function(e) { var obj = {}; obj[k] = Number(e.target.value); setSliders(Object.assign({}, sliders, obj)); }, style: { width: "100%", accentColor: "#534AB7" } })
          ); }),
          Object.entries(SELECTS).map(function(entry) { var k = entry[0]; var v = entry[1]; return React.createElement("div", { key: k },
            React.createElement("label", { style: { fontSize: 12, color: "#888", display: "block", marginBottom: 4 } }, v.label),
            React.createElement("select", { value: selects[k], onChange: function(e) { var obj = {}; obj[k] = e.target.value; setSelects(Object.assign({}, selects, obj)); }, style: { width: "100%", padding: "8px 10px", borderRadius: 8, border: "1px solid #ddd", fontSize: 13, fontFamily: "inherit", background: "#fff" } },
              v.opts.map(function(o) { return React.createElement("option", { key: o[0], value: o[0] }, o[1]); })
            )
          ); })
        )
      ),
      React.createElement("div", { style: Object.assign({}, card, { textAlign: "center", borderColor: riskColor + "33" }) },
        React.createElement("div", { style: { fontSize: 11, color: "#888", textTransform: "uppercase", letterSpacing: 1 } }, "Estimated stroke risk"),
        React.createElement("div", { style: { fontSize: 52, fontWeight: 700, color: riskColor, margin: "6px 0" } }, pct + "%"),
        React.createElement("span", { style: { display: "inline-block", padding: "4px 16px", borderRadius: 20, background: riskColor + "15", color: riskColor, fontSize: 13, fontWeight: 600 } }, riskLevel + " risk"),
        React.createElement("div", { style: { height: 8, borderRadius: 4, background: "#eee", margin: "18px 0 0", overflow: "hidden" } },
          React.createElement("div", { style: { height: "100%", width: pct + "%", background: riskColor, borderRadius: 4, transition: "width 0.5s ease" } })
        )
      ),
      React.createElement("div", { style: card },
        React.createElement("div", { style: { fontSize: 17, fontWeight: 600, marginBottom: 12 } }, "Individual SHAP contributions"),
        React.createElement("p", { style: { fontSize: 12, color: "#999", marginBottom: 12 } }, "How each feature pushes this prediction toward or away from stroke"),
        sortedC.map(function(entry) { var k = entry[0]; var v = entry[1]; return React.createElement("div", { key: k, style: { display: "flex", alignItems: "center", gap: 8, marginBottom: 7, fontSize: 13 } },
          React.createElement("span", { style: { width: 130, textAlign: "right", color: "#888", flexShrink: 0 } }, k.replace(/_/g, " ")),
          React.createElement("div", { style: { flex: 1 } }, React.createElement(HBar, { value: v, max: maxC, color: v > 0 ? "#D94040" : "#1D9E75" })),
          React.createElement("span", { style: { width: 55, textAlign: "right", fontSize: 12, color: v > 0 ? "#D94040" : "#1D9E75", fontWeight: 500, flexShrink: 0 } }, (v > 0 ? "+" : "") + v.toFixed(3))
        ); }),
        React.createElement("div", { style: { fontSize: 12, color: "#999", marginTop: 10 } }, "Red = increases risk. Green = decreases risk")
      ),
      React.createElement("div", { style: note("#f0f4ff", "#534AB7") },
        React.createElement("strong", null, "Clinical note: "), "This model is optimized for high sensitivity (recall = 80%) to minimize missed stroke cases. Not a diagnostic instrument."
      )
    ),
    tab === 1 && React.createElement("div", null,
      React.createElement("div", { style: { display: "flex", gap: 12, marginBottom: 20, flexWrap: "wrap" } },
        React.createElement(Metric, { label: "Best model", value: "Logistic Reg.", sub: "Balanced class weights" }),
        React.createElement(Metric, { label: "AUC-ROC", value: "0.838", sub: "Area under ROC curve" }),
        React.createElement(Metric, { label: "Recall", value: "80.0%", sub: "Sensitivity" }),
        React.createElement(Metric, { label: "Patients", value: "5,110", sub: "Training dataset" })
      ),
      React.createElement("div", { style: card },
        React.createElement("div", { style: { fontSize: 17, fontWeight: 600, marginBottom: 12 } }, "Model performance comparison"),
        React.createElement("table", { style: { width: "100%", borderCollapse: "collapse", fontSize: 13 } },
          React.createElement("thead", null,
            React.createElement("tr", { style: { borderBottom: "2px solid #eee" } },
              ["Model", "AUC", "Recall", "Precision", "F1"].map(function(h) { return React.createElement("th", { key: h, style: { padding: "8px 10px", textAlign: h === "Model" ? "left" : "center", color: "#888", fontWeight: 500, fontSize: 11, textTransform: "uppercase" } }, h); })
            )
          ),
          React.createElement("tbody", null,
            MODELS.map(function(m) { return React.createElement("tr", { key: m.name, style: { borderBottom: "1px solid #f0f0f0", background: m.best ? "#f8f7ff" : "transparent" } },
              React.createElement("td", { style: { padding: 10, fontWeight: m.best ? 600 : 400, color: m.best ? "#534AB7" : "#2d2a4a" } }, m.name, m.best ? React.createElement("span", { style: { fontSize: 10, background: "#534AB7", color: "#fff", padding: "2px 8px", borderRadius: 10, marginLeft: 6 } }, "BEST") : null),
              React.createElement("td", { style: { textAlign: "center", padding: 10 } }, m.auc.toFixed(3)),
              React.createElement("td", { style: { textAlign: "center", padding: 10 } }, React.createElement("span", { style: { color: m.recall >= 0.7 ? "#1D9E75" : m.recall >= 0.5 ? "#D4A017" : "#D94040", fontWeight: 500 } }, (m.recall * 100).toFixed(1) + "%")),
              React.createElement("td", { style: { textAlign: "center", padding: 10 } }, (m.precision * 100).toFixed(1) + "%"),
              React.createElement("td", { style: { textAlign: "center", padding: 10 } }, m.f1.toFixed(3))
            ); })
          )
        )
      ),
      React.createElement("div", { style: note("#fff8f0", "#D4A017") },
        React.createElement("strong", null, "Why recall matters most: "), "In stroke screening, a false negative (missing a stroke patient) is far more dangerous than a false positive. A naive model achieves 95.1% accuracy but 0% recall."
      ),
      React.createElement("div", { style: { height: 12 } }),
      React.createElement("div", { style: note("#f0fff5", "#1D9E75") },
        React.createElement("strong", null, "Class imbalance: "), "Only 4.87% of patients had strokes (249 of 5,110). We used SMOTE oversampling on training data only and cost-sensitive learning."
      )
    ),
    tab === 2 && React.createElement("div", null,
      React.createElement("div", { style: card },
        React.createElement("div", { style: { fontSize: 17, fontWeight: 600, marginBottom: 12 } }, "Global SHAP feature importance"),
        React.createElement("p", { style: { fontSize: 12, color: "#999", marginBottom: 14 } }, "Mean |SHAP value| across all patients"),
        globalSorted.map(function(entry) { var k = entry[0]; var v = entry[1]; return React.createElement("div", { key: k, style: { display: "flex", alignItems: "center", gap: 8, marginBottom: 8, fontSize: 13 } },
          React.createElement("span", { style: { width: 130, textAlign: "right", color: "#888", flexShrink: 0 } }, k.replace(/_/g, " ")),
          React.createElement("div", { style: { flex: 1 } }, React.createElement(HBar, { value: v, max: maxG, color: "#7F77DD" })),
          React.createElement("span", { style: { width: 50, textAlign: "right", fontSize: 12, color: "#666", flexShrink: 0 } }, v.toFixed(3))
        ); })
      ),
      React.createElement("div", { style: card },
        React.createElement("div", { style: { fontSize: 17, fontWeight: 600, marginBottom: 12 } }, "Neuroscience interpretation"),
        NEURO_FACTORS.map(function(rf, i) { return React.createElement("div", { key: i, style: { marginBottom: 8 } },
          React.createElement("button", { onClick: function() { setExpandedFactor(expandedFactor === i ? null : i); }, style: { width: "100%", textAlign: "left", padding: "12px 16px", border: "1px solid " + (expandedFactor === i ? "#534AB7" : "#eee"), borderRadius: expandedFactor === i ? "10px 10px 0 0" : 10, background: expandedFactor === i ? "#f8f7ff" : "#fff", cursor: "pointer", fontFamily: "inherit", fontSize: 13, display: "flex", justifyContent: "space-between", alignItems: "center" } },
            React.createElement("span", null, React.createElement("span", { style: { fontWeight: 600 } }, rf.feature), React.createElement("span", { style: { color: "#999", marginLeft: 10, fontSize: 12 } }, "SHAP: " + rf.shap.toFixed(3))),
            React.createElement("span", { style: { color: "#534AB7", fontSize: 18 } }, expandedFactor === i ? "-" : "+")
          ),
          expandedFactor === i && React.createElement("div", { style: { padding: "14px 16px", fontSize: 13, lineHeight: 1.8, color: "#555", background: "#faf9ff", borderRadius: "0 0 10px 10px", border: "1px solid #534AB7", borderTop: "none" } }, rf.detail)
        ); })
      ),
      React.createElement("div", { style: note("#f0f4ff", "#534AB7") },
        React.createElement("strong", null, "Research insight: "), "SHAP values confirm what neurovascular literature has established. Age-related cerebrovascular degeneration, hypertension-driven arterial damage, and glucose-mediated endothelial dysfunction are the dominant pathways."
      )
    ),
    tab === 3 && React.createElement("div", null,
      React.createElement("div", { style: card },
        React.createElement("div", { style: { fontSize: 17, fontWeight: 600, marginBottom: 12 } }, "Machine learning pipeline"),
        STEPS.map(function(step, i) { return React.createElement("div", { key: i, style: { display: "flex", gap: 16, marginBottom: 18, alignItems: "flex-start" } },
          React.createElement("div", { style: { width: 32, height: 32, borderRadius: "50%", background: "#f0eeff", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 13, fontWeight: 600, color: "#534AB7", flexShrink: 0 } }, i + 1),
          React.createElement("div", { style: { flex: 1 } },
            React.createElement("div", { style: { fontSize: 14, fontWeight: 600, marginBottom: 4 } }, step.title),
            React.createElement("div", { style: { fontSize: 13, color: "#666", lineHeight: 1.7 } }, step.desc)
          )
        ); })
      ),
      React.createElement("div", { style: card },
        React.createElement("div", { style: { fontSize: 17, fontWeight: 600, marginBottom: 12 } }, "Tools and libraries"),
        React.createElement("div", { style: { display: "flex", flexWrap: "wrap", gap: 8 } },
          ["Python", "scikit-learn", "XGBoost", "SHAP", "SMOTE", "pandas", "matplotlib", "seaborn", "React"].map(function(t) { return React.createElement("span", { key: t, style: { padding: "4px 12px", background: "#f0eeff", borderRadius: 20, fontSize: 12, color: "#534AB7", fontWeight: 500 } }, t); })
        )
      )
    ),
    tab === 4 && React.createElement("div", null,
      React.createElement("div", { style: card },
        React.createElement("div", { style: { fontSize: 17, fontWeight: 600, marginBottom: 12 } }, "About this project"),
        React.createElement("p", { style: { fontSize: 14, lineHeight: 1.8, color: "#555", marginBottom: 16 } }, "This project develops a machine learning-based screening tool for stroke risk prediction, with a focus on clinical applicability and explainability. Unlike typical ML classification projects that optimize for accuracy, this tool is designed around the asymmetric cost structure of medical screening."),
        React.createElement("div", { style: { fontSize: 14, fontWeight: 600, marginBottom: 8 } }, "Key findings"),
        React.createElement("ul", { style: { fontSize: 13, color: "#666", lineHeight: 2, paddingLeft: 20, marginBottom: 16 } },
          React.createElement("li", null, "Logistic Regression achieves 80% recall and 0.838 AUC-ROC"),
          React.createElement("li", null, "Age is the dominant predictor (SHAP: 1.713), consistent with cerebrovascular pathophysiology"),
          React.createElement("li", null, "Hypertension and glucose are the strongest modifiable risk factors"),
          React.createElement("li", null, "A naive 'predict no stroke' baseline achieves 95.1% accuracy but 0% clinical utility")
        ),
        React.createElement("div", { style: { fontSize: 14, fontWeight: 600, marginBottom: 8 } }, "Limitations"),
        React.createElement("ul", { style: { fontSize: 13, color: "#666", lineHeight: 2, paddingLeft: 20 } },
          React.createElement("li", null, "Cross-sectional dataset with no temporal modeling"),
          React.createElement("li", null, "Missing features: atrial fibrillation, cholesterol, family history"),
          React.createElement("li", null, "SMOTE may introduce synthetic artifacts"),
          React.createElement("li", null, "Single-source data limits generalizability")
        )
      ),
      React.createElement("div", { style: card },
        React.createElement("div", { style: { fontSize: 17, fontWeight: 600, marginBottom: 12 } }, "Author"),
        React.createElement("p", { style: { fontSize: 14, color: "#666", lineHeight: 1.7 } }, "Jahnavi Nimmakayala"),
        React.createElement("div", { style: { marginTop: 12 } },
          React.createElement("a", { href: "https://github.com/jahnavin123/stroke-risk-predictor", target: "_blank", rel: "noreferrer", style: { padding: "6px 14px", background: "#2d2a4a", color: "#fff", borderRadius: 8, fontSize: 12, textDecoration: "none", fontWeight: 500 } }, "GitHub Repository")
        )
      ),
      React.createElement("div", { style: note("#fff0f0", "#D94040") },
        React.createElement("strong", null, "Disclaimer: "), "This tool is for educational and research purposes only. Not a medical device."
      )
    ),
    React.createElement("footer", { style: { textAlign: "center", padding: "28px 0 12px", fontSize: 11, color: "#ccc" } }, "Built with scikit-learn, XGBoost, SHAP, React")
  );
}
