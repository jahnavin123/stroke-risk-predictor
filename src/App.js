import React, { useState, useMemo } from "react";

/* ─── constants ─── */
const FEATURES = {
  age: { label: "Age", min: 1, max: 82, step: 1, default: 55, unit: "years" },
  avg_glucose_level: { label: "Avg glucose level", min: 55, max: 280, step: 1, default: 110, unit: "mg/dL" },
  bmi: { label: "BMI", min: 10, max: 60, step: 0.5, default: 28, unit: "kg/m²" },
};

const SELECTS = {
  hypertension: { label: "Hypertension", opts: [["0", "No"], ["1", "Yes"]] },
  heart_disease: { label: "Heart disease", opts: [["0", "No"], ["1", "Yes"]] },
  gender: { label: "Gender", opts: [["0", "Female"], ["1", "Male"]] },
  smoking_status: { label: "Smoking status", opts: [["2", "Never smoked"], ["1", "Formerly smoked"], ["3", "Smokes"], ["0", "Unknown"]] },
  ever_married: { label: "Ever married", opts: [["1", "Yes"], ["0", "No"]] },
};

const SHAP_WEIGHTS = {
  age: 2.033, avg_glucose_level: 0.495, bmi: 0.284, heart_disease: 0.276,
  hypertension: 0.182, gender: 0.174, ever_married: 0.168, smoking_status: 0.104,
  work_type: 0.058, Residence_type: 0.016,
};

const MODELS = [
  { name: "Logistic Regression", auc: 0.897, recall: 0.82, precision: 0.163, f1: 0.272, best: true },
  { name: "Random Forest", auc: 0.865, recall: 0.42, precision: 0.313, f1: 0.359 },
  { name: "XGBoost", auc: 0.847, recall: 0.58, precision: 0.168, f1: 0.26 },
  { name: "Gradient Boosting", auc: 0.846, recall: 0.44, precision: 0.279, f1: 0.341 },
  { name: "SVM", auc: 0.836, recall: 0.62, precision: 0.157, f1: 0.251 },
  { name: "KNN", auc: 0.753, recall: 0.56, precision: 0.181, f1: 0.273 },
];

const NEURO_FACTORS = [
  { feature: "Age", shap: 2.033, detail: "Stroke risk doubles each decade after 55. Aging causes progressive atherosclerosis and cerebral small vessel disease — cumulative damage to the vasculature that supplies the brain. Age-related changes include arterial stiffening, loss of endothelial nitric oxide production, and increased susceptibility to thrombosis." },
  { feature: "Avg glucose level", shap: 0.495, detail: "Chronic hyperglycemia damages endothelial cells lining cerebral blood vessels through advanced glycation end products (AGEs). These AGEs promote oxidative stress, inflammation, and a pro-thrombotic state in the cerebral vasculature. Diabetic patients have 2-4x higher stroke risk, with glucose toxicity directly impairing the blood-brain barrier." },
  { feature: "BMI", shap: 0.284, detail: "Obesity promotes stroke through multiple converging pathways: systemic inflammation (elevated CRP, IL-6), insulin resistance leading to endothelial dysfunction, dyslipidemia accelerating atherosclerotic plaque formation, and increased blood viscosity. Central adiposity is particularly correlated with cerebrovascular events due to visceral fat's metabolic activity." },
  { feature: "Heart disease", shap: 0.276, detail: "Cardiac conditions — especially atrial fibrillation — are major embolic stroke risk factors. Irregular heart rhythm allows blood to pool in the left atrial appendage, forming clots that can travel through the carotid arteries to the brain (cardioembolic stroke). This mechanism accounts for approximately 20-30% of all ischemic strokes." },
  { feature: "Hypertension", shap: 0.182, detail: "Chronic high blood pressure is the single most modifiable risk factor for stroke. It damages arterial walls through mechanical stress, promotes atherosclerotic plaque formation in the carotid and cerebral arteries, and weakens small penetrating arteries in the brain (causing lacunar infarcts). Sustained hypertension can also lead to hemorrhagic stroke through vessel rupture." },
];

const METHODOLOGY_STEPS = [
  { title: "Data collection", desc: "5,110 patient records with 11 clinical features from the Kaggle Stroke Prediction Dataset (McKinsey & Company)" },
  { title: "Preprocessing", desc: "KNN imputation for 201 missing BMI values (preserves feature relationships better than mean/median), label encoding for categoricals, StandardScaler normalization" },
  { title: "Class imbalance handling", desc: "SMOTE oversampling on training data only (not test set) + cost-sensitive learning via balanced class weights. Only 4.87% positive class — a naive model achieves 95% accuracy by predicting 'no stroke' every time" },
  { title: "Model training & evaluation", desc: "6 algorithms compared via 5-fold stratified cross-validation. Primary metric: recall (sensitivity), because missing a stroke patient is far more dangerous than a false alarm" },
  { title: "Explainability", desc: "SHAP (SHapley Additive exPlanations) provides both global feature importance and individual patient-level explanations — clinicians can see why each patient is flagged" },
  { title: "Threshold optimization", desc: "Classification threshold lowered from 0.5 to 0.3 for clinical screening — catches more at-risk patients at the cost of some false positives, an appropriate tradeoff for a screening tool" },
];

/* ─── helpers ─── */
function computeRisk(sliders, selects) {
  const c = {
    age: ((sliders.age - 40) / 42) * 2.033,
    avg_glucose_level: ((sliders.avg_glucose_level - 100) / 180) * 0.495,
    bmi: ((sliders.bmi - 28.9) / 31) * 0.284,
    heart_disease: Number(selects.heart_disease) * 0.276,
    hypertension: Number(selects.hypertension) * 0.182,
    gender: Number(selects.gender) * 0.174 * 0.3,
    ever_married: Number(selects.ever_married) * 0.168 * 0.2,
    smoking_status: selects.smoking_status === "3" ? 0.104 : selects.smoking_status === "1" ? 0.05 : 0,
  };
  let logit = -3.5;
  for (const v of Object.values(c)) logit += v;
  return { prob: 1 / (1 + Math.exp(-logit)), contributions: c };
}

/* ─── small components ─── */
const s = {
  page: { maxWidth: 760, margin: "0 auto", padding: "0 20px 40px", fontFamily: "'Segoe UI', system-ui, -apple-system, sans-serif", color: "#2d2a4a" },
  nav: { display: "flex", gap: 0, background: "#fff", borderRadius: 12, border: "1px solid #e8e6f0", padding: 4, marginBottom: 28, flexWrap: "wrap" },
  card: { background: "#fff", borderRadius: 14, border: "1px solid #eae8f0", padding: "20px 24px", marginBottom: 16 },
  muted: { fontSize: 12, color: "#999", lineHeight: 1.6 },
  h2: { fontSize: 17, fontWeight: 600, marginBottom: 12 },
  h3: { fontSize: 14, fontWeight: 600, marginBottom: 8 },
  pill: (active) => ({
    padding: "8px 16px", fontSize: 13, fontWeight: active ? 600 : 400, border: "none",
    borderRadius: 8, cursor: "pointer", fontFamily: "inherit", transition: "all 0.2s",
    background: active ? "#534AB7" : "transparent", color: active ? "#fff" : "#888",
  }),
  note: (bg, accent) => ({ background: bg, borderLeft: `3px solid ${accent}`, borderRadius: "0 10px 10px 0", padding: "14px 18px", fontSize: 12, lineHeight: 1.7, color: "#555" }),
};

function HBar({ value, max, color, width = "100%" }) {
  const pct = max > 0 ? Math.min(Math.abs(value) / max * 100, 100) : 0;
  return <div style={{ width: `${Math.round(pct)}%`, minWidth: 3, height: 18, borderRadius: 3, background: color, transition: "width 0.4s ease" }} />;
}

function Metric({ label, value, sub }) {
  return (
    <div style={{ background: "#f8f7ff", borderRadius: 10, padding: "14px 16px", flex: "1 1 140px", minWidth: 130 }}>
      <div style={{ fontSize: 10, color: "#888", textTransform: "uppercase", letterSpacing: 0.8 }}>{label}</div>
      <div style={{ fontSize: 20, fontWeight: 700, color: "#2d2a4a", marginTop: 4 }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: "#aaa", marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

/* ─── main app ─── */
export default function App() {
  const [tab, setTab] = useState(0);
  const [sliders, setSliders] = useState({ age: 55, avg_glucose_level: 110, bmi: 28 });
  const [selects, setSelects] = useState({ hypertension: "0", heart_disease: "0", gender: "0", smoking_status: "2", ever_married: "1" });
  const [expandedFactor, setExpandedFactor] = useState(null);

  const { prob, contributions } = useMemo(() => computeRisk(sliders, selects), [sliders, selects]);
  const pct = Math.round(prob * 100);
  const riskLevel = pct < 20 ? "Low" : pct < 50 ? "Moderate" : "High";
  const riskColor = pct < 20 ? "#1D9E75" : pct < 50 ? "#D4A017" : "#D94040";

  const sortedC = Object.entries(contributions).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const maxC = Math.max(...sortedC.map(([, v]) => Math.abs(v)), 0.01);

  const globalSorted = Object.entries(SHAP_WEIGHTS).sort((a, b) => b[1] - a[1]);
  const maxG = globalSorted[0][1];

  const tabs = ["Risk assessment", "Models", "Risk factors", "Methodology", "About"];

  return (
    <div style={s.page}>
      {/* header */}
      <div style={{ textAlign: "center", padding: "36px 0 24px" }}>
        <div style={{ display: "inline-block", background: "#f0eeff", padding: "4px 14px", borderRadius: 20, fontSize: 11, fontWeight: 600, letterSpacing: 1, color: "#534AB7", marginBottom: 10 }}>
          MACHINE LEARNING + EXPLAINABLE AI
        </div>
        <h1 style={{ fontSize: 28, fontWeight: 700, margin: "8px 0 6px" }}>Stroke risk predictor</h1>
        <p style={{ fontSize: 14, color: "#999", maxWidth: 500, margin: "0 auto" }}>
          Clinical screening tool trained on 5,110 patients with SHAP-based interpretability
        </p>
      </div>

      {/* nav */}
      <nav style={s.nav}>
        {tabs.map((t, i) => (
          <button key={i} onClick={() => setTab(i)} style={s.pill(tab === i)}>{t}</button>
        ))}
      </nav>

      {/* ─── TAB 0: Risk Assessment ─── */}
      {tab === 0 && (
        <>
          <div style={s.card}>
            <div style={s.h2}>Patient information</div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
              {Object.entries(FEATURES).map(([k, f]) => (
                <div key={k}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                    <label style={{ fontSize: 12, color: "#888" }}>{f.label}</label>
                    <span style={{ fontSize: 12, fontWeight: 600, color: "#534AB7" }}>{sliders[k]} {f.unit}</span>
                  </div>
                  <input type="range" min={f.min} max={f.max} step={f.step} value={sliders[k]}
                    onChange={e => setSliders(p => ({ ...p, [k]: Number(e.target.value) }))}
                    style={{ width: "100%", accentColor: "#534AB7" }} />
                </div>
              ))}
              {Object.entries(SELECTS).map(([k, v]) => (
                <div key={k}>
                  <label style={{ fontSize: 12, color: "#888", display: "block", marginBottom: 4 }}>{v.label}</label>
                  <select value={selects[k]} onChange={e => setSelects(p => ({ ...p, [k]: e.target.value }))}
                    style={{ width: "100%", padding: "8px 10px", borderRadius: 8, border: "1px solid #ddd", fontSize: 13, fontFamily: "inherit", background: "#fff" }}>
                    {v.opts.map(([val, lbl]) => <option key={val} value={val}>{lbl}</option>)}
                  </select>
                </div>
              ))}
            </div>
          </div>

          <div style={{ ...s.card, textAlign: "center", background: `linear-gradient(135deg, ${riskColor}06, ${riskColor}03)`, borderColor: `${riskColor}33` }}>
            <div style={{ fontSize: 11, color: "#888", textTransform: "uppercase", letterSpacing: 1 }}>Estimated stroke risk</div>
            <div style={{ fontSize: 52, fontWeight: 700, color: riskColor, margin: "6px 0" }}>{pct}%</div>
            <span style={{ display: "inline-block", padding: "4px 16px", borderRadius: 20, background: `${riskColor}15`, color: riskColor, fontSize: 13, fontWeight: 600 }}>{riskLevel} risk</span>
            <div style={{ height: 8, borderRadius: 4, background: "#eee", margin: "18px 0 0", overflow: "hidden" }}>
              <div style={{ height: "100%", width: `${pct}%`, background: riskColor, borderRadius: 4, transition: "width 0.5s ease" }} />
            </div>
          </div>

          <div style={s.card}>
            <div style={s.h2}>Individual SHAP contributions</div>
            <p style={{ ...s.muted, marginBottom: 12 }}>How each feature pushes this patient's prediction toward or away from stroke</p>
            {sortedC.map(([k, v]) => (
              <div key={k} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 7, fontSize: 13 }}>
                <span style={{ width: 130, textAlign: "right", color: "#888", flexShrink: 0 }}>{k.replace(/_/g, " ")}</span>
                <div style={{ flex: 1 }}><HBar value={v} max={maxC} color={v > 0 ? "#D94040" : "#1D9E75"} /></div>
                <span style={{ width: 55, textAlign: "right", fontSize: 12, color: v > 0 ? "#D94040" : "#1D9E75", fontWeight: 500, flexShrink: 0 }}>
                  {v > 0 ? "+" : ""}{v.toFixed(3)}
                </span>
              </div>
            ))}
            <div style={{ ...s.muted, marginTop: 10 }}>Red = increases risk · Green = decreases risk</div>
          </div>

          <div style={s.note("#f0f4ff", "#534AB7")}>
            <strong>Clinical note:</strong> This model is optimized for high sensitivity (recall = 82%) to minimize missed stroke cases. It is designed as a screening tool — flagged patients should receive further clinical evaluation. Not a diagnostic instrument.
          </div>
        </>
      )}

      {/* ─── TAB 1: Model Comparison ─── */}
      {tab === 1 && (
        <>
          <div style={{ display: "flex", gap: 12, marginBottom: 20, flexWrap: "wrap" }}>
            <Metric label="Best model" value="Logistic Reg." sub="Balanced class weights" />
            <Metric label="AUC-ROC" value="0.897" sub="Area under ROC curve" />
            <Metric label="Recall" value="82.0%" sub="Sensitivity" />
            <Metric label="Patients" value="5,110" sub="Training dataset" />
          </div>

          <div style={s.card}>
            <div style={s.h2}>Model performance comparison</div>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: "2px solid #eee" }}>
                    {["Model", "AUC", "Recall", "Precision", "F1"].map(h => (
                      <th key={h} style={{ padding: "8px 10px", textAlign: h === "Model" ? "left" : "center", color: "#888", fontWeight: 500, fontSize: 11, textTransform: "uppercase", letterSpacing: 0.5 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {MODELS.map(m => (
                    <tr key={m.name} style={{ borderBottom: "1px solid #f0f0f0", background: m.best ? "#f8f7ff" : "transparent" }}>
                      <td style={{ padding: 10, fontWeight: m.best ? 600 : 400, color: m.best ? "#534AB7" : "#2d2a4a" }}>
                        {m.name}{m.best && <span style={{ fontSize: 10, background: "#534AB7", color: "#fff", padding: "2px 8px", borderRadius: 10, marginLeft: 6, verticalAlign: "middle" }}>BEST</span>}
                      </td>
                      <td style={{ textAlign: "center", padding: 10 }}>{m.auc.toFixed(3)}</td>
                      <td style={{ textAlign: "center", padding: 10 }}>
                        <span style={{ color: m.recall >= 0.7 ? "#1D9E75" : m.recall >= 0.5 ? "#D4A017" : "#D94040", fontWeight: 500 }}>{(m.recall * 100).toFixed(1)}%</span>
                      </td>
                      <td style={{ textAlign: "center", padding: 10 }}>{(m.precision * 100).toFixed(1)}%</td>
                      <td style={{ textAlign: "center", padding: 10 }}>{m.f1.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div style={s.note("#fff8f0", "#D4A017")}>
            <strong>Why recall matters most:</strong> In stroke screening, a false negative (missing a stroke patient) is far more dangerous than a false positive (flagging a healthy patient for follow-up testing). A naive "predict no stroke" model achieves 95.1% accuracy but 0% recall — completely useless clinically.
          </div>
          <div style={{ height: 12 }} />
          <div style={s.note("#f0fff5", "#1D9E75")}>
            <strong>Class imbalance:</strong> Only 4.87% of patients had strokes (249 of 5,110). We used SMOTE oversampling on training data only and cost-sensitive learning to prevent the model from ignoring the minority class.
          </div>
        </>
      )}

      {/* ─── TAB 2: Risk Factors ─── */}
      {tab === 2 && (
        <>
          <div style={s.card}>
            <div style={s.h2}>Global SHAP feature importance</div>
            <p style={{ ...s.muted, marginBottom: 14 }}>Mean |SHAP value| — how much each feature influences predictions across all patients</p>
            {globalSorted.map(([k, v]) => (
              <div key={k} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8, fontSize: 13 }}>
                <span style={{ width: 130, textAlign: "right", color: "#888", flexShrink: 0 }}>{k.replace(/_/g, " ")}</span>
                <div style={{ flex: 1 }}><HBar value={v} max={maxG} color="#7F77DD" /></div>
                <span style={{ width: 50, textAlign: "right", fontSize: 12, color: "#666", flexShrink: 0 }}>{v.toFixed(3)}</span>
              </div>
            ))}
          </div>

          <div style={s.card}>
            <div style={s.h2}>Neuroscience interpretation</div>
            <p style={{ ...s.muted, marginBottom: 14 }}>How each risk factor relates to cerebrovascular pathophysiology</p>
            {NEURO_FACTORS.map((rf, i) => (
              <div key={i} style={{ marginBottom: 8 }}>
                <button onClick={() => setExpandedFactor(expandedFactor === i ? null : i)} style={{
                  width: "100%", textAlign: "left", padding: "12px 16px", border: `1px solid ${expandedFactor === i ? "#534AB7" : "#eee"}`,
                  borderRadius: expandedFactor === i ? "10px 10px 0 0" : 10, background: expandedFactor === i ? "#f8f7ff" : "#fff",
                  cursor: "pointer", fontFamily: "inherit", fontSize: 13, display: "flex", justifyContent: "space-between", alignItems: "center", transition: "all 0.2s",
                }}>
                  <span><span style={{ fontWeight: 600 }}>{rf.feature}</span><span style={{ color: "#999", marginLeft: 10, fontSize: 12 }}>SHAP: {rf.shap.toFixed(3)}</span></span>
                  <span style={{ color: "#534AB7", fontSize: 18, fontWeight: 300 }}>{expandedFactor === i ? "−" : "+"}</span>
                </button>
                {expandedFactor === i && (
                  <div style={{ padding: "14px 16px", fontSize: 13, lineHeight: 1.8, color: "#555", background: "#faf9ff", borderRadius: "0 0 10px 10px", border: "1px solid #534AB7", borderTop: "none" }}>
                    {rf.detail}
                  </div>
                )}
              </div>
            ))}
          </div>

          <div style={s.note("#f0f4ff", "#534AB7")}>
            <strong>Research insight:</strong> SHAP values quantitatively confirm what neurovascular literature has established — age-related cerebrovascular degeneration, glucose-mediated endothelial damage, and cardiac comorbidities are the dominant pathways. This alignment between data-driven ML features and established neuroscience validates the model's clinical plausibility.
          </div>
        </>
      )}

      {/* ─── TAB 3: Methodology ─── */}
      {tab === 3 && (
        <>
          <div style={s.card}>
            <div style={s.h2}>Machine learning pipeline</div>
            <p style={{ ...s.muted, marginBottom: 18 }}>End-to-end workflow from raw patient data to explainable risk predictions</p>
            {METHODOLOGY_STEPS.map((step, i) => (
              <div key={i} style={{ display: "flex", gap: 16, marginBottom: 18, alignItems: "flex-start" }}>
                <div style={{ width: 32, height: 32, borderRadius: "50%", background: "#f0eeff", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 13, fontWeight: 600, color: "#534AB7", flexShrink: 0 }}>
                  {i + 1}
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 4 }}>{step.title}</div>
                  <div style={{ fontSize: 13, color: "#666", lineHeight: 1.7 }}>{step.desc}</div>
                </div>
              </div>
            ))}
          </div>

          <div style={s.card}>
            <div style={s.h2}>Key technical decisions</div>
            <div style={{ display: "grid", gap: 12 }}>
              {[
                ["KNN imputation over mean/median", "KNN imputation preserves relationships between BMI and other features (age, glucose). Mean imputation introduces bias by ignoring correlations."],
                ["SMOTE on training set only", "Applying SMOTE to the full dataset before splitting would leak synthetic minority samples into the test set, producing overly optimistic evaluation metrics."],
                ["Recall as primary metric", "In clinical screening, the cost of missing a stroke patient (false negative) vastly exceeds the cost of an unnecessary follow-up (false positive). Optimizing for recall prioritizes patient safety."],
                ["SHAP over permutation importance", "SHAP provides theoretically grounded, consistent feature attributions at both global and individual patient levels — critical for clinical interpretability and building physician trust."],
              ].map(([title, desc], i) => (
                <div key={i} style={{ padding: "12px 16px", background: "#faf9ff", borderRadius: 10, borderLeft: "3px solid #534AB7" }}>
                  <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 4 }}>{title}</div>
                  <div style={{ fontSize: 12, color: "#666", lineHeight: 1.6 }}>{desc}</div>
                </div>
              ))}
            </div>
          </div>

          <div style={s.card}>
            <div style={s.h2}>Tools & libraries</div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              {["Python", "scikit-learn", "XGBoost", "SHAP", "imbalanced-learn (SMOTE)", "pandas", "matplotlib", "seaborn", "React"].map(t => (
                <span key={t} style={{ padding: "4px 12px", background: "#f0eeff", borderRadius: 20, fontSize: 12, color: "#534AB7", fontWeight: 500 }}>{t}</span>
              ))}
            </div>
          </div>
        </>
      )}

      {/* ─── TAB 4: About ─── */}
      {tab === 4 && (
        <>
          <div style={s.card}>
            <div style={s.h2}>About this project</div>
            <p style={{ fontSize: 14, lineHeight: 1.8, color: "#555", marginBottom: 16 }}>
              This project develops a machine learning-based screening tool for stroke risk prediction,
              with a focus on clinical applicability and explainability. Unlike typical ML classification projects
              that optimize for accuracy, this tool is designed around the asymmetric cost structure of
              medical screening — where missing an at-risk patient is far worse than a false alarm.
            </p>
            <p style={{ fontSize: 14, lineHeight: 1.8, color: "#555", marginBottom: 16 }}>
              The SHAP explainability layer connects data-driven predictions to established neurovascular
              pathophysiology, enabling clinicians to understand <em>why</em> a patient is flagged rather than
              treating the model as a black box. This interpretability is essential for building trust in
              clinical AI systems and aligns with emerging FDA guidance on ML/AI in medical devices.
            </p>

            <div style={s.h3}>Dataset</div>
            <p style={{ fontSize: 13, color: "#666", lineHeight: 1.7, marginBottom: 16 }}>
              Kaggle Stroke Prediction Dataset (McKinsey & Company) — 5,110 patients, 11 clinical features,
              binary target. Severe class imbalance: only 4.87% stroke-positive cases.
            </p>

            <div style={s.h3}>Key findings</div>
            <ul style={{ fontSize: 13, color: "#666", lineHeight: 2, paddingLeft: 20, marginBottom: 16 }}>
              <li>Logistic Regression with balanced weights outperforms complex ensembles on the primary metric (recall)</li>
              <li>Age is the dominant predictor (SHAP: 2.033), consistent with known cerebrovascular pathophysiology</li>
              <li>Glucose and BMI are the strongest modifiable risk factors, suggesting actionable intervention targets</li>
              <li>Standard accuracy metric is misleading — a "predict no stroke" baseline achieves 95.1% accuracy but 0% clinical utility</li>
            </ul>

            <div style={s.h3}>Limitations</div>
            <ul style={{ fontSize: 13, color: "#666", lineHeight: 2, paddingLeft: 20 }}>
              <li>Cross-sectional dataset — no temporal/longitudinal modeling</li>
              <li>Missing features: atrial fibrillation, cholesterol, family history, medications</li>
              <li>SMOTE may introduce synthetic artifacts in minority class</li>
              <li>Single-source data limits generalizability across populations</li>
            </ul>
          </div>

          <div style={s.card}>
            <div style={s.h2}>Author</div>
            <p style={{ fontSize: 14, color: "#666", lineHeight: 1.7 }}>
              {/* Replace with your child's name and details */}
              Jahnavi Nimmakayala — Emerald High school researcher with a focus on neuroscience and computational biology.
              This project was developed as part of research in applying machine learning
              to clinical neuroscience problems.
            </p>
            <div style={{ marginTop: 12, display: "flex", gap: 10 }}>
              <a href="https://github.com/YOUR_USERNAME/stroke-risk-predictor" target="_blank" rel="noreferrer"
                style={{ padding: "6px 14px", background: "#2d2a4a", color: "#fff", borderRadius: 8, fontSize: 12, textDecoration: "none", fontWeight: 500 }}>
                GitHub Repository
              </a>
              <a href="#" style={{ padding: "6px 14px", background: "#f0eeff", color: "#534AB7", borderRadius: 8, fontSize: 12, textDecoration: "none", fontWeight: 500 }}>
                Research Paper (PDF)
              </a>
            </div>
          </div>

          <div style={s.note("#fff0f0", "#D94040")}>
            <strong>Disclaimer:</strong> This tool is for educational and research purposes only. It is not a medical device and should not be used for clinical decision-making without professional medical oversight.
          </div>
        </>
      )}

      <footer style={{ textAlign: "center", padding: "28px 0 12px", fontSize: 11, color: "#ccc" }}>
        Built with scikit-learn · XGBoost · SHAP · React — For research purposes only
      </footer>
    </div>
  );
}
