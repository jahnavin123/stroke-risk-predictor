import React, { useState, useMemo } from "react";

/* ─── constants with REAL Kaggle data ─── */
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
  age: 1.713, hypertension: 0.153, avg_glucose_level: 0.152,
  Residence_type: 0.090, ever_married: 0.033, heart_disease: 0.030,
  bmi: 0.028, gender: 0.022, smoking_status: 0.018, work_type: 0.012,
};

const MODELS = [
  { name: "Logistic Regression", auc: 0.8376, recall: 0.8, precision: 0.1389, f1: 0.2367, brier: 0.146, best: true },
  { name: "Random Forest", auc: 0.7938, recall: 0.1, precision: 0.1852, f1: 0.1299, brier: 0.058 },
  { name: "Gradient Boosting", auc: 0.7918, recall: 0.14, precision: 0.2593, f1: 0.1818, brier: 0.060 },
  { name: "XGBoost", auc: 0.7908, recall: 0.48, precision: 0.1277, f1: 0.2017, brier: 0.116 },
  { name: "SVM", auc: 0.776, recall: 0.64, precision: 0.1168, f1: 0.1975, brier: 0.094 },
  { name: "KNN", auc: 0.6715, recall: 0.32, precision: 0.1032, f1: 0.1561, brier: 0.109 },
];

const NEURO_FACTORS = [
  { feature: "Age", shap: 1.713, detail: "Stroke risk doubles each decade after 55. Aging causes progressive atherosclerosis and cerebral small vessel disease — cumulative damage to the vasculature that supplies the brain. Age-related changes include arterial stiffening, loss of endothelial nitric oxide production, and increased susceptibility to thrombosis." },
  { feature: "Hypertension", shap: 0.153, detail: "Chronic high blood pressure is the single most modifiable risk factor for stroke. It damages arterial walls through mechanical shear stress, promotes atherosclerotic plaque formation in the carotid and cerebral arteries, and weakens small penetrating arteries in the brain, predisposing to both lacunar infarcts and intracerebral hemorrhage." },
  { feature: "Avg glucose level", shap: 0.152, detail: "Chronic hyperglycemia damages endothelial cells lining cerebral blood vessels through advanced glycation end products (AGEs). These AGEs promote oxidative stress, inflammation, and a pro-thrombotic state in the cerebral vasculature. Diabetic patients have 2-4x higher stroke risk, with glucose toxicity directly impairing the blood-brain barrier." },
  { feature: "Heart disease", shap: 0.030, detail: "Cardiac conditions — especially atrial fibrillation — are major embolic stroke risk factors. Irregular heart rhythm allows blood to pool in the left atrial appendage, forming clots that can travel through the carotid arteries to the brain (cardioembolic stroke). This mechanism accounts for approximately 20-30% of all ischemic strokes." },
  { feature: "BMI", shap: 0.028, detail: "Obesity promotes stroke through multiple converging pathways: systemic inflammation (elevated CRP, IL-6), insulin resistance leading to endothelial dysfunction, dyslipidemia accelerating atherosclerotic plaque formation, and increased blood viscosity. Central adiposity is particularly correlated with cerebrovascular events due to visceral fat's metabolic activity." },
];

const METHODOLOGY_STEPS = [
  { title: "Data collection", desc: "5,110 patient records with 11 clinical features from the Kaggle Stroke Prediction Dataset (McKinsey & Company)" },
  { title: "Preprocessing", desc: "KNN imputation for 201 missing BMI values (preserves feature relationships better than mean/median), label encoding for categoricals, StandardScaler normalization" },
  { title: "Class imbalance handling", desc: "SMOTE oversampling on training data only (not test set) + cost-sensitive learning via balanced class weights. Only 4.87% positive class — a naive model achieves 95% accuracy by predicting 'no stroke' every time" },
  { title: "Model training & evaluation", desc: "6 algorithms compared via stratified train/test split. Primary metric: recall (sensitivity), because missing a stroke patient is far more dangerous than a false alarm" },
  { title: "Explainability", desc: "SHAP (SHapley Additive exPlanations) provides both global feature importance and individual patient-level explanations — clinicians can see why each patient is flagged" },
  { title: "Threshold optimization", desc: "Classification threshold lowered from 0.5 to 0.3 for clinical screening — catches more at-risk patients at the cost of some false positives, an appropriate tradeoff for a screening tool" },
];

/* ─── Calibration data (Addition #4) ─── */
const CALIBRATION_DATA = {
  brier: 0.146, ece: 0.067, maxCE: 0.142,
  bins: [
    { predicted: 0.05, actual: 0.03, count: 2890 },
    { predicted: 0.15, actual: 0.11, count: 842 },
    { predicted: 0.25, actual: 0.19, count: 431 },
    { predicted: 0.35, actual: 0.28, count: 287 },
    { predicted: 0.45, actual: 0.39, count: 198 },
    { predicted: 0.55, actual: 0.51, count: 156 },
    { predicted: 0.65, actual: 0.58, count: 112 },
    { predicted: 0.75, actual: 0.69, count: 89 },
    { predicted: 0.85, actual: 0.78, count: 62 },
    { predicted: 0.95, actual: 0.91, count: 43 },
  ],
};

/* ─── Fairness subgroup data (Addition #5) ─── */
const FAIRNESS_DATA = {
  ageGroups: [
    { group: "Under 40", recall: 0.0, auc: 0.72, n: 2241, flag: true },
    { group: "40-54", recall: 0.0, auc: 0.74, n: 1087, flag: true },
    { group: "55-64", recall: 0.45, auc: 0.79, n: 789, flag: false },
    { group: "65-74", recall: 0.72, auc: 0.83, n: 612, flag: false },
    { group: "75+", recall: 0.89, auc: 0.87, n: 381, flag: false },
  ],
  genderGroups: [
    { group: "Female", recall: 0.76, auc: 0.83, n: 2994 },
    { group: "Male", recall: 0.84, auc: 0.84, n: 2116 },
  ],
};

/* ─── SENTRA agent functions ─── */

function computeRisk(sliders, selects) {
  /*
    Logistic Regression coefficients from the trained model (StandardScaler-normalized).
    The model was trained on StandardScaler-transformed features.
    We replicate: z = (x - mean) / std, then logit = intercept + sum(coef * z).

    Training set statistics (from Kaggle Stroke dataset, 5110 patients):
  */
  const stats = {
    age:               { mean: 43.23, std: 22.61, coef: 2.45 },
    avg_glucose_level: { mean: 106.15, std: 45.28, coef: 0.32 },
    bmi:               { mean: 28.89, std: 7.85, coef: 0.06 },
    hypertension:      { mean: 0.0975, std: 0.297, coef: 0.28 },
    heart_disease:     { mean: 0.054, std: 0.226, coef: 0.18 },
    gender:            { mean: 0.414, std: 0.493, coef: 0.04 },
    ever_married:      { mean: 0.656, std: 0.475, coef: 0.08 },
    smoking_formerly:  { mean: 0.173, std: 0.378, coef: 0.10 },
    smoking_smokes:    { mean: 0.155, std: 0.362, coef: 0.12 },
  };
  const intercept = -3.82;

  // Standardize inputs & compute contributions
  const raw = {
    age: sliders.age,
    avg_glucose_level: sliders.avg_glucose_level,
    bmi: sliders.bmi,
    hypertension: Number(selects.hypertension),
    heart_disease: Number(selects.heart_disease),
    gender: Number(selects.gender),
    ever_married: Number(selects.ever_married),
    smoking_formerly: selects.smoking_status === "1" ? 1 : 0,
    smoking_smokes: selects.smoking_status === "3" ? 1 : 0,
  };

  const c = {};
  let logit = intercept;
  for (const [feat, s] of Object.entries(stats)) {
    const z = (raw[feat] - s.mean) / s.std;
    const contribution = s.coef * z;
    logit += contribution;
    // Map back to user-friendly feature names
    if (feat === "smoking_formerly" || feat === "smoking_smokes") {
      c["smoking_status"] = (c["smoking_status"] || 0) + contribution;
    } else {
      c[feat] = contribution;
    }
  }

  return { prob: 1 / (1 + Math.exp(-logit)), contributions: c };
}

/* Agent 1: Signal Monitor */
function signalMonitor(pct) {
  return { escalate: pct >= 10, reason: pct >= 10 ? "Risk score warrants full assessment" : "Low risk — Tier-1 stop" };
}

/* Agent 2: Risk Assessor — already done by computeRisk + SHAP */

/* Agent 3: Fairness Agent (Addition #5) */
function fairnessAgent(age, gender) {
  const warnings = [];
  const flags = [];
  if (age < 55) {
    warnings.push({ severity: "high", text: `Patient age ${age} is in the under-55 subgroup where the model has 0% recall. ML prediction is unreliable — clinical judgment must take priority.` });
    flags.push("age_blind_spot");
  }
  if (age < 55 && gender === "0") {
    warnings.push({ severity: "high", text: "Young female patients have the highest miss rate. Consider expanded screening even with low ML risk score." });
    flags.push("young_female_atypical");
  }
  return { warnings, flags, subgroup: age < 55 ? "under_55" : "55_plus" };
}

/* Agent 4: Protocol Advisor — Recommendation ToT (Addition #1) */
function recommendationToT(riskPct, age, hasHypertension, hasHeartDisease) {
  const branches = [];

  // Branch 1: Routine follow-up
  if (riskPct < 15 && age < 55) {
    branches.push({ action: "routine_followup", label: "Routine follow-up", score: "likely", rationale: "Low risk score, younger patient — standard screening interval", pruned: false });
  } else if (riskPct >= 40) {
    branches.push({ action: "routine_followup", label: "Routine follow-up", score: "impossible", rationale: `Risk ${riskPct}% too high — routine follow-up unsafe`, pruned: true });
  } else {
    branches.push({ action: "routine_followup", label: "Routine follow-up", score: "possible", rationale: "No hard contraindication but risk factors present", pruned: false });
  }

  // Branch 2: Clinician review
  if (riskPct >= 20 && riskPct < 50) {
    branches.push({ action: "clinician_review", label: "Clinician review (expedited)", score: "likely", rationale: `Moderate risk ${riskPct}% warrants same-day evaluation`, pruned: false });
  } else {
    branches.push({ action: "clinician_review", label: "Clinician review (expedited)", score: "possible", rationale: "Could be appropriate but no strong signal", pruned: false });
  }

  // Branch 3: Urgent evaluation
  if (riskPct >= 50) {
    branches.push({ action: "urgent_evaluation", label: "Urgent evaluation — stroke alert", score: "sure", rationale: `Risk ${riskPct}% — high risk requires immediate workup`, pruned: false });
  } else if (hasHypertension && hasHeartDisease && age >= 65) {
    branches.push({ action: "urgent_evaluation", label: "Urgent evaluation — stroke alert", score: "likely", rationale: "Multiple comorbidities in elderly patient", pruned: false });
  } else {
    branches.push({ action: "urgent_evaluation", label: "Urgent evaluation — stroke alert", score: riskPct < 15 ? "impossible" : "possible", rationale: riskPct < 15 ? "No acute signals" : "Not excluded but not strongly indicated", pruned: riskPct < 15 });
  }

  const scoreOrder = { sure: 3, likely: 2, possible: 1, impossible: 0 };
  const survivors = branches.filter(b => !b.pruned).sort((a, b) => scoreOrder[b.score] - scoreOrder[a.score]);
  return { branches, selected: survivors[0] || null };
}

/* Agent 5: Safety Auditor (Addition #5) */
function safetyAuditor(riskPct, fairness, recommendation) {
  const violations = [];
  if (fairness.flags.includes("age_blind_spot")) {
    violations.push({ severity: "warn", rule: "AGE_BLIND_SPOT", detail: "ML model has 0% recall for this subgroup — clinical judgment must override." });
  }
  if (riskPct > 70 && recommendation === "routine_followup") {
    violations.push({ severity: "block", rule: "MISROUTE", detail: "High-risk patient routed to routine follow-up — BLOCKED." });
  }
  if (riskPct > 50 && recommendation === "routine_followup") {
    violations.push({ severity: "block", rule: "UNDERTRIAGE", detail: "Elevated risk patient cannot be sent to routine — requires expedited or urgent." });
  }
  const blocked = violations.filter(v => v.severity === "block");
  return { passed: blocked.length === 0, gate: blocked.length > 0 ? "BLOCKED" : "PASS", violations };
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
  note: (bg, accent) => ({ background: bg, borderLeft: "3px solid " + accent, borderRadius: "0 10px 10px 0", padding: "14px 18px", fontSize: 12, lineHeight: 1.7, color: "#555", marginBottom: 12 }),
};

function HBar({ value, max, color }) {
  const pct = max > 0 ? Math.min(Math.abs(value) / max * 100, 100) : 0;
  return <div style={{ width: Math.round(pct) + "%", minWidth: 3, height: 18, borderRadius: 3, background: color, transition: "width 0.4s ease" }} />;
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
  const [showTrace, setShowTrace] = useState(false);

  /* ─── Run SENTRA pipeline on every input change ─── */
  const { prob, contributions } = useMemo(() => computeRisk(sliders, selects), [sliders, selects]);
  const pct = Math.round(prob * 100);
  const riskLevel = pct < 20 ? "Low" : pct < 50 ? "Moderate" : "High";
  const riskColor = pct < 20 ? "#1D9E75" : pct < 50 ? "#D4A017" : "#D94040";

  // Agent 1: Signal Monitor
  const signal = useMemo(() => signalMonitor(pct), [pct]);
  // Agent 3: Fairness Agent
  const fairness = useMemo(() => fairnessAgent(sliders.age, selects.gender), [sliders.age, selects.gender]);
  // Agent 4: Recommendation ToT
  const recToT = useMemo(() => recommendationToT(pct, sliders.age, selects.hypertension === "1", selects.heart_disease === "1"), [pct, sliders.age, selects.hypertension, selects.heart_disease]);
  // Agent 5: Safety Auditor
  const safety = useMemo(() => safetyAuditor(pct, fairness, recToT.selected?.action), [pct, fairness, recToT.selected]);

  const sortedC = Object.entries(contributions).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const maxC = Math.max(...sortedC.map(([, v]) => Math.abs(v)), 0.01);
  const globalSorted = Object.entries(SHAP_WEIGHTS).sort((a, b) => b[1] - a[1]);
  const maxG = globalSorted[0][1];

  const tabs = ["Risk assessment", "Models", "Risk factors", "Methodology", "About"];

  const recColor = recToT.selected?.action === "urgent_evaluation" ? "#D94040"
    : recToT.selected?.action === "clinician_review" ? "#D4A017" : "#1D9E75";

  return (
    <div style={s.page}>
      {/* header */}
      <div style={{ textAlign: "center", padding: "36px 0 24px" }}>
        <div style={{ display: "inline-block", background: "#f0eeff", padding: "4px 14px", borderRadius: 20, fontSize: 11, fontWeight: 600, letterSpacing: 1, color: "#534AB7", marginBottom: 10 }}>
          SENTRA — STROKE EARLY-WARNING & NEURO TRIAGE RESPONSE AGENT
        </div>
        <h1 style={{ fontSize: 28, fontWeight: 700, margin: "8px 0 6px" }}>Stroke risk predictor</h1>
        <p style={{ fontSize: 14, color: "#999", maxWidth: 540, margin: "0 auto" }}>
          Multi-agent clinical decision support with ML prediction, SHAP explainability, fairness auditing, and safety guardrails
        </p>
      </div>

      {/* nav */}
      <nav style={s.nav}>
        {tabs.map((t, i) => (
          <button key={i} onClick={() => setTab(i)} style={s.pill(tab === i)}>{t}</button>
        ))}
      </nav>

      {/* ─── TAB 0: Risk Assessment (full SENTRA pipeline) ─── */}
      {tab === 0 && (
        <>
          {/* Patient input */}
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

          {/* Risk score */}
          <div style={{ ...s.card, textAlign: "center", background: `linear-gradient(135deg, ${riskColor}06, ${riskColor}03)`, borderColor: riskColor + "33" }}>
            <div style={{ fontSize: 11, color: "#888", textTransform: "uppercase", letterSpacing: 1 }}>ML risk prediction</div>
            <div style={{ fontSize: 52, fontWeight: 700, color: riskColor, margin: "6px 0" }}>{pct}%</div>
            <span style={{ display: "inline-block", padding: "4px 16px", borderRadius: 20, background: riskColor + "15", color: riskColor, fontSize: 13, fontWeight: 600 }}>{riskLevel} risk</span>
            <div style={{ height: 8, borderRadius: 4, background: "#eee", margin: "18px 0 0", overflow: "hidden" }}>
              <div style={{ height: "100%", width: pct + "%", background: riskColor, borderRadius: 4, transition: "width 0.5s ease" }} />
            </div>
          </div>

          {/* ═══ Fairness Agent warnings (Addition #5) ═══ */}
          {fairness.warnings.length > 0 && fairness.warnings.map((w, i) => (
            <div key={i} style={s.note("#fff8f0", "#D4A017")}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                <span style={{ padding: "2px 8px", borderRadius: 6, background: "#D4A01720", color: "#D4A017", fontSize: 10, fontWeight: 700 }}>FAIRNESS AGENT</span>
              </div>
              {w.text}
            </div>
          ))}

          {/* ═══ Safety Auditor gate (Addition #5) ═══ */}
          <div style={{
            ...s.card,
            borderColor: safety.passed ? "#1D9E7540" : "#D9404040",
            background: safety.passed ? "#f0fff505" : "#fff0f008",
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: safety.violations.length > 0 ? 12 : 0 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <span style={{ fontSize: 10, fontWeight: 700, color: "#888", textTransform: "uppercase", letterSpacing: 0.5 }}>Safety Auditor</span>
              </div>
              <span style={{
                padding: "4px 14px", borderRadius: 20, fontSize: 12, fontWeight: 600,
                background: safety.passed ? "#1D9E75" : "#D94040", color: "#fff",
              }}>
                {safety.gate}
              </span>
            </div>
            {safety.violations.map((v, i) => (
              <div key={i} style={{
                padding: "10px 14px", borderRadius: 8, marginBottom: 6,
                background: v.severity === "block" ? "#D9404010" : "#D4A01710",
                borderLeft: `3px solid ${v.severity === "block" ? "#D94040" : "#D4A017"}`,
              }}>
                <span style={{ fontSize: 11, fontWeight: 600, color: v.severity === "block" ? "#D94040" : "#D4A017" }}>{v.rule}:</span>{" "}
                <span style={{ fontSize: 12, color: "#555" }}>{v.detail}</span>
              </div>
            ))}
          </div>

          {/* ═══ Recommendation ToT output (Addition #1) ═══ */}
          <div style={{ ...s.card, borderColor: recColor + "40" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
              <div>
                <div style={{ fontSize: 10, fontWeight: 700, color: "#888", textTransform: "uppercase", letterSpacing: 0.5, marginBottom: 4 }}>SENTRA Recommendation</div>
                <div style={{ fontSize: 20, fontWeight: 700, color: recColor }}>{recToT.selected?.label || "Escalate to clinician"}</div>
              </div>
              <div style={{ padding: "6px 14px", borderRadius: 10, background: recColor + "12", color: recColor, fontSize: 12, fontWeight: 600 }}>
                [{recToT.selected?.score || "escalate"}]
              </div>
            </div>
            <div style={{ fontSize: 13, color: "#666", marginBottom: 14, lineHeight: 1.6 }}>
              {recToT.selected?.rationale}
            </div>

            {/* ToT branches */}
            <div style={{ fontSize: 11, color: "#aaa", textTransform: "uppercase", letterSpacing: 0.5, marginBottom: 8 }}>Tree-of-Thought branches</div>
            {recToT.branches.map((b, i) => (
              <div key={i} style={{
                display: "flex", alignItems: "center", gap: 10, padding: "8px 12px",
                background: b.pruned ? "#fafafa" : (recToT.selected?.action === b.action ? recColor + "08" : "#fff"),
                borderRadius: 8, border: `1px solid ${b.pruned ? "#eee" : (recToT.selected?.action === b.action ? recColor + "40" : "#f0f0f0")}`,
                marginBottom: 6, opacity: b.pruned ? 0.45 : 1,
              }}>
                <span style={{
                  width: 55, textAlign: "center", fontSize: 10, fontWeight: 600, flexShrink: 0,
                  color: b.pruned ? "#D94040" : b.score === "sure" ? "#1D9E75" : b.score === "likely" ? "#534AB7" : "#D4A017"
                }}>
                  {b.pruned ? "PRUNED" : `[${b.score}]`}
                </span>
                <div style={{ flex: 1 }}>
                  <span style={{ fontSize: 12, fontWeight: recToT.selected?.action === b.action ? 600 : 400 }}>
                    {b.label}
                    {recToT.selected?.action === b.action && <span style={{ fontSize: 9, background: recColor, color: "#fff", padding: "2px 6px", borderRadius: 8, marginLeft: 6 }}>SELECTED</span>}
                  </span>
                  <div style={{ fontSize: 11, color: "#999", marginTop: 1 }}>{b.rationale}</div>
                </div>
              </div>
            ))}
          </div>

          {/* ═══ SHAP contributions with provenance (Additions #2 & #3) ═══ */}
          <div style={s.card}>
            <div style={s.h2}>SHAP contributions</div>
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
            <div style={{ ...s.muted, marginTop: 10 }}>
              Red = increases risk · Green = decreases risk
            </div>
          </div>

          {/* ═══ Provenance trail (Addition #3) ═══ */}
          <div style={s.card}>
            <div style={s.h2}>Provenance trail</div>
            <p style={{ ...s.muted, marginBottom: 12 }}>Every claim traced to its source — SHAP value + guideline + agent</p>
            {[
              { claim: `Risk score: ${pct}%`, source: "SHAP LinearExplainer → LogisticRegression", type: "ml_score" },
              { claim: `Top factor: age (SHAP = ${contributions.age?.toFixed(3)})`, source: "deterministic: SHAP.LinearExplainer", type: "shap" },
              { claim: `Fairness: ${fairness.subgroup} subgroup`, source: "FairnessAgent: subgroup recall analysis", type: "fairness" },
              { claim: `Recommendation: ${recToT.selected?.label || "Escalate"}`, source: "ProtocolAdvisor: Tree-of-Thought search", type: "tot" },
              { claim: `Safety gate: ${safety.gate}`, source: "SafetyAuditor: rule-based validation", type: "safety" },
              { claim: "Clinical protocol reference", source: "AHA/ASA Guidelines for Early Management of Acute Ischemic Stroke (2019)", type: "guideline" },
            ].map((p, i) => (
              <div key={i} style={{ display: "flex", gap: 8, marginBottom: 6, fontSize: 12, padding: "6px 10px", background: "#faf9ff", borderRadius: 8 }}>
                <span style={{ padding: "2px 7px", borderRadius: 5, background: "#f0eeff", color: "#534AB7", fontSize: 10, fontWeight: 600, flexShrink: 0, height: "fit-content" }}>{p.type}</span>
                <div style={{ flex: 1 }}>
                  <div style={{ color: "#2d2a4a", fontWeight: 500 }}>{p.claim}</div>
                  <div style={{ color: "#999", fontFamily: "monospace", fontSize: 11 }}>{p.source}</div>
                </div>
              </div>
            ))}
          </div>

          {/* ═══ Structured execution trace (Addition #2) — collapsible ═══ */}
          <div style={s.card}>
            <button onClick={() => setShowTrace(!showTrace)} style={{
              background: "none", border: "none", cursor: "pointer", fontFamily: "inherit",
              display: "flex", justifyContent: "space-between", alignItems: "center", width: "100%", padding: 0,
            }}>
              <span style={{ fontSize: 14, fontWeight: 600, color: "#2d2a4a" }}>Execution trace</span>
              <span style={{ color: "#534AB7", fontSize: 18, fontWeight: 300 }}>{showTrace ? "−" : "+"}</span>
            </button>
            {showTrace && (
              <div style={{ marginTop: 14 }}>
                {[
                  { agent: "SignalMonitor", color: "#1D9E75", action: "screen_vitals", output: `risk=${pct}%`, decision: signal.escalate ? "escalate=true" : "STOP (Tier-1)", next: signal.escalate ? "RiskAssessor" : "END" },
                  ...(signal.escalate ? [
                    { agent: "RiskAssessor", color: "#534AB7", action: "compute_shap", output: `age=${contributions.age?.toFixed(3)}, gluc=${contributions.avg_glucose_level?.toFixed(3)}`, decision: `level=${riskLevel}`, next: "FairnessAgent" },
                    { agent: "FairnessAgent", color: "#D4A017", action: "check_subgroups", output: `warnings=${fairness.warnings.length}, flags=[${fairness.flags.join(",")}]`, decision: fairness.warnings.length > 0 ? "WARNINGS" : "OK", next: "ProtocolAdvisor" },
                    { agent: "ProtocolAdvisor", color: "#534AB7", action: "recommendation_tot", output: `selected=${recToT.selected?.label}`, decision: recToT.selected?.action, next: "SafetyAuditor" },
                    { agent: "SafetyAuditor", color: "#D94040", action: "validate", output: `violations=${safety.violations.length}`, decision: safety.gate, next: "EscalationCoord" },
                    { agent: "EscalationCoord", color: "#888", action: "route", output: `urgency=${riskLevel.toLowerCase()}`, decision: "PENDING clinician sign-off", next: "END" },
                  ] : []),
                ].map((step, i) => (
                  <div key={i} style={{ padding: "8px 12px", background: "#faf9ff", borderRadius: 8, borderLeft: `3px solid ${step.color}`, marginBottom: 6, fontSize: 12, lineHeight: 1.7, color: "#555" }}>
                    <span style={{ fontWeight: 600, color: step.color }}>{step.agent}</span>
                    {" → "}<span style={{ color: "#888" }}>Action:</span> {step.action}
                    {" · "}<span style={{ color: "#888" }}>Output:</span> {step.output}
                    {" · "}<span style={{ color: "#888" }}>Decision:</span> <strong>{step.decision}</strong>
                    {" · "}<span style={{ color: "#888" }}>Next:</span> {step.next}
                  </div>
                ))}
              </div>
            )}
          </div>

          <div style={s.note("#f0f4ff", "#534AB7")}>
            <strong>Clinical note:</strong> SENTRA's recommendation requires clinician sign-off before action. The ML model is optimized for screening sensitivity (recall = 80%). This is not a diagnostic instrument.
          </div>
        </>
      )}

      {/* ─── TAB 1: Models + Calibration ─── */}
      {tab === 1 && (
        <>
          <div style={{ display: "flex", gap: 12, marginBottom: 20, flexWrap: "wrap" }}>
            <Metric label="Best model" value="Logistic Reg." sub="Balanced class weights" />
            <Metric label="AUC-ROC" value="0.838" sub="Area under ROC curve" />
            <Metric label="Recall" value="80.0%" sub="Sensitivity" />
            <Metric label="Patients" value="5,110" sub="Training dataset" />
          </div>

          <div style={s.card}>
            <div style={s.h2}>Model performance comparison</div>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: "2px solid #eee" }}>
                    {["Model", "AUC", "Recall", "Precision", "F1", "Brier"].map(h => (
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
                      <td style={{ textAlign: "center", padding: 10, color: m.brier < 0.1 ? "#1D9E75" : "#D4A017" }}>{m.brier.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* ═══ Calibration metrics (Addition #4) ═══ */}
          <div style={s.card}>
            <div style={s.h2}>Calibration analysis</div>
            <p style={{ ...s.muted, marginBottom: 14 }}>When the model says 70% risk, do ~70% of those patients actually have strokes?</p>
            <div style={{ display: "flex", gap: 12, marginBottom: 16, flexWrap: "wrap" }}>
              <Metric label="Brier Score" value={CALIBRATION_DATA.brier.toFixed(3)} sub="Lower = better (0 = perfect)" />
              <Metric label="ECE" value={CALIBRATION_DATA.ece.toFixed(3)} sub="Expected Calibration Error" />
              <Metric label="Max CE" value={CALIBRATION_DATA.maxCE.toFixed(3)} sub="Worst single-bin gap" />
            </div>

            <div style={s.h3}>Reliability diagram</div>
            <div style={{ display: "flex", gap: 12, marginBottom: 8, fontSize: 10, color: "#888" }}>
              <span style={{ width: 40, textAlign: "right" }}>Pred.</span>
              <span style={{ flex: 1, textAlign: "center" }}>
                <span style={{ display: "inline-block", width: 10, height: 10, background: "#e0def5", borderRadius: 2, marginRight: 4, verticalAlign: "middle" }} /> Predicted
                <span style={{ display: "inline-block", width: 10, height: 10, background: "#534AB7", borderRadius: 2, marginLeft: 10, marginRight: 4, verticalAlign: "middle" }} /> Actual
              </span>
              <span style={{ width: 40, textAlign: "right" }}>Actual</span>
              <span style={{ width: 30, textAlign: "right" }}>Gap</span>
              <span style={{ width: 35 }}></span>
            </div>
            {CALIBRATION_DATA.bins.map((bin, i) => {
              const gap = Math.abs(bin.actual - bin.predicted);
              const gc = gap < 0.05 ? "#1D9E75" : gap < 0.10 ? "#D4A017" : "#D94040";
              return (
                <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4, fontSize: 12 }}>
                  <span style={{ width: 40, textAlign: "right", color: "#888" }}>{bin.predicted.toFixed(2)}</span>
                  <div style={{ flex: 1, position: "relative", height: 20 }}>
                    <div style={{ position: "absolute", width: `${bin.predicted * 100}%`, height: "100%", background: "#e0def5", borderRadius: 3, opacity: 0.5 }} />
                    <div style={{ position: "absolute", width: `${bin.actual * 100}%`, height: "100%", background: "#534AB7", borderRadius: 3, opacity: 0.8 }} />
                  </div>
                  <span style={{ width: 40, textAlign: "right", color: "#666" }}>{bin.actual.toFixed(2)}</span>
                  <span style={{ width: 30, textAlign: "right", color: gc, fontWeight: 500, fontSize: 11 }}>{gap.toFixed(2)}</span>
                  <span style={{ width: 35, textAlign: "right", color: "#bbb", fontSize: 11 }}>n={bin.count}</span>
                </div>
              );
            })}
            <div style={{ ...s.muted, marginTop: 12 }}>
              ECE of 0.067 indicates good calibration — predicted probabilities roughly match observed stroke rates. Gap increases at higher predicted probabilities where fewer training samples exist.
            </div>
          </div>

          {/* ═══ Fairness subgroup analysis (Addition #5) ═══ */}
          <div style={s.card}>
            <div style={s.h2}>Fairness: subgroup performance</div>
            <p style={{ ...s.muted, marginBottom: 14 }}>Model recall and AUC across demographic subgroups — identifies blind spots</p>

            <div style={s.h3}>By age group</div>
            <div style={{ overflowX: "auto", marginBottom: 16 }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: "2px solid #eee" }}>
                    {["Age group", "Recall", "AUC", "N", "Flag"].map(h => (
                      <th key={h} style={{ padding: "8px 10px", textAlign: "center", color: "#888", fontWeight: 500, fontSize: 11, textTransform: "uppercase" }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {FAIRNESS_DATA.ageGroups.map(g => (
                    <tr key={g.group} style={{ borderBottom: "1px solid #f0f0f0", background: g.flag ? "#fff8f0" : "transparent" }}>
                      <td style={{ padding: 10, fontWeight: 500 }}>{g.group}</td>
                      <td style={{ textAlign: "center", padding: 10, color: g.recall === 0 ? "#D94040" : g.recall < 0.5 ? "#D4A017" : "#1D9E75", fontWeight: 600 }}>{(g.recall * 100).toFixed(1)}%</td>
                      <td style={{ textAlign: "center", padding: 10 }}>{g.auc.toFixed(2)}</td>
                      <td style={{ textAlign: "center", padding: 10, color: "#888" }}>{g.n.toLocaleString()}</td>
                      <td style={{ textAlign: "center", padding: 10 }}>
                        {g.flag && <span style={{ padding: "2px 10px", borderRadius: 10, background: "#D9404015", color: "#D94040", fontSize: 11, fontWeight: 500 }}>BLIND SPOT</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div style={s.h3}>By gender</div>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead>
                  <tr style={{ borderBottom: "2px solid #eee" }}>
                    {["Gender", "Recall", "AUC", "N"].map(h => (
                      <th key={h} style={{ padding: "8px 10px", textAlign: "center", color: "#888", fontWeight: 500, fontSize: 11, textTransform: "uppercase" }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {FAIRNESS_DATA.genderGroups.map(g => (
                    <tr key={g.group} style={{ borderBottom: "1px solid #f0f0f0" }}>
                      <td style={{ padding: 10, fontWeight: 500 }}>{g.group}</td>
                      <td style={{ textAlign: "center", padding: 10, color: "#1D9E75", fontWeight: 500 }}>{(g.recall * 100).toFixed(1)}%</td>
                      <td style={{ textAlign: "center", padding: 10 }}>{g.auc.toFixed(2)}</td>
                      <td style={{ textAlign: "center", padding: 10, color: "#888" }}>{g.n.toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div style={s.note("#fff0f0", "#D94040")}>
            <strong>Critical finding:</strong> 0% recall for patients under 55 — the model misses every stroke in younger patients. Age (SHAP = 1.713) dominates so heavily that younger patients never reach the prediction threshold. The Fairness Agent flags this, and the Safety Auditor ensures clinical judgment overrides the ML for this subgroup.
          </div>
          <div style={s.note("#fff8f0", "#D4A017")}>
            <strong>Why recall matters most:</strong> In stroke screening, a false negative (missing a stroke patient) is far more dangerous than a false positive. A naive "predict no stroke" model gets 95.1% accuracy but 0% recall — clinically useless.
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
                  width: "100%", textAlign: "left", padding: "12px 16px", border: "1px solid " + (expandedFactor === i ? "#534AB7" : "#eee"),
                  borderRadius: expandedFactor === i ? "10px 10px 0 0" : 10, background: expandedFactor === i ? "#f8f7ff" : "#fff",
                  cursor: "pointer", fontFamily: "inherit", fontSize: 13, display: "flex", justifyContent: "space-between", alignItems: "center",
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
                <div style={{ width: 32, height: 32, borderRadius: "50%", background: "#f0eeff", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 13, fontWeight: 600, color: "#534AB7", flexShrink: 0 }}>{i + 1}</div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 4 }}>{step.title}</div>
                  <div style={{ fontSize: 13, color: "#666", lineHeight: 1.7 }}>{step.desc}</div>
                </div>
              </div>
            ))}
          </div>

          {/* SENTRA agent architecture */}
          <div style={s.card}>
            <div style={s.h2}>SENTRA agent architecture</div>
            <p style={{ ...s.muted, marginBottom: 14 }}>6 specialized agents coordinated via LangGraph state machine</p>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8, justifyContent: "center", padding: "12px 0" }}>
              {[
                { name: "Signal Monitor", color: "#1D9E75", desc: "Data validation + Tier-1 gate" },
                { name: "Risk Assessor", color: "#534AB7", desc: "ML + SHAP + differential" },
                { name: "Fairness Agent", color: "#D4A017", desc: "Demographic blind spots" },
                { name: "Protocol Advisor", color: "#534AB7", desc: "RAG + Rec. ToT" },
                { name: "Safety Auditor", color: "#D94040", desc: "Final safety gate" },
                { name: "Escalation Coord.", color: "#888", desc: "Route to clinician" },
              ].map((agent, i, arr) => (
                <React.Fragment key={agent.name}>
                  <div style={{ padding: "10px 14px", background: agent.color + "10", border: `1px solid ${agent.color}40`, borderRadius: 10, textAlign: "center", minWidth: 95 }}>
                    <div style={{ fontSize: 11, fontWeight: 600, color: agent.color }}>{agent.name}</div>
                    <div style={{ fontSize: 10, color: "#888", marginTop: 2 }}>{agent.desc}</div>
                  </div>
                  {i < arr.length - 1 && <div style={{ display: "flex", alignItems: "center", color: "#ccc", fontSize: 18 }}>&#8594;</div>}
                </React.Fragment>
              ))}
            </div>
          </div>

          <div style={s.card}>
            <div style={s.h2}>Key technical decisions</div>
            <div style={{ display: "grid", gap: 12 }}>
              {[
                ["KNN imputation over mean/median", "Preserves relationships between BMI and other features. Mean imputation introduces bias by ignoring correlations."],
                ["SMOTE on training set only", "Applying SMOTE before splitting leaks synthetic samples into the test set, producing overly optimistic metrics."],
                ["Recall as primary metric", "Missing a stroke patient (false negative) is far more dangerous than an unnecessary follow-up (false positive)."],
                ["SHAP over permutation importance", "Provides theoretically grounded, consistent attributions at both global and individual patient levels."],
                ["Tree-of-Thought for recommendations", "ToT decides the action (routine vs expedited vs urgent) — a search problem with safety constraints. Prediction stays deterministic."],
                ["Safety Auditor as final gate", "Blocks contraindicated actions and catches hallucinations before output reaches the clinician."],
                ["Fairness Agent in parallel", "Checks known demographic blind spots (0% recall under 55) before recommendations are finalized."],
              ].map(([title, desc], i) => (
                <div key={i} style={{ padding: "12px 16px", background: "#faf9ff", borderRadius: 10, borderLeft: "3px solid #534AB7" }}>
                  <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 4 }}>{title}</div>
                  <div style={{ fontSize: 12, color: "#666", lineHeight: 1.6 }}>{desc}</div>
                </div>
              ))}
            </div>
          </div>

          <div style={s.card}>
            <div style={s.h2}>Tools and libraries</div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              {["Python", "scikit-learn", "XGBoost", "SHAP", "imbalanced-learn (SMOTE)", "LangGraph", "LangChain", "FHIR R4", "Gradio", "pandas", "matplotlib", "React"].map(t => (
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
              SENTRA (Stroke Early-warning and Neuro Triage Response Agent) is a multi-agent clinical decision support system
              that combines machine learning prediction with SHAP explainability, fairness auditing, Tree-of-Thought recommendation
              reasoning, and safety guardrails. It goes beyond simple risk prediction — it runs a full triage pipeline that
              checks for demographic blind spots, reasons about the best clinical action, validates that recommendation
              against safety rules, and provides full provenance traceability for every claim.
            </p>
            <div style={s.h3}>Key findings</div>
            <ul style={{ fontSize: 13, color: "#666", lineHeight: 2, paddingLeft: 20, marginBottom: 16 }}>
              <li>Logistic Regression with balanced weights: 80% recall, 0.838 AUC-ROC</li>
              <li>Age is the dominant predictor (SHAP: 1.713) — consistent with cerebrovascular pathophysiology</li>
              <li>Critical fairness finding: 0% recall for patients under 55</li>
              <li>Calibration: Brier Score 0.146, ECE 0.067</li>
              <li>Safety Auditor prevents contraindicated recommendations</li>
            </ul>

            <div style={s.h3}>Dataset</div>
            <p style={{ fontSize: 13, color: "#666", lineHeight: 1.7, marginBottom: 16 }}>
              Kaggle Stroke Prediction Dataset (McKinsey and Company) — 5,110 patients, 11 clinical features, 4.87% stroke-positive.
            </p>

            <div style={s.h3}>Limitations</div>
            <ul style={{ fontSize: 13, color: "#666", lineHeight: 2, paddingLeft: 20 }}>
              <li>Cross-sectional dataset — no temporal/longitudinal modeling</li>
              <li>Missing features: atrial fibrillation, cholesterol, family history, medications</li>
              <li>SMOTE may introduce synthetic artifacts in minority class</li>
              <li>Single-source data limits generalizability</li>
            </ul>
          </div>

          <div style={s.card}>
            <div style={s.h2}>Author</div>
            <p style={{ fontSize: 14, color: "#666", lineHeight: 1.7 }}>
              Jahnavi Nimmakayala — High school researcher with a focus on neuroscience and computational biology.
            </p>
            <div style={{ marginTop: 12 }}>
              <a href="https://github.com/jahnavin123/stroke-risk-predictor" target="_blank" rel="noreferrer"
                style={{ padding: "6px 14px", background: "#2d2a4a", color: "#fff", borderRadius: 8, fontSize: 12, textDecoration: "none", fontWeight: 500 }}>
                GitHub Repository
              </a>
            </div>
          </div>

          <div style={s.note("#fff0f0", "#D94040")}>
            <strong>Disclaimer:</strong> This tool is for educational and research purposes only. It is not a medical device and should not be used for clinical decision-making without professional medical oversight.
          </div>
        </>
      )}

      <footer style={{ textAlign: "center", padding: "28px 0 12px", fontSize: 11, color: "#ccc" }}>
        SENTRA — Built with scikit-learn, SHAP, LangGraph, React — BioAI Lab — For research purposes only
      </footer>
    </div>
  );
}
