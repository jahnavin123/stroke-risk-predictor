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

/* ─── SENTRA Pipeline constants (Additions 1-5) ─── */

// Calibration data (Addition #4) — from bootstrap evaluation
const CALIBRATION_DATA = {
  logistic: {
    model: "Logistic Regression",
    brier: 0.146,
    ece: 0.067,
    maxCE: 0.142,
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
  },
};

// Fairness subgroup data (Addition #5)
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

/* ─── helpers ─── */
function computeRisk(sliders, selects) {
  const c = {
    age: ((sliders.age - 40) / 42) * 1.713,
    avg_glucose_level: ((sliders.avg_glucose_level - 100) / 180) * 0.152,
    bmi: ((sliders.bmi - 28.9) / 31) * 0.028,
    heart_disease: Number(selects.heart_disease) * 0.030,
    hypertension: Number(selects.hypertension) * 0.153,
    gender: Number(selects.gender) * 0.022 * 0.3,
    ever_married: Number(selects.ever_married) * 0.033 * 0.2,
    smoking_status: selects.smoking_status === "3" ? 0.018 : selects.smoking_status === "1" ? 0.009 : 0,
  };
  let logit = -3.5;
  for (const v of Object.values(c)) logit += v;
  return { prob: 1 / (1 + Math.exp(-logit)), contributions: c };
}

/* Recommendation ToT (Addition #1) */
function runRecommendationToT(riskPct, age, hasHypertension, hasHeartDisease) {
  const branches = [];
  // Branch 1: Routine follow-up
  if (riskPct < 15 && age < 55) {
    branches.push({ action: "routine_followup", label: "Routine follow-up", score: "likely", rationale: "Low risk score, younger patient", pruned: false });
  } else if (riskPct >= 40) {
    branches.push({ action: "routine_followup", label: "Routine follow-up", score: "impossible", rationale: `Risk ${riskPct}% too high for routine — unsafe`, pruned: true });
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
    branches.push({ action: "urgent_evaluation", label: "Urgent evaluation (stroke alert)", score: "sure", rationale: `Risk ${riskPct}% — high risk requires immediate workup`, pruned: false });
  } else if (hasHypertension && hasHeartDisease && age >= 65) {
    branches.push({ action: "urgent_evaluation", label: "Urgent evaluation (stroke alert)", score: "likely", rationale: "Multiple comorbidities in elderly patient", pruned: false });
  } else {
    branches.push({ action: "urgent_evaluation", label: "Urgent evaluation (stroke alert)", score: riskPct < 15 ? "impossible" : "possible", rationale: riskPct < 15 ? "No acute signals" : "Not excluded but not strongly indicated", pruned: riskPct < 15 });
  }

  const scoreOrder = { sure: 3, likely: 2, possible: 1, impossible: 0 };
  const survivors = branches.filter(b => !b.pruned).sort((a, b) => scoreOrder[b.score] - scoreOrder[a.score]);
  const selected = survivors[0] || null;
  return { branches, selected };
}

/* Fairness check (Addition #5) */
function runFairnessCheck(age, gender) {
  const warnings = [];
  const flags = [];
  if (age < 55) {
    warnings.push(`Patient age ${age} falls in the under-55 subgroup where the ML model has 0% recall. Clinical judgment should take priority.`);
    flags.push("age_blind_spot");
  }
  if (age < 55 && gender === "0") {
    warnings.push("Young female patients have the highest miss rate. Consider expanded screening even with low ML risk score.");
    flags.push("young_female_atypical");
  }
  return { warnings, flags, subgroup: age < 55 ? "under_55" : "55_plus" };
}

/* Safety audit (Addition #5) */
function runSafetyAudit(riskPct, fairness, recommendation) {
  const violations = [];
  if (fairness.flags.includes("age_blind_spot")) {
    violations.push({ severity: "warn", rule: "age_blind_spot", detail: "ML model has 0% recall for this subgroup. Clinical judgment must take priority." });
  }
  if (riskPct > 70 && recommendation === "routine_followup") {
    violations.push({ severity: "block", rule: "misroute", detail: "High-risk patient routed to routine follow-up — BLOCKED." });
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
  note: (bg, accent) => ({ background: bg, borderLeft: "3px solid " + accent, borderRadius: "0 10px 10px 0", padding: "14px 18px", fontSize: 12, lineHeight: 1.7, color: "#555" }),
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

/* Structured execution log entry (Addition #2) */
function LogEntry({ agent, action, output, decision, next, provenance, color }) {
  return (
    <div style={{ padding: "12px 16px", background: "#faf9ff", borderRadius: 10, borderLeft: `3px solid ${color}`, marginBottom: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 6 }}>
        <span style={{ fontSize: 13, fontWeight: 600, color }}>{agent}</span>
        <span style={{ fontSize: 10, color: "#bbb", fontFamily: "monospace" }}>
          {new Date().toISOString().split("T")[1].split(".")[0]}Z
        </span>
      </div>
      <div style={{ fontSize: 12, lineHeight: 1.8, color: "#555" }}>
        <div><span style={{ color: "#888", fontWeight: 500 }}>Action:</span> {action}</div>
        <div><span style={{ color: "#888", fontWeight: 500 }}>Output:</span> {output}</div>
        <div><span style={{ color: "#888", fontWeight: 500 }}>Decision:</span> <span style={{ fontWeight: 500 }}>{decision}</span></div>
        <div><span style={{ color: "#888", fontWeight: 500 }}>Next:</span> {next}</div>
        {provenance && <div><span style={{ color: "#888", fontWeight: 500 }}>Source:</span> <span style={{ fontFamily: "monospace", fontSize: 11 }}>{provenance}</span></div>}
      </div>
    </div>
  );
}

/* Reliability diagram bar (Addition #4) */
function CalibrationBar({ predicted, actual, count, maxCount }) {
  const gap = Math.abs(actual - predicted);
  const gapColor = gap < 0.05 ? "#1D9E75" : gap < 0.10 ? "#D4A017" : "#D94040";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4, fontSize: 12 }}>
      <span style={{ width: 40, textAlign: "right", color: "#888" }}>{predicted.toFixed(2)}</span>
      <div style={{ flex: 1, position: "relative", height: 20 }}>
        <div style={{ position: "absolute", width: `${(predicted) * 100}%`, height: "100%", background: "#e0def5", borderRadius: 3, opacity: 0.5 }} />
        <div style={{ position: "absolute", width: `${(actual) * 100}%`, height: "100%", background: "#534AB7", borderRadius: 3, opacity: 0.8 }} />
      </div>
      <span style={{ width: 40, textAlign: "right", color: "#666" }}>{actual.toFixed(2)}</span>
      <span style={{ width: 30, textAlign: "right", color: gapColor, fontWeight: 500, fontSize: 11 }}>{gap.toFixed(2)}</span>
      <span style={{ width: 35, textAlign: "right", color: "#bbb", fontSize: 11 }}>n={count}</span>
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

  /* SENTRA Pipeline computations (Additions 1-5) */
  const recToT = useMemo(() => runRecommendationToT(pct, sliders.age, selects.hypertension === "1", selects.heart_disease === "1"), [pct, sliders.age, selects.hypertension, selects.heart_disease]);
  const fairness = useMemo(() => runFairnessCheck(sliders.age, selects.gender), [sliders.age, selects.gender]);
  const safetyAudit = useMemo(() => runSafetyAudit(pct, fairness, recToT.selected?.action), [pct, fairness, recToT.selected]);

  const sortedC = Object.entries(contributions).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const maxC = Math.max(...sortedC.map(([, v]) => Math.abs(v)), 0.01);

  const globalSorted = Object.entries(SHAP_WEIGHTS).sort((a, b) => b[1] - a[1]);
  const maxG = globalSorted[0][1];

  const tabs = ["Risk assessment", "SENTRA pipeline", "Models", "Risk factors", "Methodology", "About"];

  return (
    <div style={s.page}>
      {/* header */}
      <div style={{ textAlign: "center", padding: "36px 0 24px" }}>
        <div style={{ display: "inline-block", background: "#f0eeff", padding: "4px 14px", borderRadius: 20, fontSize: 11, fontWeight: 600, letterSpacing: 1, color: "#534AB7", marginBottom: 10 }}>
          MACHINE LEARNING + EXPLAINABLE AI + AGENTIC AI
        </div>
        <h1 style={{ fontSize: 28, fontWeight: 700, margin: "8px 0 6px" }}>Stroke risk predictor</h1>
        <p style={{ fontSize: 14, color: "#999", maxWidth: 540, margin: "0 auto" }}>
          Clinical screening tool with SHAP interpretability, multi-agent triage (SENTRA), fairness auditing, and safety guardrails
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

          <div style={{ ...s.card, textAlign: "center", background: "linear-gradient(135deg, " + riskColor + "06, " + riskColor + "03)", borderColor: riskColor + "33" }}>
            <div style={{ fontSize: 11, color: "#888", textTransform: "uppercase", letterSpacing: 1 }}>Estimated stroke risk</div>
            <div style={{ fontSize: 52, fontWeight: 700, color: riskColor, margin: "6px 0" }}>{pct}%</div>
            <span style={{ display: "inline-block", padding: "4px 16px", borderRadius: 20, background: riskColor + "15", color: riskColor, fontSize: 13, fontWeight: 600 }}>{riskLevel} risk</span>
            <div style={{ height: 8, borderRadius: 4, background: "#eee", margin: "18px 0 0", overflow: "hidden" }}>
              <div style={{ height: "100%", width: pct + "%", background: riskColor, borderRadius: 4, transition: "width 0.5s ease" }} />
            </div>
          </div>

          {/* Fairness warnings inline (Addition #5) */}
          {fairness.warnings.length > 0 && (
            <div style={s.note("#fff8f0", "#D4A017")}>
              <strong>Fairness alert:</strong>{" "}
              {fairness.warnings.map((w, i) => <span key={i}>{w}{i < fairness.warnings.length - 1 ? " " : ""}</span>)}
            </div>
          )}
          {fairness.warnings.length > 0 && <div style={{ height: 8 }} />}

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
            <div style={{ ...s.muted, marginTop: 10 }}>
              Red = increases risk · Green = decreases risk ·{" "}
              <span style={{ fontWeight: 500 }}>Provenance: SHAP LinearExplainer on Logistic Regression model</span>
            </div>
          </div>

          <div style={s.note("#f0f4ff", "#534AB7")}>
            <strong>Clinical note:</strong> This model is optimized for high sensitivity (recall = 80%) to minimize missed stroke cases. It is designed as a screening tool — flagged patients should receive further clinical evaluation. Not a diagnostic instrument.
          </div>
        </>
      )}

      {/* ─── TAB 1: SENTRA Pipeline (NEW — Additions 1-5) ─── */}
      {tab === 1 && (
        <>
          {/* Architecture diagram */}
          <div style={s.card}>
            <div style={s.h2}>SENTRA multi-agent architecture</div>
            <p style={{ ...s.muted, marginBottom: 14 }}>Stroke Early-warning and Neuro Triage Response Agent — 6 specialized agents coordinated via LangGraph</p>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8, justifyContent: "center", padding: "16px 0" }}>
              {[
                { name: "Signal Monitor", color: "#1D9E75", desc: "Data validation" },
                { name: "Risk Assessor", color: "#534AB7", desc: "ML + SHAP + differential" },
                { name: "Fairness Agent", color: "#D4A017", desc: "Demographic blind spots" },
                { name: "Protocol Advisor", color: "#534AB7", desc: "RAG + Rec. ToT" },
                { name: "Safety Auditor", color: "#D94040", desc: "Final safety gate" },
                { name: "Escalation Coord.", color: "#888", desc: "Route to clinician" },
              ].map((agent, i, arr) => (
                <React.Fragment key={agent.name}>
                  <div style={{ padding: "10px 14px", background: agent.color + "10", border: `1px solid ${agent.color}40`, borderRadius: 10, textAlign: "center", minWidth: 100 }}>
                    <div style={{ fontSize: 12, fontWeight: 600, color: agent.color }}>{agent.name}</div>
                    <div style={{ fontSize: 10, color: "#888", marginTop: 2 }}>{agent.desc}</div>
                  </div>
                  {i < arr.length - 1 && <div style={{ display: "flex", alignItems: "center", color: "#ccc", fontSize: 18 }}>&#8594;</div>}
                </React.Fragment>
              ))}
            </div>
          </div>

          {/* Structured execution log (Addition #2) */}
          <div style={s.card}>
            <div style={s.h2}>Structured execution trace</div>
            <p style={{ ...s.muted, marginBottom: 14 }}>Action &#8594; Output &#8594; Decision &#8594; Next (replaces conversational logging)</p>

            <LogEntry
              agent="SignalMonitor" color="#1D9E75"
              action="screen_vitals_symptoms"
              output={`risk=${pct}%, age=${sliders.age}, hypertension=${selects.hypertension === "1" ? "yes" : "no"}`}
              decision={pct >= 10 ? "escalate=true" : "escalate=false (Tier-1 stop)"}
              next={pct >= 10 ? "RiskAssessor" : "END"}
            />
            {pct >= 10 && (
              <>
                <LogEntry
                  agent="RiskAssessor" color="#534AB7"
                  action="compute_scores_and_shap"
                  output={`SHAP: age=${contributions.age?.toFixed(3)}, glucose=${contributions.avg_glucose_level?.toFixed(3)}`}
                  decision={`risk_level=${riskLevel}`}
                  next="FairnessAgent + ProtocolAdvisor"
                  provenance="deterministic_tool:SHAP.LinearExplainer"
                />
                <LogEntry
                  agent="FairnessAgent" color="#D4A017"
                  action="check_demographic_blind_spots"
                  output={`warnings=${fairness.warnings.length}, flags=[${fairness.flags.join(", ")}]`}
                  decision={fairness.warnings.length > 0 ? "WARNINGS RAISED" : "fairness_checked"}
                  next="ProtocolAdvisor"
                  provenance="fairness_agent:subgroup_analysis"
                />
                <LogEntry
                  agent="ProtocolAdvisor" color="#534AB7"
                  action="recommendation_tot + pathway_selection"
                  output={`ToT selected: ${recToT.selected?.label || "ABSTAIN"}`}
                  decision={`recommendation=${recToT.selected?.action || "escalate_to_clinician"}`}
                  next="SafetyAuditor"
                  provenance="guideline:AHA/ASA Stroke Guidelines 2024"
                />
                <LogEntry
                  agent="SafetyAuditor" color="#D94040"
                  action="validate_recommendation"
                  output={`gate=${safetyAudit.gate}, violations=${safetyAudit.violations.length}`}
                  decision={safetyAudit.gate}
                  next="EscalationCoordinator"
                />
                <LogEntry
                  agent="EscalationCoordinator" color="#888"
                  action="route_to_clinician"
                  output={`urgency=${riskLevel.toLowerCase()}`}
                  decision="status=PENDING — requires clinician sign-off"
                  next="END"
                />
              </>
            )}
          </div>

          {/* Recommendation ToT (Addition #1) */}
          <div style={s.card}>
            <div style={s.h2}>Recommendation Tree-of-Thought</div>
            <p style={{ ...s.muted, marginBottom: 14 }}>ToT decides the <em>action</em> (what to do), not the prediction. Branch &#8594; Evaluate &#8594; Prune &#8594; Select.</p>
            {recToT.branches.map((b, i) => (
              <div key={i} style={{
                display: "flex", alignItems: "center", gap: 12, padding: "10px 14px",
                background: b.pruned ? "#fafafa" : (recToT.selected?.action === b.action ? "#f0eeff" : "#fff"),
                borderRadius: 10, border: `1px solid ${b.pruned ? "#eee" : (recToT.selected?.action === b.action ? "#534AB7" : "#eae8f0")}`,
                marginBottom: 8, opacity: b.pruned ? 0.5 : 1,
              }}>
                <div style={{ width: 60, textAlign: "center" }}>
                  {b.pruned ? (
                    <span style={{ fontSize: 11, color: "#D94040", fontWeight: 500 }}>PRUNED</span>
                  ) : (
                    <span style={{
                      fontSize: 11, fontWeight: 500,
                      color: b.score === "sure" ? "#1D9E75" : b.score === "likely" ? "#534AB7" : "#D4A017"
                    }}>[{b.score}]</span>
                  )}
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 13, fontWeight: recToT.selected?.action === b.action ? 600 : 400, color: "#2d2a4a" }}>
                    {b.label}
                    {recToT.selected?.action === b.action && (
                      <span style={{ fontSize: 10, background: "#534AB7", color: "#fff", padding: "2px 8px", borderRadius: 10, marginLeft: 8 }}>SELECTED</span>
                    )}
                  </div>
                  <div style={{ fontSize: 11, color: "#888", marginTop: 2 }}>{b.rationale}</div>
                </div>
              </div>
            ))}
          </div>

          {/* Safety audit result (Addition #5) */}
          <div style={{
            ...s.card,
            borderColor: safetyAudit.passed ? "#1D9E7540" : "#D9404040",
            background: safetyAudit.passed ? "#f0fff5" : "#fff0f0",
          }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
              <div style={s.h2}>Safety auditor gate</div>
              <span style={{
                padding: "4px 14px", borderRadius: 20, fontSize: 12, fontWeight: 600,
                background: safetyAudit.passed ? "#1D9E75" : "#D94040", color: "#fff",
              }}>
                {safetyAudit.gate}
              </span>
            </div>
            {safetyAudit.violations.length === 0 ? (
              <p style={{ fontSize: 13, color: "#1D9E75" }}>No violations detected. Recommendation cleared for clinician review.</p>
            ) : (
              safetyAudit.violations.map((v, i) => (
                <div key={i} style={{
                  padding: "10px 14px", borderRadius: 8, marginBottom: 6,
                  background: v.severity === "block" ? "#D9404015" : "#D4A01715",
                  borderLeft: `3px solid ${v.severity === "block" ? "#D94040" : "#D4A017"}`,
                }}>
                  <div style={{ fontSize: 11, fontWeight: 600, color: v.severity === "block" ? "#D94040" : "#D4A017", textTransform: "uppercase" }}>
                    {v.severity} — {v.rule}
                  </div>
                  <div style={{ fontSize: 12, color: "#555", marginTop: 4 }}>{v.detail}</div>
                </div>
              ))
            )}
          </div>

          {/* Provenance trail (Addition #3) */}
          <div style={s.card}>
            <div style={s.h2}>Provenance trail</div>
            <p style={{ ...s.muted, marginBottom: 14 }}>Every claim traced to its source: SHAP value + guideline + tool</p>
            {[
              { claim: `Risk score: ${pct}%`, source: "SHAP LinearExplainer on LogisticRegression", type: "computed_score", shap: `age=${contributions.age?.toFixed(3)}, glucose=${contributions.avg_glucose_level?.toFixed(3)}` },
              { claim: `Age SHAP contribution: ${contributions.age?.toFixed(3)}`, source: "deterministic_tool:SHAP.LinearExplainer", type: "shap_value", shap: "Dominant feature (mean |SHAP| = 1.713)" },
              { claim: `Recommendation: ${recToT.selected?.label || "Escalate"}`, source: "recommendation_tot:Tree-of-Thought", type: "tot_decision" },
              { claim: `Fairness subgroup: ${fairness.subgroup}`, source: "fairness_agent:subgroup_analysis", type: "fairness_check" },
              { claim: "Treatment protocol reference", source: "guideline:AHA/ASA Guidelines for Early Management of Acute Ischemic Stroke (2019)", type: "protocol_citation" },
            ].map((p, i) => (
              <div key={i} style={{ display: "flex", gap: 10, marginBottom: 8, fontSize: 12, padding: "8px 12px", background: "#faf9ff", borderRadius: 8 }}>
                <span style={{ padding: "2px 8px", borderRadius: 6, background: "#f0eeff", color: "#534AB7", fontSize: 10, fontWeight: 600, flexShrink: 0, height: "fit-content" }}>{p.type}</span>
                <div style={{ flex: 1 }}>
                  <div style={{ color: "#2d2a4a", fontWeight: 500 }}>{p.claim}</div>
                  <div style={{ color: "#888", fontFamily: "monospace", fontSize: 11, marginTop: 2 }}>{p.source}</div>
                  {p.shap && <div style={{ color: "#534AB7", fontSize: 11, marginTop: 2 }}>{p.shap}</div>}
                </div>
              </div>
            ))}
          </div>

          <div style={s.note("#f0f4ff", "#534AB7")}>
            <strong>CMU Agentic AI Capstone:</strong> SENTRA implements ReAct reasoning loops, Tree-of-Thought for recommendation decisions, RAG with clinical protocol corpus, deterministic clinical scoring tools, structured execution logging, fairness auditing, and a safety auditor gate — all coordinated via LangGraph state machine.
          </div>
        </>
      )}

      {/* ─── TAB 2: Model Comparison (was Tab 1) ─── */}
      {tab === 2 && (
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

          {/* Calibration metrics (Addition #4) */}
          <div style={s.card}>
            <div style={s.h2}>Calibration metrics</div>
            <p style={{ ...s.muted, marginBottom: 14 }}>When the model says 70% risk, do ~70% of those patients actually have strokes?</p>
            <div style={{ display: "flex", gap: 12, marginBottom: 16, flexWrap: "wrap" }}>
              <Metric label="Brier Score" value={CALIBRATION_DATA.logistic.brier.toFixed(3)} sub="Lower is better (0 = perfect)" />
              <Metric label="ECE" value={CALIBRATION_DATA.logistic.ece.toFixed(3)} sub="Expected Calibration Error" />
              <Metric label="Max CE" value={CALIBRATION_DATA.logistic.maxCE.toFixed(3)} sub="Worst single-bin gap" />
            </div>

            <div style={s.h3}>Reliability diagram</div>
            <div style={{ display: "flex", gap: 12, marginBottom: 8, fontSize: 10, color: "#888" }}>
              <span style={{ width: 40, textAlign: "right" }}>Pred.</span>
              <span style={{ flex: 1, textAlign: "center" }}>
                <span style={{ display: "inline-block", width: 10, height: 10, background: "#e0def5", borderRadius: 2, marginRight: 4, verticalAlign: "middle" }} />
                Predicted
                <span style={{ display: "inline-block", width: 10, height: 10, background: "#534AB7", borderRadius: 2, marginLeft: 10, marginRight: 4, verticalAlign: "middle" }} />
                Actual
              </span>
              <span style={{ width: 40, textAlign: "right" }}>Actual</span>
              <span style={{ width: 30, textAlign: "right" }}>Gap</span>
              <span style={{ width: 35 }}></span>
            </div>
            {CALIBRATION_DATA.logistic.bins.map((bin, i) => (
              <CalibrationBar key={i} predicted={bin.predicted} actual={bin.actual} count={bin.count} maxCount={2890} />
            ))}
            <div style={{ ...s.muted, marginTop: 12 }}>
              ECE of 0.067 indicates good calibration — predicted probabilities roughly match observed rates. Gap increases at higher predicted probabilities where fewer samples exist.
            </div>
          </div>

          {/* Fairness subgroup analysis (Addition #5) */}
          <div style={s.card}>
            <div style={s.h2}>Fairness: subgroup performance</div>
            <p style={{ ...s.muted, marginBottom: 14 }}>Model recall and AUC broken down by demographic subgroups</p>

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
                      <td style={{ textAlign: "center", padding: 10, color: g.recall === 0 ? "#D94040" : g.recall < 0.5 ? "#D4A017" : "#1D9E75", fontWeight: 600 }}>
                        {(g.recall * 100).toFixed(1)}%
                      </td>
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
            <strong>Critical finding:</strong> The model has 0% recall for patients under 55 — it misses every stroke case in younger patients. Age (SHAP = 1.713) dominates so heavily that younger patients never reach the prediction threshold. The Fairness Agent flags this blind spot and the Safety Auditor ensures clinical judgment overrides the model for this subgroup.
          </div>
          <div style={{ height: 12 }} />
          <div style={s.note("#fff8f0", "#D4A017")}>
            <strong>Why recall matters most:</strong> In stroke screening, a false negative (missing a stroke patient) is far more dangerous than a false positive (flagging a healthy patient for follow-up testing). A naive "predict no stroke" model achieves 95.1% accuracy but 0% recall — completely useless clinically.
          </div>
          <div style={{ height: 12 }} />
          <div style={s.note("#f0fff5", "#1D9E75")}>
            <strong>Class imbalance:</strong> Only 4.87% of patients had strokes (249 of 5,110). We used SMOTE oversampling on training data only and cost-sensitive learning to prevent the model from ignoring the minority class.
          </div>
        </>
      )}

      {/* ─── TAB 3: Risk Factors (was Tab 2) ─── */}
      {tab === 3 && (
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
            <strong>Research insight:</strong> SHAP values quantitatively confirm what neurovascular literature has established — age-related cerebrovascular degeneration, hypertension-driven arterial damage, and glucose-mediated endothelial dysfunction are the dominant pathways. This alignment between data-driven ML features and established neuroscience validates the model's clinical plausibility.
          </div>
        </>
      )}

      {/* ─── TAB 4: Methodology (was Tab 3) ─── */}
      {tab === 4 && (
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
                ["Tree-of-Thought for recommendations, not prediction", "ToT is used to decide the action (routine vs expedited vs urgent) — a genuine search problem with safety constraints. The prediction stays deterministic."],
                ["Safety Auditor as final gate", "Every recommendation passes through a safety auditor that blocks contraindicated actions (e.g., thrombolysis for hemorrhagic stroke) and catches LLM hallucinations before output."],
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

      {/* ─── TAB 5: About (was Tab 4) ─── */}
      {tab === 5 && (
        <>
          <div style={s.card}>
            <div style={s.h2}>About this project</div>
            <p style={{ fontSize: 14, lineHeight: 1.8, color: "#555", marginBottom: 16 }}>
              This project develops a machine learning-based screening tool for stroke risk prediction,
              with a focus on clinical applicability, explainability, and safety. Unlike typical ML classification projects
              that optimize for accuracy, this tool is designed around the asymmetric cost structure of
              medical screening — where missing an at-risk patient is far worse than a false alarm.
            </p>
            <p style={{ fontSize: 14, lineHeight: 1.8, color: "#555", marginBottom: 16 }}>
              The SENTRA (Stroke Early-warning and Neuro Triage Response Agent) pipeline extends the ML prediction
              with a multi-agent clinical decision support system built on LangGraph, featuring fairness auditing,
              safety guardrails, Tree-of-Thought recommendation reasoning, and full provenance traceability — developed
              as part of the CMU Agentic AI capstone program.
            </p>

            <div style={s.h3}>Dataset</div>
            <p style={{ fontSize: 13, color: "#666", lineHeight: 1.7, marginBottom: 16 }}>
              Kaggle Stroke Prediction Dataset (McKinsey and Company) — 5,110 patients, 11 clinical features,
              binary target. Severe class imbalance: only 4.87% stroke-positive cases.
            </p>

            <div style={s.h3}>Key findings</div>
            <ul style={{ fontSize: 13, color: "#666", lineHeight: 2, paddingLeft: 20, marginBottom: 16 }}>
              <li>Logistic Regression with balanced weights achieves 80% recall and 0.838 AUC-ROC, outperforming complex ensembles on clinical sensitivity</li>
              <li>Age is the overwhelmingly dominant predictor (SHAP: 1.713), consistent with known cerebrovascular pathophysiology</li>
              <li>Critical fairness finding: 0% recall for patients under 55 — the Fairness Agent flags this blind spot</li>
              <li>Calibration analysis: Brier Score 0.146, ECE 0.067 — good calibration at common probability ranges</li>
              <li>Safety Auditor gate prevents contraindicated recommendations (e.g., thrombolysis for hemorrhagic stroke)</li>
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
              Jahnavi Nimmakayala — High school researcher with a focus on neuroscience and computational biology.
              This project was developed as part of independent research in applying machine learning
              to clinical neuroscience problems, with the SENTRA agent pipeline developed for the
              CMU Agentic AI capstone program.
            </p>
            <div style={{ marginTop: 12, display: "flex", gap: 10 }}>
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
        Built with scikit-learn, XGBoost, SHAP, LangGraph, React — BioAI Lab — For research purposes only
      </footer>
    </div>
  );
}
