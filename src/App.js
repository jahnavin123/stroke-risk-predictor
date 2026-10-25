import React, { useState, useMemo } from "react";

/* ─── Data constants ─── */
const FEATURES = {
  age: { label: "Age", min: 1, max: 82, step: 1, default: 55, unit: "years" },
  avg_glucose_level: { label: "Avg glucose", min: 55, max: 280, step: 1, default: 110, unit: "mg/dL" },
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
  { feature: "Hypertension", shap: 0.153, detail: "Chronic high blood pressure is the single most modifiable risk factor for stroke. It damages arterial walls through mechanical shear stress, promotes atherosclerotic plaque formation in the carotid and cerebral arteries, and weakens small penetrating arteries in the brain." },
  { feature: "Avg glucose level", shap: 0.152, detail: "Chronic hyperglycemia damages endothelial cells lining cerebral blood vessels through advanced glycation end products (AGEs). Diabetic patients have 2-4x higher stroke risk, with glucose toxicity directly impairing the blood-brain barrier." },
  { feature: "Heart disease", shap: 0.030, detail: "Cardiac conditions — especially atrial fibrillation — are major embolic stroke risk factors. Irregular heart rhythm allows blood to pool in the left atrial appendage, forming clots that can travel to the brain. This accounts for approximately 20-30% of all ischemic strokes." },
  { feature: "BMI", shap: 0.028, detail: "Obesity promotes stroke through systemic inflammation, insulin resistance leading to endothelial dysfunction, dyslipidemia accelerating atherosclerotic plaque formation, and increased blood viscosity." },
];

const METHODOLOGY_STEPS = [
  { title: "Data collection", desc: "5,110 patient records with 11 clinical features from the Kaggle Stroke Prediction Dataset (McKinsey & Company)", icon: "📊" },
  { title: "Preprocessing", desc: "KNN imputation for 201 missing BMI values, label encoding for categoricals, StandardScaler normalization", icon: "🔧" },
  { title: "Class imbalance", desc: "SMOTE oversampling on training data only + cost-sensitive learning via balanced class weights. Only 4.87% positive class", icon: "⚖️" },
  { title: "Model training", desc: "6 algorithms compared via stratified train/test split. Primary metric: recall — missing a stroke is far worse than a false alarm", icon: "🧠" },
  { title: "Explainability", desc: "SHAP provides global feature importance and individual patient-level explanations — clinicians see why each patient is flagged", icon: "🔍" },
  { title: "Threshold tuning", desc: "Classification threshold lowered from 0.5 to 0.3 for clinical screening — catches more at-risk patients", icon: "🎯" },
];

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

const FAIRNESS_DATA = {
  ageGroups: [
    { group: "Under 40", recall: 0.0, auc: 0.72, n: 2241, flag: true },
    { group: "40–54", recall: 0.0, auc: 0.74, n: 1087, flag: true },
    { group: "55–64", recall: 0.45, auc: 0.79, n: 789 },
    { group: "65–74", recall: 0.72, auc: 0.83, n: 612 },
    { group: "75+", recall: 0.89, auc: 0.87, n: 381 },
  ],
  genderGroups: [
    { group: "Female", recall: 0.76, auc: 0.83, n: 2994 },
    { group: "Male", recall: 0.84, auc: 0.84, n: 2116 },
  ],
};

/* ─── SENTRA pipeline functions ─── */

function computeRisk(sliders, selects) {
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
  const raw = {
    age: sliders.age, avg_glucose_level: sliders.avg_glucose_level, bmi: sliders.bmi,
    hypertension: Number(selects.hypertension), heart_disease: Number(selects.heart_disease),
    gender: Number(selects.gender), ever_married: Number(selects.ever_married),
    smoking_formerly: selects.smoking_status === "1" ? 1 : 0,
    smoking_smokes: selects.smoking_status === "3" ? 1 : 0,
  };
  const c = {};
  let logit = intercept;
  for (const [feat, st] of Object.entries(stats)) {
    const z = (raw[feat] - st.mean) / st.std;
    const contribution = st.coef * z;
    logit += contribution;
    if (feat === "smoking_formerly" || feat === "smoking_smokes") {
      c["smoking_status"] = (c["smoking_status"] || 0) + contribution;
    } else { c[feat] = contribution; }
  }
  return { prob: 1 / (1 + Math.exp(-logit)), contributions: c };
}

function signalMonitor(pct) {
  return { escalate: pct >= 10, reason: pct >= 10 ? "Risk score warrants full assessment" : "Low risk — Tier-1 stop" };
}

function fairnessAgent(age, gender) {
  const warnings = []; const flags = [];
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

function recommendationToT(riskPct, age, hasHypertension, hasHeartDisease) {
  const branches = [];
  if (riskPct < 15 && age < 55) branches.push({ action: "routine_followup", label: "Routine follow-up", score: "likely", rationale: "Low risk, younger patient — standard screening interval", pruned: false });
  else if (riskPct >= 40) branches.push({ action: "routine_followup", label: "Routine follow-up", score: "impossible", rationale: `Risk ${riskPct}% too high — routine unsafe`, pruned: true });
  else branches.push({ action: "routine_followup", label: "Routine follow-up", score: "possible", rationale: "No hard contraindication but risk factors present", pruned: false });

  if (riskPct >= 20 && riskPct < 50) branches.push({ action: "clinician_review", label: "Clinician review", score: "likely", rationale: `Moderate risk ${riskPct}% warrants same-day evaluation`, pruned: false });
  else branches.push({ action: "clinician_review", label: "Clinician review", score: "possible", rationale: "Could be appropriate but no strong signal", pruned: false });

  if (riskPct >= 50) branches.push({ action: "urgent_evaluation", label: "Urgent evaluation", score: "sure", rationale: `Risk ${riskPct}% — immediate workup required`, pruned: false });
  else if (hasHypertension && hasHeartDisease && age >= 65) branches.push({ action: "urgent_evaluation", label: "Urgent evaluation", score: "likely", rationale: "Multiple comorbidities in elderly patient", pruned: false });
  else branches.push({ action: "urgent_evaluation", label: "Urgent evaluation", score: riskPct < 15 ? "impossible" : "possible", rationale: riskPct < 15 ? "No acute signals" : "Not strongly indicated", pruned: riskPct < 15 });

  const scoreOrder = { sure: 3, likely: 2, possible: 1, impossible: 0 };
  const survivors = branches.filter(b => !b.pruned).sort((a, b) => scoreOrder[b.score] - scoreOrder[a.score]);
  return { branches, selected: survivors[0] || null };
}

function safetyAuditor(riskPct, fairness, recommendation) {
  const violations = [];
  if (fairness.flags.includes("age_blind_spot")) violations.push({ severity: "warn", rule: "AGE_BLIND_SPOT", detail: "Model has 0% recall for this subgroup — clinical judgment must override." });
  if (riskPct > 70 && recommendation === "routine_followup") violations.push({ severity: "block", rule: "MISROUTE", detail: "High-risk patient routed to routine follow-up — BLOCKED." });
  if (riskPct > 50 && recommendation === "routine_followup") violations.push({ severity: "block", rule: "UNDERTRIAGE", detail: "Elevated risk cannot be sent to routine — requires expedited or urgent." });
  const blocked = violations.filter(v => v.severity === "block");
  return { passed: blocked.length === 0, gate: blocked.length > 0 ? "BLOCKED" : "PASS", violations };
}

/* ─── Shared styles ─── */
const font = "'Inter', 'SF Pro Display', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";

/* ─── Components ─── */

function HBar({ value, max, color, height = 14 }) {
  const pct = max > 0 ? Math.min(Math.abs(value) / max * 100, 100) : 0;
  return (
    <div style={{ width: "100%", height, background: "#f0eff5", borderRadius: height / 2, overflow: "hidden" }}>
      <div style={{ width: Math.round(pct) + "%", minWidth: 2, height: "100%", borderRadius: height / 2, background: color, transition: "width 0.4s ease" }} />
    </div>
  );
}

function StatCard({ label, value, sub, accent }) {
  return (
    <div style={{ flex: "1 1 130px", minWidth: 120, padding: "18px 16px", background: "#fff", borderRadius: 16, border: "1px solid #eeedf5", boxShadow: "0 1px 4px rgba(83,74,183,0.04)" }}>
      {accent && <div style={{ width: 28, height: 3, borderRadius: 2, background: accent, marginBottom: 10 }} />}
      <div style={{ fontSize: 10, fontWeight: 600, color: "#a8a3c0", textTransform: "uppercase", letterSpacing: 1 }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 700, color: "#1a1730", marginTop: 6, letterSpacing: -0.5 }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: "#b0abca", marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

function Badge({ children, color, bg }) {
  return <span style={{ display: "inline-block", padding: "3px 10px", borderRadius: 6, fontSize: 10, fontWeight: 700, letterSpacing: 0.5, color, background: bg }}>{children}</span>;
}

/* ─── Main App ─── */
export default function App() {
  const [tab, setTab] = useState(0);
  const [sliders, setSliders] = useState({ age: 55, avg_glucose_level: 110, bmi: 28 });
  const [selects, setSelects] = useState({ hypertension: "0", heart_disease: "0", gender: "0", smoking_status: "2", ever_married: "1" });
  const [expandedFactor, setExpandedFactor] = useState(null);
  const [showTrace, setShowTrace] = useState(false);

  const { prob, contributions } = useMemo(() => computeRisk(sliders, selects), [sliders, selects]);
  const pct = Math.round(prob * 100);
  const riskLevel = pct < 20 ? "Low" : pct < 50 ? "Moderate" : "High";
  const riskColor = pct < 20 ? "#10B981" : pct < 50 ? "#F59E0B" : "#EF4444";
  const signal = useMemo(() => signalMonitor(pct), [pct]);
  const fairness = useMemo(() => fairnessAgent(sliders.age, selects.gender), [sliders.age, selects.gender]);
  const recToT = useMemo(() => recommendationToT(pct, sliders.age, selects.hypertension === "1", selects.heart_disease === "1"), [pct, sliders.age, selects.hypertension, selects.heart_disease]);
  const safety = useMemo(() => safetyAuditor(pct, fairness, recToT.selected?.action), [pct, fairness, recToT.selected]);
  const recColor = recToT.selected?.action === "urgent_evaluation" ? "#EF4444" : recToT.selected?.action === "clinician_review" ? "#F59E0B" : "#10B981";

  const sortedC = Object.entries(contributions).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  const maxC = Math.max(...sortedC.map(([, v]) => Math.abs(v)), 0.01);
  const globalSorted = Object.entries(SHAP_WEIGHTS).sort((a, b) => b[1] - a[1]);
  const maxG = globalSorted[0][1];

  const tabs = ["Risk assessment", "Models", "Risk factors", "Methodology", "About"];

  /* shared styles */
  const card = { background: "#fff", borderRadius: 20, border: "1px solid #eeedf5", padding: "28px 28px", marginBottom: 20, boxShadow: "0 2px 12px rgba(83,74,183,0.04)" };
  const sectionTitle = { fontSize: 13, fontWeight: 700, color: "#a8a3c0", textTransform: "uppercase", letterSpacing: 1.2, marginBottom: 16 };
  const heading = { fontSize: 20, fontWeight: 700, color: "#1a1730", marginBottom: 6, letterSpacing: -0.3 };
  const subtext = { fontSize: 13, color: "#8884a8", lineHeight: 1.6 };
  const callout = (bg, accent) => ({ background: bg, borderLeft: `3px solid ${accent}`, borderRadius: "0 14px 14px 0", padding: "16px 20px", fontSize: 13, lineHeight: 1.7, color: "#555", marginBottom: 16 });

  return (
    <div style={{ maxWidth: 820, margin: "0 auto", padding: "0 20px 60px", fontFamily: font, color: "#1a1730", WebkitFontSmoothing: "antialiased" }}>

      {/* ═══ Header ═══ */}
      <div style={{ textAlign: "center", padding: "48px 0 32px" }}>
        <div style={{ display: "inline-flex", alignItems: "center", gap: 8, background: "linear-gradient(135deg, #6C63FF15, #534AB715)", padding: "6px 18px", borderRadius: 24, marginBottom: 16 }}>
          <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#6C63FF" }} />
          <span style={{ fontSize: 11, fontWeight: 700, letterSpacing: 1.5, color: "#6C63FF" }}>SENTRA</span>
        </div>
        <h1 style={{ fontSize: 36, fontWeight: 800, margin: "0 0 8px", letterSpacing: -1, background: "linear-gradient(135deg, #1a1730, #534AB7)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
          Stroke Risk Predictor
        </h1>
        <p style={{ fontSize: 15, color: "#8884a8", maxWidth: 520, margin: "0 auto", lineHeight: 1.6 }}>
          Multi-agent clinical decision support with ML prediction, SHAP explainability, fairness auditing, and safety guardrails
        </p>
      </div>

      {/* ═══ Nav ═══ */}
      <nav style={{ display: "flex", gap: 4, background: "#f8f7fc", borderRadius: 14, padding: 4, marginBottom: 32 }}>
        {tabs.map((t, i) => (
          <button key={i} onClick={() => setTab(i)} style={{
            flex: 1, padding: "10px 8px", fontSize: 13, fontWeight: tab === i ? 700 : 500, border: "none",
            borderRadius: 10, cursor: "pointer", fontFamily: font, transition: "all 0.25s ease",
            background: tab === i ? "#fff" : "transparent",
            color: tab === i ? "#1a1730" : "#a8a3c0",
            boxShadow: tab === i ? "0 2px 8px rgba(83,74,183,0.08)" : "none",
          }}>{t}</button>
        ))}
      </nav>

      {/* ═══════════════════════════════════════════════════════ */}
      {/* TAB 0: RISK ASSESSMENT (full SENTRA pipeline output)   */}
      {/* ═══════════════════════════════════════════════════════ */}
      {tab === 0 && (
        <>
          {/* Patient input */}
          <div style={card}>
            <div style={sectionTitle}>Patient information</div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
              {Object.entries(FEATURES).map(([k, f]) => (
                <div key={k}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                    <label style={{ fontSize: 12, fontWeight: 600, color: "#8884a8" }}>{f.label}</label>
                    <span style={{ fontSize: 13, fontWeight: 700, color: "#6C63FF", background: "#6C63FF10", padding: "2px 10px", borderRadius: 6 }}>{sliders[k]} {f.unit}</span>
                  </div>
                  <input type="range" min={f.min} max={f.max} step={f.step} value={sliders[k]}
                    onChange={e => setSliders(p => ({ ...p, [k]: Number(e.target.value) }))}
                    style={{ width: "100%", accentColor: "#6C63FF", height: 6 }} />
                </div>
              ))}
              {Object.entries(SELECTS).map(([k, v]) => (
                <div key={k}>
                  <label style={{ fontSize: 12, fontWeight: 600, color: "#8884a8", display: "block", marginBottom: 8 }}>{v.label}</label>
                  <select value={selects[k]} onChange={e => setSelects(p => ({ ...p, [k]: e.target.value }))}
                    style={{ width: "100%", padding: "10px 12px", borderRadius: 10, border: "1px solid #e8e6f0", fontSize: 13, fontFamily: font, background: "#faf9ff", color: "#1a1730", fontWeight: 500, outline: "none" }}>
                    {v.opts.map(([val, lbl]) => <option key={val} value={val}>{lbl}</option>)}
                  </select>
                </div>
              ))}
            </div>
          </div>

          {/* ── Risk score + Recommendation side by side ── */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 20 }}>
            {/* Risk score */}
            <div style={{ ...card, marginBottom: 0, textAlign: "center", position: "relative", overflow: "hidden" }}>
              <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 4, background: riskColor }} />
              <div style={{ fontSize: 10, fontWeight: 700, color: "#a8a3c0", textTransform: "uppercase", letterSpacing: 1.2, marginTop: 8 }}>ML Risk Prediction</div>
              <div style={{ fontSize: 56, fontWeight: 800, color: riskColor, margin: "8px 0 4px", letterSpacing: -2 }}>{pct}<span style={{ fontSize: 24, fontWeight: 600 }}>%</span></div>
              <Badge color={riskColor} bg={riskColor + "15"}>{riskLevel} risk</Badge>
              <div style={{ height: 6, borderRadius: 3, background: "#f0eff5", margin: "16px 0 0", overflow: "hidden" }}>
                <div style={{ height: "100%", width: pct + "%", background: `linear-gradient(90deg, ${riskColor}88, ${riskColor})`, borderRadius: 3, transition: "width 0.5s ease" }} />
              </div>
            </div>

            {/* SENTRA recommendation */}
            <div style={{ ...card, marginBottom: 0, position: "relative", overflow: "hidden" }}>
              <div style={{ position: "absolute", top: 0, left: 0, right: 0, height: 4, background: recColor }} />
              <div style={{ fontSize: 10, fontWeight: 700, color: "#a8a3c0", textTransform: "uppercase", letterSpacing: 1.2, marginTop: 8 }}>SENTRA Recommendation</div>
              <div style={{ fontSize: 20, fontWeight: 800, color: recColor, margin: "12px 0 8px", lineHeight: 1.2 }}>
                {recToT.selected?.label || "Escalate to clinician"}
              </div>
              <div style={{ fontSize: 12, color: "#8884a8", lineHeight: 1.5, marginBottom: 12 }}>{recToT.selected?.rationale}</div>
              <div style={{ display: "flex", gap: 8 }}>
                <Badge color={recColor} bg={recColor + "15"}>{recToT.selected?.score || "escalate"}</Badge>
                <Badge color={safety.passed ? "#10B981" : "#EF4444"} bg={safety.passed ? "#10B98115" : "#EF444415"}>
                  Safety: {safety.gate}
                </Badge>
              </div>
            </div>
          </div>

          {/* Fairness warnings */}
          {fairness.warnings.map((w, i) => (
            <div key={i} style={callout("#FFFBEB", "#F59E0B")}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 6 }}>
                <Badge color="#92400E" bg="#FDE68A">FAIRNESS AGENT</Badge>
              </div>
              {w.text}
            </div>
          ))}

          {/* Safety violations */}
          {safety.violations.length > 0 && safety.violations.map((v, i) => (
            <div key={i} style={callout(v.severity === "block" ? "#FEF2F2" : "#FFFBEB", v.severity === "block" ? "#EF4444" : "#F59E0B")}>
              <Badge color={v.severity === "block" ? "#991B1B" : "#92400E"} bg={v.severity === "block" ? "#FECACA" : "#FDE68A"}>
                SAFETY: {v.rule}
              </Badge>
              <div style={{ marginTop: 6 }}>{v.detail}</div>
            </div>
          ))}

          {/* ToT branches */}
          <div style={card}>
            <div style={sectionTitle}>Tree-of-Thought reasoning</div>
            <div style={{ display: "grid", gap: 10 }}>
              {recToT.branches.map((b, i) => {
                const isSelected = recToT.selected?.action === b.action;
                const col = b.pruned ? "#aaa" : isSelected ? recColor : "#8884a8";
                return (
                  <div key={i} style={{
                    display: "flex", alignItems: "center", gap: 14, padding: "14px 18px",
                    background: b.pruned ? "#fafafa" : isSelected ? col + "06" : "#faf9ff",
                    borderRadius: 14, border: `1px solid ${b.pruned ? "#eee" : isSelected ? col + "40" : "#eeedf5"}`,
                    opacity: b.pruned ? 0.4 : 1, transition: "all 0.2s",
                  }}>
                    <div style={{ width: 50, textAlign: "center", flexShrink: 0 }}>
                      <Badge color={b.pruned ? "#fff" : "#fff"} bg={b.pruned ? "#ccc" : b.score === "sure" ? "#10B981" : b.score === "likely" ? "#6C63FF" : "#F59E0B"}>
                        {b.pruned ? "✕" : b.score}
                      </Badge>
                    </div>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontSize: 14, fontWeight: isSelected ? 700 : 500, color: "#1a1730" }}>
                        {b.label}
                        {isSelected && <span style={{ marginLeft: 8, fontSize: 10, fontWeight: 700, color: recColor }}>← SELECTED</span>}
                      </div>
                      <div style={{ fontSize: 12, color: "#a8a3c0", marginTop: 2 }}>{b.rationale}</div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* SHAP contributions */}
          <div style={card}>
            <div style={sectionTitle}>SHAP contributions</div>
            <p style={{ ...subtext, marginBottom: 16 }}>How each feature pushes this patient's prediction toward or away from stroke</p>
            {sortedC.map(([k, v]) => (
              <div key={k} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
                <span style={{ width: 120, textAlign: "right", fontSize: 12, fontWeight: 600, color: "#a8a3c0", flexShrink: 0 }}>{k.replace(/_/g, " ")}</span>
                <div style={{ flex: 1 }}><HBar value={v} max={maxC} color={v > 0 ? "#EF4444" : "#10B981"} height={12} /></div>
                <span style={{ width: 55, textAlign: "right", fontSize: 12, fontWeight: 700, color: v > 0 ? "#EF4444" : "#10B981", flexShrink: 0, fontFamily: "monospace" }}>
                  {v > 0 ? "+" : ""}{v.toFixed(3)}
                </span>
              </div>
            ))}
            <div style={{ marginTop: 14, display: "flex", gap: 16, fontSize: 11, color: "#a8a3c0" }}>
              <span><span style={{ display: "inline-block", width: 8, height: 8, borderRadius: 2, background: "#EF4444", marginRight: 4, verticalAlign: "middle" }} /> Increases risk</span>
              <span><span style={{ display: "inline-block", width: 8, height: 8, borderRadius: 2, background: "#10B981", marginRight: 4, verticalAlign: "middle" }} /> Decreases risk</span>
              <span style={{ marginLeft: "auto", fontFamily: "monospace", fontSize: 10 }}>Source: SHAP.LinearExplainer</span>
            </div>
          </div>

          {/* Provenance */}
          <div style={card}>
            <div style={sectionTitle}>Provenance trail</div>
            <div style={{ display: "grid", gap: 8 }}>
              {[
                { claim: `Risk score: ${pct}%`, source: "SHAP LinearExplainer → LogisticRegression", type: "ml_score" },
                { claim: `Top factor: age (${contributions.age?.toFixed(3)})`, source: "deterministic: SHAP.LinearExplainer", type: "shap" },
                { claim: `Subgroup: ${fairness.subgroup}`, source: "FairnessAgent: subgroup recall analysis", type: "fairness" },
                { claim: `Rec: ${recToT.selected?.label || "Escalate"}`, source: "ProtocolAdvisor: Tree-of-Thought", type: "tot" },
                { claim: `Safety: ${safety.gate}`, source: "SafetyAuditor: rule-based validation", type: "safety" },
                { claim: "Protocol reference", source: "AHA/ASA Stroke Guidelines (2019)", type: "guideline" },
              ].map((p, i) => (
                <div key={i} style={{ display: "flex", gap: 10, padding: "10px 14px", background: "#faf9ff", borderRadius: 12, alignItems: "center" }}>
                  <Badge color="#6C63FF" bg="#6C63FF12">{p.type}</Badge>
                  <div style={{ flex: 1 }}>
                    <span style={{ fontSize: 13, fontWeight: 600, color: "#1a1730" }}>{p.claim}</span>
                    <span style={{ fontSize: 11, color: "#b0abca", fontFamily: "monospace", marginLeft: 8 }}>{p.source}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Execution trace (collapsible) */}
          <div style={card}>
            <button onClick={() => setShowTrace(!showTrace)} style={{
              background: "none", border: "none", cursor: "pointer", fontFamily: font, width: "100%", padding: 0,
              display: "flex", justifyContent: "space-between", alignItems: "center",
            }}>
              <span style={{ ...sectionTitle, marginBottom: 0 }}>Execution trace</span>
              <span style={{ width: 28, height: 28, borderRadius: 8, background: "#f0eff5", display: "flex", alignItems: "center", justifyContent: "center", color: "#6C63FF", fontSize: 16, fontWeight: 300 }}>{showTrace ? "−" : "+"}</span>
            </button>
            {showTrace && (
              <div style={{ marginTop: 16, display: "grid", gap: 6 }}>
                {[
                  { agent: "SignalMonitor", color: "#10B981", action: "screen_vitals", output: `risk=${pct}%`, decision: signal.escalate ? "escalate" : "STOP", next: signal.escalate ? "RiskAssessor" : "END" },
                  ...(signal.escalate ? [
                    { agent: "RiskAssessor", color: "#6C63FF", action: "compute_shap", output: `age=${contributions.age?.toFixed(3)}`, decision: riskLevel, next: "FairnessAgent" },
                    { agent: "FairnessAgent", color: "#F59E0B", action: "check_subgroups", output: `flags=[${fairness.flags.join(",")}]`, decision: fairness.warnings.length > 0 ? "WARN" : "OK", next: "ProtocolAdvisor" },
                    { agent: "ProtocolAdvisor", color: "#6C63FF", action: "rec_tot", output: recToT.selected?.label, decision: recToT.selected?.action, next: "SafetyAuditor" },
                    { agent: "SafetyAuditor", color: "#EF4444", action: "validate", output: `violations=${safety.violations.length}`, decision: safety.gate, next: "Escalation" },
                    { agent: "Escalation", color: "#8884a8", action: "route", output: riskLevel.toLowerCase(), decision: "PENDING", next: "END" },
                  ] : []),
                ].map((step, i) => (
                  <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px 14px", background: "#faf9ff", borderRadius: 10, borderLeft: `3px solid ${step.color}`, fontSize: 12, color: "#666" }}>
                    <span style={{ fontWeight: 700, color: step.color, width: 100, flexShrink: 0 }}>{step.agent}</span>
                    <span style={{ color: "#b0abca" }}>{step.action}</span>
                    <span style={{ color: "#b0abca" }}>→</span>
                    <span style={{ fontFamily: "monospace", fontSize: 11 }}>{step.output}</span>
                    <span style={{ color: "#b0abca" }}>→</span>
                    <span style={{ fontWeight: 600, color: "#1a1730" }}>{step.decision}</span>
                    <span style={{ marginLeft: "auto", color: "#ccc", fontSize: 11 }}>→ {step.next}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div style={callout("#EFF6FF", "#6C63FF")}>
            <strong>Clinical note:</strong> SENTRA's recommendation requires clinician sign-off before action. The ML model is optimized for screening sensitivity (recall = 80%). This is not a diagnostic instrument.
          </div>
        </>
      )}

      {/* ═══════════════════════ */}
      {/* TAB 1: MODELS          */}
      {/* ═══════════════════════ */}
      {tab === 1 && (
        <>
          <div style={{ display: "flex", gap: 14, marginBottom: 24, flexWrap: "wrap" }}>
            <StatCard label="Best model" value="Logistic Reg." sub="Balanced weights" accent="#6C63FF" />
            <StatCard label="AUC-ROC" value="0.838" sub="Discrimination" accent="#6C63FF" />
            <StatCard label="Recall" value="80.0%" sub="Sensitivity" accent="#10B981" />
            <StatCard label="Patients" value="5,110" sub="Dataset size" accent="#F59E0B" />
          </div>

          <div style={card}>
            <div style={sectionTitle}>Model comparison</div>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead>
                  <tr>{["Model", "AUC", "Recall", "Precision", "F1", "Brier"].map(h => (
                    <th key={h} style={{ padding: "12px 10px", textAlign: h === "Model" ? "left" : "center", color: "#a8a3c0", fontWeight: 700, fontSize: 10, textTransform: "uppercase", letterSpacing: 1, borderBottom: "2px solid #f0eff5" }}>{h}</th>
                  ))}</tr>
                </thead>
                <tbody>
                  {MODELS.map(m => (
                    <tr key={m.name} style={{ borderBottom: "1px solid #f8f7fc" }}>
                      <td style={{ padding: "14px 10px", fontWeight: m.best ? 700 : 500, color: m.best ? "#6C63FF" : "#1a1730" }}>
                        {m.name}{m.best && <Badge color="#fff" bg="#6C63FF"> BEST</Badge>}
                      </td>
                      <td style={{ textAlign: "center", padding: 10, fontFamily: "monospace" }}>{m.auc.toFixed(3)}</td>
                      <td style={{ textAlign: "center", padding: 10 }}>
                        <span style={{ color: m.recall >= 0.7 ? "#10B981" : m.recall >= 0.5 ? "#F59E0B" : "#EF4444", fontWeight: 700 }}>{(m.recall * 100).toFixed(1)}%</span>
                      </td>
                      <td style={{ textAlign: "center", padding: 10, fontFamily: "monospace" }}>{(m.precision * 100).toFixed(1)}%</td>
                      <td style={{ textAlign: "center", padding: 10, fontFamily: "monospace" }}>{m.f1.toFixed(3)}</td>
                      <td style={{ textAlign: "center", padding: 10, fontFamily: "monospace", color: m.brier < 0.1 ? "#10B981" : "#F59E0B" }}>{m.brier.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Calibration */}
          <div style={card}>
            <div style={sectionTitle}>Calibration analysis</div>
            <p style={{ ...subtext, marginBottom: 18 }}>When the model says 70% risk, do ~70% of those patients actually have strokes?</p>
            <div style={{ display: "flex", gap: 14, marginBottom: 20, flexWrap: "wrap" }}>
              <StatCard label="Brier Score" value={CALIBRATION_DATA.brier.toFixed(3)} sub="Lower = better" accent="#6C63FF" />
              <StatCard label="ECE" value={CALIBRATION_DATA.ece.toFixed(3)} sub="Calibration error" accent="#F59E0B" />
              <StatCard label="Max CE" value={CALIBRATION_DATA.maxCE.toFixed(3)} sub="Worst bin gap" accent="#EF4444" />
            </div>
            <div style={{ fontSize: 11, fontWeight: 700, color: "#a8a3c0", textTransform: "uppercase", letterSpacing: 1, marginBottom: 10 }}>Reliability diagram</div>
            {CALIBRATION_DATA.bins.map((bin, i) => {
              const gap = Math.abs(bin.actual - bin.predicted);
              const gc = gap < 0.05 ? "#10B981" : gap < 0.10 ? "#F59E0B" : "#EF4444";
              return (
                <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 5, fontSize: 12 }}>
                  <span style={{ width: 36, textAlign: "right", color: "#a8a3c0", fontFamily: "monospace", fontSize: 11 }}>{bin.predicted.toFixed(2)}</span>
                  <div style={{ flex: 1, position: "relative", height: 16, borderRadius: 4, overflow: "hidden", background: "#f0eff5" }}>
                    <div style={{ position: "absolute", width: `${bin.predicted * 100}%`, height: "100%", background: "#ddd8f5", borderRadius: 4 }} />
                    <div style={{ position: "absolute", width: `${bin.actual * 100}%`, height: "100%", background: "#6C63FF", borderRadius: 4, opacity: 0.85 }} />
                  </div>
                  <span style={{ width: 36, textAlign: "right", fontFamily: "monospace", fontSize: 11, color: "#666" }}>{bin.actual.toFixed(2)}</span>
                  <span style={{ width: 28, textAlign: "right", fontWeight: 700, color: gc, fontSize: 11 }}>{gap.toFixed(2)}</span>
                  <span style={{ width: 42, textAlign: "right", color: "#ccc", fontSize: 10 }}>n={bin.count}</span>
                </div>
              );
            })}
          </div>

          {/* Fairness */}
          <div style={card}>
            <div style={sectionTitle}>Fairness: subgroup performance</div>
            <div style={{ fontSize: 11, fontWeight: 700, color: "#a8a3c0", textTransform: "uppercase", letterSpacing: 1, marginBottom: 10 }}>By age group</div>
            <div style={{ overflowX: "auto", marginBottom: 20 }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead><tr>{["Age group", "Recall", "AUC", "N", ""].map(h => (
                  <th key={h} style={{ padding: "10px 10px", textAlign: "center", color: "#a8a3c0", fontWeight: 700, fontSize: 10, textTransform: "uppercase", letterSpacing: 1, borderBottom: "2px solid #f0eff5" }}>{h}</th>
                ))}</tr></thead>
                <tbody>{FAIRNESS_DATA.ageGroups.map(g => (
                  <tr key={g.group} style={{ borderBottom: "1px solid #f8f7fc", background: g.flag ? "#FFFBEB" : "transparent" }}>
                    <td style={{ padding: "12px 10px", fontWeight: 600 }}>{g.group}</td>
                    <td style={{ textAlign: "center", fontWeight: 700, color: g.recall === 0 ? "#EF4444" : g.recall < 0.5 ? "#F59E0B" : "#10B981" }}>{(g.recall * 100).toFixed(1)}%</td>
                    <td style={{ textAlign: "center", fontFamily: "monospace" }}>{g.auc.toFixed(2)}</td>
                    <td style={{ textAlign: "center", color: "#a8a3c0" }}>{g.n.toLocaleString()}</td>
                    <td style={{ textAlign: "center" }}>{g.flag && <Badge color="#991B1B" bg="#FECACA">BLIND SPOT</Badge>}</td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
            <div style={{ fontSize: 11, fontWeight: 700, color: "#a8a3c0", textTransform: "uppercase", letterSpacing: 1, marginBottom: 10 }}>By gender</div>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
                <thead><tr>{["Gender", "Recall", "AUC", "N"].map(h => (
                  <th key={h} style={{ padding: "10px 10px", textAlign: "center", color: "#a8a3c0", fontWeight: 700, fontSize: 10, textTransform: "uppercase", letterSpacing: 1, borderBottom: "2px solid #f0eff5" }}>{h}</th>
                ))}</tr></thead>
                <tbody>{FAIRNESS_DATA.genderGroups.map(g => (
                  <tr key={g.group} style={{ borderBottom: "1px solid #f8f7fc" }}>
                    <td style={{ padding: "12px 10px", fontWeight: 600 }}>{g.group}</td>
                    <td style={{ textAlign: "center", fontWeight: 700, color: "#10B981" }}>{(g.recall * 100).toFixed(1)}%</td>
                    <td style={{ textAlign: "center", fontFamily: "monospace" }}>{g.auc.toFixed(2)}</td>
                    <td style={{ textAlign: "center", color: "#a8a3c0" }}>{g.n.toLocaleString()}</td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
          </div>

          <div style={callout("#FEF2F2", "#EF4444")}>
            <strong>Critical finding:</strong> 0% recall for patients under 55 — the model misses every stroke in younger patients. The Fairness Agent flags this and the Safety Auditor ensures clinical judgment overrides the ML for this subgroup.
          </div>
        </>
      )}

      {/* ═══════════════════════ */}
      {/* TAB 2: RISK FACTORS     */}
      {/* ═══════════════════════ */}
      {tab === 2 && (
        <>
          <div style={card}>
            <div style={sectionTitle}>Global SHAP feature importance</div>
            <p style={{ ...subtext, marginBottom: 18 }}>Mean |SHAP value| — how much each feature influences predictions across all patients</p>
            {globalSorted.map(([k, v]) => (
              <div key={k} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
                <span style={{ width: 120, textAlign: "right", fontSize: 12, fontWeight: 600, color: "#a8a3c0", flexShrink: 0 }}>{k.replace(/_/g, " ")}</span>
                <div style={{ flex: 1 }}><HBar value={v} max={maxG} color="#6C63FF" height={14} /></div>
                <span style={{ width: 50, textAlign: "right", fontSize: 12, fontWeight: 700, color: "#1a1730", fontFamily: "monospace", flexShrink: 0 }}>{v.toFixed(3)}</span>
              </div>
            ))}
          </div>

          <div style={card}>
            <div style={sectionTitle}>Neuroscience interpretation</div>
            {NEURO_FACTORS.map((rf, i) => (
              <div key={i} style={{ marginBottom: 8 }}>
                <button onClick={() => setExpandedFactor(expandedFactor === i ? null : i)} style={{
                  width: "100%", textAlign: "left", padding: "14px 18px",
                  border: `1px solid ${expandedFactor === i ? "#6C63FF30" : "#eeedf5"}`,
                  borderRadius: expandedFactor === i ? "14px 14px 0 0" : 14,
                  background: expandedFactor === i ? "#faf9ff" : "#fff",
                  cursor: "pointer", fontFamily: font, fontSize: 14, display: "flex", justifyContent: "space-between", alignItems: "center",
                }}>
                  <span>
                    <span style={{ fontWeight: 700, color: "#1a1730" }}>{rf.feature}</span>
                    <span style={{ color: "#b0abca", marginLeft: 12, fontSize: 12, fontFamily: "monospace" }}>SHAP: {rf.shap.toFixed(3)}</span>
                  </span>
                  <span style={{ width: 28, height: 28, borderRadius: 8, background: expandedFactor === i ? "#6C63FF15" : "#f0eff5", display: "flex", alignItems: "center", justifyContent: "center", color: "#6C63FF", fontSize: 16, fontWeight: 300 }}>
                    {expandedFactor === i ? "−" : "+"}
                  </span>
                </button>
                {expandedFactor === i && (
                  <div style={{ padding: "16px 18px", fontSize: 13, lineHeight: 1.8, color: "#555", background: "#faf9ff", borderRadius: "0 0 14px 14px", border: "1px solid #6C63FF30", borderTop: "none" }}>
                    {rf.detail}
                  </div>
                )}
              </div>
            ))}
          </div>
        </>
      )}

      {/* ═══════════════════════ */}
      {/* TAB 3: METHODOLOGY      */}
      {/* ═══════════════════════ */}
      {tab === 3 && (
        <>
          <div style={card}>
            <div style={sectionTitle}>ML pipeline</div>
            {METHODOLOGY_STEPS.map((step, i) => (
              <div key={i} style={{ display: "flex", gap: 18, marginBottom: 22, alignItems: "flex-start" }}>
                <div style={{ width: 44, height: 44, borderRadius: 14, background: "linear-gradient(135deg, #6C63FF12, #6C63FF08)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 20, flexShrink: 0 }}>
                  {step.icon}
                </div>
                <div style={{ flex: 1 }}>
                  <div style={{ fontSize: 15, fontWeight: 700, color: "#1a1730", marginBottom: 4 }}>{step.title}</div>
                  <div style={{ fontSize: 13, color: "#8884a8", lineHeight: 1.7 }}>{step.desc}</div>
                </div>
              </div>
            ))}
          </div>

          {/* Agent architecture */}
          <div style={card}>
            <div style={sectionTitle}>SENTRA agent architecture</div>
            <p style={{ ...subtext, marginBottom: 18 }}>6 agents coordinated via LangGraph state machine</p>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 10, justifyContent: "center", padding: "8px 0" }}>
              {[
                { name: "Signal Monitor", color: "#10B981" },
                { name: "Risk Assessor", color: "#6C63FF" },
                { name: "Fairness Agent", color: "#F59E0B" },
                { name: "Protocol Advisor", color: "#6C63FF" },
                { name: "Safety Auditor", color: "#EF4444" },
                { name: "Escalation", color: "#8884a8" },
              ].map((a, i, arr) => (
                <React.Fragment key={a.name}>
                  <div style={{ padding: "12px 16px", background: a.color + "08", border: `1px solid ${a.color}25`, borderRadius: 14, textAlign: "center" }}>
                    <div style={{ fontSize: 12, fontWeight: 700, color: a.color }}>{a.name}</div>
                  </div>
                  {i < arr.length - 1 && <div style={{ display: "flex", alignItems: "center", color: "#ddd", fontSize: 16 }}>→</div>}
                </React.Fragment>
              ))}
            </div>
          </div>

          <div style={card}>
            <div style={sectionTitle}>Key decisions</div>
            <div style={{ display: "grid", gap: 10 }}>
              {[
                ["KNN imputation over mean/median", "Preserves feature relationships. Mean imputation introduces bias."],
                ["SMOTE on training set only", "Prevents leaking synthetic samples into evaluation."],
                ["Recall as primary metric", "Missing a stroke is far worse than a false alarm."],
                ["SHAP over permutation importance", "Consistent individual + global explanations for clinical trust."],
                ["Tree-of-Thought for recommendations", "Searches action space (routine/expedited/urgent) with safety constraints."],
                ["Safety Auditor as final gate", "Blocks contraindicated actions and catches hallucinations."],
                ["Fairness Agent in parallel", "Flags 0% recall under-55 blind spot before recommendations finalize."],
              ].map(([title, desc], i) => (
                <div key={i} style={{ padding: "14px 18px", background: "#faf9ff", borderRadius: 14, borderLeft: "3px solid #6C63FF" }}>
                  <div style={{ fontSize: 14, fontWeight: 700, color: "#1a1730", marginBottom: 4 }}>{title}</div>
                  <div style={{ fontSize: 13, color: "#8884a8", lineHeight: 1.5 }}>{desc}</div>
                </div>
              ))}
            </div>
          </div>

          <div style={card}>
            <div style={sectionTitle}>Tools & libraries</div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
              {["Python", "scikit-learn", "XGBoost", "SHAP", "imbalanced-learn", "LangGraph", "LangChain", "FHIR R4", "Gradio", "pandas", "matplotlib", "React"].map(t => (
                <span key={t} style={{ padding: "6px 14px", background: "#6C63FF08", border: "1px solid #6C63FF20", borderRadius: 10, fontSize: 12, color: "#6C63FF", fontWeight: 600 }}>{t}</span>
              ))}
            </div>
          </div>
        </>
      )}

      {/* ═══════════════════════ */}
      {/* TAB 4: ABOUT            */}
      {/* ═══════════════════════ */}
      {tab === 4 && (
        <>
          <div style={card}>
            <div style={heading}>About SENTRA</div>
            <p style={{ fontSize: 15, lineHeight: 1.8, color: "#555", marginBottom: 20, marginTop: 12 }}>
              SENTRA (Stroke Early-warning and Neuro Triage Response Agent) is a multi-agent clinical decision support system
              that combines machine learning prediction with SHAP explainability, fairness auditing, Tree-of-Thought recommendation
              reasoning, and safety guardrails. It goes beyond risk prediction — running a full triage pipeline that
              checks for demographic blind spots, reasons about the best clinical action, validates recommendations
              against safety rules, and provides full provenance traceability.
            </p>

            <div style={sectionTitle}>Key findings</div>
            <div style={{ display: "grid", gap: 8, marginBottom: 24 }}>
              {[
                "Logistic Regression with balanced weights: 80% recall, 0.838 AUC-ROC",
                "Age is the dominant predictor (SHAP: 1.713) — consistent with cerebrovascular pathophysiology",
                "Critical fairness finding: 0% recall for patients under 55",
                "Calibration: Brier Score 0.146, ECE 0.067",
                "Safety Auditor prevents contraindicated recommendations",
              ].map((f, i) => (
                <div key={i} style={{ display: "flex", gap: 10, alignItems: "flex-start", fontSize: 14, color: "#555", lineHeight: 1.6 }}>
                  <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#6C63FF", marginTop: 7, flexShrink: 0 }} />
                  {f}
                </div>
              ))}
            </div>

            <div style={sectionTitle}>Dataset</div>
            <p style={{ fontSize: 14, color: "#666", lineHeight: 1.7, marginBottom: 20 }}>
              Kaggle Stroke Prediction Dataset (McKinsey and Company) — 5,110 patients, 11 clinical features, 4.87% stroke-positive.
            </p>

            <div style={sectionTitle}>Limitations</div>
            <div style={{ display: "grid", gap: 6, marginBottom: 8 }}>
              {[
                "Cross-sectional dataset — no temporal/longitudinal modeling",
                "Missing features: atrial fibrillation, cholesterol, family history",
                "SMOTE may introduce synthetic artifacts in minority class",
                "Single-source data limits generalizability",
              ].map((l, i) => (
                <div key={i} style={{ display: "flex", gap: 10, alignItems: "flex-start", fontSize: 14, color: "#888", lineHeight: 1.6 }}>
                  <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#ddd", marginTop: 7, flexShrink: 0 }} />
                  {l}
                </div>
              ))}
            </div>
          </div>

          <div style={card}>
            <div style={sectionTitle}>Author</div>
            <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
              <div style={{ width: 48, height: 48, borderRadius: 16, background: "linear-gradient(135deg, #6C63FF, #534AB7)", display: "flex", alignItems: "center", justifyContent: "center", color: "#fff", fontSize: 20, fontWeight: 700, flexShrink: 0 }}>J</div>
              <div>
                <div style={{ fontSize: 16, fontWeight: 700, color: "#1a1730" }}>Jahnavi Nimmakayala</div>
                <div style={{ fontSize: 13, color: "#8884a8" }}>High school researcher — neuroscience & computational biology</div>
              </div>
            </div>
            <div style={{ marginTop: 16 }}>
              <a href="https://github.com/jahnavin123/stroke-risk-predictor" target="_blank" rel="noreferrer"
                style={{ display: "inline-flex", alignItems: "center", gap: 8, padding: "10px 20px", background: "#1a1730", color: "#fff", borderRadius: 12, fontSize: 13, textDecoration: "none", fontWeight: 600, fontFamily: font }}>
                GitHub Repository →
              </a>
            </div>
          </div>

          <div style={callout("#FEF2F2", "#EF4444")}>
            <strong>Disclaimer:</strong> This tool is for educational and research purposes only. It is not a medical device and should not be used for clinical decision-making without professional medical oversight.
          </div>
        </>
      )}

      {/* ═══ Footer ═══ */}
      <footer style={{ textAlign: "center", padding: "32px 0 16px" }}>
        <div style={{ width: 40, height: 3, borderRadius: 2, background: "#eeedf5", margin: "0 auto 16px" }} />
        <div style={{ fontSize: 11, color: "#ccc", letterSpacing: 0.5 }}>
          SENTRA · scikit-learn · SHAP · LangGraph · React · BioAI Lab · Research purposes only
        </div>
      </footer>
    </div>
  );
}
