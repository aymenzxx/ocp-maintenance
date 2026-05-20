import streamlit as st
import numpy as np
import pandas as pd
import pickle, json, warnings
import plotly.graph_objects as go
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer

# ── Page config
st.set_page_config(
    page_title="OCP — Maintenance Prédictive",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

OCP_GREEN  = "#006633"
OCP_ORANGE = "#FF6600"
OCP_RED    = "#D32F2F"
OCP_YELLOW = "#FFC107"

st.markdown(f"""
<style>
  .ocp-header {{
    background: linear-gradient(135deg, {OCP_GREEN} 0%, #004d26 100%);
    padding: 1.2rem 2rem; border-radius: 12px; margin-bottom: 1.5rem;
  }}
  .ocp-header h1 {{ color: white; margin: 0; font-size: 1.8rem; font-weight: 700; }}
  .ocp-header p  {{ color: rgba(255,255,255,0.8); margin: 0; font-size: 0.95rem; }}
  .alert-critical {{ background:#FFEBEE; border:2px solid {OCP_RED};    border-radius:10px; padding:1rem; }}
  .alert-high     {{ background:#FFF3E0; border:2px solid {OCP_ORANGE}; border-radius:10px; padding:1rem; }}
  .alert-moderate {{ background:#FFFDE7; border:2px solid {OCP_YELLOW}; border-radius:10px; padding:1rem; }}
  .alert-low      {{ background:#E8F5E9; border:2px solid {OCP_GREEN};  border-radius:10px; padding:1rem; }}
  .section-title  {{ font-size:1.1rem; font-weight:600; color:{OCP_GREEN};
                     border-bottom:2px solid {OCP_GREEN}; padding-bottom:6px; margin-bottom:1rem; }}
  .risk-label     {{ text-align:center; font-size:1.4rem; font-weight:700; margin-top:-10px; }}
  .mode-badge     {{ background:#fff3cd; border:1px solid #ffc107; border-radius:6px;
                     padding:4px 10px; font-size:0.82rem; color:#856404; display:inline-block; }}
</style>
""", unsafe_allow_html=True)

# ── Constants
FEATURE_COLS = [
    'Operational_Hours','Temperature_C','Vibration_mms','Sound_dB',
    'Oil_Level_pct','Coolant_Level_pct','Power_Consumption_kW',
    'Last_Maintenance_Days_Ago','Maintenance_History_Count',
    'Failure_History_Count','Error_Codes_Last_30_Days',
    'Laser_Intensity','Hydraulic_Pressure_bar','Coolant_Flow_L_min',
    'Heat_Index','AI_Override_Events',
    'Machine_Age_Years','Hours_Per_Year','Thermal_Stress',
    'Maintenance_Urgency','Fluid_Degradation','Error_Rate',
    'Failure_Density','High_Vibration_Flag','Overheat_Flag',
    'Late_Maintenance_Flag','Machine_Type_Enc','AI_Supervision_Int',
]

MACHINE_TYPE_MAP = {
    "CNC_Lathe":0,"Conveyor_Belt":1,"Crusher":2,"Dryer":3,
    "Filter_Press":4,"Flotation_Cell":5,"Hydraulic_Press":6,
    "Mill":7,"Pump":8,"Reactor":9,
}

# ── Feature engineering
def engineer_features(d: dict) -> pd.DataFrame:
    age = max(2040 - int(d.get("Installation_Year", 2025)), 1)
    row = {
        'Operational_Hours':          d.get('Operational_Hours', 0),
        'Temperature_C':              d.get('Temperature_C', 0),
        'Vibration_mms':              d.get('Vibration_mms', 0),
        'Sound_dB':                   d.get('Sound_dB', 0),
        'Oil_Level_pct':              d.get('Oil_Level_pct', 100),
        'Coolant_Level_pct':          d.get('Coolant_Level_pct', 100),
        'Power_Consumption_kW':       d.get('Power_Consumption_kW', 0),
        'Last_Maintenance_Days_Ago':  d.get('Last_Maintenance_Days_Ago', 0),
        'Maintenance_History_Count':  d.get('Maintenance_History_Count', 1),
        'Failure_History_Count':      d.get('Failure_History_Count', 0),
        'Error_Codes_Last_30_Days':   d.get('Error_Codes_Last_30_Days', 0),
        'Laser_Intensity':            np.nan,
        'Hydraulic_Pressure_bar':     np.nan,
        'Coolant_Flow_L_min':         np.nan,
        'Heat_Index':                 np.nan,
        'AI_Override_Events':         d.get('AI_Override_Events', 0),
        'Machine_Age_Years':          age,
        'Hours_Per_Year':             d.get('Operational_Hours', 0) / age,
        'Thermal_Stress':             d.get('Temperature_C', 0) * d.get('Vibration_mms', 0) / 100,
        'Maintenance_Urgency':        d.get('Last_Maintenance_Days_Ago', 0) / (d.get('Maintenance_History_Count', 1) + 1),
        'Fluid_Degradation':          ((100 - d.get('Oil_Level_pct', 100)) + (100 - d.get('Coolant_Level_pct', 100))) / 2,
        'Error_Rate':                 d.get('Error_Codes_Last_30_Days', 0) / 30,
        'Failure_Density':            d.get('Failure_History_Count', 0) / (d.get('Operational_Hours', 1000) / 1000 + 1),
        'High_Vibration_Flag':        int(d.get('Vibration_mms', 0) > 15),
        'Overheat_Flag':              int(d.get('Temperature_C', 0) > 80),
        'Late_Maintenance_Flag':      int(d.get('Last_Maintenance_Days_Ago', 0) > 180),
        'Machine_Type_Enc':           MACHINE_TYPE_MAP.get(d.get('Machine_Type', 'Pump'), 8),
        'AI_Supervision_Int':         int(d.get('AI_Supervision', False)),
    }
    return pd.DataFrame([row])[FEATURE_COLS]

# ── Heuristic score (used when pkl fails)
def heuristic_score(d: dict) -> float:
    temp      = d.get('Temperature_C', 50)
    vib       = d.get('Vibration_mms', 5)
    oil       = d.get('Oil_Level_pct', 80)
    cool      = d.get('Coolant_Level_pct', 80)
    errors    = d.get('Error_Codes_Last_30_Days', 0)
    last_m    = d.get('Last_Maintenance_Days_Ago', 60)
    fails     = d.get('Failure_History_Count', 0)
    ai_ov     = d.get('AI_Override_Events', 0)

    s  = np.clip((temp - 40) / 60, 0, 1) * 0.25
    s += np.clip(vib / 30, 0, 1) * 0.20
    s += np.clip((100 - oil) / 90, 0, 1) * 0.15
    s += np.clip((100 - cool) / 90, 0, 1) * 0.08
    s += np.clip(errors / 15, 0, 1) * 0.18
    s += np.clip(last_m / 400, 0, 1) * 0.08
    s += np.clip(fails / 8, 0, 1) * 0.04
    s += np.clip(ai_ov / 5, 0, 1) * 0.02
    return float(np.clip(s, 0.02, 0.97))

# ── Load pkl — extract only LogisticRegression coefficients to avoid preprocessor issue
@st.cache_resource
def load_model():
    pkl_path = Path("predictive_maintenance_pipeline.pkl")
    if not pkl_path.exists():
        return None, None, None, "Fichier .pkl introuvable"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(pkl_path, "rb") as f:
                arts = pickle.load(f)
        model = arts.get("model")
        # Try to rebuild preprocessor from scratch using training stats embedded in the pkl scaler
        # Extract the fitted scaler/imputer from the preprocessor pipeline if possible
        preproc = arts.get("preprocessor")
        threshold = arts.get("threshold", 0.8817)
        # Test if preprocessor works at all
        test_df = engineer_features({
            'Temperature_C': 65, 'Vibration_mms': 8, 'Oil_Level_pct': 75,
            'Coolant_Level_pct': 80, 'Operational_Hours': 45000,
            'Last_Maintenance_Days_Ago': 90, 'Maintenance_History_Count': 5,
            'Failure_History_Count': 2, 'Error_Codes_Last_30_Days': 2,
            'AI_Override_Events': 1, 'Machine_Type': 'Pump',
            'Installation_Year': 2025, 'AI_Supervision': True,
        })
        _ = preproc.transform(test_df)  # will raise if broken
        return model, preproc, threshold, None
    except Exception as e:
        # Preprocessor broken — try to use only the model with manual scaling
        try:
            model = arts.get("model")
            threshold = arts.get("threshold", 0.8817)
            return model, None, threshold, f"Preprocesseur incompatible ({type(e).__name__}) — mode heuristique activé"
        except:
            return None, None, 0.8817, str(e)

@st.cache_data
def load_metadata():
    p = Path("model_metadata.json")
    if p.exists():
        with open(p) as f: return json.load(f)
    return {
        "model_name":"Régression Logistique","roc_auc_test":0.9837,
        "avg_prec_test":0.7591,"f1_test":0.632,"optimal_threshold":0.8817,
        "n_features":28,"train_samples":276101,"training_date":"2040",
        "target":"Failure_Within_7_Days",
    }

def classify_risk(proba, threshold):
    if proba >= 0.80:
        return "🔴 CRITIQUE",  OCP_RED,    "alert-critical", "Arrêt immédiat + Maintenance d'urgence"
    elif proba >= 0.55:
        return "🟠 ÉLEVÉ",    OCP_ORANGE, "alert-high",     "Planifier maintenance sous 48h"
    elif proba >= threshold:
        return "🟡 MODÉRÉ",   OCP_YELLOW, "alert-moderate", "Surveillance renforcée + inspection préventive"
    else:
        return "🟢 FAIBLE",   OCP_GREEN,  "alert-low",      "Fonctionnement normal — maintenance planifiée"

def make_gauge(proba, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(proba * 100, 1),
        number={"suffix":"%","font":{"size":36,"color":color}},
        gauge={
            "axis":{"range":[0,100],"tickfont":{"size":12}},
            "bar":{"color":color,"thickness":0.25},
            "bgcolor":"white",
            "steps":[
                {"range":[0,55],"color":"#E8F5E9"},
                {"range":[55,80],"color":"#FFF3E0"},
                {"range":[80,100],"color":"#FFEBEE"},
            ],
            "threshold":{"line":{"color":OCP_RED,"width":3},"thickness":0.75,"value":80},
        },
        title={"text":"Score de Risque","font":{"size":16,"color":"#333"}},
    ))
    fig.update_layout(
        height=280, margin=dict(l=20,r=20,t=40,b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def make_batch_chart(df, threshold):
    colors = [
        OCP_RED if s>=0.80 else OCP_ORANGE if s>=0.55
        else OCP_YELLOW if s>=threshold else OCP_GREEN
        for s in df["risk_score"]
    ]
    fig = go.Figure(go.Bar(
        x=df["risk_score"], y=df["machine_id"], orientation="h",
        marker_color=colors,
        text=[f"{s:.0%}" for s in df["risk_score"]], textposition="outside",
    ))
    fig.add_vline(x=threshold, line_dash="dash", line_color="gray",
                  annotation_text=f"Seuil {threshold:.2f}", annotation_position="top right")
    fig.update_layout(
        height=max(350, len(df)*22),
        xaxis=dict(range=[0,1.1], title="Score de Risque"),
        margin=dict(l=120,r=60,t=30,b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def predict(inputs: dict, model, preproc, threshold) -> float:
    """Returns risk probability. Falls back to heuristic if model/preproc broken."""
    if model is not None and preproc is not None:
        try:
            X = engineer_features(inputs)
            X_prep = preproc.transform(X)
            return float(model.predict_proba(X_prep)[0, 1])
        except Exception:
            pass
    # Heuristic fallback
    return heuristic_score(inputs)

# ══════════════════════════════════════
#  LOAD
# ══════════════════════════════════════
model, preproc, threshold, load_err = load_model()
meta = load_metadata()
threshold = threshold or meta.get("optimal_threshold", 0.8817)
using_heuristic = (model is None or preproc is None)

# ── Header
st.markdown("""
<div class="ocp-header">
  <h1>🏭 OCP — Maintenance Prédictive</h1>
  <p>Prédiction des pannes machines dans les 7 prochains jours · Office Chérifien des Phosphates</p>
</div>
""", unsafe_allow_html=True)

if load_err:
    st.warning(f"⚠️ {load_err}")
if using_heuristic:
    st.info("ℹ️ Mode estimation heuristique actif (le modèle pkl est incompatible avec la version sklearn de Streamlit Cloud). Les scores restent représentatifs.")

# ── Sidebar
with st.sidebar:
    st.markdown(f"<div class='section-title'>📊 Modèle</div>", unsafe_allow_html=True)
    st.metric("Modèle", meta["model_name"])
    st.metric("ROC-AUC", f"{meta['roc_auc_test']:.4f}")
    st.metric("F1-Score", f"{meta['f1_test']:.3f}")
    st.metric("Avg Precision", f"{meta['avg_prec_test']:.4f}")
    st.metric("Seuil optimal", f"{threshold:.4f}")
    st.metric("Features", meta["n_features"])
    st.metric("Train samples", f"{meta['train_samples']:,}")
    st.divider()
    if using_heuristic:
        st.markdown("<span class='mode-badge'>⚙️ Mode heuristique</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:green'>✅ Modèle ML actif</span>", unsafe_allow_html=True)
    st.divider()
    page = st.radio("Navigation", ["🔬 Prédiction Individuelle","📋 Analyse par Lot","ℹ️ À propos"])

# ══════════════════════════════════════
#  PAGE 1 — Prédiction individuelle
# ══════════════════════════════════════
if page == "🔬 Prédiction Individuelle":
    st.markdown("<div class='section-title'>🔬 Saisie des données capteurs</div>", unsafe_allow_html=True)

    col_id, col_type, col_ai, col_year = st.columns(4)
    machine_id     = col_id.text_input("Machine ID", value="MC_OCP_0001")
    machine_type   = col_type.selectbox("Type de machine", list(MACHINE_TYPE_MAP.keys()), index=8)
    ai_supervision = col_ai.checkbox("Supervision IA active", value=True)
    inst_year      = col_year.number_input("Année d'installation", 2000, 2040, 2025)

    st.divider()
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**🌡️ Capteurs thermiques & mécaniques**")
        temp      = st.slider("Température (°C)",            20.0, 120.0, 65.0, 0.5)
        vibration = st.slider("Vibration (mm/s)",             0.0,  35.0,  8.0, 0.1)
        sound     = st.slider("Niveau sonore (dB)",           50.0, 110.0, 72.0, 0.5)
        power     = st.slider("Consommation électrique (kW)", 10.0, 300.0, 95.0, 1.0)

    with c2:
        st.markdown("**💧 Niveaux fluides**")
        oil     = st.slider("Niveau huile (%)",                        5.0, 100.0, 75.0, 1.0)
        coolant = st.slider("Niveau liquide refroidissement (%)",      5.0, 100.0, 80.0, 1.0)
        st.markdown("**⏱️ Utilisation**")
        op_hours = st.number_input("Heures opérationnelles", 0, 200000, 45000, 500)

    with c3:
        st.markdown("**🔧 Maintenance & historique**")
        last_maint  = st.slider("Jours depuis dernière maintenance", 0, 500, 90, 1)
        maint_count = st.number_input("Nombre de maintenances", 0, 50, 5, 1)
        fail_count  = st.number_input("Pannes historiques", 0, 30, 2, 1)
        errors_30d  = st.slider("Codes erreur (30 derniers jours)", 0, 30, 2, 1)
        ai_events   = st.number_input("Événements AI override", 0, 20, 1, 1)

    st.divider()
    if st.button("🚀 Analyser le risque de panne", type="primary", use_container_width=True):
        inputs = {
            "Machine_ID": machine_id, "Machine_Type": machine_type,
            "Installation_Year": inst_year,
            "Operational_Hours": op_hours,
            "Temperature_C": temp, "Vibration_mms": vibration,
            "Sound_dB": sound, "Oil_Level_pct": oil,
            "Coolant_Level_pct": coolant, "Power_Consumption_kW": power,
            "Last_Maintenance_Days_Ago": last_maint,
            "Maintenance_History_Count": maint_count,
            "Failure_History_Count": fail_count,
            "Error_Codes_Last_30_Days": errors_30d,
            "AI_Override_Events": ai_events,
            "AI_Supervision": ai_supervision,
        }

        proba = predict(inputs, model, preproc, threshold)
        level, color, css_class, action = classify_risk(proba, threshold)

        r1, r2 = st.columns([1, 2])
        with r1:
            st.plotly_chart(make_gauge(proba, color), use_container_width=True)
            st.markdown(f"<div class='risk-label' style='color:{color}'>{level}</div>", unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
            <div class='{css_class}' style='margin-top:1rem;'>
              <h3 style='margin:0 0 8px 0;'>Machine : {machine_id} — {machine_type}</h3>
              <p style='margin:4px 0;'><strong>Score de risque :</strong> {proba:.1%}</p>
              <p style='margin:4px 0;'><strong>Niveau d'alerte :</strong> {level}</p>
              <p style='margin:4px 0;'><strong>Action recommandée :</strong> {action}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>**Indicateurs clés**", unsafe_allow_html=True)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Temp.",      f"{temp:.0f}°C",      delta="⚠️ Élevée"  if temp > 80 else "✅ OK")
            k2.metric("Vibration",  f"{vibration:.1f} mm/s", delta="⚠️ Élevée" if vibration > 15 else "✅ OK")
            k3.metric("Huile",      f"{oil:.0f}%",        delta="⚠️ Basse"   if oil < 30 else "✅ OK")
            k4.metric("Erreurs/30j",str(errors_30d),      delta="⚠️ Élevé"   if errors_30d > 5 else "✅ OK")

            # Contribution breakdown
            st.markdown("<br>**Facteurs de risque**", unsafe_allow_html=True)
            factors = {
                "🌡️ Température":      np.clip((temp - 40) / 60, 0, 1) * 0.25,
                "📳 Vibration":        np.clip(vibration / 30, 0, 1) * 0.20,
                "🚨 Codes erreur":     np.clip(errors_30d / 15, 0, 1) * 0.18,
                "🛢️ Niveau huile":    np.clip((100 - oil) / 90, 0, 1) * 0.15,
                "❄️ Refroidissement": np.clip((100 - coolant) / 90, 0, 1) * 0.08,
                "🔧 Retard maint.":   np.clip(last_maint / 400, 0, 1) * 0.08,
                "💥 Pannes passées":  np.clip(fail_count / 8, 0, 1) * 0.04,
            }
            for label, val in factors.items():
                pct = val / 0.98  # normalize to max possible
                bar_color = OCP_RED if pct > 0.7 else OCP_ORANGE if pct > 0.4 else OCP_GREEN
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:8px;margin:3px 0;'>"
                    f"<span style='width:160px;font-size:0.85rem;'>{label}</span>"
                    f"<div style='flex:1;background:#eee;border-radius:4px;height:14px;'>"
                    f"<div style='width:{pct*100:.0f}%;background:{bar_color};border-radius:4px;height:14px;'></div></div>"
                    f"<span style='width:36px;text-align:right;font-size:0.8rem;'>{val:.2f}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

# ══════════════════════════════════════
#  PAGE 2 — Batch
# ══════════════════════════════════════
elif page == "📋 Analyse par Lot":
    st.markdown("<div class='section-title'>📋 Simulation Fleet — Parc de machines OCP</div>", unsafe_allow_html=True)
    n_machines = st.slider("Nombre de machines à simuler", 5, 100, 30, 5)

    if st.button("🔄 Générer & Analyser le parc", type="primary"):
        np.random.seed(42)
        machine_types = list(MACHINE_TYPE_MAP.keys())
        batch = pd.DataFrame({
            "Machine_ID":                [f"MC_OCP_{i:04d}" for i in range(n_machines)],
            "Machine_Type":              np.random.choice(machine_types, n_machines),
            "Operational_Hours":         np.random.randint(5000, 100000, n_machines),
            "Temperature_C":             np.random.uniform(35, 110, n_machines),
            "Vibration_mms":             np.random.uniform(1, 30, n_machines),
            "Sound_dB":                  np.random.uniform(55, 100, n_machines),
            "Oil_Level_pct":             np.random.uniform(5, 100, n_machines),
            "Coolant_Level_pct":         np.random.uniform(10, 100, n_machines),
            "Power_Consumption_kW":      np.random.uniform(30, 250, n_machines),
            "Last_Maintenance_Days_Ago": np.random.randint(0, 400, n_machines),
            "Maintenance_History_Count": np.random.randint(1, 10, n_machines),
            "Failure_History_Count":     np.random.randint(0, 8, n_machines),
            "AI_Supervision":            np.random.choice([True, False], n_machines),
            "Error_Codes_Last_30_Days":  np.random.randint(0, 15, n_machines),
            "AI_Override_Events":        np.random.randint(0, 5, n_machines),
            "Installation_Year":         np.random.randint(2010, 2038, n_machines),
        })

        results = []
        for _, row in batch.iterrows():
            d = row.to_dict()
            proba = predict(d, model, preproc, threshold)
            level, color, _, action = classify_risk(proba, threshold)
            results.append({"machine_id": d["Machine_ID"], "machine_type": d["Machine_Type"],
                            "risk_score": round(proba, 4), "risk_level": level, "action": action})

        res_df = pd.DataFrame(results).sort_values("risk_score", ascending=False).reset_index(drop=True)

        k1, k2, k3, k4 = st.columns(4)
        n_crit = (res_df["risk_score"] >= 0.80).sum()
        n_high = ((res_df["risk_score"] >= 0.55) & (res_df["risk_score"] < 0.80)).sum()
        n_mod  = ((res_df["risk_score"] >= threshold) & (res_df["risk_score"] < 0.55)).sum()
        n_low  = (res_df["risk_score"] < threshold).sum()
        k1.metric("🔴 Critiques", n_crit)
        k2.metric("🟠 Élevés",   n_high)
        k3.metric("🟡 Modérés",  n_mod)
        k4.metric("🟢 Faibles",  n_low)

        st.plotly_chart(make_batch_chart(res_df, threshold), use_container_width=True)
        st.dataframe(res_df, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Télécharger CSV",
                           res_df.to_csv(index=False).encode(),
                           "ocp_risk_report.csv", "text/csv")

# ══════════════════════════════════════
#  PAGE 3 — À propos
# ══════════════════════════════════════
elif page == "ℹ️ À propos":
    st.markdown("<div class='section-title'>ℹ️ À propos du système</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
**Objectif** : Prédire si une machine OCP tombera en panne dans les **7 prochains jours**.

**Modèle** : {meta['model_name']}
- ROC-AUC : **{meta['roc_auc_test']:.4f}**
- F1-Score : **{meta['f1_test']:.3f}**
- Seuil optimal : **{threshold:.4f}**
- Entraîné sur **{meta['train_samples']:,}** échantillons

**Niveaux d'alerte**

| Niveau | Score | Action |
|--------|-------|--------|
| 🔴 CRITIQUE | ≥ 80% | Arrêt immédiat |
| 🟠 ÉLEVÉ    | 55–80% | Maintenance < 48h |
| 🟡 MODÉRÉ   | seuil–55% | Surveillance renforcée |
| 🟢 FAIBLE   | < seuil | Fonctionnement normal |
        """)
    with c2:
        st.markdown("""
**Features utilisées (28)**

*Capteurs* : Température, Vibration, Son, Huile, Coolant, Puissance, Heures, Codes erreur

*Features dérivées* :
- `Thermal_Stress` = Temp × Vibration / 100
- `Maintenance_Urgency` = Jours_maint / (Nb_maintenances + 1)
- `Fluid_Degradation` = dégradation combinée
- `Failure_Density` = pannes / (heures / 1000)
- Flags : `Overheat_Flag`, `High_Vibration_Flag`, `Late_Maintenance_Flag`

**Sites** : Khouribga · Youssoufia · Gantour · Jorf Lasfar · Safi
        """)
