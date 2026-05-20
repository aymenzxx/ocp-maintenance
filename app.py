import streamlit as st
import numpy as np
import pandas as pd
import pickle
import json
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── Page config
st.set_page_config(
    page_title="OCP — Maintenance Prédictive",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── OCP Brand Colors
OCP_GREEN  = "#006633"
OCP_ORANGE = "#FF6600"
OCP_RED    = "#D32F2F"
OCP_YELLOW = "#FFC107"

# ── Custom CSS
st.markdown(f"""
<style>
  /* Header bar */
  .ocp-header {{
    background: linear-gradient(135deg, {OCP_GREEN} 0%, #004d26 100%);
    padding: 1.2rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
  }}
  .ocp-header h1 {{ color: white; margin: 0; font-size: 1.8rem; font-weight: 700; }}
  .ocp-header p  {{ color: rgba(255,255,255,0.8); margin: 0; font-size: 0.95rem; }}

  /* Metric cards */
  .metric-card {{
    background: white;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    border-left: 5px solid {OCP_GREEN};
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    margin-bottom: 0.8rem;
  }}
  .metric-card.orange {{ border-left-color: {OCP_ORANGE}; }}
  .metric-card.red    {{ border-left-color: {OCP_RED};    }}
  .metric-card h3 {{ margin: 0 0 4px 0; font-size: 0.8rem; color: #666; text-transform: uppercase; letter-spacing: 0.05em; }}
  .metric-card .val {{ font-size: 1.8rem; font-weight: 700; color: #222; }}
  .metric-card .sub {{ font-size: 0.8rem; color: #888; }}

  /* Alert boxes */
  .alert-critical {{ background:#FFEBEE; border:2px solid {OCP_RED};    border-radius:10px; padding:1rem; }}
  .alert-high     {{ background:#FFF3E0; border:2px solid {OCP_ORANGE}; border-radius:10px; padding:1rem; }}
  .alert-moderate {{ background:#FFFDE7; border:2px solid {OCP_YELLOW}; border-radius:10px; padding:1rem; }}
  .alert-low      {{ background:#E8F5E9; border:2px solid {OCP_GREEN};  border-radius:10px; padding:1rem; }}

  /* Section titles */
  .section-title {{
    font-size: 1.1rem; font-weight: 600; color: {OCP_GREEN};
    border-bottom: 2px solid {OCP_GREEN}; padding-bottom: 6px; margin-bottom: 1rem;
  }}

  /* Gauge label */
  .risk-label {{ text-align: center; font-size: 1.4rem; font-weight: 700; margin-top:-10px; }}

  /* Streamlit tweaks */
  .stSlider > div[data-baseweb="slider"] {{ padding-top: 8px; }}
  div[data-testid="metric-container"] {{ background: white; border-radius: 8px; padding: 0.8rem; box-shadow: 0 1px 4px rgba(0,0,0,0.07); }}
</style>
""", unsafe_allow_html=True)

# ── Feature columns (from notebook)
FEATURE_COLS = [
    'Operational_Hours', 'Temperature_C', 'Vibration_mms', 'Sound_dB',
    'Oil_Level_pct', 'Coolant_Level_pct', 'Power_Consumption_kW',
    'Last_Maintenance_Days_Ago', 'Maintenance_History_Count',
    'Failure_History_Count', 'Error_Codes_Last_30_Days',
    'Laser_Intensity', 'Hydraulic_Pressure_bar', 'Coolant_Flow_L_min',
    'Heat_Index', 'AI_Override_Events',
    'Machine_Age_Years', 'Hours_Per_Year', 'Thermal_Stress',
    'Maintenance_Urgency', 'Fluid_Degradation', 'Error_Rate',
    'Failure_Density', 'High_Vibration_Flag', 'Overheat_Flag',
    'Late_Maintenance_Flag', 'Machine_Type_Enc', 'AI_Supervision_Int',
]

MACHINE_TYPE_MAP = {
    "CNC_Lathe": 0, "Conveyor_Belt": 1, "Crusher": 2,
    "Dryer": 3, "Filter_Press": 4, "Flotation_Cell": 5,
    "Hydraulic_Press": 6, "Mill": 7, "Pump": 8, "Reactor": 9,
}

# ── Load model
@st.cache_resource
def load_model():
    pkl_path = Path("predictive_maintenance_pipeline.pkl")
    if not pkl_path.exists():
        return None, None
    with open(pkl_path, "rb") as f:
        artifacts = pickle.load(f)
    return artifacts, None

@st.cache_data
def load_metadata():
    meta_path = Path("model_metadata.json")
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    # Fallback to hardcoded metadata from provided file
    return {
        "model_name": "Régression Logistique",
        "roc_auc_test": 0.9837,
        "avg_prec_test": 0.7591,
        "f1_test": 0.632,
        "optimal_threshold": 0.8817,
        "n_features": 28,
        "train_samples": 276101,
        "training_date": "2040",
        "target": "Failure_Within_7_Days",
    }

# ── Feature engineering (mirrors notebook)
def engineer_features(inputs: dict) -> pd.DataFrame:
    row = inputs.copy()
    year = 2040
    inst_year = row.get("Installation_Year", 2025)
    age = max(year - inst_year, 1)

    row["Machine_Age_Years"]     = age
    row["Hours_Per_Year"]        = row["Operational_Hours"] / age
    row["Thermal_Stress"]        = row["Temperature_C"] * row["Vibration_mms"] / 100
    row["Maintenance_Urgency"]   = row["Last_Maintenance_Days_Ago"] / (row["Maintenance_History_Count"] + 1)
    row["Fluid_Degradation"]     = ((100 - row["Oil_Level_pct"]) + (100 - row["Coolant_Level_pct"])) / 2
    row["Error_Rate"]            = row["Error_Codes_Last_30_Days"] / 30
    row["Failure_Density"]       = row["Failure_History_Count"] / (row["Operational_Hours"] / 1000 + 1)
    row["High_Vibration_Flag"]   = int(row["Vibration_mms"] > 15)
    row["Overheat_Flag"]         = int(row["Temperature_C"] > 80)
    row["Late_Maintenance_Flag"] = int(row["Last_Maintenance_Days_Ago"] > 180)
    row["Machine_Type_Enc"]      = MACHINE_TYPE_MAP.get(row.get("Machine_Type", "Pump"), 8)
    row["AI_Supervision_Int"]    = int(row.get("AI_Supervision", False))

    df = pd.DataFrame([row])
    return df.reindex(columns=FEATURE_COLS, fill_value=0)

# ── Risk classification
def classify_risk(proba: float, threshold: float):
    if proba >= 0.80:
        return "🔴 CRITIQUE",  OCP_RED,    "alert-critical", "Arrêt immédiat + Maintenance d'urgence"
    elif proba >= 0.55:
        return "🟠 ÉLEVÉ",    OCP_ORANGE,  "alert-high",     "Planifier maintenance sous 48h"
    elif proba >= threshold:
        return "🟡 MODÉRÉ",   OCP_YELLOW,  "alert-moderate", "Surveillance renforcée + inspection préventive"
    else:
        return "🟢 FAIBLE",   OCP_GREEN,   "alert-low",      "Fonctionnement normal — maintenance planifiée"

# ── Gauge chart
def make_gauge(proba: float, color: str):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(proba * 100, 1),
        number={"suffix": "%", "font": {"size": 36, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"size": 12}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "white",
            "steps": [
                {"range": [0,  55],  "color": "#E8F5E9"},
                {"range": [55, 80],  "color": "#FFF3E0"},
                {"range": [80, 100], "color": "#FFEBEE"},
            ],
            "threshold": {
                "line": {"color": OCP_RED, "width": 3},
                "thickness": 0.75, "value": 88,
            },
        },
        title={"text": "Score de Risque", "font": {"size": 16, "color": "#333"}},
    ))
    fig.update_layout(
        height=280, margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

# ── Batch simulation chart
def make_batch_chart(results_df: pd.DataFrame, threshold: float):
    colors = [
        OCP_RED if s >= 0.80 else OCP_ORANGE if s >= 0.55
        else OCP_YELLOW if s >= threshold else OCP_GREEN
        for s in results_df["risk_score"]
    ]
    fig = go.Figure(go.Bar(
        x=results_df["risk_score"],
        y=results_df["machine_id"],
        orientation="h",
        marker_color=colors,
        text=[f"{s:.0%}" for s in results_df["risk_score"]],
        textposition="outside",
    ))
    fig.add_vline(x=threshold, line_dash="dash", line_color="gray",
                  annotation_text=f"Seuil {threshold:.2f}", annotation_position="top right")
    fig.update_layout(
        height=max(350, len(results_df) * 22),
        xaxis=dict(range=[0, 1.1], title="Score de Risque"),
        yaxis=dict(title=""),
        margin=dict(l=120, r=60, t=30, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

# ═══════════════════════════════════════════════════════════
#  MAIN APP
# ═══════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="ocp-header">
  <div>
    <h1>🏭 OCP — Maintenance Prédictive</h1>
    <p>Prédiction des pannes machines dans les 7 prochains jours · Office Chérifien des Phosphates</p>
  </div>
</div>
""", unsafe_allow_html=True)

# Load model & metadata
artifacts, err = load_model()
meta = load_metadata()
threshold = meta.get("optimal_threshold", 0.8817)

# ── Sidebar: Model info + navigation
with st.sidebar:
    st.markdown(f"<div class='section-title'>📊 Modèle</div>", unsafe_allow_html=True)
    st.metric("Modèle", meta["model_name"])
    st.metric("ROC-AUC", f"{meta['roc_auc_test']:.4f}")
    st.metric("F1-Score", f"{meta['f1_test']:.3f}")
    st.metric("Avg Precision", f"{meta['avg_prec_test']:.4f}")
    st.metric("Seuil optimal", f"{threshold:.4f}")
    st.metric("Features", meta["n_features"])
    st.metric("Échantillons train", f"{meta['train_samples']:,}")

    st.divider()
    page = st.radio("Navigation", ["🔬 Prédiction Individuelle", "📋 Analyse par Lot", "ℹ️ À propos"])

if artifacts is None:
    st.error("⚠️  Fichier `predictive_maintenance_pipeline.pkl` introuvable dans le répertoire de l'app. Placez-le à côté de `app.py`.")
    st.info("Le modèle doit être chargé pour effectuer des prédictions. Les autres onglets restent accessibles.")
    model = None
    preprocessor = None
else:
    model = artifacts["model"]
    preprocessor = artifacts["preprocessor"]

# ═══════════════════════════════════════════
#  PAGE 1 — Prédiction individuelle
# ═══════════════════════════════════════════
if page == "🔬 Prédiction Individuelle":

    st.markdown("<div class='section-title'>🔬 Saisie des données capteurs</div>", unsafe_allow_html=True)

    col_id, col_type, col_ai, col_year = st.columns(4)
    machine_id   = col_id.text_input("Machine ID", value="MC_OCP_0001")
    machine_type = col_type.selectbox("Type de machine", list(MACHINE_TYPE_MAP.keys()), index=8)
    ai_supervision = col_ai.checkbox("Supervision IA active", value=True)
    inst_year    = col_year.number_input("Année d'installation", 2000, 2040, 2025)

    st.divider()
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**🌡️ Capteurs thermiques & mécaniques**")
        temp        = st.slider("Température (°C)",         20.0, 120.0, 65.0, 0.5)
        vibration   = st.slider("Vibration (mm/s)",          0.0,  35.0,  8.0, 0.1)
        sound       = st.slider("Niveau sonore (dB)",        50.0, 110.0, 72.0, 0.5)
        power       = st.slider("Consommation électrique (kW)", 10.0, 300.0, 95.0, 1.0)

    with c2:
        st.markdown("**💧 Niveaux fluides**")
        oil         = st.slider("Niveau huile (%)",          5.0, 100.0, 75.0, 1.0)
        coolant     = st.slider("Niveau liquide refroidissement (%)", 5.0, 100.0, 80.0, 1.0)
        st.markdown("**⏱️ Utilisation**")
        op_hours    = st.number_input("Heures opérationnelles", 0, 200000, 45000, 500)

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
            "Laser_Intensity": np.nan, "Hydraulic_Pressure_bar": np.nan,
            "Coolant_Flow_L_min": np.nan, "Heat_Index": np.nan,
            "AI_Supervision": ai_supervision,
        }

        X_new = engineer_features(inputs)

        if model is not None and preprocessor is not None:
            try:
                X_prep = preprocessor.transform(X_new)
                proba  = model.predict_proba(X_prep)[0, 1]
            except Exception as e:
                st.warning(f"Erreur préprocesseur : {e}. Prédiction demo.")
                proba = float(np.clip(
                    0.1 + (temp - 50) / 200 + vibration / 60 + (100 - oil) / 300
                    + errors_30d / 60 + last_maint / 1000, 0.05, 0.98))
        else:
            # Demo mode
            proba = float(np.clip(
                0.1 + (temp - 50) / 200 + vibration / 60 + (100 - oil) / 300
                + errors_30d / 60 + last_maint / 1000, 0.05, 0.98))

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
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.metric("Temp.", f"{temp:.0f}°C", delta="⚠️" if temp > 80 else "✅")
            kpi2.metric("Vibration", f"{vibration:.1f} mm/s", delta="⚠️" if vibration > 15 else "✅")
            kpi3.metric("Huile", f"{oil:.0f}%", delta="⚠️" if oil < 30 else "✅")
            kpi4.metric("Erreurs/30j", str(errors_30d), delta="⚠️" if errors_30d > 5 else "✅")


# ═══════════════════════════════════════════
#  PAGE 2 — Batch
# ═══════════════════════════════════════════
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
            "Temperature_C":             np.random.uniform(35, 100, n_machines),
            "Vibration_mms":             np.random.uniform(1, 30, n_machines),
            "Sound_dB":                  np.random.uniform(55, 100, n_machines),
            "Oil_Level_pct":             np.random.uniform(10, 100, n_machines),
            "Coolant_Level_pct":         np.random.uniform(15, 100, n_machines),
            "Power_Consumption_kW":      np.random.uniform(30, 250, n_machines),
            "Last_Maintenance_Days_Ago": np.random.randint(0, 400, n_machines),
            "Maintenance_History_Count": np.random.randint(1, 10, n_machines),
            "Failure_History_Count":     np.random.randint(0, 8, n_machines),
            "AI_Supervision":            np.random.choice([True, False], n_machines),
            "Error_Codes_Last_30_Days":  np.random.randint(0, 15, n_machines),
            "AI_Override_Events":        np.random.randint(0, 5, n_machines),
            "Installation_Year":         np.random.randint(2010, 2038, n_machines),
            "Laser_Intensity":           np.nan,
            "Hydraulic_Pressure_bar":    np.nan,
            "Coolant_Flow_L_min":        np.nan,
            "Heat_Index":                np.nan,
        })

        results = []
        for _, row in batch.iterrows():
            d = row.to_dict()
            X_new = engineer_features(d)

            if model is not None and preprocessor is not None:
                try:
                    X_prep = preprocessor.transform(X_new)
                    proba  = model.predict_proba(X_prep)[0, 1]
                except:
                    proba = float(np.clip(
                        0.1 + (d["Temperature_C"] - 50)/200 + d["Vibration_mms"]/60
                        + (100 - d["Oil_Level_pct"])/300 + d["Error_Codes_Last_30_Days"]/60
                        + d["Last_Maintenance_Days_Ago"]/1000, 0.05, 0.98))
            else:
                proba = float(np.clip(
                    0.1 + (d["Temperature_C"] - 50)/200 + d["Vibration_mms"]/60
                    + (100 - d["Oil_Level_pct"])/300 + d["Error_Codes_Last_30_Days"]/60
                    + d["Last_Maintenance_Days_Ago"]/1000, 0.05, 0.98))

            level, color, _, action = classify_risk(proba, threshold)
            results.append({
                "machine_id":   d["Machine_ID"],
                "machine_type": d["Machine_Type"],
                "risk_score":   round(proba, 4),
                "risk_level":   level,
                "action":       action,
            })

        res_df = pd.DataFrame(results).sort_values("risk_score", ascending=False).reset_index(drop=True)

        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        n_crit  = (res_df["risk_score"] >= 0.80).sum()
        n_high  = ((res_df["risk_score"] >= 0.55) & (res_df["risk_score"] < 0.80)).sum()
        n_mod   = ((res_df["risk_score"] >= threshold) & (res_df["risk_score"] < 0.55)).sum()
        n_low   = (res_df["risk_score"] < threshold).sum()
        k1.metric("🔴 Critiques",  n_crit,  delta=f"{n_crit/n_machines:.0%}")
        k2.metric("🟠 Élevés",    n_high,  delta=f"{n_high/n_machines:.0%}")
        k3.metric("🟡 Modérés",   n_mod,   delta=f"{n_mod/n_machines:.0%}")
        k4.metric("🟢 Faibles",   n_low,   delta=f"{n_low/n_machines:.0%}")

        st.plotly_chart(make_batch_chart(res_df, threshold), use_container_width=True)

        st.markdown("**📄 Tableau complet**")
        st.dataframe(
            res_df.style.background_gradient(subset=["risk_score"], cmap="RdYlGn_r"),
            use_container_width=True, hide_index=True
        )

        csv = res_df.to_csv(index=False).encode()
        st.download_button("⬇️ Télécharger CSV", csv, "ocp_risk_report.csv", "text/csv")


# ═══════════════════════════════════════════
#  PAGE 3 — À propos
# ═══════════════════════════════════════════
elif page == "ℹ️ À propos":

    st.markdown("<div class='section-title'>ℹ️ À propos du système</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        **Objectif**

        Prédire si une machine OCP tombera en panne dans les **7 prochains jours**
        à partir de données capteurs en temps réel, afin de réduire les arrêts non planifiés.

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

        *Capteurs bruts* : Température, Vibration, Son, Huile, Liquide de refroidissement,
        Puissance, Heures opérationnelles, Codes erreur

        *Features dérivées (Feature Engineering)* :
        - `Thermal_Stress` = Temp × Vibration / 100
        - `Maintenance_Urgency` = Jours_depuis_maint / (Nb_maintenances + 1)
        - `Fluid_Degradation` = dégradation combinée huile + refroidissement
        - `Failure_Density` = pannes historiques / (heures / 1000)
        - Flags binaires : `Overheat_Flag`, `High_Vibration_Flag`, `Late_Maintenance_Flag`

        **Sites couverts** : Khouribga · Youssoufia · Gantour · Jorf Lasfar · Safi
        """)

    st.info("💡 Pour mettre à jour le modèle, replacez `predictive_maintenance_pipeline.pkl` et `model_metadata.json` dans le répertoire de l'application puis rechargez.")
