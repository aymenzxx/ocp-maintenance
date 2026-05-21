import streamlit as st
import pickle
import numpy as np
import pandas as pd
import base64
from pathlib import Path

# ── Resolve base directory (works locally and on Streamlit Cloud)
BASE_DIR = Path(__file__).parent

# ── Page config
st.set_page_config(
    page_title="OCP — Maintenance Prédictive",
    page_icon="🏭",
    layout="wide",
)

# ── OCP Colors
OCP_GREEN  = "#006633"
OCP_ORANGE = "#FF6600"

st.markdown(f"""
<style>
    .main-header {{
        background: linear-gradient(135deg, {OCP_GREEN} 0%, #004d26 100%);
        padding: 1.2rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
        display: flex;
        align-items: center;
        gap: 1.5rem;
    }}
    .main-header-logo {{
        width: 72px; height: 72px; flex-shrink: 0;
        background: white; border-radius: 50%; padding: 6px;
    }}
    .main-header-logo img {{ width: 100%; height: 100%; object-fit: contain; }}
    .main-header-text h1 {{ margin: 0; font-size: 1.7rem; }}
    .main-header-text p  {{ margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.9rem; }}

    .metric-card {{
        background: white;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }}
    .metric-card .label {{ font-size: 0.8rem; color: #666; text-transform: uppercase; letter-spacing: 0.05em; }}
    .metric-card .value {{ font-size: 2rem; font-weight: 700; }}

    .alert-box {{
        border-radius: 10px;
        padding: 1.5rem;
        font-size: 1.05rem;
        font-weight: 600;
    }}
    .alert-critique {{ background: #FFEBEE; border-left: 5px solid #D32F2F; color: #B71C1C; }}
    .alert-eleve    {{ background: #FFF3E0; border-left: 5px solid #FF6600; color: #E65100; }}
    .alert-modere   {{ background: #FFFDE7; border-left: 5px solid #FBC02D; color: #F57F17; }}
    .alert-faible   {{ background: #E8F5E9; border-left: 5px solid {OCP_GREEN}; color: #1B5E20; }}

    .section-title {{
        font-size: 1rem;
        font-weight: 700;
        color: {OCP_GREEN};
        border-bottom: 2px solid {OCP_GREEN};
        padding-bottom: 0.3rem;
        margin-bottom: 1rem;
    }}
    .stSlider > div > div {{ color: {OCP_GREEN}; }}
</style>
""", unsafe_allow_html=True)

# ── Load model
@st.cache_resource
def load_model():
    model_path = BASE_DIR / "predictive_maintenance_pipeline.pkl"
    with open(model_path, "rb") as f:
        return pickle.load(f)

try:
    artifacts    = load_model()
    model        = artifacts["model"]
    preprocessor = artifacts["preprocessor"]
    FEATURE_COLS = artifacts["feature_cols"]
    THRESHOLD    = artifacts["threshold"]
    le           = artifacts["label_encoder"]
except Exception as e:
    st.error(f"❌ Erreur chargement modèle : {e}\n\nVérifiez que `predictive_maintenance_pipeline.pkl` est bien dans le dépôt.")
    st.stop()

MACHINE_TYPES = ['Conveyor_Belt', 'CNC_Lathe', 'Hydraulic_Press',
                 'Crusher', 'Flotation_Cell', 'Dryer', 'Mixer', 'Pump']

# ── Feature engineering (mirrors notebook)
def engineer_features(d: dict) -> pd.DataFrame:
    row = pd.DataFrame([d])
    row["Machine_Age_Years"]    = 2040 - row["Installation_Year"]
    row["Hours_Per_Year"]       = row["Operational_Hours"] / row["Machine_Age_Years"].replace(0, 1)
    row["Thermal_Stress"]       = row["Temperature_C"] * row["Vibration_mms"] / 100
    row["Maintenance_Urgency"]  = row["Last_Maintenance_Days_Ago"] / (row["Maintenance_History_Count"] + 1)
    row["Fluid_Degradation"]    = ((100 - row["Oil_Level_pct"]) + (100 - row["Coolant_Level_pct"])) / 2
    row["Error_Rate"]           = row["Error_Codes_Last_30_Days"] / 30
    row["Failure_Density"]      = row["Failure_History_Count"] / (row["Operational_Hours"] / 1000 + 1)
    row["High_Vibration_Flag"]  = (row["Vibration_mms"] > 15).astype(int)
    row["Overheat_Flag"]        = (row["Temperature_C"] > 80).astype(int)
    row["Late_Maintenance_Flag"]= (row["Last_Maintenance_Days_Ago"] > 180).astype(int)
    row["Machine_Type_Enc"]     = 0
    row["AI_Supervision_Int"]   = int(d.get("AI_Supervision", False))
    return row[FEATURE_COLS].reindex(columns=FEATURE_COLS, fill_value=0)

def predict(d: dict):
    X = engineer_features(d)
    X_prep = preprocessor.transform(X)
    proba  = model.predict_proba(X_prep)[0, 1]
    if proba >= 0.80:
        level = "🔴 CRITIQUE";  css = "critique"; action = "Arrêt immédiat + Maintenance d'urgence"
    elif proba >= 0.55:
        level = "🟠 ÉLEVÉ";    css = "eleve";    action = "Planifier maintenance sous 48h"
    elif proba >= THRESHOLD:
        level = "🟡 MODÉRÉ";   css = "modere";   action = "Surveillance renforcée + inspection préventive"
    else:
        level = "🟢 FAIBLE";   css = "faible";   action = "Fonctionnement normal — maintenance planifiée"
    return proba, level, css, action

# ── Load OCP logo (inline base64, fallback to emoji)
_logo_html = ""
_logo_path = BASE_DIR / "ocp_logo.png"
if _logo_path.exists():
    _logo_b64 = base64.b64encode(_logo_path.read_bytes()).decode()
    _logo_html = f'<div class="main-header-logo"><img src="data:image/png;base64,{_logo_b64}" alt="OCP"/></div>'

# ── Header
st.markdown(f"""
<div class="main-header">
  {_logo_html}
  <div class="main-header-text">
    <h1>OCP — Système de Maintenance Prédictive</h1>
    <p>Prédiction de pannes machines dans les 7 prochains jours · Groupe OCP</p>
  </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🔍 Analyse Machine", "📊 Simulation Flotte"])

# ═══════════════════════════════
# TAB 1 — Single machine
# ═══════════════════════════════
with tab1:
    col_left, col_right = st.columns([1.1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-title">⚙️ Données Machine</div>', unsafe_allow_html=True)

        machine_id   = st.text_input("ID Machine", "MC_OCP_4521")
        machine_type = st.selectbox("Type de Machine", MACHINE_TYPES, index=1)
        install_year = st.slider("Année d'installation", 2005, 2038, 2025)
        op_hours     = st.number_input("Heures opérationnelles", 0, 200000, 85000, step=500)

        st.markdown('<div class="section-title" style="margin-top:1.2rem">🌡️ Capteurs Temps Réel</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        temp       = c1.slider("Température (°C)",  20.0, 120.0, 87.5, 0.5)
        vibration  = c2.slider("Vibration (mm/s)",   0.0,  40.0, 18.3, 0.1)
        sound      = c1.slider("Son (dB)",           40.0, 120.0, 91.0, 0.5)
        power      = c2.slider("Consommation (kW)",  10.0, 300.0, 145.0, 1.0)
        oil        = c1.slider("Niveau Huile (%)",    0.0, 100.0, 22.0, 1.0)
        coolant    = c2.slider("Liquide Refroid. (%)",0.0, 100.0, 31.0, 1.0)

        st.markdown('<div class="section-title" style="margin-top:1.2rem">🔧 Historique Maintenance</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        last_maint   = c3.number_input("Jours depuis maintenance", 0, 500, 210)
        maint_count  = c4.number_input("Nb maintenances",          0, 30,  3)
        fail_count   = c3.number_input("Nb pannes historique",     0, 20,  4)
        error_codes  = c4.number_input("Codes erreur (30j)",       0, 50,  7)
        ai_sup       = st.checkbox("Supervision IA active", value=True)
        ai_overrides = st.number_input("Événements AI Override", 0, 20, 3)

    with col_right:
        st.markdown('<div class="section-title">🎯 Résultat de Prédiction</div>', unsafe_allow_html=True)

        machine_data = {
            "Machine_ID": machine_id, "Machine_Type": machine_type,
            "Installation_Year": install_year, "Operational_Hours": op_hours,
            "Temperature_C": temp, "Vibration_mms": vibration,
            "Sound_dB": sound, "Power_Consumption_kW": power,
            "Oil_Level_pct": oil, "Coolant_Level_pct": coolant,
            "Last_Maintenance_Days_Ago": last_maint,
            "Maintenance_History_Count": maint_count,
            "Failure_History_Count": fail_count,
            "Error_Codes_Last_30_Days": error_codes,
            "AI_Supervision": ai_sup, "AI_Override_Events": ai_overrides,
            "Laser_Intensity": np.nan, "Hydraulic_Pressure_bar": np.nan,
            "Coolant_Flow_L_min": np.nan, "Heat_Index": np.nan,
            "Remaining_Useful_Life_days": 365,
        }

        proba, level, css, action = predict(machine_data)
        pct = proba * 100

        # Risk gauge (SVG)
        angle = -90 + 180 * proba
        needle_x = 100 + 75 * np.cos(np.radians(angle - 90))
        needle_y = 100 - 75 * np.sin(np.radians(angle - 90)) + 30

        st.markdown(f"""
        <svg viewBox="0 0 200 140" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:280px;display:block;margin:0 auto">
          <path d="M 25 130 A 75 75 0 0 1 62 57" fill="none" stroke="#1B5E20" stroke-width="16" stroke-linecap="round"/>
          <path d="M 62 57  A 75 75 0 0 1 100 25" fill="none" stroke="#FBC02D" stroke-width="16" stroke-linecap="round"/>
          <path d="M 100 25 A 75 75 0 0 1 138 57" fill="none" stroke="#FF6600" stroke-width="16" stroke-linecap="round"/>
          <path d="M 138 57 A 75 75 0 0 1 175 130" fill="none" stroke="#D32F2F" stroke-width="16" stroke-linecap="round"/>
          <line x1="100" y1="130" x2="{needle_x:.1f}" y2="{needle_y:.1f}"
                stroke="#333" stroke-width="3" stroke-linecap="round"/>
          <circle cx="100" cy="130" r="6" fill="#333"/>
          <text x="100" y="118" text-anchor="middle" font-size="22" font-weight="bold" fill="#1a1a1a">{pct:.1f}%</text>
          <text x="100" y="138" text-anchor="middle" font-size="9" fill="#666">Score de Risque</text>
        </svg>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="alert-box alert-{css}" style="margin-top:1rem">
          <div style="font-size:1.3rem">{level}</div>
          <div style="margin-top:0.4rem;font-size:0.95rem">{action}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        km1, km2 = st.columns(2)
        age = 2040 - install_year
        km1.metric("🌡️ Température", f"{temp}°C",  delta=f"+{temp-75:.0f}°C vs normal" if temp > 75 else None)
        km2.metric("📳 Vibration",   f"{vibration} mm/s", delta=f"+{vibration-15:.1f}" if vibration > 15 else None)
        km1.metric("🔧 Âge Machine", f"{age} ans")
        km2.metric("📅 Dernière maint.", f"{last_maint}j", delta="Retard" if last_maint > 180 else None, delta_color="inverse")

        st.markdown('<div class="section-title" style="margin-top:1rem">⚠️ Indicateurs d\'Alerte</div>', unsafe_allow_html=True)
        flags = []
        if temp > 80:       flags.append("🔴 Surchauffe détectée")
        if vibration > 15:  flags.append("🔴 Vibration élevée")
        if last_maint > 180:flags.append("🟠 Maintenance en retard")
        if oil < 30:        flags.append("🟠 Niveau huile critique")
        if coolant < 30:    flags.append("🟠 Liquide refroidissement bas")
        if error_codes > 5: flags.append("🟡 Codes erreur fréquents")
        if not flags:
            st.success("✅ Aucune alerte active — tous paramètres normaux")
        for f in flags:
            st.warning(f)

# ═══════════════════════════════
# TAB 2 — Fleet simulation
# ═══════════════════════════════
with tab2:
    st.markdown('<div class="section-title">📊 Simulation Flotte OCP</div>', unsafe_allow_html=True)

    n_sim = st.slider("Nombre de machines à simuler", 10, 100, 50, 5)

    if st.button("🚀 Lancer la Simulation", type="primary"):
        np.random.seed(42)
        sim = pd.DataFrame({
            "Machine_ID"                : [f"MC_OCP_{i:04d}" for i in range(n_sim)],
            "Machine_Type"              : np.random.choice(MACHINE_TYPES, n_sim),
            "Installation_Year"         : np.random.randint(2010, 2038, n_sim),
            "Operational_Hours"         : np.random.randint(5000, 100000, n_sim),
            "Temperature_C"             : np.random.uniform(35, 100, n_sim),
            "Vibration_mms"             : np.random.uniform(1, 30, n_sim),
            "Sound_dB"                  : np.random.uniform(55, 100, n_sim),
            "Oil_Level_pct"             : np.random.uniform(10, 100, n_sim),
            "Coolant_Level_pct"         : np.random.uniform(15, 100, n_sim),
            "Power_Consumption_kW"      : np.random.uniform(30, 250, n_sim),
            "Last_Maintenance_Days_Ago" : np.random.randint(0, 400, n_sim),
            "Maintenance_History_Count" : np.random.randint(1, 10, n_sim),
            "Failure_History_Count"     : np.random.randint(0, 8, n_sim),
            "AI_Supervision"            : np.random.choice([True, False], n_sim),
            "Error_Codes_Last_30_Days"  : np.random.randint(0, 15, n_sim),
            "Remaining_Useful_Life_days": np.random.uniform(0, 500, n_sim),
            "Laser_Intensity"           : np.nan,
            "Hydraulic_Pressure_bar"    : np.nan,
            "Coolant_Flow_L_min"        : np.nan,
            "Heat_Index"                : np.nan,
            "AI_Override_Events"        : np.random.randint(0, 5, n_sim),
        })

        results = []
        for _, row in sim.iterrows():
            proba, level, css, action = predict(row.to_dict())
            results.append({"Machine_ID": row["Machine_ID"],
                             "Type": row["Machine_Type"],
                             "Score (%)": round(proba * 100, 1),
                             "Niveau": level, "Action": action})
        df_res = pd.DataFrame(results).sort_values("Score (%)", ascending=False)

        total     = len(df_res)
        critiques = (df_res["Score (%)"] >= 80).sum()
        eleves    = ((df_res["Score (%)"] >= 55) & (df_res["Score (%)"] < 80)).sum()
        normaux   = (df_res["Score (%)"] < 55).sum()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("🏭 Machines Analysées", total)
        k2.metric("🔴 CRITIQUES",   critiques)
        k3.metric("🟠 ÉLEVÉ",       eleves)
        k4.metric("🟢 Normal",      normaux)

        st.markdown("---")

        # Color-coded table — explicit dark text to prevent disappearing text
        def color_row(row):
            s = row["Score (%)"]
            if s >= 80:   bg, fg = "#FFCDD2", "#7f0000"
            elif s >= 55: bg, fg = "#FFE0B2", "#bf360c"
            elif s >= THRESHOLD * 100: bg, fg = "#FFF9C4", "#827717"
            else:         bg, fg = "#C8E6C9", "#1b5e20"
            style = f"background-color: {bg}; color: {fg}; font-weight: 600;"
            return [style] * len(row)

        st.dataframe(
            df_res.style.apply(color_row, axis=1)
                        .set_properties(**{"color": "#111111"}),
            use_container_width=True,
            height=420,
        )

        urgents = df_res[df_res["Score (%)"] >= 80]
        if not urgents.empty:
            st.error(f"🚨 {len(urgents)} machine(s) nécessitent une intervention immédiate !")
            st.dataframe(urgents[["Machine_ID", "Type", "Score (%)", "Action"]],
                         use_container_width=True)
        else:
            st.success("✅ Aucune machine en état critique dans cet échantillon.")

# ── Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#999;font-size:0.8rem'>"
    "OCP Group — Système de Maintenance Prédictive · Modèle scikit-learn · ROC-AUC > 0.95"
    "</div>",
    unsafe_allow_html=True,
)
