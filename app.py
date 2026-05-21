import streamlit as st
import numpy as np
import pandas as pd
import pickle, json, warnings
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime, timedelta
import time

st.set_page_config(
    page_title="OCP — Maintenance Prédictive",
    page_icon="🏭", layout="wide", initial_sidebar_state="expanded",
)

OCP_GREEN  = "#006633"
OCP_ORANGE = "#FF6600"
OCP_RED    = "#D32F2F"
OCP_YELLOW = "#FFC107"
OCP_BLUE   = "#1565C0"

st.markdown(f"""
<style>
  .ocp-header {{
    background: linear-gradient(135deg, {OCP_GREEN} 0%, #004d26 100%);
    padding: 1.4rem 2rem; border-radius: 14px; margin-bottom: 1.5rem;
    box-shadow: 0 4px 15px rgba(0,102,51,.3);
    display: flex; align-items: center; gap: 1rem;
  }}
  .ocp-header h1 {{ color: white; margin: 0; font-size: 1.9rem; font-weight: 800; }}
  .ocp-header p  {{ color: rgba(255,255,255,.8); margin: 4px 0 0; font-size: .92rem; }}
  .kpi-card {{
    background: white; border-radius: 12px; padding: 1rem 1.2rem;
    text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,.08);
    border-top: 4px solid var(--green);
  }}
  .kpi-card .kpi-value {{ font-size: 2rem; font-weight: 800; margin: .3rem 0; }}
  .kpi-card .kpi-label {{ font-size: .82rem; color: #666; text-transform: uppercase; }}
  .alert-critical {{
    background: linear-gradient(135deg,#FFEBEE,#FFCDD2);
    border: 2px solid {OCP_RED}; border-radius: 12px; padding: 1.2rem;
  }}
  .alert-high {{
    background: linear-gradient(135deg,#FFF3E0,#FFE0B2);
    border: 2px solid {OCP_ORANGE}; border-radius: 12px; padding: 1.2rem;
  }}
  .alert-moderate {{
    background: linear-gradient(135deg,#FFFDE7,#FFF9C4);
    border: 2px solid {OCP_YELLOW}; border-radius: 12px; padding: 1.2rem;
  }}
  .alert-low {{
    background: linear-gradient(135deg,#E8F5E9,#C8E6C9);
    border: 2px solid {OCP_GREEN}; border-radius: 12px; padding: 1.2rem;
  }}
  .section-title {{
    font-size: 1.1rem; font-weight: 700; color: {OCP_GREEN};
    border-bottom: 3px solid {OCP_GREEN}; padding-bottom: 6px; margin-bottom: 1rem;
  }}
  .risk-label {{
    text-align: center; font-size: 1.5rem; font-weight: 800; margin-top: -8px;
  }}
  .factor-bar-wrap {{ display: flex; align-items: center; gap: 8px; margin: 4px 0; }}
  .factor-label  {{ width: 175px; font-size: .85rem; }}
  .factor-track  {{ flex: 1; background: #eee; border-radius: 6px; height: 14px; overflow: hidden; }}
  .factor-fill   {{ height: 14px; border-radius: 6px; }}
  .factor-val    {{ width: 40px; text-align: right; font-size: .8rem; color: #555; }}
  .sidebar-badge {{
    background: {OCP_GREEN}22; border-radius: 8px; padding: 8px 12px; margin: 4px 0;
    border-left: 3px solid {OCP_GREEN}; font-size: .85rem;
  }}
  .live-score-box {{
    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    border: 2px solid #dee2e6; border-radius: 12px;
    padding: 1rem; text-align: center; margin-bottom: 1rem;
  }}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════
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

MACHINE_ICONS = {
    "CNC_Lathe":"⚙️","Conveyor_Belt":"🔄","Crusher":"💥","Dryer":"🌡️",
    "Filter_Press":"🔩","Flotation_Cell":"🫧","Hydraulic_Press":"🔧",
    "Mill":"⚡","Pump":"💧","Reactor":"⚗️",
}

SITES_OCP = ["Khouribga","Youssoufia","Gantour","Jorf Lasfar","Safi","Benguerir"]

# ══════════════════════════════════════════════════════════════
# FONCTIONS MÉTIER  ← BUG FIXÉ ICI
# ══════════════════════════════════════════════════════════════
def heuristic_score(temp, vib, oil, cool, errors, last_m, fails, ai_ov) -> float:
    """
    Score heuristique pondéré — retourne toujours une valeur correcte.
    Poids total = 1.0
    """
    # Chaque terme est clippé [0,1] puis multiplié par son poids
    s  = float(np.clip((float(temp)   - 40) / 60,  0.0, 1.0)) * 0.25   # température
    s += float(np.clip( float(vib)    / 30,         0.0, 1.0)) * 0.20   # vibration
    s += float(np.clip((100 - float(oil))  / 90,    0.0, 1.0)) * 0.15   # huile faible
    s += float(np.clip((100 - float(cool)) / 90,    0.0, 1.0)) * 0.08   # coolant faible
    s += float(np.clip( float(errors) / 15,         0.0, 1.0)) * 0.18   # erreurs
    s += float(np.clip( float(last_m) / 400,        0.0, 1.0)) * 0.08   # retard maint.
    s += float(np.clip( float(fails)  / 8,          0.0, 1.0)) * 0.04   # pannes passées
    s += float(np.clip( float(ai_ov)  / 5,          0.0, 1.0)) * 0.02   # AI events

    # Garantit une valeur minimale visible et max < 1
    return float(np.clip(s, 0.02, 0.97))


def engineer_features(d: dict) -> pd.DataFrame:
    age = max(2040 - int(d.get("Installation_Year", 2020)), 1)
    oh  = float(d.get("Operational_Hours", 1000))
    row = {
        'Operational_Hours':          oh,
        'Temperature_C':              float(d.get('Temperature_C', 50)),
        'Vibration_mms':              float(d.get('Vibration_mms', 5)),
        'Sound_dB':                   float(d.get('Sound_dB', 70)),
        'Oil_Level_pct':              float(d.get('Oil_Level_pct', 80)),
        'Coolant_Level_pct':          float(d.get('Coolant_Level_pct', 80)),
        'Power_Consumption_kW':       float(d.get('Power_Consumption_kW', 100)),
        'Last_Maintenance_Days_Ago':  float(d.get('Last_Maintenance_Days_Ago', 60)),
        'Maintenance_History_Count':  float(d.get('Maintenance_History_Count', 3)),
        'Failure_History_Count':      float(d.get('Failure_History_Count', 0)),
        'Error_Codes_Last_30_Days':   float(d.get('Error_Codes_Last_30_Days', 0)),
        'Laser_Intensity':            float('nan'),
        'Hydraulic_Pressure_bar':     float('nan'),
        'Coolant_Flow_L_min':         float('nan'),
        'Heat_Index':                 float('nan'),
        'AI_Override_Events':         float(d.get('AI_Override_Events', 0)),
        'Machine_Age_Years':          float(age),
        'Hours_Per_Year':             oh / age,
        'Thermal_Stress':             float(d.get('Temperature_C', 50)) * float(d.get('Vibration_mms', 5)) / 100,
        'Maintenance_Urgency':        float(d.get('Last_Maintenance_Days_Ago', 60)) / (float(d.get('Maintenance_History_Count', 3)) + 1),
        'Fluid_Degradation':         ((100 - float(d.get('Oil_Level_pct', 80))) + (100 - float(d.get('Coolant_Level_pct', 80)))) / 2,
        'Error_Rate':                 float(d.get('Error_Codes_Last_30_Days', 0)) / 30,
        'Failure_Density':            float(d.get('Failure_History_Count', 0)) / (oh / 1000 + 1),
        'High_Vibration_Flag':        int(float(d.get('Vibration_mms', 5)) > 15),
        'Overheat_Flag':              int(float(d.get('Temperature_C', 50)) > 80),
        'Late_Maintenance_Flag':      int(float(d.get('Last_Maintenance_Days_Ago', 60)) > 180),
        'Machine_Type_Enc':           float(MACHINE_TYPE_MAP.get(str(d.get('Machine_Type', 'Pump')), 8)),
        'AI_Supervision_Int':         int(bool(d.get('AI_Supervision', False))),
    }
    return pd.DataFrame([row])[FEATURE_COLS]


@st.cache_resource(show_spinner="🔄 Chargement du modèle ML…")
def load_artifacts():
    pkl_path = Path("predictive_maintenance_pipeline.pkl")
    if not pkl_path.exists():
        return None, None, 0.8817, "pkl introuvable"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(pkl_path, "rb") as f:
                arts = pickle.load(f)
    except Exception as e:
        return None, None, 0.8817, f"Erreur chargement : {e}"

    model     = arts.get("model")
    preproc   = arts.get("preprocessor")
    threshold = float(arts.get("threshold", 0.8817))

    try:
        test = engineer_features({
            'Temperature_C':65,'Vibration_mms':8,'Oil_Level_pct':75,
            'Coolant_Level_pct':80,'Operational_Hours':45000,
            'Last_Maintenance_Days_Ago':90,'Maintenance_History_Count':5,
            'Failure_History_Count':2,'Error_Codes_Last_30_Days':2,
            'AI_Override_Events':1,'Machine_Type':'Pump','Installation_Year':2020
        })
        _ = model.predict_proba(preproc.transform(test))[0, 1]
        return model, preproc, threshold, "ok"
    except Exception as e:
        return None, None, threshold, f"Incompatible : {type(e).__name__}"


@st.cache_data(show_spinner=False)
def load_metadata():
    p = Path("model_metadata.json")
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {
        "model_name": "Régression Logistique",
        "roc_auc_test": 0.9837, "avg_prec_test": 0.7591,
        "f1_test": 0.632, "optimal_threshold": 0.8817,
        "n_features": 28, "train_samples": 276101,
        "training_date": "2040", "target": "Failure_Within_7_Days",
    }


def get_score(d: dict, model, preproc) -> tuple:
    """
    Retourne (score: float, source: str).
    Essaie le modèle ML, sinon fallback heuristique.
    Ne lève JAMAIS d'exception.
    """
    # ── Calcul heuristique (référence fiable)
    h = heuristic_score(
        temp   = float(d.get('Temperature_C',   50)),
        vib    = float(d.get('Vibration_mms',    5)),
        oil    = float(d.get('Oil_Level_pct',   80)),
        cool   = float(d.get('Coolant_Level_pct',80)),
        errors = float(d.get('Error_Codes_Last_30_Days', 0)),
        last_m = float(d.get('Last_Maintenance_Days_Ago',60)),
        fails  = float(d.get('Failure_History_Count',    0)),
        ai_ov  = float(d.get('AI_Override_Events',       0)),
    )

    # Vérifie que h est bien un float non-nul
    if not (0 < h <= 1):
        h = 0.05

    if model is None or preproc is None:
        return h, "heuristique"

    try:
        X   = engineer_features(d)
        X_p = preproc.transform(X)
        ml_score = float(model.predict_proba(X_p)[0, 1])
        if 0 <= ml_score <= 1:
            return ml_score, "modèle ML"
        return h, "heuristique (fallback)"
    except Exception:
        return h, "heuristique"


def classify_risk(p: float, thr: float) -> tuple:
    if p >= 0.80:
        return "🔴 CRITIQUE", OCP_RED,    "alert-critical", "Arrêt immédiat + Maintenance d'urgence", 4
    if p >= 0.55:
        return "🟠 ÉLEVÉ",   OCP_ORANGE,  "alert-high",     "Planifier maintenance sous 48h", 3
    if p >= thr:
        return "🟡 MODÉRÉ",  OCP_YELLOW,  "alert-moderate", "Surveillance renforcée + inspection", 2
    return     "🟢 FAIBLE",  OCP_GREEN,   "alert-low",      "Fonctionnement normal — maintenance planifiée", 1


def get_maintenance_date(score: float) -> str:
    days = 0 if score >= 0.80 else 2 if score >= 0.55 else 7 if score >= 0.40 else 30
    if days == 0:
        return "⚡ IMMÉDIAT"
    return (datetime.now() + timedelta(days=days)).strftime("%d/%m/%Y")


# ══════════════════════════════════════════════════════════════
# GRAPHIQUES
# ══════════════════════════════════════════════════════════════
def make_gauge(proba: float, color: str) -> go.Figure:
    """Jauge — accepte proba ∈ [0, 1]."""
    pct = round(float(proba) * 100, 1)          # ← conversion explicite
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = pct,
        number = {"suffix": "%", "font": {"size": 44, "color": color}},
        gauge  = {
            "axis": {"range": [0, 100], "ticksuffix": "%", "tickfont": {"size": 11}},
            "bar":  {"color": color, "thickness": 0.30},
            "bgcolor": "white",
            "steps": [
                {"range": [0,  55], "color": "#E8F5E9"},
                {"range": [55, 80], "color": "#FFF3E0"},
                {"range": [80,100], "color": "#FFEBEE"},
            ],
            "threshold": {
                "line": {"color": OCP_RED, "width": 3},
                "thickness": 0.80, "value": 80,
            },
        },
        title = {"text": "Score de Risque", "font": {"size": 14, "color": "#333"}},
    ))
    fig.update_layout(
        height=290, margin=dict(l=20, r=20, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_radar(d: dict) -> go.Figure:
    cats = ['Température','Vibration','Erreurs','Huile','Refroidissement','Maintenance']
    vals = [
        float(np.clip((float(d.get('Temperature_C',50)) - 20) / 100,          0, 1)),
        float(np.clip( float(d.get('Vibration_mms',5))         / 35,           0, 1)),
        float(np.clip( float(d.get('Error_Codes_Last_30_Days',0)) / 15,        0, 1)),
        float(np.clip((100 - float(d.get('Oil_Level_pct',80)))     / 95,       0, 1)),
        float(np.clip((100 - float(d.get('Coolant_Level_pct',80))) / 95,       0, 1)),
        float(np.clip( float(d.get('Last_Maintenance_Days_Ago',60)) / 400,     0, 1)),
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[0.7]*len(cats)+[0.7], theta=cats+[cats[0]],
        fill='toself', fillcolor='rgba(211,47,47,.08)',
        line=dict(color=OCP_RED, dash='dot', width=1),
        name='Seuil danger', showlegend=False,
    ))
    fig.add_trace(go.Scatterpolar(
        r=vals+[vals[0]], theta=cats+[cats[0]],
        fill='toself', fillcolor='rgba(0,102,51,.15)',
        line=dict(color=OCP_GREEN, width=2.5),
        name='Capteurs', marker=dict(size=6, color=OCP_GREEN),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        height=300, margin=dict(l=40,r=40,t=30,b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_history_sim(proba: float) -> go.Figure:
    np.random.seed(int(proba * 1000) % 1000)
    days   = list(range(-29, 1))
    trend  = np.linspace(max(0.05, proba - 0.35), proba, 30)
    scores = np.clip(trend + np.random.normal(0, 0.025, 30), 0.02, 0.97)
    fig = go.Figure()
    fig.add_hrect(y0=0.80, y1=1.0, fillcolor="rgba(211,47,47,.1)", line_width=0)
    fig.add_hrect(y0=0.55, y1=0.80, fillcolor="rgba(255,102,0,.08)", line_width=0)
    fig.add_trace(go.Scatter(
        x=days, y=scores.tolist(), mode='lines+markers',
        line=dict(color=OCP_GREEN, width=2.5, shape='spline'),
        marker=dict(size=5, color=scores.tolist(),
                    colorscale=[[0,OCP_GREEN],[0.55,OCP_YELLOW],[1,OCP_RED]],
                    cmin=0, cmax=1),
        name='Score',
    ))
    fig.add_trace(go.Scatter(
        x=[0], y=[float(proba)], mode='markers',
        marker=dict(size=14, color=OCP_RED, symbol='star',
                    line=dict(color='white', width=2)),
        name="Aujourd'hui",
    ))
    fig.update_layout(
        title="Évolution simulée — 30 jours",
        xaxis=dict(title="Jours", gridcolor="#eee"),
        yaxis=dict(title="Score", range=[0,1.05], tickformat=".0%", gridcolor="#eee"),
        height=280, margin=dict(l=10,r=10,t=40,b=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_fleet_bar(res: pd.DataFrame, threshold: float) -> go.Figure:
    colors = [
        OCP_RED if s >= 0.80 else OCP_ORANGE if s >= 0.55
        else OCP_YELLOW if s >= threshold else OCP_GREEN
        for s in res["Score"]
    ]
    fig = go.Figure(go.Bar(
        x=res["Score"], y=res["Machine"], orientation="h",
        marker=dict(color=colors),
        text=[f"{s:.0%}" for s in res["Score"]], textposition="outside",
        hovertemplate="<b>%{y}</b><br>Score : %{x:.1%}<extra></extra>",
    ))
    fig.add_vline(x=threshold, line_dash="dash", line_color="gray",
                  annotation_text=f"Seuil {threshold:.2f}")
    fig.update_layout(
        height=max(380, len(res)*24),
        xaxis=dict(range=[0,1.15], tickformat=".0%", gridcolor="#eee"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=130,r=70,t=20,b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_pie(res: pd.DataFrame, threshold: float) -> go.Figure:
    vals = [
        int((res["Score"] >= 0.80).sum()),
        int(((res["Score"] >= 0.55) & (res["Score"] < 0.80)).sum()),
        int(((res["Score"] >= threshold) & (res["Score"] < 0.55)).sum()),
        int((res["Score"] < threshold).sum()),
    ]
    fig = go.Figure(go.Pie(
        labels=["🔴 Critique","🟠 Élevé","🟡 Modéré","🟢 Faible"],
        values=vals,
        marker=dict(colors=[OCP_RED,OCP_ORANGE,OCP_YELLOW,OCP_GREEN],
                    line=dict(color='white',width=2)),
        hole=0.55, textinfo="label+percent",
    ))
    fig.update_layout(
        title="Répartition niveaux", height=280,
        margin=dict(l=10,r=10,t=40,b=10),
        paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
    )
    return fig


# ══════════════════════════════════════════════════════════════
# CHARGEMENT
# ══════════════════════════════════════════════════════════════
model, preproc, DEFAULT_THRESHOLD, status = load_artifacts()
meta      = load_metadata()
threshold = DEFAULT_THRESHOLD or meta.get("optimal_threshold", 0.8817)
ml_active = (status == "ok")

# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="ocp-header">
  <div style="font-size:3rem;">🏭</div>
  <div>
    <h1>OCP — Maintenance Prédictive</h1>
    <p>Prédiction des pannes · Office Chérifien des Phosphates ·
       {datetime.now().strftime('%d %B %Y, %H:%M')}</p>
  </div>
</div>""", unsafe_allow_html=True)

if ml_active:
    st.success("✅ Modèle ML actif — prédictions haute précision")
else:
    st.info(f"⚙️ Mode heuristique ({status})")

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<div class='section-title'>📊 Modèle</div>", unsafe_allow_html=True)
    for label, val in [
        ("🤖 Modèle",        meta["model_name"]),
        ("📈 ROC-AUC",       f"{meta['roc_auc_test']:.4f}"),
        ("🎯 F1-Score",      f"{meta['f1_test']:.3f}"),
        ("⚖️ Seuil",         f"{threshold:.4f}"),
        ("🗃️ Train samples", f"{meta['train_samples']:,}"),
    ]:
        st.markdown(
            f"<div class='sidebar-badge'><b>{label}</b><br>{val}</div>",
            unsafe_allow_html=True,
        )
    st.divider()
    st.markdown("✅ ML actif" if ml_active else "⚙️ Heuristique")
    st.divider()
    page = st.radio("🧭 Navigation", [
        "🔬 Prédiction Individuelle",
        "📋 Analyse par Lot",
        "ℹ️ À propos",
    ], label_visibility="collapsed")

# ══════════════════════════════════════════════════════════════
# PAGE 1 — PRÉDICTION INDIVIDUELLE
# ══════════════════════════════════════════════════════════════
if page == "🔬 Prédiction Individuelle":

    st.markdown("<div class='section-title'>🔬 Saisie des données capteurs</div>",
                unsafe_allow_html=True)

    # ── Identité
    c0a, c0b, c0c, c0d, c0e = st.columns([2,2,2,1,1])
    machine_id     = c0a.text_input("🏷️ Machine ID", "MC_OCP_0001")
    machine_type   = c0b.selectbox("⚙️ Type", list(MACHINE_TYPE_MAP.keys()), index=8)
    site           = c0c.selectbox("📍 Site", SITES_OCP)
    ai_supervision = c0d.checkbox("🤖 IA", True)
    inst_year      = c0e.number_input("📅 Install.", 2000, 2040, 2020)

    st.divider()
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**🌡️ Thermique & Mécanique**")
        temp      = st.slider("Température (°C)",    20.0, 120.0, 65.0, 0.5)
        vibration = st.slider("Vibration (mm/s)",     0.0,  35.0,  8.0, 0.1)
        sound     = st.slider("Niveau sonore (dB)",  50.0, 110.0, 72.0, 0.5)
        power     = st.slider("Consommation (kW)",   10.0, 300.0, 95.0, 1.0)

    with c2:
        st.markdown("**💧 Fluides**")
        oil     = st.slider("Niveau huile (%)",          5.0, 100.0, 75.0, 1.0)
        coolant = st.slider("Refroidissement (%)",       5.0, 100.0, 80.0, 1.0)
        st.markdown("**⏱️ Utilisation**")
        op_hours = st.number_input("Heures opérationnelles", 0, 200_000, 45_000, 500)

    with c3:
        st.markdown("**🔧 Maintenance & Historique**")
        last_maint  = st.slider("Jours depuis maintenance", 0, 500, 90, 1)
        maint_count = st.number_input("Nb maintenances",    0,  50,  5, 1)
        fail_count  = st.number_input("Pannes historiques", 0,  30,  2, 1)
        errors_30d  = st.slider("Codes erreur / 30j",       0,  30,  2, 1)
        ai_events   = st.number_input("AI override events", 0,  20,  1, 1)

    st.divider()

    # ══════════════════════════════════════════════════════════
    # ← CALCUL EN TEMPS RÉEL (sans bouton) ← FIX PRINCIPAL
    # ══════════════════════════════════════════════════════════
    d_live = {
        "Machine_ID": machine_id, "Machine_Type": machine_type,
        "Site": site, "Installation_Year": int(inst_year),
        "Operational_Hours":         float(op_hours),
        "Temperature_C":             float(temp),
        "Vibration_mms":             float(vibration),
        "Sound_dB":                  float(sound),
        "Oil_Level_pct":             float(oil),
        "Coolant_Level_pct":         float(coolant),
        "Power_Consumption_kW":      float(power),
        "Last_Maintenance_Days_Ago": float(last_maint),
        "Maintenance_History_Count": float(maint_count),
        "Failure_History_Count":     float(fail_count),
        "Error_Codes_Last_30_Days":  float(errors_30d),
        "AI_Override_Events":        float(ai_events),
        "AI_Supervision":            bool(ai_supervision),
    }

    # Score calculé EN PERMANENCE à chaque changement de widget
    proba_live, source_live = get_score(d_live, model, preproc)
    level_live, color_live, css_live, action_live, _ = classify_risk(proba_live, threshold)

    # ── Aperçu live (toujours visible)
    st.markdown("### 📡 Score en temps réel")
    lc1, lc2, lc3, lc4 = st.columns(4)
    lc1.metric("🎯 Score", f"{proba_live:.1%}", help="Mis à jour à chaque modification")
    lc2.metric("🚦 Niveau", level_live)
    lc3.metric("🔧 Maintenance", get_maintenance_date(proba_live))
    lc4.metric("📡 Source", source_live)

    # Barre de progression colorée
    bar_color = (OCP_RED if proba_live >= 0.80 else
                 OCP_ORANGE if proba_live >= 0.55 else
                 OCP_YELLOW if proba_live >= threshold else OCP_GREEN)
    st.markdown(
        f"<div style='background:#eee;border-radius:8px;height:18px;margin:6px 0 16px;'>"
        f"<div style='width:{proba_live*100:.1f}%;background:{bar_color};"
        f"border-radius:8px;height:18px;transition:width .4s ease;'></div></div>",
        unsafe_allow_html=True,
    )

    # ── Bouton pour afficher les détails complets
    show_details = st.checkbox("📊 Afficher l'analyse détaillée", value=False)

    if show_details:
        st.markdown("---")
        st.markdown("<div class='section-title'>📊 Analyse détaillée</div>",
                    unsafe_allow_html=True)

        r1, r2 = st.columns([1, 2])

        with r1:
            # ← On passe directement proba_live à la jauge
            st.plotly_chart(make_gauge(proba_live, color_live), use_container_width=True)
            st.markdown(
                f"<div class='risk-label' style='color:{color_live};'>{level_live}</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"Source : {source_live}  |  Seuil : {threshold:.4f}")

        with r2:
            icon = MACHINE_ICONS.get(machine_type, "⚙️")
            st.markdown(f"""
            <div class='{css_live}'>
              <h3 style='margin:0 0 10px;'>{icon} {machine_id} — {machine_type}
                <span style='font-size:.75rem;color:#666;'> · {site}</span>
              </h3>
              <p><strong>Score :</strong>
                <span style='font-size:1.2rem;font-weight:800;color:{color_live};'>
                  {proba_live:.1%}
                </span>
              </p>
              <p><strong>Niveau :</strong> {level_live}</p>
              <p><strong>Action :</strong> {action_live}</p>
              <p><strong>Maintenance :</strong> {get_maintenance_date(proba_live)}</p>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>**📏 Indicateurs clés**", unsafe_allow_html=True)
            kcols = st.columns(4)
            kpi_data = [
                ("🌡️ Temp.",    f"{temp:.0f}°C",        temp > 80,      "Surchauffe"),
                ("📳 Vibration",f"{vibration:.1f} mm/s", vibration > 15, "Élevée"),
                ("🛢️ Huile",   f"{oil:.0f}%",           oil < 30,       "Critique"),
                ("🚨 Erreurs",  str(int(errors_30d)),    errors_30d > 5, "Excessif"),
            ]
            for col, (lbl, val, warn, wlbl) in zip(kcols, kpi_data):
                bg  = "#FFEBEE" if warn else "#E8F5E9"
                clr = OCP_RED   if warn else OCP_GREEN
                col.markdown(
                    f"<div style='background:{bg};border-radius:8px;padding:10px;"
                    f"text-align:center;border-top:3px solid {clr};'>"
                    f"<div style='font-size:.75rem;color:#666;'>{lbl}</div>"
                    f"<div style='font-size:1.3rem;font-weight:800;color:{clr};'>{val}</div>"
                    f"{'<div style=font-size:.7rem;color:'+clr+';>'+wlbl+'</div>' if warn else ''}"
                    f"</div>", unsafe_allow_html=True,
                )

        # ── Tabs détails
        tab1, tab2, tab3 = st.tabs(
            ["📊 Facteurs de risque", "🕸️ Radar capteurs", "📈 Historique simulé"]
        )

        with tab1:
            factors = [
                ("🌡️ Température",     float(np.clip((temp-40)/60,    0,1)) * 0.25, 0.25),
                ("📳 Vibration",        float(np.clip(vibration/30,    0,1)) * 0.20, 0.20),
                ("🚨 Codes erreur",     float(np.clip(errors_30d/15,   0,1)) * 0.18, 0.18),
                ("🛢️ Huile",           float(np.clip((100-oil)/90,    0,1)) * 0.15, 0.15),
                ("❄️ Refroidissement", float(np.clip((100-coolant)/90, 0,1)) * 0.08, 0.08),
                ("🔧 Retard maint.",   float(np.clip(last_maint/400,   0,1)) * 0.08, 0.08),
                ("💥 Pannes passées",  float(np.clip(fail_count/8,     0,1)) * 0.04, 0.04),
                ("🤖 AI override",     float(np.clip(ai_events/5,      0,1)) * 0.02, 0.02),
            ]
            for label, val, weight in factors:
                pct     = min(val / weight * 100, 100) if weight > 0 else 0
                bar_col = OCP_RED if pct > 70 else OCP_ORANGE if pct > 40 else OCP_GREEN
                st.markdown(
                    f"<div class='factor-bar-wrap'>"
                    f"<span class='factor-label'>{label}</span>"
                    f"<div class='factor-track'>"
                    f"<div class='factor-fill' style='width:{pct:.0f}%;background:{bar_col};'>"
                    f"</div></div>"
                    f"<span style='width:50px;text-align:right;font-size:.8rem;"
                    f"color:{bar_col};font-weight:600;'>{pct:.0f}%</span>"
                    f"<span class='factor-val'>{val:.3f}</span>"
                    f"</div>", unsafe_allow_html=True,
                )

        with tab2:
            st.plotly_chart(make_radar(d_live), use_container_width=True)

        with tab3:
            st.plotly_chart(make_history_sim(proba_live), use_container_width=True)

        # Export JSON
        with st.expander("📥 Exporter le rapport JSON"):
            report = {
                "machine_id": machine_id, "machine_type": machine_type,
                "site": site, "timestamp": datetime.now().isoformat(),
                "score": round(proba_live, 4), "level": level_live,
                "action": action_live, "source": source_live,
                "sensors": {
                    "Temperature_C": temp, "Vibration_mms": vibration,
                    "Oil_Level_pct": oil, "Coolant_Level_pct": coolant,
                    "Error_Codes_Last_30_Days": errors_30d,
                    "Last_Maintenance_Days_Ago": last_maint,
                },
            }
            st.json(report)
            st.download_button(
                "⬇️ Télécharger JSON",
                json.dumps(report, indent=2, ensure_ascii=False).encode(),
                f"rapport_{machine_id}.json", "application/json",
            )

# ══════════════════════════════════════════════════════════════
# PAGE 2 — ANALYSE PAR LOT
# ══════════════════════════════════════════════════════════════
elif page == "📋 Analyse par Lot":

    st.markdown("<div class='section-title'>📋 Analyse Fleet</div>", unsafe_allow_html=True)

    col_s1, col_s2 = st.columns([3,1])
    n_machines = col_s1.slider("Nombre de machines", 5, 150, 40, 5)
    seed_val   = col_s2.number_input("Graine", 0, 999, 42, 1)

    if st.button("🔄 Générer & Analyser", type="primary", use_container_width=True):
        np.random.seed(int(seed_val))
        mt = list(MACHINE_TYPE_MAP.keys())

        with st.spinner(f"⏳ Analyse de {n_machines} machines…"):
            batch = [{
                "Machine_ID":                f"MC_OCP_{i:04d}",
                "Machine_Type":              np.random.choice(mt),
                "Site":                      np.random.choice(SITES_OCP),
                "Operational_Hours":         float(np.random.randint(5_000,100_000)),
                "Temperature_C":             float(np.random.uniform(35,110)),
                "Vibration_mms":             float(np.random.uniform(1,30)),
                "Sound_dB":                  float(np.random.uniform(55,100)),
                "Oil_Level_pct":             float(np.random.uniform(5,100)),
                "Coolant_Level_pct":         float(np.random.uniform(10,100)),
                "Power_Consumption_kW":      float(np.random.uniform(30,250)),
                "Last_Maintenance_Days_Ago": float(np.random.randint(0,400)),
                "Maintenance_History_Count": float(np.random.randint(1,10)),
                "Failure_History_Count":     float(np.random.randint(0,8)),
                "AI_Supervision":            bool(np.random.choice([True,False])),
                "Error_Codes_Last_30_Days":  float(np.random.randint(0,15)),
                "AI_Override_Events":        float(np.random.randint(0,5)),
                "Installation_Year":         int(np.random.randint(2010,2038)),
            } for i in range(n_machines)]

            prog = st.progress(0)
            rows = []
            for idx, rec in enumerate(batch):
                proba, _ = get_score(rec, model, preproc)
                lvl, _, _, act, sev = classify_risk(proba, threshold)
                rows.append({
                    "Machine": rec["Machine_ID"], "Type": rec["Machine_Type"],
                    "Site":    rec["Site"],
                    "Score":   round(proba, 4),
                    "Niveau":  lvl, "Action": act,
                })
                prog.progress((idx+1)/n_machines)
            prog.empty()

        res = (pd.DataFrame(rows)
               .sort_values("Score", ascending=False)
               .reset_index(drop=True))

        # KPIs
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("🔴 Critiques", int((res["Score"]>=0.80).sum()))
        k2.metric("🟠 Élevés",    int(((res["Score"]>=0.55)&(res["Score"]<0.80)).sum()))
        k3.metric("🟡 Modérés",   int(((res["Score"]>=threshold)&(res["Score"]<0.55)).sum()))
        k4.metric("🟢 Faibles",   int((res["Score"]<threshold).sum()))

        t1, t2, t3 = st.tabs(["📊 Scores", "🍩 Répartition", "📋 Tableau"])
        with t1:
            st.plotly_chart(make_fleet_bar(res, threshold), use_container_width=True)
        with t2:
            st.plotly_chart(make_pie(res, threshold), use_container_width=True)
        with t3:
            st.dataframe(
                res.style
                .background_gradient(subset=["Score"], cmap="RdYlGn_r", vmin=0, vmax=1)
                .format({"Score": "{:.2%}"}),
                use_container_width=True, hide_index=True, height=400,
            )

        st.download_button(
            "⬇️ Télécharger CSV",
            res.to_csv(index=False).encode("utf-8"),
            f"ocp_risk_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv",
            use_container_width=True,
        )

# ══════════════════════════════════════════════════════════════
# PAGE 3 — À PROPOS
# ══════════════════════════════════════════════════════════════
elif page == "ℹ️ À propos":
    st.markdown("<div class='section-title'>ℹ️ À propos</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
**Modèle :** {meta['model_name']}
**ROC-AUC :** {meta['roc_auc_test']:.4f}
**F1-Score :** {meta['f1_test']:.3f}
**Seuil :** {threshold:.4f}
**Train samples :** {meta['train_samples']:,}
        """)
    with c2:
        st.markdown("""
**Features clés :**
- `Thermal_Stress` = Temp × Vib / 100
- `Maintenance_Urgency` = Jours / (Maint+1)
- `Fluid_Degradation` = (Huile+Coolant dégradés)/2
- `Failure_Density` = Pannes / (Heures/1000)

**Sites :** Khouribga · Youssoufia · Gantour · Jorf Lasfar · Safi · Benguerir
        """)
    st.caption(f"© {datetime.now().year} OCP Group — Maintenance IA v2.0")