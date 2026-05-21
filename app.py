import streamlit as st
import numpy as np
import pandas as pd
import pickle, json, warnings
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta
import time

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="OCP — Maintenance Prédictive",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Palette OCP
OCP_GREEN  = "#006633"
OCP_ORANGE = "#FF6600"
OCP_RED    = "#D32F2F"
OCP_YELLOW = "#FFC107"
OCP_BLUE   = "#1565C0"
OCP_PURPLE = "#6A1B9A"

# ══════════════════════════════════════════════════════════════
# CSS GLOBAL AMÉLIORÉ
# ══════════════════════════════════════════════════════════════
st.markdown(f"""
<style>
  /* ── Variables */
  :root {{
    --green:  {OCP_GREEN};
    --orange: {OCP_ORANGE};
    --red:    {OCP_RED};
    --yellow: {OCP_YELLOW};
    --blue:   {OCP_BLUE};
  }}

  /* ── Header */
  .ocp-header {{
    background: linear-gradient(135deg, {OCP_GREEN} 0%, #004d26 60%, #003d1f 100%);
    padding: 1.4rem 2rem;
    border-radius: 14px;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 15px rgba(0,102,51,.3);
    display: flex;
    align-items: center;
    gap: 1rem;
  }}
  .ocp-header h1 {{
    color: white; margin: 0;
    font-size: 1.9rem; font-weight: 800; letter-spacing: -.5px;
  }}
  .ocp-header p {{
    color: rgba(255,255,255,.8); margin: 4px 0 0;
    font-size: .92rem;
  }}
  .ocp-logo {{
    font-size: 3rem; line-height: 1;
  }}

  /* ── KPI Cards */
  .kpi-card {{
    background: white;
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    text-align: center;
    box-shadow: 0 2px 10px rgba(0,0,0,.08);
    border-top: 4px solid var(--green);
    transition: transform .2s, box-shadow .2s;
    height: 100%;
  }}
  .kpi-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(0,0,0,.12);
  }}
  .kpi-card .kpi-value {{
    font-size: 2rem; font-weight: 800; margin: .3rem 0;
  }}
  .kpi-card .kpi-label {{
    font-size: .82rem; color: #666; text-transform: uppercase;
    letter-spacing: .5px;
  }}
  .kpi-card .kpi-delta {{
    font-size: .78rem; margin-top: .3rem; font-weight: 600;
  }}

  /* ── Alert boxes */
  .alert-critical {{
    background: linear-gradient(135deg,#FFEBEE,#FFCDD2);
    border: 2px solid {OCP_RED};
    border-radius: 12px; padding: 1.2rem;
    box-shadow: 0 3px 10px rgba(211,47,47,.2);
  }}
  .alert-high {{
    background: linear-gradient(135deg,#FFF3E0,#FFE0B2);
    border: 2px solid {OCP_ORANGE};
    border-radius: 12px; padding: 1.2rem;
    box-shadow: 0 3px 10px rgba(255,102,0,.2);
  }}
  .alert-moderate {{
    background: linear-gradient(135deg,#FFFDE7,#FFF9C4);
    border: 2px solid {OCP_YELLOW};
    border-radius: 12px; padding: 1.2rem;
    box-shadow: 0 3px 10px rgba(255,193,7,.2);
  }}
  .alert-low {{
    background: linear-gradient(135deg,#E8F5E9,#C8E6C9);
    border: 2px solid {OCP_GREEN};
    border-radius: 12px; padding: 1.2rem;
    box-shadow: 0 3px 10px rgba(0,102,51,.15);
  }}

  /* ── Section titles */
  .section-title {{
    font-size: 1.1rem; font-weight: 700; color: {OCP_GREEN};
    border-bottom: 3px solid {OCP_GREEN};
    padding-bottom: 6px; margin-bottom: 1rem;
    letter-spacing: -.2px;
  }}

  /* ── Risk label */
  .risk-label {{
    text-align: center; font-size: 1.5rem;
    font-weight: 800; margin-top: -8px;
    text-shadow: 0 1px 3px rgba(0,0,0,.15);
  }}

  /* ── Progress bar factor */
  .factor-bar-wrap {{
    display: flex; align-items: center;
    gap: 8px; margin: 4px 0;
  }}
  .factor-label  {{ width: 175px; font-size: .85rem; }}
  .factor-track  {{
    flex: 1; background: #eee;
    border-radius: 6px; height: 14px; overflow: hidden;
  }}
  .factor-fill   {{
    height: 14px; border-radius: 6px;
    transition: width .6s ease;
  }}
  .factor-val    {{ width: 40px; text-align: right; font-size: .8rem; color: #555; }}

  /* ── Timeline badge */
  .timeline-badge {{
    display: inline-block;
    padding: 3px 10px; border-radius: 20px;
    font-size: .75rem; font-weight: 700;
    letter-spacing: .3px;
  }}

  /* ── Sensor chips */
  .sensor-chip {{
    display: inline-flex; align-items: center; gap: 6px;
    background: #f5f5f5; border-radius: 20px;
    padding: 5px 12px; font-size: .83rem;
    border: 1px solid #ddd; margin: 3px;
  }}

  /* ── Sidebar */
  .sidebar-badge {{
    background: {OCP_GREEN}22; border-radius: 8px;
    padding: 8px 12px; margin: 4px 0;
    border-left: 3px solid {OCP_GREEN};
    font-size: .85rem;
  }}

  /* ── Tooltip */
  .tooltip-text {{
    font-size: .78rem; color: #888;
    font-style: italic; margin-top: 2px;
  }}

  /* ── Scrollbar */
  ::-webkit-scrollbar {{ width: 6px; }}
  ::-webkit-scrollbar-thumb {{
    background: {OCP_GREEN}55; border-radius: 3px;
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
    "CNC_Lathe":0, "Conveyor_Belt":1, "Crusher":2, "Dryer":3,
    "Filter_Press":4, "Flotation_Cell":5, "Hydraulic_Press":6,
    "Mill":7, "Pump":8, "Reactor":9,
}

MACHINE_ICONS = {
    "CNC_Lathe":"⚙️", "Conveyor_Belt":"🔄", "Crusher":"💥",
    "Dryer":"🌡️", "Filter_Press":"🔩", "Flotation_Cell":"🫧",
    "Hydraulic_Press":"🔧", "Mill":"⚡", "Pump":"💧", "Reactor":"⚗️",
}

SITES_OCP = ["Khouribga","Youssoufia","Gantour","Jorf Lasfar","Safi","Benguerir"]

# ══════════════════════════════════════════════════════════════
# FONCTIONS MÉTIER
# ══════════════════════════════════════════════════════════════
def heuristic_score(temp, vib, oil, cool, errors, last_m, fails, ai_ov) -> float:
    """Score heuristique pondéré — toujours disponible."""
    s  = float(np.clip((temp  - 40) / 60,   0, 1)) * 0.25
    s += float(np.clip(vib   / 30,           0, 1)) * 0.20
    s += float(np.clip((100 - oil)  / 90,   0, 1)) * 0.15
    s += float(np.clip((100 - cool) / 90,   0, 1)) * 0.08
    s += float(np.clip(errors / 15,          0, 1)) * 0.18
    s += float(np.clip(last_m / 400,         0, 1)) * 0.08
    s += float(np.clip(fails  / 8,           0, 1)) * 0.04
    s += float(np.clip(ai_ov  / 5,           0, 1)) * 0.02
    return float(np.clip(s, 0.02, 0.97))


def engineer_features(d: dict) -> pd.DataFrame:
    """Construit le vecteur de features à partir du dict brut."""
    age = max(2040 - int(d.get("Installation_Year", 2025)), 1)
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
        'Thermal_Stress':             float(d.get('Temperature_C',50)) * float(d.get('Vibration_mms',5)) / 100,
        'Maintenance_Urgency':        float(d.get('Last_Maintenance_Days_Ago',60)) / (float(d.get('Maintenance_History_Count',3)) + 1),
        'Fluid_Degradation':         ((100 - float(d.get('Oil_Level_pct',80))) + (100 - float(d.get('Coolant_Level_pct',80)))) / 2,
        'Error_Rate':                 float(d.get('Error_Codes_Last_30_Days',0)) / 30,
        'Failure_Density':            float(d.get('Failure_History_Count',0)) / (oh / 1000 + 1),
        'High_Vibration_Flag':        int(float(d.get('Vibration_mms',5)) > 15),
        'Overheat_Flag':              int(float(d.get('Temperature_C',50)) > 80),
        'Late_Maintenance_Flag':      int(float(d.get('Last_Maintenance_Days_Ago',60)) > 180),
        'Machine_Type_Enc':           float(MACHINE_TYPE_MAP.get(str(d.get('Machine_Type','Pump')), 8)),
        'AI_Supervision_Int':         int(bool(d.get('AI_Supervision', False))),
    }
    return pd.DataFrame([row])[FEATURE_COLS]


@st.cache_resource(show_spinner="🔄 Chargement du modèle ML…")
def load_artifacts():
    """Charge le pkl — tolère toutes les erreurs."""
    pkl_path = Path("predictive_maintenance_pipeline.pkl")
    if not pkl_path.exists():
        return None, None, 0.8817, "pkl introuvable"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(pkl_path, "rb") as f:
                arts = pickle.load(f)
    except Exception as e:
        return None, None, 0.8817, f"Erreur chargement pkl : {e}"

    model     = arts.get("model")
    preproc   = arts.get("preprocessor")
    threshold = float(arts.get("threshold", 0.8817))

    # Validation rapide
    try:
        test = engineer_features({
            'Temperature_C':65,'Vibration_mms':8,'Oil_Level_pct':75,
            'Coolant_Level_pct':80,'Operational_Hours':45000,
            'Last_Maintenance_Days_Ago':90,'Maintenance_History_Count':5,
            'Failure_History_Count':2,'Error_Codes_Last_30_Days':2,
            'AI_Override_Events':1,'Machine_Type':'Pump','Installation_Year':2025
        })
        _ = model.predict_proba(preproc.transform(test))[0, 1]
        return model, preproc, threshold, "ok"
    except Exception as e:
        return None, None, threshold, f"Préprocesseur incompatible : {type(e).__name__}"


@st.cache_data(show_spinner=False)
def load_metadata():
    p = Path("model_metadata.json")
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {
        "model_name":      "Régression Logistique",
        "roc_auc_test":    0.9837,
        "avg_prec_test":   0.7591,
        "f1_test":         0.632,
        "optimal_threshold": 0.8817,
        "n_features":      28,
        "train_samples":   276101,
        "training_date":   "2040",
        "target":          "Failure_Within_7_Days",
    }


def get_score(d: dict, model, preproc) -> tuple[float, str]:
    """Retourne (score, source). Ne lève jamais d'exception."""
    h = heuristic_score(
        temp   = float(d.get('Temperature_C', 50)),
        vib    = float(d.get('Vibration_mms', 5)),
        oil    = float(d.get('Oil_Level_pct', 80)),
        cool   = float(d.get('Coolant_Level_pct', 80)),
        errors = float(d.get('Error_Codes_Last_30_Days', 0)),
        last_m = float(d.get('Last_Maintenance_Days_Ago', 60)),
        fails  = float(d.get('Failure_History_Count', 0)),
        ai_ov  = float(d.get('AI_Override_Events', 0)),
    )
    if model is None or preproc is None:
        return h, "heuristique"
    try:
        X   = engineer_features(d)
        X_p = preproc.transform(X)
        return float(model.predict_proba(X_p)[0, 1]), "modèle ML"
    except Exception:
        return h, "heuristique"


def classify_risk(p: float, thr: float) -> tuple:
    if p >= 0.80:
        return "🔴 CRITIQUE", OCP_RED,    "alert-critical", "Arrêt immédiat + Maintenance d'urgence", 4
    if p >= 0.55:
        return "🟠 ÉLEVÉ",   OCP_ORANGE,  "alert-high",     "Planifier maintenance sous 48h",          3
    if p >= thr:
        return "🟡 MODÉRÉ",  OCP_YELLOW,  "alert-moderate", "Surveillance renforcée + inspection",     2
    return     "🟢 FAIBLE",  OCP_GREEN,   "alert-low",      "Fonctionnement normal — maintenance planifiée", 1


def get_maintenance_date(score: float, last_days: int) -> str:
    """Estime la date de maintenance recommandée."""
    if score >= 0.80: days = 0
    elif score >= 0.55: days = 2
    elif score >= 0.40: days = 7
    else: days = 30
    dt = datetime.now() + timedelta(days=days)
    return dt.strftime("%d/%m/%Y") if days > 0 else "⚡ IMMÉDIAT"


# ══════════════════════════════════════════════════════════════
# GRAPHIQUES
# ══════════════════════════════════════════════════════════════
def make_gauge(proba: float, color: str, title: str = "Score de Risque") -> go.Figure:
    """Jauge animée avec zones colorées."""
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = round(proba * 100, 1),
        delta = {"reference": 50, "valueformat": ".1f",
                 "increasing": {"color": OCP_RED},
                 "decreasing": {"color": OCP_GREEN}},
        number = {"suffix": "%", "font": {"size": 40, "color": color, "family": "Arial Black"}},
        gauge  = {
            "axis": {"range": [0, 100], "tickfont": {"size": 11}, "ticksuffix": "%"},
            "bar":  {"color": color, "thickness": 0.30},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#ddd",
            "steps": [
                {"range": [0,  55], "color": "#E8F5E9"},
                {"range": [55, 80], "color": "#FFF3E0"},
                {"range": [80,100], "color": "#FFEBEE"},
            ],
            "threshold": {
                "line": {"color": OCP_RED, "width": 3},
                "thickness": 0.80,
                "value": 80,
            },
        },
        title = {"text": title, "font": {"size": 14, "color": "#333"}},
    ))
    fig.update_layout(
        height=290,
        margin=dict(l=20, r=20, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
    )
    return fig


def make_radar(d: dict) -> go.Figure:
    """Radar chart des indicateurs normalisés."""
    categories = ['Température','Vibration','Codes Erreur',
                  'Huile','Refroidissement','Maintenance']
    values = [
        float(np.clip((float(d.get('Temperature_C',50)) - 20) / 100, 0, 1)),
        float(np.clip(float(d.get('Vibration_mms',5))    / 35,       0, 1)),
        float(np.clip(float(d.get('Error_Codes_Last_30_Days',0)) / 15, 0, 1)),
        float(np.clip((100 - float(d.get('Oil_Level_pct',80)))   / 95, 0, 1)),
        float(np.clip((100 - float(d.get('Coolant_Level_pct',80)))/ 95, 0, 1)),
        float(np.clip(float(d.get('Last_Maintenance_Days_Ago',60)) / 400, 0, 1)),
    ]
    fig = go.Figure()
    # Zone de danger
    fig.add_trace(go.Scatterpolar(
        r=[0.7]*len(categories) + [0.7],
        theta=categories + [categories[0]],
        fill='toself', fillcolor='rgba(211,47,47,.08)',
        line=dict(color=OCP_RED, dash='dot', width=1),
        name='Zone danger', showlegend=False,
    ))
    # Valeurs actuelles
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(0,102,51,.15)',
        line=dict(color=OCP_GREEN, width=2.5),
        name='Capteurs actuels',
        marker=dict(size=6, color=OCP_GREEN),
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0,1],
                            tickfont=dict(size=9), gridcolor="#ddd"),
            angularaxis=dict(tickfont=dict(size=11)),
            bgcolor="white",
        ),
        showlegend=False,
        height=300,
        margin=dict(l=40, r=40, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_history_simulation(proba: float) -> go.Figure:
    """Simule l'évolution du score sur 30 jours."""
    np.random.seed(int(proba * 100))
    days  = list(range(-29, 1))
    trend = np.linspace(max(0.05, proba - 0.35), proba, 30)
    noise = np.random.normal(0, 0.025, 30)
    scores = np.clip(trend + noise, 0.02, 0.97)

    fig = go.Figure()
    # Zone critique
    fig.add_hrect(y0=0.80, y1=1.0, fillcolor="rgba(211,47,47,.1)",
                  line_width=0, annotation_text="⚠️ Critique")
    fig.add_hrect(y0=0.55, y1=0.80, fillcolor="rgba(255,102,0,.08)",
                  line_width=0, annotation_text="Élevé")

    # Courbe avec gradient via scatter
    fig.add_trace(go.Scatter(
        x=days, y=scores.tolist(),
        mode='lines+markers',
        line=dict(color=OCP_GREEN, width=2.5, shape='spline'),
        marker=dict(
            size=5,
            color=scores.tolist(),
            colorscale=[[0,'#006633'],[0.55,'#FFC107'],[0.80,'#D32F2F'],[1,'#B71C1C']],
            cmin=0, cmax=1,
        ),
        name='Score risque',
        fill='tonexty',
        fillcolor='rgba(0,102,51,.06)',
    ))
    # Point actuel
    fig.add_trace(go.Scatter(
        x=[0], y=[float(proba)],
        mode='markers',
        marker=dict(size=14, color=OCP_RED, symbol='star',
                    line=dict(color='white', width=2)),
        name='Aujourd\'hui',
    ))

    fig.update_layout(
        title="Évolution du score — 30 derniers jours (simulation)",
        xaxis=dict(title="Jours relatifs", gridcolor="#eee"),
        yaxis=dict(title="Score de risque", range=[0,1.05],
                   tickformat=".0%", gridcolor="#eee"),
        height=280,
        margin=dict(l=10, r=10, t=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
    )
    return fig


def make_fleet_bar(res: pd.DataFrame, threshold: float) -> go.Figure:
    """Graphique en barres horizontales avec couleurs par niveau."""
    colors = [
        OCP_RED    if s >= 0.80 else
        OCP_ORANGE if s >= 0.55 else
        OCP_YELLOW if s >= threshold else
        OCP_GREEN
        for s in res["Score"]
    ]
    fig = go.Figure(go.Bar(
        x=res["Score"], y=res["Machine"],
        orientation="h",
        marker=dict(
            color=colors,
            line=dict(color='white', width=0.5),
            cornerradius=4,
        ),
        text=[f"{s:.0%}" for s in res["Score"]],
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Score : %{x:.1%}<br>"
            "<extra></extra>"
        ),
    ))
    fig.add_vline(
        x=threshold, line_dash="dash", line_color="#555", line_width=1.5,
        annotation=dict(text=f"Seuil {threshold:.2f}",
                        font=dict(color="#555", size=11), bgcolor="white"),
    )
    fig.update_layout(
        height=max(380, len(res) * 24),
        xaxis=dict(range=[0, 1.15], title="Score de risque",
                   tickformat=".0%", gridcolor="#eee"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=130, r=70, t=20, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
    )
    return fig


def make_distribution_pie(res: pd.DataFrame, threshold: float) -> go.Figure:
    """Camembert de distribution des niveaux de risque."""
    critiques = int((res["Score"] >= 0.80).sum())
    eleves    = int(((res["Score"] >= 0.55) & (res["Score"] < 0.80)).sum())
    moderes   = int(((res["Score"] >= threshold) & (res["Score"] < 0.55)).sum())
    faibles   = int((res["Score"] < threshold).sum())

    fig = go.Figure(go.Pie(
        labels=["🔴 Critique","🟠 Élevé","🟡 Modéré","🟢 Faible"],
        values=[critiques, eleves, moderes, faibles],
        marker=dict(colors=[OCP_RED, OCP_ORANGE, OCP_YELLOW, OCP_GREEN],
                    line=dict(color='white', width=2)),
        hole=0.55,
        textinfo="label+percent",
        textfont=dict(size=11),
        pull=[0.05 if critiques > 0 else 0, 0, 0, 0],
    ))
    fig.update_layout(
        title="Répartition des niveaux",
        height=280,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def make_type_risk_heatmap(res: pd.DataFrame) -> go.Figure:
    """Score moyen par type de machine."""
    avg = (res.groupby("Type")["Score"]
               .mean()
               .sort_values(ascending=False)
               .reset_index())
    colors = [
        OCP_RED    if s >= 0.80 else
        OCP_ORANGE if s >= 0.55 else
        OCP_YELLOW if s >= 0.30 else
        OCP_GREEN
        for s in avg["Score"]
    ]
    fig = go.Figure(go.Bar(
        x=avg["Score"], y=avg["Type"],
        orientation='h',
        marker=dict(color=colors, cornerradius=4),
        text=[f"{s:.0%}" for s in avg["Score"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Score moyen par type de machine",
        xaxis=dict(range=[0,1.15], tickformat=".0%", gridcolor="#eee"),
        yaxis=dict(autorange="reversed"),
        height=300,
        margin=dict(l=120,r=60,t=40,b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
    )
    return fig


# ══════════════════════════════════════════════════════════════
# CHARGEMENT ARTEFACTS
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
  <div class="ocp-logo">🏭</div>
  <div>
    <h1>OCP — Maintenance Prédictive</h1>
    <p>Prédiction des pannes machines dans les 7 prochains jours &nbsp;·&nbsp;
       Office Chérifien des Phosphates &nbsp;·&nbsp;
       {datetime.now().strftime('%d %B %Y, %H:%M')}</p>
  </div>
</div>
""", unsafe_allow_html=True)

# Bandeau de statut
if ml_active:
    st.success("✅ Modèle ML chargé et opérationnel — Prédictions haute précision actives")
else:
    st.info(f"⚙️ Mode estimation heuristique ({status}). "
            "Scores représentatifs basés sur 8 indicateurs capteurs.")

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"<div class='section-title'>📊 Modèle & Performance</div>",
                unsafe_allow_html=True)

    # Métriques du modèle
    for label, val in [
        ("🤖 Modèle",        meta["model_name"]),
        ("📈 ROC-AUC",       f"{meta['roc_auc_test']:.4f}"),
        ("🎯 F1-Score",      f"{meta['f1_test']:.3f}"),
        ("⚖️ Seuil optimal", f"{threshold:.4f}"),
        ("🗃️ Train samples", f"{meta['train_samples']:,}"),
        ("📅 Date training", str(meta.get('training_date','N/A'))),
    ]:
        st.markdown(
            f"<div class='sidebar-badge'><b>{label}</b><br>"
            f"<span style='font-size:.95rem;color:#333;'>{val}</span></div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # Statut ML
    if ml_active:
        st.markdown("✅ **Modèle ML actif**")
    else:
        st.markdown("⚙️ **Mode heuristique**")

    st.divider()

    # Navigation
    page = st.radio(
        "🧭 Navigation",
        ["🔬 Prédiction Individuelle",
         "📋 Analyse par Lot",
         "📈 Tableau de Bord",
         "ℹ️ À propos"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption(f"© {datetime.now().year} OCP Group · Maintenance IA")

# ══════════════════════════════════════════════════════════════
# PAGE 1 — PRÉDICTION INDIVIDUELLE
# ══════════════════════════════════════════════════════════════
if page == "🔬 Prédiction Individuelle":

    st.markdown("<div class='section-title'>🔬 Saisie des données capteurs</div>",
                unsafe_allow_html=True)

    # ── Identité machine
    with st.container():
        c0a, c0b, c0c, c0d, c0e = st.columns([2,2,2,1,1])
        machine_id     = c0a.text_input("🏷️ Machine ID", "MC_OCP_0001")
        machine_type   = c0b.selectbox("⚙️ Type de machine",
                                        list(MACHINE_TYPE_MAP.keys()), index=8)
        site           = c0c.selectbox("📍 Site OCP", SITES_OCP)
        ai_supervision = c0d.checkbox("🤖 Supervision IA", True)
        inst_year      = c0e.number_input("📅 Année install.", 2000, 2040, 2025)

    st.divider()

    # ── Capteurs : 3 colonnes
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**🌡️ Thermique & Mécanique**")
        temp      = st.slider("Température (°C)",             20.0, 120.0, 65.0, 0.5,
                               help="Température interne mesurée par capteur thermique")
        vibration = st.slider("Vibration (mm/s)",              0.0,  35.0,  8.0, 0.1,
                               help="Amplitude de vibration RMS")
        sound     = st.slider("Niveau sonore (dB)",           50.0, 110.0, 72.0, 0.5,
                               help="Niveau sonore ambiant mesuré en dB(A)")
        power     = st.slider("Consommation (kW)",            10.0, 300.0, 95.0, 1.0,
                               help="Puissance électrique instantanée")

        # Chips de statut inline
        chips = []
        if temp > 80:     chips.append(("🌡️ Surchauffe",  OCP_RED))
        if vibration > 15:chips.append(("📳 Vib. élevée", OCP_ORANGE))
        if sound > 90:    chips.append(("🔊 Bruit excessif", OCP_YELLOW))
        if chips:
            html = "".join(
                f"<span class='sensor-chip' style='border-color:{c};color:{c};'>{l}</span>"
                for l, c in chips
            )
            st.markdown(html, unsafe_allow_html=True)

    with c2:
        st.markdown("**💧 Fluides**")
        oil     = st.slider("Niveau huile (%)",               5.0, 100.0, 75.0, 1.0)
        coolant = st.slider("Liquide refroidissement (%)",    5.0, 100.0, 80.0, 1.0)

        # Mini jauges visuelles
        for label, val, warn in [("Huile", oil, 30), ("Refroidissement", coolant, 25)]:
            pct = int(val)
            col = OCP_RED if pct < warn else OCP_ORANGE if pct < 50 else OCP_GREEN
            st.markdown(
                f"<div style='margin:4px 0;font-size:.82rem;color:#555;'>{label} : "
                f"<b style='color:{col};'>{pct}%</b></div>"
                f"<div style='background:#eee;border-radius:6px;height:8px;'>"
                f"<div style='width:{pct}%;background:{col};border-radius:6px;height:8px;'>"
                f"</div></div>", unsafe_allow_html=True,
            )

        st.markdown("**⏱️ Utilisation**")
        op_hours = st.number_input("Heures opérationnelles", 0, 200_000, 45_000, 500)

    with c3:
        st.markdown("**🔧 Maintenance & Historique**")
        last_maint  = st.slider("Jours depuis maintenance",   0, 500, 90, 1)
        maint_count = st.number_input("Nb maintenances",      0,  50,  5, 1)
        fail_count  = st.number_input("Pannes historiques",   0,  30,  2, 1)
        errors_30d  = st.slider("Codes erreur / 30j",         0,  30,  2, 1)
        ai_events   = st.number_input("AI override events",   0,  20,  1, 1)

        # Alerte maintenance retardée
        if last_maint > 180:
            st.warning("⚠️ Maintenance en retard significatif !")
        if fail_count >= 5:
            st.error("🔴 Historique de pannes élevé !")

    st.divider()

    # ── Dict de données
    d = {
        "Machine_ID": machine_id, "Machine_Type": machine_type,
        "Site": site, "Installation_Year": inst_year,
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

    # ── Bouton d'analyse
    col_btn1, col_btn2 = st.columns([3,1])
    analyze = col_btn1.button(
        "🚀 Analyser le risque de panne", type="primary", use_container_width=True
    )
    show_details = col_btn2.checkbox("📊 Détails avancés", True)

    if analyze:
        with st.spinner("⏳ Analyse en cours…"):
            time.sleep(0.4)   # petite pause pour UX
            proba, source = get_score(d, model, preproc)

        level, color, css_class, action, severity = classify_risk(proba, threshold)
        maint_date = get_maintenance_date(proba, last_maint)

        # ════ RÉSULTATS ════
        st.markdown("---")
        st.markdown("<div class='section-title'>📊 Résultats de l'analyse</div>",
                    unsafe_allow_html=True)

        r1, r2 = st.columns([1, 2])

        # ── Jauge + niveau
        with r1:
            st.plotly_chart(make_gauge(proba, color), use_container_width=True)
            st.markdown(
                f"<div class='risk-label' style='color:{color};'>{level}</div>",
                unsafe_allow_html=True,
            )
            st.caption(f"Source : {source}  |  Seuil : {threshold:.4f}")

            # Badge maintenance date
            st.markdown(
                f"<div style='text-align:center;margin-top:.8rem;'>"
                f"<span class='timeline-badge' style='background:{color}22;color:{color};"
                f"border:1px solid {color};'>"
                f"🔧 Maintenance : {maint_date}</span></div>",
                unsafe_allow_html=True,
            )

        # ── Carte alerte + KPIs
        with r2:
            icon = MACHINE_ICONS.get(machine_type, "⚙️")
            st.markdown(f"""
            <div class='{css_class}'>
              <h3 style='margin:0 0 10px;'>{icon} {machine_id} — {machine_type}
                <span style='font-size:.75rem;font-weight:400;color:#666;'>
                  &nbsp;·&nbsp;{site}</span>
              </h3>
              <p style='margin:5px 0;'>
                <strong>Score de risque :</strong>
                <span style='font-size:1.1rem;font-weight:700;color:{color};'>
                  {proba:.1%}
                </span>
              </p>
              <p style='margin:5px 0;'><strong>Niveau d'alerte :</strong> {level}</p>
              <p style='margin:5px 0;'>
                <strong>Action recommandée :</strong>
                <span style='font-weight:600;'>{action}</span>
              </p>
              <p style='margin:5px 0;'>
                <strong>Date maintenance :</strong> {maint_date}
              </p>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>**📏 Indicateurs clés**", unsafe_allow_html=True)

            # KPI cards inline
            kpi_data = [
                ("🌡️ Temp.",    f"{temp:.0f}°C",      temp > 80,      "Surchauffe"),
                ("📳 Vibration",f"{vibration:.1f} mm/s",vibration > 15, "Élevée"),
                ("🛢️ Huile",   f"{oil:.0f}%",          oil < 30,       "Critique"),
                ("🚨 Erreurs",  str(errors_30d),         errors_30d > 5, "Excessif"),
            ]
            kcols = st.columns(4)
            for col, (lbl, val, warn, warn_lbl) in zip(kcols, kpi_data):
                bg   = "#FFEBEE" if warn else "#E8F5E9"
                clr  = OCP_RED   if warn else OCP_GREEN
                col.markdown(
                    f"<div style='background:{bg};border-radius:8px;padding:10px;"
                    f"text-align:center;border-top:3px solid {clr};'>"
                    f"<div style='font-size:.75rem;color:#666;'>{lbl}</div>"
                    f"<div style='font-size:1.3rem;font-weight:800;color:{clr};'>{val}</div>"
                    f"{'<div style=font-size:.7rem;color:'+clr+';>'+warn_lbl+'</div>' if warn else ''}"
                    f"</div>", unsafe_allow_html=True,
                )

        # ── Détails avancés
        if show_details:
            st.markdown("<br>", unsafe_allow_html=True)
            tab1, tab2, tab3 = st.tabs(
                ["📊 Facteurs de risque", "🕸️ Radar capteurs", "📈 Historique simulé"]
            )

            with tab1:
                st.markdown("**Contribution de chaque facteur au score global**")
                factors = [
                    ("🌡️ Température",     float(np.clip((temp-40)/60,   0,1)) * 0.25, 0.25),
                    ("📳 Vibration",        float(np.clip(vibration/30,   0,1)) * 0.20, 0.20),
                    ("🚨 Codes erreur",     float(np.clip(errors_30d/15,  0,1)) * 0.18, 0.18),
                    ("🛢️ Huile",           float(np.clip((100-oil)/90,   0,1)) * 0.15, 0.15),
                    ("❄️ Refroidissement", float(np.clip((100-coolant)/90,0,1)) * 0.08, 0.08),
                    ("🔧 Retard maint.",   float(np.clip(last_maint/400,  0,1)) * 0.08, 0.08),
                    ("💥 Pannes passées",  float(np.clip(fail_count/8,    0,1)) * 0.04, 0.04),
                    ("🤖 AI override",     float(np.clip(ai_events/5,     0,1)) * 0.02, 0.02),
                ]
                for label, val, weight in factors:
                    pct      = min(val / weight * 100, 100) if weight > 0 else 0
                    bar_col  = OCP_RED if pct > 70 else OCP_ORANGE if pct > 40 else OCP_GREEN
                    intensity = f"{pct:.0f}%"
                    st.markdown(
                        f"<div class='factor-bar-wrap'>"
                        f"<span class='factor-label'>{label}</span>"
                        f"<div class='factor-track'>"
                        f"<div class='factor-fill' style='width:{pct:.0f}%;background:{bar_col};'>"
                        f"</div></div>"
                        f"<span style='width:40px;text-align:right;font-size:.8rem;"
                        f"color:{bar_col};font-weight:600;'>{intensity}</span>"
                        f"<span class='factor-val'>{val:.3f}</span>"
                        f"</div>", unsafe_allow_html=True,
                    )
                st.caption("Barre = % d'utilisation du poids max. Valeur = contribution absolue.")

            with tab2:
                st.plotly_chart(make_radar(d), use_container_width=True)
                st.caption("La zone rouge en pointillé représente le seuil de danger (70%).")

            with tab3:
                st.plotly_chart(make_history_simulation(proba), use_container_width=True)
                st.caption("Simulation basée sur la tendance actuelle des capteurs.")

        # ── Export JSON du rapport
        with st.expander("📥 Exporter le rapport JSON"):
            report = {
                "machine_id":    machine_id,
                "machine_type":  machine_type,
                "site":          site,
                "timestamp":     datetime.now().isoformat(),
                "score":         round(proba, 4),
                "level":         level,
                "action":        action,
                "maintenance_date": maint_date,
                "source":        source,
                "sensors":       {k: v for k, v in d.items()
                                   if k not in ["Machine_ID","Machine_Type","Site"]},
            }
            st.json(report)
            st.download_button(
                "⬇️ Télécharger JSON",
                json.dumps(report, indent=2, ensure_ascii=False).encode(),
                f"rapport_{machine_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                "application/json",
            )

# ══════════════════════════════════════════════════════════════
# PAGE 2 — ANALYSE PAR LOT
# ══════════════════════════════════════════════════════════════
elif page == "📋 Analyse par Lot":

    st.markdown("<div class='section-title'>📋 Analyse Fleet — Parc de machines OCP</div>",
                unsafe_allow_html=True)

    col_s1, col_s2, col_s3 = st.columns([2,1,1])
    n_machines = col_s1.slider("Nombre de machines", 5, 150, 40, 5)
    seed_val   = col_s2.number_input("Graine aléatoire", 0, 999, 42, 1)
    site_filter= col_s3.multiselect("Sites", SITES_OCP, default=SITES_OCP)

    if st.button("🔄 Générer & Analyser le parc", type="primary", use_container_width=True):
        np.random.seed(int(seed_val))
        mt    = list(MACHINE_TYPE_MAP.keys())

        with st.spinner(f"⏳ Analyse de {n_machines} machines…"):
            batch = [{
                "Machine_ID":                f"MC_OCP_{i:04d}",
                "Machine_Type":              np.random.choice(mt),
                "Site":                      np.random.choice(site_filter or SITES_OCP),
                "Operational_Hours":         int(np.random.randint(5_000, 100_000)),
                "Temperature_C":             float(np.random.uniform(35, 110)),
                "Vibration_mms":             float(np.random.uniform(1, 30)),
                "Sound_dB":                  float(np.random.uniform(55, 100)),
                "Oil_Level_pct":             float(np.random.uniform(5, 100)),
                "Coolant_Level_pct":         float(np.random.uniform(10, 100)),
                "Power_Consumption_kW":      float(np.random.uniform(30, 250)),
                "Last_Maintenance_Days_Ago": int(np.random.randint(0, 400)),
                "Maintenance_History_Count": int(np.random.randint(1, 10)),
                "Failure_History_Count":     int(np.random.randint(0, 8)),
                "AI_Supervision":            bool(np.random.choice([True, False])),
                "Error_Codes_Last_30_Days":  int(np.random.randint(0, 15)),
                "AI_Override_Events":        int(np.random.randint(0, 5)),
                "Installation_Year":         int(np.random.randint(2010, 2038)),
            } for i in range(n_machines)]

            progress = st.progress(0, "Analyse en cours…")
            rows = []
            for idx, rec in enumerate(batch):
                proba, _ = get_score(rec, model, preproc)
                lvl, _, _, act, sev = classify_risk(proba, threshold)
                rows.append({
                    "Machine":  rec["Machine_ID"],
                    "Type":     rec["Machine_Type"],
                    "Site":     rec["Site"],
                    "Score":    round(proba, 4),
                    "Niveau":   lvl,
                    "Sévérité": sev,
                    "Action":   act,
                    "Maint. recommandée": get_maintenance_date(proba, rec["Last_Maintenance_Days_Ago"]),
                })
                progress.progress((idx+1)/n_machines)
            progress.empty()

        res = (pd.DataFrame(rows)
               .sort_values("Score", ascending=False)
               .reset_index(drop=True))

        # ── KPI résumé
        st.markdown("<div class='section-title'>📊 Synthèse du parc</div>",
                    unsafe_allow_html=True)

        critiques = int((res["Score"] >= 0.80).sum())
        eleves    = int(((res["Score"] >= 0.55) & (res["Score"] < 0.80)).sum())
        moderes   = int(((res["Score"] >= threshold) & (res["Score"] < 0.55)).sum())
        faibles   = int((res["Score"] < threshold).sum())
        avg_score = res["Score"].mean()

        kpi_cols = st.columns(5)
        kpi_defs = [
            ("🔴 Critiques",  critiques, OCP_RED,    f"{critiques/len(res):.0%}"),
            ("🟠 Élevés",     eleves,    OCP_ORANGE,  f"{eleves/len(res):.0%}"),
            ("🟡 Modérés",    moderes,   OCP_YELLOW,  f"{moderes/len(res):.0%}"),
            ("🟢 Faibles",    faibles,   OCP_GREEN,   f"{faibles/len(res):.0%}"),
            ("📊 Score moy.", f"{avg_score:.1%}", OCP_BLUE, f"{len(res)} machines"),
        ]
        for col, (lbl, val, clr, sub) in zip(kpi_cols, kpi_defs):
            col.markdown(
                f"<div class='kpi-card' style='border-top-color:{clr};'>"
                f"<div class='kpi-label'>{lbl}</div>"
                f"<div class='kpi-value' style='color:{clr};'>{val}</div>"
                f"<div class='kpi-delta' style='color:#888;'>{sub}</div>"
                f"</div>", unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Onglets de visualisation
        t1, t2, t3, t4 = st.tabs(
            ["📊 Scores Fleet", "🍩 Répartition", "🏭 Par type", "📋 Tableau"]
        )

        with t1:
            st.plotly_chart(make_fleet_bar(res, threshold), use_container_width=True)

        with t2:
            pie_col, stats_col = st.columns([1,1])
            with pie_col:
                st.plotly_chart(make_distribution_pie(res, threshold),
                                use_container_width=True)
            with stats_col:
                st.markdown("**Statistiques descriptives**")
                st.dataframe(
                    res["Score"].describe().rename("Score (%)").apply(lambda x: f"{x:.2%}"),
                    use_container_width=True,
                )
                st.markdown("**Top 5 machines à risque**")
                st.dataframe(
                    res[["Machine","Type","Score","Niveau"]].head(5),
                    use_container_width=True, hide_index=True,
                )

        with t3:
            st.plotly_chart(make_type_risk_heatmap(res), use_container_width=True)

        with t4:
            # Filtres interactifs
            fc1, fc2 = st.columns(2)
            filter_level = fc1.multiselect(
                "Filtrer par niveau",
                res["Niveau"].unique().tolist(),
                default=res["Niveau"].unique().tolist(),
            )
            filter_type = fc2.multiselect(
                "Filtrer par type",
                res["Type"].unique().tolist(),
                default=res["Type"].unique().tolist(),
            )
            df_filtered = res[
                res["Niveau"].isin(filter_level) & res["Type"].isin(filter_type)
            ]

            st.dataframe(
                df_filtered.style
                .background_gradient(subset=["Score"], cmap="RdYlGn_r", vmin=0, vmax=1)
                .format({"Score": "{:.2%}"}),
                use_container_width=True,
                hide_index=True,
                height=400,
            )

        # ── Export CSV
        st.download_button(
            "⬇️ Télécharger le rapport CSV",
            res.to_csv(index=False).encode("utf-8"),
            f"ocp_risk_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
            use_container_width=True,
        )

# ══════════════════════════════════════════════════════════════
# PAGE 3 — TABLEAU DE BORD
# ══════════════════════════════════════════════════════════════
elif page == "📈 Tableau de Bord":

    st.markdown("<div class='section-title'>📈 Tableau de Bord Opérationnel</div>",
                unsafe_allow_html=True)
    st.info("💡 Ce tableau de bord présente des données de démonstration simulées.")

    np.random.seed(0)
    n = 80
    mt = list(MACHINE_TYPE_MAP.keys())

    # Génération fleet démo
    demo_batch = [{
        "Machine_ID":                f"MC_OCP_{i:04d}",
        "Machine_Type":              np.random.choice(mt),
        "Site":                      np.random.choice(SITES_OCP),
        "Operational_Hours":         int(np.random.randint(5_000, 100_000)),
        "Temperature_C":             float(np.random.uniform(35, 110)),
        "Vibration_mms":             float(np.random.uniform(1, 30)),
        "Sound_dB":                  float(np.random.uniform(55, 100)),
        "Oil_Level_pct":             float(np.random.uniform(5, 100)),
        "Coolant_Level_pct":         float(np.random.uniform(10, 100)),
        "Power_Consumption_kW":      float(np.random.uniform(30, 250)),
        "Last_Maintenance_Days_Ago": int(np.random.randint(0, 400)),
        "Maintenance_History_Count": int(np.random.randint(1, 10)),
        "Failure_History_Count":     int(np.random.randint(0, 8)),
        "AI_Supervision":            bool(np.random.choice([True, False])),
        "Error_Codes_Last_30_Days":  int(np.random.randint(0, 15)),
        "AI_Override_Events":        int(np.random.randint(0, 5)),
        "Installation_Year":         int(np.random.randint(2010, 2038)),
    } for i in range(n)]

    demo_rows = []
    for rec in demo_batch:
        proba, _ = get_score(rec, model, preproc)
        lvl, clr, _, act, sev = classify_risk(proba, threshold)
        demo_rows.append({
            "Machine":  rec["Machine_ID"],
            "Type":     rec["Machine_Type"],
            "Site":     rec["Site"],
            "Score":    round(proba, 4),
            "Niveau":   lvl,
            "Sévérité": sev,
            "Temp":     rec["Temperature_C"],
            "Vib":      rec["Vibration_mms"],
        })
    df_demo = pd.DataFrame(demo_rows)

    # ── KPIs globaux
    kc = st.columns(4)
    kc[0].metric("🏭 Machines surveillées", n, delta="+3 ce mois")
    kc[1].metric("🔴 Alertes critiques",
                 int((df_demo["Score"] >= 0.80).sum()),
                 delta=f"{(df_demo['Score']>=0.80).mean():.0%} du parc")
    kc[2].metric("📊 Score moyen", f"{df_demo['Score'].mean():.1%}")
    kc[3].metric("🤖 Disponibilité ML", "100%" if ml_active else "—")

    st.divider()

    # ── Graphiques dashboard
    row1_c1, row1_c2 = st.columns([3, 2])

    with row1_c1:
        # Score par site
        site_avg = df_demo.groupby("Site")["Score"].mean().sort_values(ascending=False)
        fig_site = go.Figure(go.Bar(
            x=site_avg.values,
            y=site_avg.index,
            orientation='h',
            marker=dict(
                color=site_avg.values,
                colorscale=[[0,OCP_GREEN],[0.55,OCP_YELLOW],[0.80,OCP_RED],[1,'#B71C1C']],
                cmin=0, cmax=1,
                cornerradius=4,
            ),
            text=[f"{v:.0%}" for v in site_avg.values],
            textposition='outside',
        ))
        fig_site.update_layout(
            title="Score moyen par site",
            xaxis=dict(range=[0,1.15], tickformat=".0%", gridcolor="#eee"),
            height=260, margin=dict(l=100,r=60,t=40,b=30),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_site, use_container_width=True)

    with row1_c2:
        st.plotly_chart(make_distribution_pie(df_demo, threshold),
                        use_container_width=True)

    row2_c1, row2_c2 = st.columns(2)

    with row2_c1:
        # Scatter Temp vs Vib coloré par score
        fig_scat = px.scatter(
            df_demo, x="Temp", y="Vib",
            color="Score", color_continuous_scale=["#006633","#FFC107","#D32F2F"],
            size=[10]*len(df_demo),
            hover_data=["Machine","Type","Site"],
            labels={"Temp":"Température (°C)","Vib":"Vibration (mm/s)"},
            title="Température vs Vibration (couleur = score risque)",
            range_color=[0,1],
        )
        fig_scat.update_layout(
            height=300, margin=dict(l=10,r=10,t=40,b=30),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_scat, use_container_width=True)

    with row2_c2:
        st.plotly_chart(make_type_risk_heatmap(df_demo), use_container_width=True)

    # ── Top alertes
    st.markdown("<div class='section-title'>🚨 Top 10 — Machines prioritaires</div>",
                unsafe_allow_html=True)
    top10 = df_demo.nlargest(10, "Score")[["Machine","Type","Site","Score","Niveau"]]
    st.dataframe(
        top10.style
        .background_gradient(subset=["Score"], cmap="RdYlGn_r", vmin=0, vmax=1)
        .format({"Score": "{:.2%}"}),
        use_container_width=True, hide_index=True,
    )

# ══════════════════════════════════════════════════════════════
# PAGE 4 — À PROPOS
# ══════════════════════════════════════════════════════════════
elif page == "ℹ️ À propos":

    st.markdown("<div class='section-title'>ℹ️ À propos du système</div>",
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 🎯 Objectif")
        st.markdown("""
Détecter les pannes machines **7 jours à l'avance** dans les sites de
production OCP afin de :
- Réduire les arrêts non planifiés
- Optimiser les interventions de maintenance
- Prévenir les accidents industriels
- Maximiser la disponibilité du parc machines
        """)

        st.markdown("### 📈 Performance du modèle")
        perf_data = {
            "Métrique":["ROC-AUC","F1-Score","Avg. Precision","Seuil optimal","Nb features","Train samples"],
            "Valeur":[
                f"{meta['roc_auc_test']:.4f}",
                f"{meta['f1_test']:.3f}",
                f"{meta['avg_prec_test']:.4f}",
                f"{threshold:.4f}",
                str(meta['n_features']),
                f"{meta['train_samples']:,}",
            ],
        }
        st.dataframe(pd.DataFrame(perf_data), hide_index=True, use_container_width=True)

        st.markdown("### 🚦 Niveaux de risque")
        levels = [
            ("🔴 CRITIQUE", "≥ 80%",       "Arrêt immédiat",         OCP_RED),
            ("🟠 ÉLEVÉ",    "55 – 80%",    "Maintenance < 48h",      OCP_ORANGE),
            ("🟡 MODÉRÉ",   "seuil – 55%", "Surveillance renforcée", OCP_YELLOW),
            ("🟢 FAIBLE",   "< seuil",     "Normal",                 OCP_GREEN),
        ]
        for lvl, sco, act, clr in levels:
            st.markdown(
                f"<div style='background:{clr}15;border-left:4px solid {clr};"
                f"border-radius:6px;padding:8px 12px;margin:5px 0;'>"
                f"<b>{lvl}</b> &nbsp;|&nbsp; Score : {sco} &nbsp;|&nbsp; {act}</div>",
                unsafe_allow_html=True,
            )

    with c2:
        st.markdown("### 🧬 Features engineering")
        features_info = {
            "Feature":["Thermal_Stress","Maintenance_Urgency","Fluid_Degradation",
                       "Failure_Density","Error_Rate","High_Vibration_Flag",
                       "Overheat_Flag","Late_Maintenance_Flag"],
            "Formule":[
                "Temp × Vibration / 100",
                "Jours / (Nb_maint + 1)",
                "((100-Huile) + (100-Coolant)) / 2",
                "Pannes / (Heures/1000 + 1)",
                "Erreurs / 30",
                "Vib > 15 mm/s",
                "Temp > 80°C",
                "Jours > 180",
            ],
        }
        st.dataframe(pd.DataFrame(features_info), hide_index=True, use_container_width=True)

        st.markdown("### 🏭 Sites couverts")
        for site in SITES_OCP:
            st.markdown(f"📍 **{site}**")

        st.markdown("### 🛠️ Stack technique")
        tech = ["Python 3.11+","Scikit-learn","Streamlit","Plotly",
                "Pandas / NumPy","Pickle (sérialisation modèle)"]
        for t in tech:
            st.markdown(f"• {t}")

        st.markdown("### 📞 Contact & Support")
        st.markdown("""
> **OCP Group — Direction Numérique & IA**
> Pôle Maintenance Prédictive
> 📧 maintenance-ia@ocpgroup.ma
        """)

    st.divider()
    st.caption(
        f"Version 2.0 · Dernière mise à jour : {datetime.now().strftime('%d/%m/%Y')} · "
        f"© {datetime.now().year} OCP Group · Tous droits réservés"
    )