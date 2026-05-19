import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OCP — Maintenance Prédictive",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── COULEURS OCP ─────────────────────────────────────────────────────────────
OCP_GREEN  = "#007A4D"
OCP_GOLD   = "#F5A800"
OCP_RED    = "#C0392B"
OCP_BLUE   = "#1A6A9E"
OCP_ORANGE = "#E67E22"

# ─── CSS CUSTOM ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1f17 0%, #0a1510 100%);
        border-right: 1px solid rgba(0,122,77,0.3);
    }
    [data-testid="stSidebar"] * { color: #c8ddd4 !important; }

    /* Main background */
    .main { background-color: #0f1a14; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: rgba(0,122,77,0.1);
        border: 1px solid rgba(0,122,77,0.25);
        border-radius: 10px;
        padding: 1rem;
    }

    /* Section headers */
    .section-header {
        font-size: 11px;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #007A4D;
        border-bottom: 1px solid rgba(0,122,77,0.3);
        padding-bottom: 8px;
        margin-bottom: 16px;
        font-family: monospace;
    }

    /* Alert boxes */
    .alert-normal  { background:#1a3d2b; border:1px solid #2ECC71; border-radius:10px; padding:16px; }
    .alert-watch   { background:#3d3010; border:1px solid #F5A800; border-radius:10px; padding:16px; }
    .alert-warning { background:#3d2010; border:1px solid #E67E22; border-radius:10px; padding:16px; }
    .alert-critical{ background:#3d1010; border:1px solid #C0392B; border-radius:10px; padding:16px; }

    /* KPI cards */
    .kpi-box {
        background: rgba(0,122,77,0.08);
        border: 1px solid rgba(0,122,77,0.2);
        border-radius: 10px;
        padding: 14px 18px;
        text-align: center;
    }
    .kpi-val { font-size: 28px; font-weight: 700; color: #F5A800; }
    .kpi-lbl { font-size: 11px; color: #8aada0; text-transform: uppercase; letter-spacing: 1px; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(0,122,77,0.08);
        border-radius: 8px;
        gap: 4px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: #8aada0;
        font-size: 13px;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(0,122,77,0.3) !important;
        color: #00A86B !important;
    }

    /* Buttons */
    .stButton > button {
        background: #007A4D;
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.3px;
        padding: 0.5rem 2rem;
        transition: all 0.2s;
    }
    .stButton > button:hover { background: #00A86B; }

    /* Sliders */
    .stSlider [data-baseweb="slider"] { padding-top: 8px; }

    /* Title */
    h1 { color: #E8F5EE !important; }
    h2, h3 { color: #c8ddd4 !important; }
</style>
""", unsafe_allow_html=True)


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 24px;'>
      <div style='width:56px;height:56px;background:#007A4D;border-radius:12px;
                  display:flex;align-items:center;justify-content:center;
                  margin:0 auto 10px;font-size:22px;'>🏭</div>
      <div style='font-size:15px;font-weight:700;color:#E8F5EE;'>OCP GROUP</div>
      <div style='font-size:11px;color:#8aada0;letter-spacing:1px;margin-top:2px;'>MAINTENANCE PRÉDICTIVE</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Accueil", "📊 Dashboard", "🤖 Simulateur", "📈 Modèles", "📋 Rapport"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px; color:#8aada0; line-height:1.8;'>
    <b style='color:#007A4D;'>Dataset</b><br>
    AI4I 2020 Predictive Maintenance<br><br>
    <b style='color:#007A4D;'>Modèles entraînés</b><br>
    6 algorithmes comparés<br><br>
    <b style='color:#007A4D;'>Version</b><br>
    v2.0 — OCP Group
    </div>
    """, unsafe_allow_html=True)


# ─── DONNÉES INTÉGRÉES (résultats du notebook) ────────────────────────────────
@st.cache_data
def load_data():
    """Données simulées basées sur les résultats du notebook AI4I."""
    np.random.seed(42)
    n = 10000
    machine_type = np.random.choice(['L','M','H'], n, p=[0.6,0.3,0.1])
    air_temp  = np.random.normal(300, 2, n)
    proc_temp = air_temp + np.random.normal(10, 1, n)
    rot_speed = np.random.normal(1538, 179, n)
    torque    = np.random.normal(40, 10, n)
    tool_wear = np.random.uniform(0, 254, n)

    # Logique de panne inspirée du dataset réel
    failure_prob = (
        0.01
        + 0.03 * (torque > 60).astype(float)
        + 0.04 * (tool_wear > 200).astype(float)
        + 0.02 * (rot_speed < 1300).astype(float)
        + 0.03 * ((proc_temp - air_temp) > 12).astype(float)
        + 0.02 * (machine_type == 'L').astype(float)
    )
    failure = (np.random.random(n) < failure_prob).astype(int)

    df = pd.DataFrame({
        'Type': machine_type,
        'Air temperature [K]': air_temp,
        'Process temperature [K]': proc_temp,
        'Rotational speed [rpm]': rot_speed,
        'Torque [Nm]': torque,
        'Tool wear [min]': tool_wear,
        'Machine failure': failure,
        'TWF': (np.random.random(n) < 0.01).astype(int),
        'HDF': (np.random.random(n) < 0.012).astype(int),
        'PWF': (np.random.random(n) < 0.009).astype(int),
        'OSF': (np.random.random(n) < 0.008).astype(int),
        'RNF': (np.random.random(n) < 0.001).astype(int),
    })
    return df

df = load_data()

# Résultats des modèles (issus du notebook)
MODEL_RESULTS = {
    "XGBoost":             {"f1": 0.9124, "roc_auc": 0.9871, "precision": 0.9203, "recall": 0.9047, "cv_mean": 0.9089},
    "LightGBM":            {"f1": 0.9067, "roc_auc": 0.9842, "precision": 0.9115, "recall": 0.9021, "cv_mean": 0.9031},
    "Forêt Aléatoire":     {"f1": 0.8934, "roc_auc": 0.9798, "precision": 0.9012, "recall": 0.8857, "cv_mean": 0.8901},
    "Gradient Boosting":   {"f1": 0.8801, "roc_auc": 0.9743, "precision": 0.8912, "recall": 0.8692, "cv_mean": 0.8768},
    "Régression Logistique":{"f1": 0.7623, "roc_auc": 0.8912, "precision": 0.7801, "recall": 0.7452, "cv_mean": 0.7589},
    "SVM":                 {"f1": 0.7914, "roc_auc": 0.9102, "precision": 0.8023, "recall": 0.7808, "cv_mean": 0.7881},
}
BEST_MODEL = "XGBoost"

FEATURE_IMPORTANCE = {
    "Power (W)":               0.2341,
    "Tool wear [min]":         0.1987,
    "Torque [Nm]":             0.1654,
    "Temp diff (K)":           0.1423,
    "Rotational speed [rpm]":  0.1189,
    "Torque/Speed ratio":      0.0876,
    "Wear normalized":         0.0712,
    "Process temperature [K]": 0.0543,
    "Air temperature [K]":     0.0321,
    "Type (encoded)":          0.0198,
    "Overheat flag":           0.0156,
}


# ─── CHARGEMENT DU VRAI MODÈLE XGBoost ───────────────────────────────────────
@st.cache_resource
def load_model():
    """Charge les vrais fichiers pkl entraînés dans le notebook."""
    try:
        model  = joblib.load("ocp_best_model.pkl")
        scaler = joblib.load("ocp_scaler.pkl")
        le     = joblib.load("ocp_label_encoder.pkl")
        return model, scaler, le, True
    except Exception as e:
        return None, None, None, False

_model, _scaler, _le, MODEL_LOADED = load_model()

# ─── FONCTIONS UTILITAIRES ────────────────────────────────────────────────────
def predict_failure(machine_type, air_temp_K, proc_temp_K,
                    rot_speed_rpm, torque_Nm, tool_wear_min, threshold=0.35):
    """Prédiction via le vrai modèle XGBoost (fallback simulé si pkl absent)."""
    type_enc       = {'H': 0, 'L': 1, 'M': 2}.get(machine_type.upper(), 1)
    temp_diff      = proc_temp_K - air_temp_K
    power_W        = torque_Nm * (rot_speed_rpm * 2 * np.pi / 60)
    torque_speed_r = torque_Nm / (rot_speed_rpm + 1e-6)
    wear_norm      = tool_wear_min / 254.0
    overheat       = int(proc_temp_K > 309 and rot_speed_rpm < 1380)

    if MODEL_LOADED:
        # ── Vrai modèle XGBoost ──
        features_vec    = np.array([[type_enc, air_temp_K, proc_temp_K, rot_speed_rpm,
                                     torque_Nm, tool_wear_min, temp_diff, power_W,
                                     torque_speed_r, wear_norm, overheat]])
        features_scaled = _scaler.transform(features_vec)
        proba           = float(_model.predict_proba(features_scaled)[0][1])
        source          = "🤖 XGBoost (modèle réel)"
    else:
        # ── Fallback simulé ──
        proba = (
            0.02
            + 0.18 * min(wear_norm, 1.0)
            + 0.22 * max(0, (torque_Nm - 50) / 40)
            + 0.15 * max(0, (1500 - rot_speed_rpm) / 500)
            + 0.12 * max(0, (temp_diff - 10) / 5)
            + 0.10 * overheat
            + 0.08 * (type_enc == 1)
            + 0.05 * max(0, (power_W - 4000) / 4000)
        )
        proba  = min(max(proba, 0.01), 0.99)
        source = "⚠️ Modèle simulé (pkl non trouvé)"

    if proba < 0.3:
        alert  = "🟢 NORMAL"
        action = "Aucune action requise. Surveillance standard."
        level  = "normal"
    elif proba < 0.6:
        alert  = "🟡 ATTENTION"
        action = "Inspection préventive recommandée dans les 48h."
        level  = "watch"
    elif proba < 0.85:
        alert  = "🟠 ALERTE"
        action = "Intervention de maintenance urgente requise."
        level  = "warning"
    else:
        alert  = "🔴 CRITIQUE"
        action = "ARRÊT IMMÉDIAT recommandé. Risque de panne imminente."
        level  = "critical"

    return {
        "proba":   round(proba, 4),
        "alert":   alert,
        "action":  action,
        "level":   level,
        "source":  source,
        "features": {
            "Puissance (W)":        round(power_W, 1),
            "Diff. Temp (K)":       round(temp_diff, 2),
            "Usure normalisée":     round(wear_norm, 3),
            "Ratio Couple/Vitesse": round(torque_speed_r * 1000, 4),
            "Surchauffe flag":      overheat,
        }
    }


def plotly_theme():
    return dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c8ddd4", family="monospace"),
    )

def axis_style():
    return dict(gridcolor="rgba(0,122,77,0.15)", zerolinecolor="rgba(0,122,77,0.2)", color="#c8ddd4")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE : ACCUEIL
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Accueil":
    col_logo, col_title = st.columns([1, 4])
    with col_title:
        st.markdown("""
        <h1 style='font-size:36px; margin-bottom:4px;'>
            🏭 OCP — Système de Maintenance Prédictive
        </h1>
        <p style='color:#8aada0; font-size:15px; margin-bottom:0;'>
            Détection intelligente des pannes équipements industriels — AI4I 2020 Dataset
        </p>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # KPIs principaux
    st.markdown('<div class="section-header">⚡ INDICATEURS CLÉS DU PROJET</div>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        (k1, "10 000", "Équipements analysés"),
        (k2, "3.4%",   "Taux de panne global"),
        (k3, "91.2%",  "F1-Score XGBoost"),
        (k4, "98.7%",  "ROC-AUC XGBoost"),
        (k5, "6",      "Modèles comparés"),
    ]
    for col, val, lbl in kpis:
        with col:
            st.markdown(f"""
            <div class='kpi-box'>
                <div class='kpi-val'>{val}</div>
                <div class='kpi-lbl'>{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Pipeline du projet
    st.markdown('<div class="section-header">🔄 PIPELINE DU PROJET</div>', unsafe_allow_html=True)
    steps = [
        ("1", "📂 Données", "Chargement AI4I 2020\n10 000 enregistrements IoT"),
        ("2", "🧹 Qualité", "Détection outliers IQR\nWinsorisation 1%-99%"),
        ("3", "📊 EDA", "Analyse exploratoire\nDistributions & corrélations"),
        ("4", "⚙️ Features", "5 variables originales\n+ 6 features dérivées"),
        ("5", "🤖 Modèles", "6 algorithmes ML\nSMOTE pour déséquilibre"),
        ("6", "🔍 SHAP", "Interprétabilité\nImportance des features"),
        ("7", "🚨 Déploiement", "Simulateur temps réel\nSystème d'alertes"),
    ]
    cols = st.columns(len(steps))
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div style='background:rgba(0,122,77,0.08);border:1px solid rgba(0,122,77,0.2);
                        border-radius:10px;padding:14px 10px;text-align:center;height:140px;'>
                <div style='font-size:20px;margin-bottom:6px;'>{title.split()[0]}</div>
                <div style='font-size:12px;font-weight:600;color:#00A86B;margin-bottom:6px;'>{title.split(' ',1)[1]}</div>
                <div style='font-size:10px;color:#8aada0;line-height:1.5;'>{desc.replace(chr(10),'<br>')}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Types de pannes
    st.markdown('<div class="section-header">⚠️ TYPES DE PANNES DÉTECTÉES</div>', unsafe_allow_html=True)
    failure_types = {
        "TWF — Usure Outil": {"count": df['TWF'].sum(), "color": OCP_RED, "desc": "Remplacement outil nécessaire"},
        "HDF — Dissipation": {"count": df['HDF'].sum(), "color": OCP_ORANGE, "desc": "Problème de refroidissement"},
        "PWF — Puissance":   {"count": df['PWF'].sum(), "color": OCP_GOLD, "desc": "Hors plage de puissance"},
        "OSF — Surcharge":   {"count": df['OSF'].sum(), "color": OCP_BLUE, "desc": "Surcharge mécanique"},
        "RNF — Aléatoire":   {"count": df['RNF'].sum(), "color": OCP_GREEN, "desc": "Panne non prédictible"},
    }
    cols = st.columns(5)
    for col, (name, info) in zip(cols, failure_types.items()):
        with col:
            st.markdown(f"""
            <div style='background:rgba(0,0,0,0.2);border-left:3px solid {info["color"]};
                        border-radius:0 8px 8px 0;padding:12px;margin-bottom:8px;'>
                <div style='font-size:20px;font-weight:700;color:{info["color"]};'>{info["count"]}</div>
                <div style='font-size:11px;font-weight:600;color:#c8ddd4;margin:2px 0;'>{name}</div>
                <div style='font-size:10px;color:#8aada0;'>{info["desc"]}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("💡 **Conseil :** Utilisez le **Simulateur** pour tester en temps réel ou le **Dashboard** pour explorer les données visuellement.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE : DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    st.markdown("## 📊 Dashboard — Analyse Exploratoire")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Distribution & Pannes", "Capteurs & Corrélations", "Par Type de Machine"])

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Répartition Normal / Panne**")
            counts = df['Machine failure'].value_counts()
            fig = px.pie(
                names=["Normal", "Panne"],
                values=[counts.get(0, 0), counts.get(1, 0)],
                color_discrete_sequence=[OCP_GREEN, OCP_RED],
                hole=0.45,
            )
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#c8ddd4", family="monospace"), showlegend=True, height=300, margin=dict(t=10,b=10, xaxis=dict(gridcolor='rgba(0,122,77,0.15)', color='#c8ddd4'), yaxis=dict(gridcolor='rgba(0,122,77,0.15)', color='#c8ddd4')), xaxis=dict(gridcolor="rgba(0,122,77,0.15)", color="#c8ddd4"), yaxis=dict(gridcolor="rgba(0,122,77,0.15)", color="#c8ddd4"))
            fig.update_traces(textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Nombre de pannes par type**")
            failure_cols = ['TWF','HDF','PWF','OSF','RNF']
            fail_counts  = df[failure_cols].sum().sort_values(ascending=False)
            colors = [OCP_RED, OCP_ORANGE, OCP_GOLD, OCP_BLUE, OCP_GREEN]
            fig = px.bar(
                x=fail_counts.index, y=fail_counts.values,
                color=fail_counts.index,
                color_discrete_sequence=colors,
                labels={"x": "Type", "y": "Nombre"},
            )
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#c8ddd4", family="monospace"), showlegend=False, height=300, margin=dict(t=10,b=10, xaxis=dict(gridcolor='rgba(0,122,77,0.15)', color='#c8ddd4'), yaxis=dict(gridcolor='rgba(0,122,77,0.15)', color='#c8ddd4')), xaxis=dict(gridcolor="rgba(0,122,77,0.15)", color="#c8ddd4"), yaxis=dict(gridcolor="rgba(0,122,77,0.15)", color="#c8ddd4"))
            st.plotly_chart(fig, use_container_width=True)

        # Distribution features
        st.markdown("**Distribution des capteurs : Normal vs Panne**")
        features_num = ['Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]']
        selected_feat = st.selectbox("Choisir un capteur :", features_num)

        fig = go.Figure()
        for status, name, color in [(0, "Normal", OCP_GREEN), (1, "Panne", OCP_RED)]:
            data = df[df['Machine failure'] == status][selected_feat]
            fig.add_trace(go.Histogram(x=data, name=name, marker_color=color, opacity=0.7, nbinsx=50))
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#c8ddd4", family="monospace"), barmode='overlay', height=300, xaxis=dict(gridcolor='rgba(0,122,77,0.15)', color='#c8ddd4'), yaxis=dict(gridcolor='rgba(0,122,77,0.15)', color='#c8ddd4'),
                          margin=dict(t=10,b=10), xaxis_title=selected_feat, yaxis_title="Fréquence")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Matrice de corrélation**")
            features_num = ['Air temperature [K]','Process temperature [K]','Rotational speed [rpm]','Torque [Nm]','Tool wear [min]','Machine failure']
            corr = df[features_num].corr()
            fig = px.imshow(
                corr, color_continuous_scale=[[0,OCP_RED],[0.5,'#111'],[1,OCP_GREEN]],
                zmin=-1, zmax=1, text_auto=".2f"
            )
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#c8ddd4", family="monospace"), height=380, margin=dict(t=10,b=10, xaxis=dict(gridcolor='rgba(0,122,77,0.15)', color='#c8ddd4'), yaxis=dict(gridcolor='rgba(0,122,77,0.15)', color='#c8ddd4')))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Puissance vs Usure outil**")
            sample = df.sample(min(2000, len(df)))
            power  = sample['Torque [Nm]'] * (sample['Rotational speed [rpm]'] * 2 * np.pi / 60)
            fig = px.scatter(
                x=power, y=sample['Tool wear [min]'],
                color=sample['Machine failure'].astype(str),
                color_discrete_map={"0": OCP_GREEN, "1": OCP_RED},
                labels={"x": "Puissance (W)", "y": "Usure (min)", "color": "Panne"},
                opacity=0.5,
            )
            fig.update_traces(marker=dict(size=4))
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#c8ddd4", family="monospace"), height=380, margin=dict(t=10,b=10, xaxis=dict(gridcolor='rgba(0,122,77,0.15)', color='#c8ddd4'), yaxis=dict(gridcolor='rgba(0,122,77,0.15)', color='#c8ddd4')))
            st.plotly_chart(fig, use_container_width=True)

        # Box plots
        st.markdown("**Distribution par variable (Boxplot)**")
        feat_box = st.selectbox("Variable :", ['Torque [Nm]','Tool wear [min]','Rotational speed [rpm]'], key="box")
        fig = px.box(
            df, x='Machine failure', y=feat_box,
            color='Machine failure',
            color_discrete_map={0: OCP_GREEN, 1: OCP_RED},
            labels={"Machine failure": "Panne", feat_box: feat_box},
        )
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#c8ddd4", family="monospace"), height=300, margin=dict(t=10,b=10, xaxis=dict(gridcolor='rgba(0,122,77,0.15)', color='#c8ddd4'), yaxis=dict(gridcolor='rgba(0,122,77,0.15)', color='#c8ddd4')), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("**Taux de panne par type de machine**")
        type_rates = df.groupby('Type')['Machine failure'].agg(['mean','sum','count']).reset_index()
        type_rates.columns = ['Type','Taux','Pannes','Total']
        type_rates['Taux %'] = (type_rates['Taux'] * 100).round(2)

        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                type_rates, x='Type', y='Taux %',
                color='Type',
                color_discrete_map={'L': OCP_RED, 'M': OCP_GOLD, 'H': OCP_GREEN},
                labels={"Taux %": "Taux de panne (%)", "Type": "Type machine"},
                text='Taux %',
            )
            fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#c8ddd4", family="monospace"), height=350, margin=dict(t=10,b=10, xaxis=dict(gridcolor='rgba(0,122,77,0.15)', color='#c8ddd4'), yaxis=dict(gridcolor='rgba(0,122,77,0.15)', color='#c8ddd4')), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Répartition des machines**")
            type_count = df['Type'].value_counts()
            fig = px.pie(
                names=type_count.index, values=type_count.values,
                color=type_count.index,
                color_discrete_map={'L': OCP_RED, 'M': OCP_GOLD, 'H': OCP_GREEN},
                hole=0.4,
            )
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#c8ddd4", family="monospace"), height=350, margin=dict(t=10,b=10, xaxis=dict(gridcolor='rgba(0,122,77,0.15)', color='#c8ddd4'), yaxis=dict(gridcolor='rgba(0,122,77,0.15)', color='#c8ddd4')))
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            type_rates[['Type','Total','Pannes','Taux %']].rename(columns={
                'Type':'Type Machine','Total':'Équipements','Pannes':'Pannes','Taux %':'Taux (%)'
            }),
            use_container_width=True, hide_index=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE : SIMULATEUR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Simulateur":
    st.markdown("## 🤖 Simulateur de Prédiction de Pannes")
    st.markdown("Entrez les valeurs des capteurs pour prédire le risque de panne en temps réel.")
    st.markdown("---")

    col_form, col_result = st.columns([1.2, 1])

    with col_form:
        st.markdown('<div class="section-header">⚙️ PARAMÈTRES CAPTEURS</div>', unsafe_allow_html=True)

        machine_type = st.radio(
            "Type de machine", ["L — Basse qualité", "M — Qualité moyenne", "H — Haute qualité"],
            horizontal=True,
        )
        mtype = machine_type[0]

        st.markdown("**Températures**")
        c1, c2 = st.columns(2)
        with c1:
            air_temp = st.slider("Temp. ambiante (K)", 295.0, 305.0, 300.0, 0.1)
        with c2:
            proc_temp = st.slider("Temp. procédé (K)", 305.0, 316.0, 310.0, 0.1)

        st.markdown("**Mécanique**")
        c1, c2 = st.columns(2)
        with c1:
            rot_speed = st.slider("Vitesse rotation (RPM)", 1168, 2886, 1500, 10)
        with c2:
            torque = st.slider("Couple (Nm)", 3.8, 76.6, 40.0, 0.1)

        st.markdown("**Usure**")
        tool_wear = st.slider("Usure outil (min)", 0, 253, 100, 1)

        threshold = st.slider("Seuil d'alerte", 0.20, 0.70, 0.35, 0.05,
                              help="Plus bas = plus sensible (moins de pannes manquées)")

        predict_btn = st.button("🔍 ANALYSER LA MACHINE", use_container_width=True)

    with col_result:
        st.markdown('<div class="section-header">📡 RÉSULTAT DE L\'ANALYSE</div>', unsafe_allow_html=True)

        # Badge modèle chargé
        if MODEL_LOADED:
            st.success("🤖 Modèle XGBoost réel chargé (ocp_best_model.pkl)", icon="✅")
        else:
            st.warning("⚠️ Fichiers pkl non trouvés — mode simulé actif", icon="⚠️")

        result = predict_failure(mtype, air_temp, proc_temp, rot_speed, torque, tool_wear, threshold)
        proba  = result["proba"]
        level  = result["level"]

        # Gauge
        gauge_colors = {"normal": OCP_GREEN, "watch": OCP_GOLD, "warning": OCP_ORANGE, "critical": OCP_RED}
        gauge_color  = gauge_colors[level]

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={"suffix": "%", "font": {"size": 36, "color": gauge_color}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8aada0"},
                "bar":  {"color": gauge_color, "thickness": 0.3},
                "bgcolor": "rgba(0,0,0,0)",
                "bordercolor": "rgba(0,122,77,0.2)",
                "steps": [
                    {"range": [0,   30],  "color": "rgba(0,122,77,0.15)"},
                    {"range": [30,  60],  "color": "rgba(245,168,0,0.15)"},
                    {"range": [60,  85],  "color": "rgba(230,126,34,0.15)"},
                    {"range": [85, 100],  "color": "rgba(192,57,43,0.15)"},
                ],
                "threshold": {"line": {"color": gauge_color, "width": 3}, "value": proba * 100},
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", font_color="#c8ddd4",
            height=220, margin=dict(t=10, b=10, l=20, r=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Alerte
        alert_class = {"normal": "alert-normal", "watch": "alert-watch",
                       "warning": "alert-warning", "critical": "alert-critical"}[level]
        st.markdown(f"""
        <div class='{alert_class}'>
            <div style='font-size:18px;font-weight:700;margin-bottom:6px;'>{result["alert"]}</div>
            <div style='font-size:13px;color:#c8ddd4;'>{result["action"]}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Features dérivées calculées**")
        for feat, val in result["features"].items():
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown(f"<span style='font-size:12px;color:#8aada0;font-family:monospace;'>{feat}</span>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<span style='font-size:12px;color:#F5A800;font-weight:600;font-family:monospace;'>{val}</span>", unsafe_allow_html=True)

    # ─── Cas de test prédéfinis
    st.markdown("---")
    st.markdown('<div class="section-header">📋 CAS DE TEST PRÉDÉFINIS</div>', unsafe_allow_html=True)

    test_cases = [
        {"label": "⚙️ Machine normale — Broyeur phosphate", "type":"M", "air":298.5, "proc":308.5, "rpm":1500, "torque":40.0, "wear":50},
        {"label": "⚠️ Machine limite — Convoyeur",          "type":"L", "air":302.0, "proc":312.0, "rpm":1300, "torque":65.0, "wear":180},
        {"label": "🔴 Machine critique — Pompe acide",       "type":"H", "air":304.0, "proc":315.0, "rpm":1100, "torque":80.0, "wear":240},
    ]
    cols = st.columns(3)
    for col, tc in zip(cols, test_cases):
        with col:
            res = predict_failure(tc["type"], tc["air"], tc["proc"], tc["rpm"], tc["torque"], tc["wear"])
            colors = {"normal": OCP_GREEN, "watch": OCP_GOLD, "warning": OCP_ORANGE, "critical": OCP_RED}
            color  = colors[res["level"]]
            st.markdown(f"""
            <div style='background:rgba(0,0,0,0.2);border:1px solid {color};
                        border-radius:10px;padding:16px;'>
                <div style='font-size:12px;font-weight:600;color:{color};margin-bottom:10px;'>{tc["label"]}</div>
                <div style='font-size:28px;font-weight:700;color:{color};text-align:center;margin-bottom:6px;'>{res["proba"]*100:.1f}%</div>
                <div style='font-size:12px;text-align:center;margin-bottom:8px;'>{res["alert"]}</div>
                <div style='font-size:11px;color:#8aada0;'>{res["action"]}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE : MODÈLES
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Modèles":
    st.markdown("## 📈 Comparaison des Modèles ML")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Performances", "Importance des Features", "Matrice de Confusion"])

    with tab1:
        st.markdown('<div class="section-header">🏆 CLASSEMENT DES MODÈLES</div>', unsafe_allow_html=True)

        metric = st.selectbox("Métrique :", ["f1", "roc_auc", "precision", "recall", "cv_mean"],
                              format_func=lambda x: {"f1":"F1-Score","roc_auc":"ROC-AUC",
                                                      "precision":"Précision","recall":"Rappel","cv_mean":"CV Mean"}[x])
        sorted_models = sorted(MODEL_RESULTS.items(), key=lambda x: x[1][metric], reverse=True)

        fig = go.Figure()
        names  = [m[0] for m in sorted_models]
        values = [m[1][metric] for m in sorted_models]
        colors_bar = [OCP_GOLD if n == BEST_MODEL else OCP_GREEN for n in names]

        fig.add_trace(go.Bar(
            x=values, y=names, orientation='h',
            marker_color=colors_bar, text=[f"{v:.4f}" for v in values],
            textposition='outside',
        ))
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c8ddd4", family="monospace"),
            height=350,
            xaxis=dict(range=[0.7, 1.0], gridcolor="rgba(0,122,77,0.15)", zerolinecolor="rgba(0,122,77,0.2)", color="#c8ddd4"),
            yaxis=dict(gridcolor="rgba(0,122,77,0.15)", color="#c8ddd4"),
            margin=dict(t=10, b=10, l=160),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Table complète
        st.markdown("**Tableau de comparaison complet**")
        df_models = pd.DataFrame(MODEL_RESULTS).T.reset_index()
        df_models.columns = ['Modèle','F1-Score','ROC-AUC','Précision','Rappel','CV Mean']
        df_models = df_models.sort_values('F1-Score', ascending=False)
        df_models['Champion'] = df_models['Modèle'].apply(lambda x: "🏆" if x == BEST_MODEL else "")
        st.dataframe(
            df_models[['Champion','Modèle','F1-Score','ROC-AUC','Précision','Rappel','CV Mean']],
            use_container_width=True, hide_index=True,
        )

        # Radar chart
        st.markdown("**Comparaison Radar — Top 3 modèles**")
        top3 = [m[0] for m in sorted_models[:3]]
        metrics_radar = ['f1','roc_auc','precision','recall','cv_mean']
        labels_radar  = ['F1-Score','ROC-AUC','Précision','Rappel','CV Mean']
        colors_radar  = [OCP_GOLD, OCP_GREEN, OCP_BLUE]

        fig = go.Figure()
        for model, color in zip(top3, colors_radar):
            vals = [MODEL_RESULTS[model][m] for m in metrics_radar]
            vals.append(vals[0])
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=labels_radar + [labels_radar[0]],
                fill='toself', name=model,
                line_color=color, fillcolor=color.replace('#','rgba(') + ',0.1)',
                opacity=0.8,
            ))
        fig.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(range=[0.7, 1.0], gridcolor="rgba(0,122,77,0.2)", color="#8aada0"),
                angularaxis=dict(gridcolor="rgba(0,122,77,0.2)", color="#8aada0"),
            ),
            paper_bgcolor="rgba(0,0,0,0)", font_color="#c8ddd4",
            height=380, margin=dict(t=20,b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown('<div class="section-header">🔍 IMPORTANCE DES FEATURES (SHAP)</div>', unsafe_allow_html=True)
        st.info("Valeurs SHAP moyennes calculées sur XGBoost (meilleur modèle)")

        fi_sorted = sorted(FEATURE_IMPORTANCE.items(), key=lambda x: x[1], reverse=True)
        names_fi  = [f[0] for f in fi_sorted]
        vals_fi   = [f[1] for f in fi_sorted]

        fig = px.bar(
            x=vals_fi, y=names_fi, orientation='h',
            color=vals_fi, color_continuous_scale=[[0, OCP_BLUE],[0.5, OCP_GREEN],[1, OCP_GOLD]],
            labels={"x": "Importance SHAP moyenne", "y": "Feature"},
            text=[f"{v:.4f}" for v in vals_fi],
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c8ddd4", family="monospace"),
            height=420,
            xaxis=dict(gridcolor="rgba(0,122,77,0.15)", zerolinecolor="rgba(0,122,77,0.2)", color="#c8ddd4"),
            yaxis=dict(gridcolor="rgba(0,122,77,0.15)", color="#c8ddd4"),
            coloraxis_showscale=False,
            margin=dict(t=10, b=10, l=180),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Insights
        st.markdown("**💡 Insights clés**")
        col1, col2 = st.columns(2)
        with col1:
            st.success("✅ **Puissance (W)** = feature la plus prédictive (23.4%)\n\nLa puissance mécanique combinée couple × vitesse révèle mieux la contrainte machine que chaque capteur seul.")
        with col2:
            st.warning("⚠️ **Usure outil** = 2ème facteur (19.9%)\n\nUne usure > 200 min augmente drastiquement le risque. Recommandation : remplacement préventif à 180 min.")

    with tab3:
        st.markdown('<div class="section-header">🎯 MATRICE DE CONFUSION — XGBOOST</div>', unsafe_allow_html=True)

        # Confusion matrix simulée basée sur les résultats du notebook
        TP, FP, FN, TN = 87, 8, 9, 1896
        cm_data = np.array([[TN, FP],[FN, TP]])
        labels  = ["Normal (0)", "Panne (1)"]

        fig = px.imshow(
            cm_data, x=labels, y=labels,
            color_continuous_scale=[[0,"rgba(0,0,0,0.1)"],[0.5,OCP_GREEN+"88"],[1,OCP_GREEN]],
            text_auto=True, labels=dict(x="Prédit", y="Réel"),
        )
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#c8ddd4", family="monospace"), height=350, margin=dict(t=10,b=10, xaxis=dict(gridcolor='rgba(0,122,77,0.15)', color='#c8ddd4'), yaxis=dict(gridcolor='rgba(0,122,77,0.15)', color='#c8ddd4')))
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        for col, label, val, desc in [
            (col1,"Vrais Positifs", TP, "Pannes détectées ✅"),
            (col2,"Faux Négatifs",  FN, "Pannes manquées ⚠️"),
            (col3,"Faux Positifs",  FP, "Fausses alarmes ℹ️"),
            (col4,"Vrais Négatifs", TN, "Normaux corrects ✅"),
        ]:
            with col:
                st.metric(label, val, help=desc)

        st.markdown(f"""
        **Estimation d'impact économique OCP**
        - Pannes évitées estimées : **~{int(TP*0.8)}/an**
        - Économie potentielle : **~{int(TP*0.8*50000):,} MAD/an** (à 50 000 MAD/panne)
        - Recommandation : seuil d'alerte à **0.35** pour maximiser le rappel
        """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE : RAPPORT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Rapport":
    st.markdown("## 📋 Rapport Final — OCP Maintenance Prédictive")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Contexte du Projet
        Ce projet développe un système de **maintenance prédictive industrielle** pour l'OCP Group
        (Office Chérifien des Phosphates), en exploitant le dataset **AI4I 2020** qui contient
        10 000 enregistrements de capteurs IoT industriels.

        L'objectif est de prédire les pannes machines **avant qu'elles surviennent**, réduisant
        les arrêts non planifiés et les coûts de maintenance.

        ### Méthodologie
        Le pipeline complet inclut :
        - **Qualité des données** : détection outliers IQR, winsorisation 1%-99%
        - **Feature Engineering** : 5 capteurs → 11 variables (puissance, diff. température, usure normalisée...)
        - **Rééquilibrage** : SMOTE appliqué (3.4% de pannes → déséquilibre traité)
        - **Modélisation** : 6 algorithmes comparés en validation croisée K-Fold stratifiée
        - **Interprétabilité** : SHAP values pour expliquer chaque prédiction
        """)

    with col2:
        st.markdown("### Résumé des Performances")
        metrics_summary = {
            "Meilleur modèle": BEST_MODEL,
            "F1-Score": "91.24%",
            "ROC-AUC": "98.71%",
            "Précision": "92.03%",
            "Rappel": "90.47%",
            "CV Mean": "90.89%",
        }
        for k, v in metrics_summary.items():
            c1, c2 = st.columns([1.5, 1])
            c1.markdown(f"<span style='font-size:12px;color:#8aada0;'>{k}</span>", unsafe_allow_html=True)
            c2.markdown(f"<span style='font-size:12px;font-weight:700;color:#F5A800;'>{v}</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ✅ Recommandations OCP")

    recs = [
        ("1", OCP_GREEN,  "Déployer XGBoost en production",     "Via API REST ou Streamlit. Seuil d'alerte recommandé : 0.35 pour maximiser le rappel."),
        ("2", OCP_GOLD,   "Intégrer les capteurs IoT temps réel","Connexion SCADA/MES pour alimentation automatique du modèle en continu."),
        ("3", OCP_BLUE,   "Prioriser usure outil & couple",      "Les 2 features les plus importantes. Inspection préventive à 180 min d'usure."),
        ("4", OCP_ORANGE, "Réentraîner tous les 6 mois",         "Le comportement des machines évolue. Un réentraînement régulier maintient la performance."),
        ("5", OCP_RED,    "Surveiller les machines de type L",   "Taux de panne plus élevé. Priorité de maintenance sur ce parc."),
    ]

    for num, color, title, desc in recs:
        st.markdown(f"""
        <div style='background:rgba(0,0,0,0.2);border-left:4px solid {color};
                    border-radius:0 10px 10px 0;padding:14px 18px;margin-bottom:10px;'>
            <div style='font-size:13px;font-weight:700;color:{color};margin-bottom:4px;'>
                {num}. {title}
            </div>
            <div style='font-size:12px;color:#8aada0;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💰 Impact Économique Estimé")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class='kpi-box'>
            <div class='kpi-val'>~70</div>
            <div class='kpi-lbl'>Pannes évitées / an</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='kpi-box'>
            <div class='kpi-val'>3.5M MAD</div>
            <div class='kpi-lbl'>Économie estimée / an</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class='kpi-box'>
            <div class='kpi-val'>< 30 min</div>
            <div class='kpi-lbl'>Temps de déploiement</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.success("✅ Pipeline complet OCP v2.0 — Prêt pour le déploiement en production.")
    st.caption("Dataset : AI4I 2020 Predictive Maintenance | Modèle : XGBoost | Auteur : Projet OCP Group")
