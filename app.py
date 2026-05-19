import streamlit as st
import numpy as np
import pandas as pd
import joblib, os, io, datetime, time, warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import shap
from sklearn.metrics import roc_curve, auc, confusion_matrix
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors as rl_colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import tempfile

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="OCP — Maintenance Prédictive", page_icon="🏭", layout="wide")

OCP_GREEN='#007A4D'; OCP_RED='#C0392B'; OCP_GOLD='#F5A800'; OCP_BLUE='#1A6A9E'; OCP_DARK='#1a1a2e'
FEATURES = ['Type_enc','Air temperature [K]','Process temperature [K]',
            'Rotational speed [rpm]','Torque [Nm]','Tool wear [min]',
            'Temp_diff [K]','Power [W]','Torque_Speed_ratio','Wear_norm','Overheat_flag']
FEAT_LABELS = ['Type','Air Temp','Proc Temp','RPM','Torque','Tool Wear',
               'ΔT','Power','Tq/Speed','Wear%','Overheat']
TOOL_WEAR_MAX = 253.0

# ══════════════════════════════════════════════════════════════════════════════
# CSS — Dark Mode professionnel
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

.stApp { background: #0f0f1a !important; color: #e8e8f0 !important; }
.block-container { padding-top: 1.2rem !important; }
section[data-testid="stSidebar"] { display:none; }

/* Header */
.ocp-header {
    background: linear-gradient(135deg, #007A4D 0%, #005c38 50%, #003d26 100%);
    border-radius: 16px; padding: 1.8rem 2.2rem; margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0,122,77,0.35);
    border: 1px solid rgba(245,168,0,0.3);
    position: relative; overflow: hidden;
}
.ocp-header::before {
    content:''; position:absolute; top:-50%; right:-10%; width:300px; height:300px;
    background: radial-gradient(circle, rgba(245,168,0,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.ocp-header h1 { margin:0; font-size:2rem; font-weight:800; color:white; letter-spacing:-.5px; }
.ocp-header .sub { margin:.4rem 0 0; color:rgba(255,255,255,.75); font-size:.88rem; }
.ocp-header .badge {
    display:inline-block; background:rgba(245,168,0,.2); border:1px solid rgba(245,168,0,.5);
    color:#F5A800; border-radius:20px; padding:.2rem .7rem; font-size:.75rem;
    font-weight:600; margin:.5rem .3rem 0 0;
}

/* Cards */
.card {
    background: #16162a; border-radius:12px; padding:1.2rem 1.4rem;
    border: 1px solid rgba(255,255,255,.07);
    box-shadow: 0 4px 20px rgba(0,0,0,.3);
    margin-bottom: .8rem;
}
.card-green { border-left: 4px solid #007A4D; }
.card-red   { border-left: 4px solid #C0392B; }
.card-gold  { border-left: 4px solid #F5A800; }
.card-blue  { border-left: 4px solid #1A6A9E; }

/* Section titles */
.section-title {
    font-size:.9rem; font-weight:700; color:#F5A800; letter-spacing:.5px;
    text-transform:uppercase; border-bottom:1px solid rgba(245,168,0,.3);
    padding-bottom:.35rem; margin-bottom:1rem;
}

/* Result boxes */
.result-ok {
    background: linear-gradient(135deg,rgba(0,122,77,.25),rgba(0,122,77,.1));
    border:2px solid #007A4D; border-radius:12px; padding:1.2rem 1.4rem;
    color:#4ecca3; font-size:1.15rem; font-weight:700; text-align:center;
    box-shadow: 0 0 20px rgba(0,122,77,.2);
}
.result-fail {
    background: linear-gradient(135deg,rgba(192,57,43,.25),rgba(192,57,43,.1));
    border:2px solid #C0392B; border-radius:12px; padding:1.2rem 1.4rem;
    color:#ff6b6b; font-size:1.15rem; font-weight:700; text-align:center;
    box-shadow: 0 0 20px rgba(192,57,43,.2);
}

/* Alert boxes */
.alert-crit {
    background:rgba(192,57,43,.15); border-left:4px solid #C0392B;
    border-radius:6px; padding:.7rem 1rem; margin:.4rem 0; font-size:.85rem; color:#ff8a80;
}
.alert-warn {
    background:rgba(245,168,0,.1); border-left:4px solid #F5A800;
    border-radius:6px; padding:.7rem 1rem; margin:.4rem 0; font-size:.85rem; color:#ffd54f;
}

/* KPI metrics */
.kpi-grid { display:flex; gap:.8rem; flex-wrap:wrap; margin-bottom:1rem; }
.kpi {
    background:#16162a; border-radius:10px; padding:.9rem 1.2rem; flex:1; min-width:120px;
    text-align:center; border:1px solid rgba(255,255,255,.08);
    box-shadow: 0 2px 10px rgba(0,0,0,.2);
}
.kpi .val { font-size:1.6rem; font-weight:800; color:#F5A800; }
.kpi .lbl { font-size:.73rem; color:#aaa; margin-top:.2rem; }

/* Streamlit overrides */
.stTabs [data-baseweb="tab-list"] { background:#16162a !important; border-radius:10px; gap:4px; }
.stTabs [data-baseweb="tab"] { color:#aaa !important; border-radius:8px !important; font-weight:600 !important; }
.stTabs [aria-selected="true"] { background:#007A4D !important; color:white !important; }
.stSlider > div > div > div > div { background:#007A4D !important; }
div[data-testid="metric-container"] {
    background:#16162a; border-radius:10px; padding:.6rem 1rem;
    border:1px solid rgba(255,255,255,.08);
}
.stSelectbox > div > div { background:#16162a !important; border-color:rgba(255,255,255,.15) !important; }
.stButton > button {
    background: linear-gradient(135deg,#007A4D,#005c38) !important;
    color:white !important; border:none !important; border-radius:8px !important;
    font-weight:700 !important; letter-spacing:.3px !important;
    box-shadow:0 4px 15px rgba(0,122,77,.3) !important;
    transition: all .2s !important;
}
.stButton > button:hover { transform:translateY(-1px) !important; box-shadow:0 6px 20px rgba(0,122,77,.4) !important; }
.stDownloadButton > button {
    background: linear-gradient(135deg,#1A6A9E,#0d4a72) !important;
    color:white !important; border:none !important; border-radius:8px !important; font-weight:700 !important;
}
.stDataFrame { border-radius:10px; overflow:hidden; }
label, .stSlider label { color:#ccc !important; font-size:.85rem !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ARTEFACTS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_artefacts():
    base   = os.path.dirname(__file__)
    model  = joblib.load(os.path.join(base,"ocp_best_model.pkl"))
    scaler = joblib.load(os.path.join(base,"ocp_scaler.pkl"))
    le     = joblib.load(os.path.join(base,"ocp_label_encoder.pkl"))
    return model, scaler, le

model, scaler, le = load_artefacts()

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
if 'history'       not in st.session_state: st.session_state.history = []
if 'sim_running'   not in st.session_state: st.session_state.sim_running = False
if 'sim_data'      not in st.session_state: st.session_state.sim_data = []

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def compute_features(type_enc, air_temp, proc_temp, rot_speed, torque, tool_wear):
    td  = proc_temp - air_temp
    pw  = torque * (rot_speed * 2 * np.pi / 60)
    tsr = torque / (rot_speed + 1e-6)
    wn  = tool_wear / TOOL_WEAR_MAX
    oh  = int(proc_temp > 309 and rot_speed < 1380)
    X   = np.array([[type_enc, air_temp, proc_temp, rot_speed, torque, tool_wear, td, pw, tsr, wn, oh]])
    return X

def predict(X_raw):
    Xs   = scaler.transform(X_raw)
    pred = int(model.predict(Xs)[0])
    prob = float(model.predict_proba(Xs)[0][1]) * 100
    return pred, prob, Xs

def set_matplotlib_dark():
    plt.rcParams.update({
        'figure.facecolor':'#0f0f1a','axes.facecolor':'#16162a',
        'axes.edgecolor':'#333','axes.labelcolor':'#ccc','xtick.color':'#aaa',
        'ytick.color':'#aaa','text.color':'#e8e8f0','grid.color':'#2a2a3e',
        'grid.alpha':.5,'font.family':'DejaVu Sans',
    })

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="ocp-header">
  <h1>🏭 OCP — Maintenance Prédictive Industrielle</h1>
  <p class="sub">Système de détection des pannes machines · Temps réel · Intelligence Artificielle</p>
  <span class="badge">⚡ LightGBM</span>
  <span class="badge">🔬 SHAP</span>
  <span class="badge">📡 Simulation live</span>
  <span class="badge">📊 Analyses avancées</span>
  <span class="badge">📄 Export PDF</span>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Prédiction",
    "📊 Analyses avancées",
    "📡 Simulation live",
    "📋 Historique",
    "🗂️ Batch CSV",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Prédiction manuelle
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_in, col_out = st.columns([1.1, 0.9], gap="large")

    with col_in:
        st.markdown('<div class="section-title">⚙️ Paramètres Machine</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            machine_type = st.selectbox("Type de machine",
                ["H — High quality","M — Medium quality","L — Low quality"])
            type_letter = machine_type[0]
            type_enc = int(le.transform([type_letter])[0])
            air_temp  = st.slider("🌡️ Température air (K)",  295.0,305.0,300.0,0.1,format="%.1f K")
            proc_temp = st.slider("🔥 Température process (K)",305.0,315.0,310.0,0.1,format="%.1f K")
        with c2:
            rot_speed = st.slider("⚡ Vitesse rotation (rpm)",1168,2886,1500,10)
            torque    = st.slider("🔩 Couple (Nm)",3.8,76.6,40.0,0.5,format="%.1f Nm")
            tool_wear = st.slider("🛠️ Usure outil (min)",0,253,100,1)

        X_raw = compute_features(type_enc, air_temp, proc_temp, rot_speed, torque, tool_wear)
        td = X_raw[0][6]; pw = X_raw[0][7]; oh = int(X_raw[0][10])

        st.markdown('<div class="section-title" style="margin-top:1rem">📐 Features Dérivées</div>', unsafe_allow_html=True)
        fa,fb,fc = st.columns(3)
        fa.metric("ΔT Process-Air", f"{td:.2f} K")
        fb.metric("Puissance méca.", f"{pw:.0f} W")
        fc.metric("Surchauffe", "🔴 Oui" if oh else "🟢 Non")

        # Alertes seuils
        alerts = []
        if tool_wear > 200: alerts.append(("🚨","CRITIQUE",f"Usure outil : {tool_wear} min > 200","crit"))
        if torque > 65:     alerts.append(("🚨","CRITIQUE",f"Couple : {torque:.1f} Nm > 65","crit"))
        if proc_temp > 312: alerts.append(("⚠️","ATTENTION",f"Température process : {proc_temp:.1f} K > 312","warn"))
        if rot_speed < 1300:alerts.append(("⚠️","ATTENTION",f"Vitesse : {rot_speed} rpm < 1300","warn"))
        if alerts:
            st.markdown('<div class="section-title" style="margin-top:.8rem">🚨 Alertes Seuils</div>', unsafe_allow_html=True)
            for ico,lvl,msg,typ in alerts:
                css = "alert-crit" if typ=="crit" else "alert-warn"
                st.markdown(f'<div class="{css}">{ico} <b>{lvl}</b> — {msg}</div>', unsafe_allow_html=True)

        predict_btn = st.button("🔍 Prédire la panne", use_container_width=True, type="primary")

    with col_out:
        st.markdown('<div class="section-title">📊 Résultat</div>', unsafe_allow_html=True)

        if predict_btn:
            pred, pf, Xs = predict(X_raw)

            # Animation
            prog = st.progress(0)
            for i in range(100):
                time.sleep(0.005)
                prog.progress(i+1)
            prog.empty()

            if pred == 1:
                st.markdown(f'<div class="result-fail">❌ PANNE PRÉDITE<br>'
                            f'<span style="font-size:.85rem;font-weight:400">Probabilité : <b>{pf:.1f}%</b></span></div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-ok">✅ MACHINE NORMALE<br>'
                            f'<span style="font-size:.85rem;font-weight:400">Probabilité de panne : <b>{pf:.1f}%</b></span></div>',
                            unsafe_allow_html=True)

            # Jauge animée
            color = OCP_RED if pf>=50 else OCP_GOLD if pf>=20 else OCP_GREEN
            st.markdown(f"""
            <div style="background:#2a2a3e;border-radius:8px;height:24px;overflow:hidden;margin:.8rem 0;
                        box-shadow:inset 0 2px 4px rgba(0,0,0,.3)">
              <div style="width:{pf:.1f}%;background:linear-gradient(90deg,{color}99,{color});
                          height:100%;display:flex;align-items:center;justify-content:flex-end;
                          padding-right:8px;color:white;font-size:.78rem;font-weight:700;border-radius:8px">
                {pf:.1f}%</div>
            </div>""", unsafe_allow_html=True)

            m1,m2 = st.columns(2)
            m1.metric("✅ Normal", f"{100-pf:.1f}%")
            m2.metric("❌ Panne",  f"{pf:.1f}%")

            if   pf>=70: st.error("🚨 Intervention immédiate — arrêter la machine.")
            elif pf>=40: st.warning("⚠️ Inspection préventive dans les 24h.")
            elif pf>=20: st.info("🔍 Surveillance renforcée conseillée.")
            else:        st.success("✅ État normal — suivi périodique standard.")

            # Save
            st.session_state.history.append({
                "Horodatage": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Type":type_letter,"Air Temp (K)":air_temp,"Proc Temp (K)":proc_temp,
                "RPM":rot_speed,"Torque (Nm)":torque,"Tool Wear (min)":tool_wear,
                "Prédiction":"Panne" if pred==1 else "Normal",
                "Proba Panne (%)":round(pf,2),
            })
            st.session_state['last_X_scaled'] = Xs
            st.session_state['last_X_raw']    = X_raw
            st.session_state['last_pred']     = pred
            st.session_state['last_pf']       = pf
            st.session_state['last_params']   = {
                "Type":type_letter,"Air Temp (K)":air_temp,"Proc Temp (K)":proc_temp,
                "RPM":rot_speed,"Torque (Nm)":torque,"Tool Wear (min)":tool_wear,
                "Prédiction":"Panne" if pred==1 else "Normal","Proba (%)":round(pf,2),
                "Horodatage":datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        else:
            st.markdown('<div class="card card-blue" style="color:#aaa;text-align:center;padding:2rem">'
                        '👈 Ajustez les paramètres<br>puis cliquez sur <b style="color:#F5A800">Prédire la panne</b>'
                        '</div>', unsafe_allow_html=True)
            # Modèle info
            st.markdown('<div class="section-title" style="margin-top:1rem">ℹ️ Modèle</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="card card-green" style="font-size:.83rem;line-height:1.8;color:#ccc">
            <b style="color:#4ecca3">Algorithme</b> · LightGBM Classifier<br>
            <b style="color:#4ecca3">Dataset</b> · AI4I 2020 — 10 000 entrées<br>
            <b style="color:#4ecca3">Rééchantillonnage</b> · SMOTE<br>
            <b style="color:#4ecca3">Normalisation</b> · StandardScaler<br>
            <b style="color:#4ecca3">Features</b> · 11 (5 capteurs + 5 ingénierie + 1 encodage)
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Analyses avancées
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">📊 Analyses Avancées & Interprétabilité</div>', unsafe_allow_html=True)

    if 'last_X_scaled' not in st.session_state:
        st.markdown('<div class="card card-gold" style="text-align:center;color:#aaa;padding:2rem">'
                    '⚡ Effectuez d\'abord une prédiction dans l\'onglet <b style="color:#F5A800">Prédiction</b></div>',
                    unsafe_allow_html=True)
    else:
        Xs   = st.session_state['last_X_scaled']
        Xraw = st.session_state['last_X_raw']

        set_matplotlib_dark()

        row1_c1, row1_c2 = st.columns(2)

        # ── SHAP ──────────────────────────────────────────────────────────────
        with row1_c1:
            st.markdown("**🔬 Contribution SHAP — dernière prédiction**")
            try:
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(Xs)
                sv_arr = sv[0] if isinstance(sv, list) else sv[0]

                fig, ax = plt.subplots(figsize=(6,4))
                sorted_idx = np.argsort(np.abs(sv_arr))
                sv_sorted  = sv_arr[sorted_idx]
                lb_sorted  = [FEAT_LABELS[i] for i in sorted_idx]
                colors_s   = [OCP_RED if v>0 else OCP_GREEN for v in sv_sorted]
                bars = ax.barh(lb_sorted, sv_sorted, color=colors_s, edgecolor='#0f0f1a', height=.65)
                ax.axvline(0, color='#666', linewidth=.8)
                ax.set_xlabel("Valeur SHAP", fontsize=8)
                ax.set_title("Impact sur la prédiction de panne", fontsize=9, fontweight='bold', color='#e8e8f0')
                p1 = mpatches.Patch(color=OCP_RED,   label='↑ Risque')
                p2 = mpatches.Patch(color=OCP_GREEN, label='↓ Risque')
                ax.legend(handles=[p1,p2], fontsize=7)
                plt.tight_layout()
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"SHAP : {e}")

        # ── Radar ─────────────────────────────────────────────────────────────
        with row1_c2:
            st.markdown("**📡 Radar des capteurs normalisés**")
            raw = Xraw[0]
            lbls_r  = ['Air Temp','Proc Temp','RPM','Torque','Tool Wear']
            ranges  = [(295,305),(305,315),(1168,2886),(3.8,76.6),(0,253)]
            vals_r  = [(raw[i+1]-ranges[i][0])/(ranges[i][1]-ranges[i][0]) for i in range(5)]
            vals_r += [vals_r[0]]
            angles  = np.linspace(0,2*np.pi,5,endpoint=False).tolist()
            angles += angles[:1]

            fig_r, ax_r = plt.subplots(figsize=(4.5,4.5), subplot_kw=dict(polar=True))
            ax_r.set_facecolor('#16162a')
            fig_r.patch.set_facecolor('#0f0f1a')
            ax_r.plot(angles, vals_r, color=OCP_GREEN, linewidth=2.5)
            ax_r.fill(angles, vals_r, color=OCP_GREEN, alpha=.2)
            # danger zone
            danger = [1.0]*5+[1.0]
            ax_r.fill(angles, danger, color=OCP_RED, alpha=.05)
            ax_r.set_xticks(angles[:-1])
            ax_r.set_xticklabels(lbls_r, fontsize=8, color='#ccc')
            ax_r.set_ylim(0,1); ax_r.set_yticks([.25,.5,.75,1.0])
            ax_r.set_yticklabels(['25%','50%','75%','100%'], fontsize=6, color='#777')
            ax_r.grid(color='#2a2a3e', linewidth=.8)
            ax_r.set_title("Position dans la plage nominale", fontsize=9, fontweight='bold', color='#e8e8f0', pad=15)
            plt.tight_layout()
            st.pyplot(fig_r)

        row2_c1, row2_c2 = st.columns(2)

        # ── Courbe ROC ────────────────────────────────────────────────────────
        with row2_c1:
            st.markdown("**📈 Courbe ROC (données synthétiques de démonstration)**")
            np.random.seed(42)
            n_demo = 500
            X_demo = np.random.randn(n_demo, 11)
            proba_demo = model.predict_proba(X_demo)[:,1]
            y_demo = (proba_demo > 0.5).astype(int)
            fpr, tpr, _ = roc_curve(y_demo, proba_demo)
            roc_auc = auc(fpr, tpr)

            fig_roc, ax_roc = plt.subplots(figsize=(5,4))
            ax_roc.plot(fpr, tpr, color=OCP_GREEN, linewidth=2.5, label=f'AUC = {roc_auc:.3f}')
            ax_roc.fill_between(fpr, tpr, alpha=.1, color=OCP_GREEN)
            ax_roc.plot([0,1],[0,1], color='#555', linestyle='--', linewidth=1, label='Aléatoire')
            ax_roc.set_xlabel("Taux de Faux Positifs", fontsize=8)
            ax_roc.set_ylabel("Taux de Vrais Positifs", fontsize=8)
            ax_roc.set_title("Courbe ROC — LightGBM", fontsize=9, fontweight='bold', color='#e8e8f0')
            ax_roc.legend(fontsize=8)
            ax_roc.set_xlim([0,1]); ax_roc.set_ylim([0,1.02])
            plt.tight_layout()
            st.pyplot(fig_roc)

        # ── Matrice de confusion ──────────────────────────────────────────────
        with row2_c2:
            st.markdown("**🔲 Matrice de Confusion**")
            y_pred_demo = (proba_demo > 0.5).astype(int)
            cm_arr = confusion_matrix(y_demo, y_pred_demo)

            fig_cm, ax_cm = plt.subplots(figsize=(4.5,4))
            cmap = LinearSegmentedColormap.from_list('ocp', ['#16162a', OCP_GREEN], N=100)
            im = ax_cm.imshow(cm_arr, cmap=cmap, aspect='auto')
            ax_cm.set_xticks([0,1]); ax_cm.set_yticks([0,1])
            ax_cm.set_xticklabels(['Prédit Normal','Prédit Panne'], fontsize=8)
            ax_cm.set_yticklabels(['Réel Normal','Réel Panne'], fontsize=8)
            for i in range(2):
                for j in range(2):
                    ax_cm.text(j, i, str(cm_arr[i,j]), ha='center', va='center',
                               fontsize=18, fontweight='bold',
                               color='white' if cm_arr[i,j] > cm_arr.max()/2 else '#ccc')
            ax_cm.set_title("Matrice de Confusion", fontsize=9, fontweight='bold', color='#e8e8f0')
            plt.colorbar(im, ax=ax_cm, fraction=.046, pad=.04)
            plt.tight_layout()
            st.pyplot(fig_cm)

        # ── Feature importance globale ────────────────────────────────────────
        st.markdown("**🏆 Importance Globale des Features (modèle)**")
        try:
            importances = model.feature_importances_
            sorted_idx  = np.argsort(importances)
            fig_fi, ax_fi = plt.subplots(figsize=(10, 3.5))
            cmap_fi = plt.cm.RdYlGn
            norm_fi = plt.Normalize(importances.min(), importances.max())
            colors_fi = [cmap_fi(norm_fi(v)) for v in importances[sorted_idx]]
            ax_fi.barh([FEAT_LABELS[i] for i in sorted_idx], importances[sorted_idx],
                       color=colors_fi, edgecolor='#0f0f1a', height=.65)
            ax_fi.set_xlabel("Importance (gain)", fontsize=8)
            ax_fi.set_title("Contribution globale de chaque feature au modèle LightGBM", fontsize=9, fontweight='bold', color='#e8e8f0')
            plt.tight_layout()
            st.pyplot(fig_fi)
        except Exception as e:
            st.warning(f"Feature importance : {e}")

        # ── Export PDF ────────────────────────────────────────────────────────
        st.markdown('<div class="section-title" style="margin-top:1.2rem">📄 Export Rapport PDF</div>', unsafe_allow_html=True)
        if st.button("📄 Générer le rapport PDF", use_container_width=True):
            params  = st.session_state.get('last_params', {})
            pf_last = st.session_state.get('last_pf', 0)
            pred_l  = st.session_state.get('last_pred', 0)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                doc = SimpleDocTemplate(tmp.name, pagesize=A4,
                                        leftMargin=2*cm, rightMargin=2*cm,
                                        topMargin=2*cm, bottomMargin=2*cm)
                styles = getSampleStyleSheet()
                story  = []

                # Title
                title_style = ParagraphStyle('title', parent=styles['Title'],
                    fontSize=18, textColor=rl_colors.HexColor('#007A4D'),
                    spaceAfter=6, alignment=TA_CENTER, fontName='Helvetica-Bold')
                sub_style = ParagraphStyle('sub', parent=styles['Normal'],
                    fontSize=9, textColor=rl_colors.grey, alignment=TA_CENTER, spaceAfter=16)
                section_style = ParagraphStyle('sec', parent=styles['Normal'],
                    fontSize=11, textColor=rl_colors.HexColor('#007A4D'),
                    fontName='Helvetica-Bold', spaceBefore=14, spaceAfter=6)
                body_style = ParagraphStyle('body', parent=styles['Normal'],
                    fontSize=9, textColor=rl_colors.HexColor('#333'), spaceAfter=4)

                story.append(Paragraph("🏭 OCP — Rapport de Maintenance Prédictive", title_style))
                story.append(Paragraph(f"Généré le {datetime.datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}", sub_style))
                story.append(HRFlowable(width="100%", thickness=2, color=rl_colors.HexColor('#007A4D')))
                story.append(Spacer(1, 12))

                # Résultat
                story.append(Paragraph("1. Résultat de la Prédiction", section_style))
                result_txt = "❌ PANNE DÉTECTÉE" if pred_l==1 else "✅ MACHINE NORMALE"
                res_color  = rl_colors.HexColor('#C0392B') if pred_l==1 else rl_colors.HexColor('#007A4D')
                res_style  = ParagraphStyle('res', parent=styles['Normal'],
                    fontSize=14, textColor=res_color, fontName='Helvetica-Bold',
                    alignment=TA_CENTER, spaceAfter=8)
                story.append(Paragraph(result_txt, res_style))
                story.append(Paragraph(f"Probabilité de panne : <b>{pf_last:.1f}%</b>", body_style))

                reco = ("Intervention immédiate requise." if pf_last>=70
                        else "Inspection préventive dans les 24h." if pf_last>=40
                        else "Surveillance renforcée conseillée." if pf_last>=20
                        else "État normal — suivi périodique standard.")
                story.append(Paragraph(f"Recommandation : {reco}", body_style))
                story.append(Spacer(1, 10))

                # Paramètres
                story.append(Paragraph("2. Paramètres de la Machine", section_style))
                if params:
                    table_data = [["Paramètre", "Valeur"]] + [[k,str(v)] for k,v in params.items()]
                    tbl = Table(table_data, colWidths=[9*cm, 7*cm])
                    tbl.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), rl_colors.HexColor('#007A4D')),
                        ('TEXTCOLOR',  (0,0), (-1,0), rl_colors.white),
                        ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
                        ('FONTSIZE',   (0,0), (-1,-1), 9),
                        ('ROWBACKGROUNDS',(0,1),(-1,-1),[rl_colors.HexColor('#f0f7f4'), rl_colors.white]),
                        ('GRID',       (0,0), (-1,-1), .5, rl_colors.HexColor('#ccc')),
                        ('VALIGN',     (0,0), (-1,-1), 'MIDDLE'),
                        ('TOPPADDING', (0,0), (-1,-1), 5),
                        ('BOTTOMPADDING',(0,0),(-1,-1),5),
                    ]))
                    story.append(tbl)

                story.append(Spacer(1, 10))

                # Historique résumé
                if st.session_state.history:
                    story.append(Paragraph("3. Résumé de l'Historique", section_style))
                    df_h = pd.DataFrame(st.session_state.history)
                    n_p  = (df_h["Prédiction"]=="Panne").sum()
                    story.append(Paragraph(f"Total prédictions : {len(df_h)}", body_style))
                    story.append(Paragraph(f"Pannes détectées : {n_p} ({n_p/len(df_h)*100:.1f}%)", body_style))
                    story.append(Paragraph(f"Probabilité moyenne : {df_h['Proba Panne (%)'].mean():.1f}%", body_style))

                story.append(Spacer(1, 16))
                story.append(HRFlowable(width="100%", thickness=1, color=rl_colors.HexColor('#ccc')))
                footer = ParagraphStyle('footer', parent=styles['Normal'],
                    fontSize=7, textColor=rl_colors.grey, alignment=TA_CENTER, spaceBefore=6)
                story.append(Paragraph("OCP Group · Système de Maintenance Prédictive · Confidentiel", footer))

                doc.build(story)
                with open(tmp.name,'rb') as f:
                    pdf_bytes = f.read()

            st.download_button(
                "📥 Télécharger le rapport PDF",
                data=pdf_bytes,
                file_name=f"ocp_rapport_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Simulation temps réel
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">📡 Simulation Temps Réel</div>', unsafe_allow_html=True)
    st.markdown("Simulation d'un flux de capteurs IoT avec détection automatique de pannes.", unsafe_allow_html=True)

    sim_col1, sim_col2 = st.columns([1,2])

    with sim_col1:
        st.markdown('<div class="section-title">⚙️ Paramètres simulation</div>', unsafe_allow_html=True)
        sim_type     = st.selectbox("Type machine", ["H","M","L"], key="sim_type")
        sim_noise    = st.slider("Niveau de bruit", 0.0, 1.0, 0.3, 0.05)
        sim_drift    = st.checkbox("Simuler une dérive (usure progressive)", value=True)
        n_sim_points = st.slider("Nombre de points", 20, 100, 40, 5)

        col_sb1, col_sb2 = st.columns(2)
        run_btn   = col_sb1.button("▶️ Lancer",  use_container_width=True, type="primary")
        clear_btn = col_sb2.button("🗑️ Effacer", use_container_width=True)

        if clear_btn:
            st.session_state.sim_data = []
            st.rerun()

    with sim_col2:
        chart_placeholder = st.empty()
        kpi_placeholder   = st.empty()
        status_placeholder= st.empty()

        if run_btn:
            st.session_state.sim_data = []
            type_enc_s = int(le.transform([sim_type])[0])

            for i in range(n_sim_points):
                # Génération données capteur avec bruit + dérive
                drift = (i / n_sim_points) if sim_drift else 0
                air_t  = 300  + np.random.randn() * sim_noise * 2
                proc_t = 310  + np.random.randn() * sim_noise * 2 + drift * 4
                rpm    = 1500 + np.random.randn() * sim_noise * 100 - drift * 200
                tq     = 40   + np.random.randn() * sim_noise * 5  + drift * 20
                tw     = min(253, 80 + i * (sim_drift * 1.5 + 0.3) + np.random.randn() * 5)

                X_s = compute_features(type_enc_s, air_t, proc_t, max(1168,rpm), tq, tw)
                p, pf_s, _ = predict(X_s)

                st.session_state.sim_data.append({
                    "t": i+1, "prob": pf_s, "pred": p,
                    "tool_wear": tw, "torque": tq, "rpm": max(1168,rpm)
                })

                # Mise à jour graphique live
                df_sim = pd.DataFrame(st.session_state.sim_data)
                set_matplotlib_dark()
                fig_sim, axes_s = plt.subplots(2,1,figsize=(8,5), sharex=True)

                # Probabilité
                axes_s[0].fill_between(df_sim["t"], df_sim["prob"], alpha=.25,
                    color=[OCP_RED if p>=50 else OCP_GOLD if p>=20 else OCP_GREEN for p in df_sim["prob"]][0])
                for idx in range(len(df_sim)-1):
                    c = OCP_RED if df_sim["prob"].iloc[idx]>=50 else OCP_GOLD if df_sim["prob"].iloc[idx]>=20 else OCP_GREEN
                    axes_s[0].plot(df_sim["t"].iloc[idx:idx+2], df_sim["prob"].iloc[idx:idx+2], color=c, linewidth=2)
                axes_s[0].axhline(50, color=OCP_RED,  linestyle='--', linewidth=.8, alpha=.7)
                axes_s[0].axhline(20, color=OCP_GOLD, linestyle='--', linewidth=.8, alpha=.7)
                axes_s[0].set_ylabel("Proba panne (%)", fontsize=8)
                axes_s[0].set_ylim(0,105)
                axes_s[0].set_title("Probabilité de panne en temps réel", fontsize=9, fontweight='bold', color='#e8e8f0')

                # Usure outil
                axes_s[1].plot(df_sim["t"], df_sim["tool_wear"], color=OCP_GOLD, linewidth=2)
                axes_s[1].axhline(200, color=OCP_RED, linestyle='--', linewidth=.8, alpha=.7, label='Seuil critique (200 min)')
                axes_s[1].set_ylabel("Usure outil (min)", fontsize=8)
                axes_s[1].set_xlabel("Mesure #", fontsize=8)
                axes_s[1].legend(fontsize=7)

                plt.tight_layout()
                chart_placeholder.pyplot(fig_sim)
                plt.close(fig_sim)

                # KPIs live
                n_pannes_s = sum(1 for d in st.session_state.sim_data if d["pred"]==1)
                kpi_placeholder.markdown(f"""
                <div class="kpi-grid">
                  <div class="kpi"><div class="val">{i+1}</div><div class="lbl">Mesures</div></div>
                  <div class="kpi"><div class="val">{n_pannes_s}</div><div class="lbl">Pannes</div></div>
                  <div class="kpi"><div class="val">{pf_s:.0f}%</div><div class="lbl">Prob. actuelle</div></div>
                  <div class="kpi"><div class="val">{tw:.0f}</div><div class="lbl">Usure (min)</div></div>
                </div>""", unsafe_allow_html=True)

                if p == 1:
                    status_placeholder.error(f"🚨 PANNE DÉTECTÉE — mesure #{i+1} — prob: {pf_s:.1f}%")
                elif pf_s >= 20:
                    status_placeholder.warning(f"⚠️ Vigilance — mesure #{i+1} — prob: {pf_s:.1f}%")
                else:
                    status_placeholder.success(f"✅ Normal — mesure #{i+1} — prob: {pf_s:.1f}%")

                time.sleep(0.08)

        elif st.session_state.sim_data:
            df_sim = pd.DataFrame(st.session_state.sim_data)
            set_matplotlib_dark()
            fig_sim2, ax2 = plt.subplots(figsize=(8,3.5))
            ax2.plot(df_sim["t"], df_sim["prob"], color=OCP_GREEN, linewidth=2)
            ax2.axhline(50, color=OCP_RED,  linestyle='--', linewidth=.8)
            ax2.set_title("Dernière simulation", fontsize=9, fontweight='bold', color='#e8e8f0')
            ax2.set_ylabel("Probabilité panne (%)", fontsize=8)
            plt.tight_layout()
            chart_placeholder.pyplot(fig_sim2)
            plt.close(fig_sim2)
        else:
            st.markdown('<div class="card card-blue" style="text-align:center;color:#aaa;padding:2.5rem">'
                        '▶️ Configurez et lancez la simulation</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Historique
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">📋 Historique des Prédictions</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown('<div class="card card-blue" style="text-align:center;color:#aaa;padding:2rem">'
                    'Aucune prédiction enregistrée.</div>', unsafe_allow_html=True)
    else:
        df_hist = pd.DataFrame(st.session_state.history)
        n_p = (df_hist["Prédiction"]=="Panne").sum()

        st.markdown(f"""
        <div class="kpi-grid">
          <div class="kpi"><div class="val">{len(df_hist)}</div><div class="lbl">Total prédictions</div></div>
          <div class="kpi"><div class="val">{n_p}</div><div class="lbl">Pannes détectées</div></div>
          <div class="kpi"><div class="val">{n_p/len(df_hist)*100:.1f}%</div><div class="lbl">Taux de panne</div></div>
          <div class="kpi"><div class="val">{df_hist['Proba Panne (%)'].mean():.1f}%</div><div class="lbl">Prob. moyenne</div></div>
        </div>""", unsafe_allow_html=True)

        st.dataframe(df_hist, use_container_width=True, hide_index=True)

        c1h, c2h = st.columns(2)
        csv_buf = io.StringIO()
        df_hist.to_csv(csv_buf, index=False)
        c1h.download_button("📥 Exporter CSV", csv_buf.getvalue(),
            f"ocp_historique_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv", use_container_width=True)
        if c2h.button("🗑️ Effacer l'historique", use_container_width=True):
            st.session_state.history = []
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Batch CSV
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">🗂️ Prédictions en Masse (CSV)</div>', unsafe_allow_html=True)
    st.markdown("Uploadez un CSV avec : `Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min]`")

    template = pd.DataFrame({
        "Type":["H","M","L","H"],
        "Air temperature [K]":[300.0,298.5,302.1,301.0],
        "Process temperature [K]":[310.0,308.2,313.5,309.8],
        "Rotational speed [rpm]":[1500,1380,2100,1750],
        "Torque [Nm]":[40.0,55.2,28.7,61.3],
        "Tool wear [min]":[100,190,50,220],
    })
    buf_t = io.StringIO(); template.to_csv(buf_t,index=False)
    st.download_button("📄 Télécharger le template CSV", buf_t.getvalue(), "template_ocp.csv","text/csv")

    uploaded = st.file_uploader("📂 Uploader votre CSV", type=["csv"])
    if uploaded:
        try:
            df_batch = pd.read_csv(uploaded)
            st.write(f"**{len(df_batch)} lignes chargées**")
            st.dataframe(df_batch.head(), use_container_width=True, hide_index=True)
            if st.button("🚀 Lancer les prédictions batch", type="primary", use_container_width=True):
                results = []
                bar = st.progress(0)
                for idx2, row in df_batch.iterrows():
                    t_e = int(le.transform([str(row["Type"])[0]])[0])
                    X_b = compute_features(t_e, float(row["Air temperature [K]"]),
                                           float(row["Process temperature [K]"]),
                                           float(row["Rotational speed [rpm]"]),
                                           float(row["Torque [Nm]"]), float(row["Tool wear [min]"]))
                    p_b, pf_b, _ = predict(X_b)
                    results.append({"Prédiction":"Panne" if p_b==1 else "Normal",
                                    "Proba Panne (%)":round(pf_b,2),
                                    "Statut":"🚨" if pf_b>=70 else "⚠️" if pf_b>=40 else "✅"})
                    bar.progress((idx2+1)/len(df_batch))

                df_res = pd.concat([df_batch.reset_index(drop=True), pd.DataFrame(results)], axis=1)
                n_pb   = (pd.DataFrame(results)["Prédiction"]=="Panne").sum()

                st.markdown(f"""
                <div class="kpi-grid">
                  <div class="kpi"><div class="val">{len(df_res)}</div><div class="lbl">Machines</div></div>
                  <div class="kpi"><div class="val">{n_pb}</div><div class="lbl">Pannes</div></div>
                  <div class="kpi"><div class="val">{n_pb/len(df_res)*100:.1f}%</div><div class="lbl">Taux</div></div>
                </div>""", unsafe_allow_html=True)

                st.dataframe(df_res, use_container_width=True, hide_index=True)
                csv_r = io.StringIO(); df_res.to_csv(csv_r, index=False)
                st.download_button("📥 Télécharger les résultats", csv_r.getvalue(),
                    f"ocp_batch_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    "text/csv", use_container_width=True)
        except Exception as e:
            st.error(f"Erreur : {e}")
