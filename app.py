import streamlit as st
import numpy as np
import joblib
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OCP — Maintenance Prédictive",
    page_icon="🏭",
    layout="wide",
)

# ── CSS OCP ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --ocp-green : #007A4D;
        --ocp-gold  : #F5A800;
        --ocp-red   : #C0392B;
        --ocp-blue  : #1A6A9E;
    }
    .main { background: #f7f8fa; }
    .block-container { padding-top: 1.5rem; }

    .ocp-header {
        background: linear-gradient(135deg, #007A4D 0%, #005c38 100%);
        border-radius: 12px;
        padding: 1.6rem 2rem;
        margin-bottom: 1.5rem;
        color: white;
    }
    .ocp-header h1 { margin: 0; font-size: 1.9rem; font-weight: 700; }
    .ocp-header p  { margin: .35rem 0 0; opacity: .85; font-size: .95rem; }

    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1.1rem 1.4rem;
        text-align: center;
        box-shadow: 0 1px 6px rgba(0,0,0,.08);
        border-top: 4px solid var(--ocp-green);
    }
    .metric-card.danger { border-top-color: var(--ocp-red); }
    .metric-card .value { font-size: 2rem; font-weight: 700; color: #1a1a1a; }
    .metric-card .label { font-size: .8rem; color: #666; margin-top: .2rem; }

    .result-ok {
        background: #e8f5ee;
        border: 2px solid #007A4D;
        border-radius: 10px;
        padding: 1.2rem 1.6rem;
        color: #007A4D;
        font-size: 1.2rem;
        font-weight: 700;
        text-align: center;
    }
    .result-fail {
        background: #fdecea;
        border: 2px solid #C0392B;
        border-radius: 10px;
        padding: 1.2rem 1.6rem;
        color: #C0392B;
        font-size: 1.2rem;
        font-weight: 700;
        text-align: center;
    }
    .section-title {
        font-size: 1rem;
        font-weight: 700;
        color: #007A4D;
        border-bottom: 2px solid #007A4D;
        padding-bottom: .3rem;
        margin-bottom: 1rem;
    }
    .stSlider > div > div > div > div { background: #007A4D !important; }
    div[data-testid="stSelectbox"] label { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── Load artefacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    base = os.path.dirname(__file__)
    model  = joblib.load(os.path.join(base, "ocp_best_model.pkl"))
    scaler = joblib.load(os.path.join(base, "ocp_scaler.pkl"))
    le     = joblib.load(os.path.join(base, "ocp_label_encoder.pkl"))
    return model, scaler, le

model, scaler, le = load_artefacts()

FEATURES = [
    'Type_enc',
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]',
    'Temp_diff [K]',
    'Power [W]',
    'Torque_Speed_ratio',
    'Wear_norm',
    'Overheat_flag',
]

TOOL_WEAR_MAX = 253.0   # max observé dans l'entraînement

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ocp-header">
  <h1>🏭 OCP — Maintenance Prédictive Industrielle</h1>
  <p>Prédiction des pannes machines en temps réel · Modèle : LightGBM · Dataset : AI4I 2020</p>
</div>
""", unsafe_allow_html=True)

# ── Layout ─────────────────────────────────────────────────────────────────────
col_inputs, col_result = st.columns([1.1, 0.9], gap="large")

# ─────────────────────────── INPUTS ──────────────────────────────────────────
with col_inputs:
    st.markdown('<div class="section-title">⚙️ Paramètres de la Machine</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        machine_type = st.selectbox(
            "Type de machine",
            options=["H — High quality", "M — Medium quality", "L — Low quality"],
            help="Qualité de la machine industrielle"
        )
        type_letter = machine_type[0]
        type_enc = int(le.transform([type_letter])[0])

        air_temp = st.slider(
            "🌡️ Température air (K)",
            min_value=295.0, max_value=305.0, value=300.0, step=0.1,
            format="%.1f K"
        )
        proc_temp = st.slider(
            "🔥 Température process (K)",
            min_value=305.0, max_value=315.0, value=310.0, step=0.1,
            format="%.1f K"
        )

    with c2:
        rot_speed = st.slider(
            "⚡ Vitesse rotation (rpm)",
            min_value=1168, max_value=2886, value=1500, step=10,
        )
        torque = st.slider(
            "🔩 Couple (Nm)",
            min_value=3.8, max_value=76.6, value=40.0, step=0.5,
            format="%.1f Nm"
        )
        tool_wear = st.slider(
            "🛠️ Usure outil (min)",
            min_value=0, max_value=253, value=100, step=1,
        )

    # ── Feature engineering (miroir du notebook) ──────────────────────────────
    temp_diff         = proc_temp - air_temp
    power_w           = torque * (rot_speed * 2 * np.pi / 60)
    torque_speed_ratio = torque / (rot_speed + 1e-6)
    wear_norm         = tool_wear / TOOL_WEAR_MAX
    overheat_flag     = int(proc_temp > 309 and rot_speed < 1380)

    # Affichage features dérivées
    st.markdown('<div class="section-title" style="margin-top:1.2rem">📐 Features Dérivées (auto-calculées)</div>', unsafe_allow_html=True)
    fc1, fc2, fc3 = st.columns(3)
    fc1.metric("ΔT Process-Air", f"{temp_diff:.2f} K")
    fc2.metric("Puissance mécanique", f"{power_w:.0f} W")
    fc3.metric("Surchauffe flag", "🔴 Oui" if overheat_flag else "🟢 Non")

    predict_btn = st.button("🔍 Prédire la panne", use_container_width=True, type="primary")

# ─────────────────────────── RESULT ──────────────────────────────────────────
with col_result:
    st.markdown('<div class="section-title">📊 Résultat de la Prédiction</div>', unsafe_allow_html=True)

    if predict_btn:
        X_raw = np.array([[
            type_enc, air_temp, proc_temp, rot_speed, torque, tool_wear,
            temp_diff, power_w, torque_speed_ratio, wear_norm, overheat_flag
        ]])
        X_scaled = scaler.transform(X_raw)
        pred      = int(model.predict(X_scaled)[0])
        proba     = model.predict_proba(X_scaled)[0]
        proba_fail = float(proba[1]) * 100

        # Résultat principal
        if pred == 1:
            st.markdown(f"""
            <div class="result-fail">
                ❌ PANNE PRÉDITE<br>
                <span style="font-size:.9rem;font-weight:400">
                Probabilité de panne : <b>{proba_fail:.1f}%</b>
                </span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-ok">
                ✅ MACHINE NORMALE<br>
                <span style="font-size:.9rem;font-weight:400">
                Probabilité de panne : <b>{proba_fail:.1f}%</b>
                </span>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Jauge probabilité
        st.markdown("**Probabilité de panne**")
        color = "#C0392B" if proba_fail >= 50 else "#007A4D"
        st.markdown(f"""
        <div style="background:#e0e0e0;border-radius:6px;height:22px;overflow:hidden;margin-bottom:.5rem">
          <div style="width:{proba_fail:.1f}%;background:{color};height:100%;
                      display:flex;align-items:center;justify-content:flex-end;
                      padding-right:6px;color:white;font-size:.8rem;font-weight:700;
                      border-radius:6px;transition:width .4s">
            {proba_fail:.1f}%
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Probabilités détaillées
        m1, m2 = st.columns(2)
        m1.metric("✅ Normal", f"{proba[0]*100:.1f}%")
        m2.metric("❌ Panne",  f"{proba[1]*100:.1f}%")

        # Recommandation
        st.markdown("**💡 Recommandation**")
        if proba_fail >= 70:
            st.error("🚨 Intervention immédiate requise. Arrêter la machine et planifier une maintenance corrective.")
        elif proba_fail >= 40:
            st.warning("⚠️ Risque modéré. Planifier une inspection préventive dans les 24h.")
        elif proba_fail >= 20:
            st.info("🔍 Surveiller les paramètres. Inspection de routine conseillée.")
        else:
            st.success("✅ Machine en bon état. Continuer le suivi périodique standard.")

        # Résumé des inputs
        with st.expander("📋 Voir le résumé des paramètres"):
            import pandas as pd
            summary = {
                "Paramètre": FEATURES,
                "Valeur": [
                    type_enc, air_temp, proc_temp, rot_speed, torque, tool_wear,
                    round(temp_diff,3), round(power_w,2),
                    round(torque_speed_ratio,6), round(wear_norm,4), overheat_flag
                ]
            }
            st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

    else:
        st.info("👈 Ajustez les paramètres à gauche puis cliquez sur **Prédire la panne**.")

        # Infos modèle
        st.markdown("---")
        st.markdown("**ℹ️ À propos du modèle**")
        st.markdown("""
        - **Algorithme :** LightGBM Classifier  
        - **Dataset :** AI4I 2020 Predictive Maintenance (10 000 entrées)  
        - **Rééchantillonnage :** SMOTE (classes déséquilibrées)  
        - **Normalisation :** StandardScaler  
        - **Encodage :** LabelEncoder (Type machine)  
        - **Features :** 11 (5 capteurs + 5 ingénierie + 1 encodage)  
        """)
        st.markdown("""
        | Type | Description |
        |------|-------------|
        | H    | High quality |
        | M    | Medium quality |
        | L    | Low quality |
        """)
