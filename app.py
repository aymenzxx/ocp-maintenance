import streamlit as st
import numpy as np
import pandas as pd
import joblib, os, io, datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OCP — Maintenance Prédictive",
    page_icon="🏭",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.ocp-header {
    background: linear-gradient(135deg,#007A4D 0%,#005c38 100%);
    border-radius:12px; padding:1.5rem 2rem; margin-bottom:1.4rem; color:white;
}
.ocp-header h1 { margin:0; font-size:1.8rem; font-weight:700; }
.ocp-header p  { margin:.3rem 0 0; opacity:.85; font-size:.9rem; }
.section-title {
    font-size:.95rem; font-weight:700; color:#007A4D;
    border-bottom:2px solid #007A4D; padding-bottom:.25rem; margin-bottom:.9rem;
}
.result-ok   { background:#e8f5ee; border:2px solid #007A4D; border-radius:10px;
               padding:1rem 1.4rem; color:#007A4D; font-size:1.1rem; font-weight:700; text-align:center; }
.result-fail { background:#fdecea; border:2px solid #C0392B; border-radius:10px;
               padding:1rem 1.4rem; color:#C0392B; font-size:1.1rem; font-weight:700; text-align:center; }
.alert-box   { background:#fff3cd; border-left:5px solid #F5A800;
               border-radius:6px; padding:.8rem 1rem; margin:.5rem 0; font-size:.9rem; }
.alert-crit  { background:#fdecea; border-left:5px solid #C0392B;
               border-radius:6px; padding:.8rem 1rem; margin:.5rem 0; font-size:.9rem; }
</style>
""", unsafe_allow_html=True)

# ── Artefacts ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    base   = os.path.dirname(__file__)
    model  = joblib.load(os.path.join(base,"ocp_best_model.pkl"))
    scaler = joblib.load(os.path.join(base,"ocp_scaler.pkl"))
    le     = joblib.load(os.path.join(base,"ocp_label_encoder.pkl"))
    return model, scaler, le

model, scaler, le = load_artefacts()

FEATURES = ['Type_enc','Air temperature [K]','Process temperature [K]',
            'Rotational speed [rpm]','Torque [Nm]','Tool wear [min]',
            'Temp_diff [K]','Power [W]','Torque_Speed_ratio','Wear_norm','Overheat_flag']
FEAT_LABELS = ['Type','Air Temp','Proc Temp','Rot Speed','Torque','Tool Wear',
               'ΔT','Power','Torque/Speed','Wear norm','Overheat']
TOOL_WEAR_MAX = 253.0
OCP_GREEN='#007A4D'; OCP_RED='#C0392B'; OCP_GOLD='#F5A800'; OCP_BLUE='#1A6A9E'

# ── Session state ─────────────────────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ocp-header">
  <h1>🏭 OCP — Maintenance Prédictive Industrielle</h1>
  <p>Prédiction des pannes · LightGBM · SHAP · Alertes · Historique · Batch CSV</p>
</div>""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Prédiction manuelle",
    "📊 Visualisations & SHAP",
    "📋 Historique",
    "🗂️ Batch CSV",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Prédiction manuelle
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_in, col_out = st.columns([1.1, 0.9], gap="large")

    with col_in:
        st.markdown('<div class="section-title">⚙️ Paramètres de la Machine</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            machine_type = st.selectbox("Type de machine",
                ["H — High quality","M — Medium quality","L — Low quality"])
            type_letter = machine_type[0]
            type_enc = int(le.transform([type_letter])[0])
            air_temp  = st.slider("🌡️ Température air (K)",  295.0, 305.0, 300.0, 0.1, format="%.1f K")
            proc_temp = st.slider("🔥 Température process (K)", 305.0, 315.0, 310.0, 0.1, format="%.1f K")
        with c2:
            rot_speed = st.slider("⚡ Vitesse rotation (rpm)", 1168, 2886, 1500, 10)
            torque    = st.slider("🔩 Couple (Nm)", 3.8, 76.6, 40.0, 0.5, format="%.1f Nm")
            tool_wear = st.slider("🛠️ Usure outil (min)", 0, 253, 100, 1)

        # Feature engineering
        temp_diff          = proc_temp - air_temp
        power_w            = torque * (rot_speed * 2 * np.pi / 60)
        torque_speed_ratio = torque / (rot_speed + 1e-6)
        wear_norm          = tool_wear / TOOL_WEAR_MAX
        overheat_flag      = int(proc_temp > 309 and rot_speed < 1380)

        st.markdown('<div class="section-title" style="margin-top:1rem">📐 Features Dérivées</div>', unsafe_allow_html=True)
        fa, fb, fc = st.columns(3)
        fa.metric("ΔT Process-Air", f"{temp_diff:.2f} K")
        fb.metric("Puissance", f"{power_w:.0f} W")
        fc.metric("Surchauffe", "🔴 Oui" if overheat_flag else "🟢 Non")

        # Alertes seuils
        alerts = []
        if tool_wear > 200:
            alerts.append(("🚨 CRITIQUE", f"Usure outil très élevée : {tool_wear} min (seuil 200)", "crit"))
        if torque > 65:
            alerts.append(("🚨 CRITIQUE", f"Couple dangereux : {torque:.1f} Nm (seuil 65)", "crit"))
        if proc_temp > 312:
            alerts.append(("⚠️ ATTENTION", f"Température process élevée : {proc_temp:.1f} K (seuil 312)", "warn"))
        if rot_speed < 1300:
            alerts.append(("⚠️ ATTENTION", f"Vitesse trop basse : {rot_speed} rpm (seuil 1300)", "warn"))

        if alerts:
            st.markdown('<div class="section-title" style="margin-top:1rem">🚨 Alertes Seuils</div>', unsafe_allow_html=True)
            for lvl, msg, typ in alerts:
                css = "alert-crit" if typ=="crit" else "alert-box"
                st.markdown(f'<div class="{css}"><b>{lvl}</b> — {msg}</div>', unsafe_allow_html=True)

        predict_btn = st.button("🔍 Prédire la panne", use_container_width=True, type="primary")

    with col_out:
        st.markdown('<div class="section-title">📊 Résultat</div>', unsafe_allow_html=True)

        if predict_btn:
            X_raw    = np.array([[type_enc, air_temp, proc_temp, rot_speed, torque, tool_wear,
                                   temp_diff, power_w, torque_speed_ratio, wear_norm, overheat_flag]])
            X_scaled = scaler.transform(X_raw)
            pred     = int(model.predict(X_scaled)[0])
            proba    = model.predict_proba(X_scaled)[0]
            pf       = float(proba[1]) * 100

            if pred == 1:
                st.markdown(f'<div class="result-fail">❌ PANNE PRÉDITE<br>'
                            f'<span style="font-size:.85rem;font-weight:400">Probabilité : <b>{pf:.1f}%</b></span></div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-ok">✅ MACHINE NORMALE<br>'
                            f'<span style="font-size:.85rem;font-weight:400">Probabilité de panne : <b>{pf:.1f}%</b></span></div>',
                            unsafe_allow_html=True)

            # Jauge
            color = OCP_RED if pf >= 50 else OCP_GREEN
            st.markdown(f"""
            <div style="background:#e0e0e0;border-radius:6px;height:20px;overflow:hidden;margin:.7rem 0">
              <div style="width:{pf:.1f}%;background:{color};height:100%;display:flex;align-items:center;
                          justify-content:flex-end;padding-right:6px;color:white;font-size:.75rem;
                          font-weight:700;border-radius:6px">{pf:.1f}%</div>
            </div>""", unsafe_allow_html=True)

            m1,m2 = st.columns(2)
            m1.metric("✅ Normal", f"{proba[0]*100:.1f}%")
            m2.metric("❌ Panne",  f"{pf:.1f}%")

            # Recommandation
            if pf >= 70:
                st.error("🚨 Intervention immédiate — arrêter la machine.")
            elif pf >= 40:
                st.warning("⚠️ Inspection préventive dans les 24h.")
            elif pf >= 20:
                st.info("🔍 Surveillance conseillée.")
            else:
                st.success("✅ État normal — suivi périodique standard.")

            # Sauvegarde historique
            st.session_state.history.append({
                "Horodatage": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Type": type_letter, "Air Temp (K)": air_temp, "Proc Temp (K)": proc_temp,
                "RPM": rot_speed, "Torque (Nm)": torque, "Tool Wear (min)": tool_wear,
                "Prédiction": "Panne" if pred==1 else "Normal",
                "Proba Panne (%)": round(pf, 2),
            })

            # Stocker pour SHAP
            st.session_state['last_X_scaled'] = X_scaled
            st.session_state['last_X_raw']    = X_raw
        else:
            st.info("👈 Ajustez les paramètres puis cliquez sur **Prédire la panne**.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Visualisations & SHAP
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">📊 Visualisations & Interprétabilité SHAP</div>', unsafe_allow_html=True)

    if 'last_X_scaled' not in st.session_state:
        st.info("Effectuez d'abord une prédiction dans l'onglet **Prédiction manuelle**.")
    else:
        X_scaled = st.session_state['last_X_scaled']
        X_raw    = st.session_state['last_X_raw']

        v1, v2 = st.columns(2)

        # ── SHAP waterfall ──────────────────────────────────────────────────
        with v1:
            st.markdown("**🔬 Importance SHAP — dernière prédiction**")
            try:
                explainer  = shap.TreeExplainer(model)
                shap_vals  = explainer.shap_values(X_scaled)
                sv = shap_vals[0] if isinstance(shap_vals, list) else shap_vals[0]

                fig_shap, ax = plt.subplots(figsize=(6, 4))
                fig_shap.patch.set_alpha(0)
                ax.set_facecolor('#f7f8fa')
                colors = [OCP_RED if v > 0 else OCP_GREEN for v in sv]
                bars = ax.barh(FEAT_LABELS, sv, color=colors, edgecolor='white', height=0.6)
                ax.axvline(0, color='#333', linewidth=0.8)
                ax.set_xlabel("Valeur SHAP (impact sur la prédiction)", fontsize=8)
                ax.set_title("Contribution de chaque feature", fontsize=9, fontweight='bold')
                ax.tick_params(labelsize=7)
                p1 = mpatches.Patch(color=OCP_RED,   label='↑ Risque de panne')
                p2 = mpatches.Patch(color=OCP_GREEN, label='↓ Risque de panne')
                ax.legend(handles=[p1,p2], fontsize=7, loc='lower right')
                plt.tight_layout()
                st.pyplot(fig_shap)
            except Exception as e:
                st.warning(f"SHAP indisponible : {e}")

        # ── Radar capteurs ──────────────────────────────────────────────────
        with v2:
            st.markdown("**📡 Radar des capteurs (valeurs normalisées)**")
            raw = X_raw[0]
            labels_radar = ['Air Temp','Proc Temp','RPM','Torque','Tool Wear']
            ranges = [(295,305),(305,315),(1168,2886),(3.8,76.6),(0,253)]
            vals   = [(raw[i+1]-ranges[i][0])/(ranges[i][1]-ranges[i][0]) for i in range(5)]
            vals  += [vals[0]]
            angles = np.linspace(0, 2*np.pi, 5, endpoint=False).tolist()
            angles += angles[:1]

            fig_r, ax_r = plt.subplots(figsize=(4,4), subplot_kw=dict(polar=True))
            fig_r.patch.set_alpha(0)
            ax_r.set_facecolor('#f7f8fa')
            ax_r.plot(angles, vals, color=OCP_GREEN, linewidth=2)
            ax_r.fill(angles, vals, color=OCP_GREEN, alpha=0.25)
            ax_r.set_xticks(angles[:-1])
            ax_r.set_xticklabels(labels_radar, fontsize=8)
            ax_r.set_ylim(0,1)
            ax_r.set_yticks([0.25,0.5,0.75,1.0])
            ax_r.set_yticklabels(['25%','50%','75%','100%'], fontsize=6)
            ax_r.set_title("Position dans la plage nominale", fontsize=9, fontweight='bold', pad=12)
            plt.tight_layout()
            st.pyplot(fig_r)

        # ── Histogramme historique probabilités ────────────────────────────
        if len(st.session_state.history) >= 2:
            st.markdown("**📈 Évolution de la probabilité de panne (historique)**")
            df_h = pd.DataFrame(st.session_state.history)
            fig_ev, ax_ev = plt.subplots(figsize=(10, 3))
            fig_ev.patch.set_alpha(0)
            ax_ev.set_facecolor('#f7f8fa')
            colors_ev = [OCP_RED if p >= 50 else OCP_GOLD if p >= 20 else OCP_GREEN
                         for p in df_h["Proba Panne (%)"]]
            ax_ev.bar(range(len(df_h)), df_h["Proba Panne (%)"], color=colors_ev, edgecolor='white')
            ax_ev.axhline(50, color=OCP_RED,  linestyle='--', linewidth=1, label='Seuil panne (50%)')
            ax_ev.axhline(20, color=OCP_GOLD, linestyle='--', linewidth=1, label='Seuil vigilance (20%)')
            ax_ev.set_xlabel("Prédiction #", fontsize=8)
            ax_ev.set_ylabel("Probabilité de panne (%)", fontsize=8)
            ax_ev.set_title("Historique des probabilités de panne", fontsize=9, fontweight='bold')
            ax_ev.legend(fontsize=7)
            ax_ev.tick_params(labelsize=7)
            plt.tight_layout()
            st.pyplot(fig_ev)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Historique
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">📋 Historique des Prédictions</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("Aucune prédiction enregistrée. Effectuez des prédictions dans l'onglet **Prédiction manuelle**.")
    else:
        df_hist = pd.DataFrame(st.session_state.history)

        # KPIs
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Total prédictions", len(df_hist))
        n_pannes = (df_hist["Prédiction"]=="Panne").sum()
        k2.metric("Pannes détectées", n_pannes)
        k3.metric("Taux de panne", f"{n_pannes/len(df_hist)*100:.1f}%")
        k4.metric("Prob. moy. panne", f"{df_hist['Proba Panne (%)'].mean():.1f}%")

        # Tableau
        def color_pred(val):
            if val == "Panne":  return 'background-color:#fdecea;color:#C0392B;font-weight:700'
            return 'background-color:#e8f5ee;color:#007A4D;font-weight:700'
        st.dataframe(
            df_hist.style.applymap(color_pred, subset=["Prédiction"]),
            use_container_width=True, hide_index=True
        )

        # Export CSV
        csv_buf = io.StringIO()
        df_hist.to_csv(csv_buf, index=False)
        st.download_button(
            "📥 Exporter l'historique (CSV)",
            data=csv_buf.getvalue(),
            file_name=f"ocp_historique_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if st.button("🗑️ Effacer l'historique", use_container_width=True):
            st.session_state.history = []
            st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Batch CSV
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">🗂️ Prédictions en Masse (CSV)</div>', unsafe_allow_html=True)

    st.markdown("""
    Uploadez un CSV avec les colonnes suivantes :
    `Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min]`
    """)

    # Template download
    template = pd.DataFrame({
        "Type":                      ["H","M","L","H"],
        "Air temperature [K]":       [300.0, 298.5, 302.1, 301.0],
        "Process temperature [K]":   [310.0, 308.2, 313.5, 309.8],
        "Rotational speed [rpm]":    [1500, 1380, 2100, 1750],
        "Torque [Nm]":               [40.0, 55.2, 28.7, 61.3],
        "Tool wear [min]":           [100, 190, 50, 220],
    })
    buf_t = io.StringIO()
    template.to_csv(buf_t, index=False)
    st.download_button("📄 Télécharger le template CSV", buf_t.getvalue(),
                       "template_ocp.csv", "text/csv")

    uploaded = st.file_uploader("📂 Uploader votre CSV", type=["csv"])

    if uploaded:
        try:
            df_batch = pd.read_csv(uploaded)
            st.write(f"**{len(df_batch)} lignes chargées**")
            st.dataframe(df_batch.head(), use_container_width=True, hide_index=True)

            if st.button("🚀 Lancer les prédictions batch", type="primary", use_container_width=True):
                results = []
                for _, row in df_batch.iterrows():
                    t_enc = int(le.transform([str(row["Type"])[0]])[0])
                    at    = float(row["Air temperature [K]"])
                    pt    = float(row["Process temperature [K]"])
                    rs    = float(row["Rotational speed [rpm]"])
                    tq    = float(row["Torque [Nm]"])
                    tw    = float(row["Tool wear [min]"])
                    td    = pt - at
                    pw    = tq * (rs * 2 * np.pi / 60)
                    tsr   = tq / (rs + 1e-6)
                    wn    = tw / TOOL_WEAR_MAX
                    oh    = int(pt > 309 and rs < 1380)
                    X_r   = np.array([[t_enc,at,pt,rs,tq,tw,td,pw,tsr,wn,oh]])
                    X_s   = scaler.transform(X_r)
                    pred  = int(model.predict(X_s)[0])
                    prob  = float(model.predict_proba(X_s)[0][1])*100
                    results.append({"Prédiction": "Panne" if pred==1 else "Normal",
                                    "Proba Panne (%)": round(prob,2),
                                    "Alerte": "🚨" if prob>=70 else "⚠️" if prob>=40 else "✅"})

                df_res = pd.concat([df_batch.reset_index(drop=True),
                                    pd.DataFrame(results)], axis=1)

                # Stats batch
                b1,b2,b3 = st.columns(3)
                n_p = (pd.DataFrame(results)["Prédiction"]=="Panne").sum()
                b1.metric("Total machines", len(df_res))
                b2.metric("Pannes détectées", n_p)
                b3.metric("Taux de panne", f"{n_p/len(df_res)*100:.1f}%")

                def color_batch(val):
                    if val == "Panne": return 'background-color:#fdecea;color:#C0392B;font-weight:700'
                    if val == "Normal": return 'background-color:#e8f5ee;color:#007A4D;font-weight:700'
                    return ''
                st.dataframe(
                    df_res.style.applymap(color_batch, subset=["Prédiction"]),
                    use_container_width=True, hide_index=True
                )

                csv_out = io.StringIO()
                df_res.to_csv(csv_out, index=False)
                st.download_button(
                    "📥 Télécharger les résultats",
                    csv_out.getvalue(),
                    f"ocp_batch_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    "text/csv", use_container_width=True
                )
        except Exception as e:
            st.error(f"Erreur lors du traitement : {e}")
