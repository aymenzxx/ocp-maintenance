import streamlit as st
import numpy as np
import pandas as pd
import pickle, json, warnings
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="OCP — Maintenance Prédictive",
    page_icon="🏭", layout="wide", initial_sidebar_state="expanded",
)

OCP_GREEN  = "#006633"
OCP_ORANGE = "#FF6600"
OCP_RED    = "#D32F2F"
OCP_YELLOW = "#FFC107"

st.markdown(f"""<style>
  .ocp-header{{background:linear-gradient(135deg,{OCP_GREEN} 0%,#004d26 100%);
    padding:1.2rem 2rem;border-radius:12px;margin-bottom:1.5rem;}}
  .ocp-header h1{{color:white;margin:0;font-size:1.8rem;font-weight:700;}}
  .ocp-header p{{color:rgba(255,255,255,.8);margin:0;font-size:.95rem;}}
  .alert-critical{{background:#FFEBEE;border:2px solid {OCP_RED};border-radius:10px;padding:1rem;}}
  .alert-high{{background:#FFF3E0;border:2px solid {OCP_ORANGE};border-radius:10px;padding:1rem;}}
  .alert-moderate{{background:#FFFDE7;border:2px solid {OCP_YELLOW};border-radius:10px;padding:1rem;}}
  .alert-low{{background:#E8F5E9;border:2px solid {OCP_GREEN};border-radius:10px;padding:1rem;}}
  .section-title{{font-size:1.1rem;font-weight:600;color:{OCP_GREEN};
    border-bottom:2px solid {OCP_GREEN};padding-bottom:6px;margin-bottom:1rem;}}
  .risk-label{{text-align:center;font-size:1.4rem;font-weight:700;margin-top:-10px;}}
</style>""", unsafe_allow_html=True)

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

# ── HEURISTIQUE — toujours disponible, ne peut pas échouer
def heuristic_score(temp, vib, oil, cool, errors, last_m, fails, ai_ov) -> float:
    s  = float(np.clip((temp  - 40) / 60,  0, 1)) * 0.25
    s += float(np.clip(vib         / 30,   0, 1)) * 0.20
    s += float(np.clip((100 - oil) / 90,   0, 1)) * 0.15
    s += float(np.clip((100 - cool)/ 90,   0, 1)) * 0.08
    s += float(np.clip(errors      / 15,   0, 1)) * 0.18
    s += float(np.clip(last_m      / 400,  0, 1)) * 0.08
    s += float(np.clip(fails       / 8,    0, 1)) * 0.04
    s += float(np.clip(ai_ov       / 5,    0, 1)) * 0.02
    return float(np.clip(s, 0.02, 0.97))

def engineer_features(d):
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
        'Fluid_Degradation':          ((100 - float(d.get('Oil_Level_pct',80))) + (100 - float(d.get('Coolant_Level_pct',80)))) / 2,
        'Error_Rate':                 float(d.get('Error_Codes_Last_30_Days',0)) / 30,
        'Failure_Density':            float(d.get('Failure_History_Count',0)) / (oh / 1000 + 1),
        'High_Vibration_Flag':        int(float(d.get('Vibration_mms',5)) > 15),
        'Overheat_Flag':              int(float(d.get('Temperature_C',50)) > 80),
        'Late_Maintenance_Flag':      int(float(d.get('Last_Maintenance_Days_Ago',60)) > 180),
        'Machine_Type_Enc':           float(MACHINE_TYPE_MAP.get(str(d.get('Machine_Type','Pump')), 8)),
        'AI_Supervision_Int':         int(bool(d.get('AI_Supervision', False))),
    }
    return pd.DataFrame([row])[FEATURE_COLS]

# ── Charger le pkl (tolérant aux erreurs)
@st.cache_resource
def load_artifacts():
    """Retourne (model, preproc, threshold, status_msg)"""
    pkl_path = Path("predictive_maintenance_pipeline.pkl")
    if not pkl_path.exists():
        return None, None, 0.8817, "pkl introuvable"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open(pkl_path, "rb") as f:
                arts = pickle.load(f)
    except Exception as e:
        return None, None, 0.8817, f"Erreur chargement pkl: {e}"

    model     = arts.get("model")
    preproc   = arts.get("preprocessor")
    threshold = float(arts.get("threshold", 0.8817))

    # Tester si le preprocesseur fonctionne
    try:
        test = engineer_features({'Temperature_C':65,'Vibration_mms':8,'Oil_Level_pct':75,
            'Coolant_Level_pct':80,'Operational_Hours':45000,'Last_Maintenance_Days_Ago':90,
            'Maintenance_History_Count':5,'Failure_History_Count':2,'Error_Codes_Last_30_Days':2,
            'AI_Override_Events':1,'Machine_Type':'Pump','Installation_Year':2025})
        _ = preproc.transform(test)
        _ = model.predict_proba(preproc.transform(test))[0, 1]
        return model, preproc, threshold, "ok"
    except Exception as e:
        return None, None, threshold, f"Preprocesseur incompatible: {type(e).__name__}"

@st.cache_data
def load_metadata():
    p = Path("model_metadata.json")
    if p.exists():
        with open(p) as f: return json.load(f)
    return {"model_name":"Régression Logistique","roc_auc_test":0.9837,
            "avg_prec_test":0.7591,"f1_test":0.632,"optimal_threshold":0.8817,
            "n_features":28,"train_samples":276101,"training_date":"2040",
            "target":"Failure_Within_7_Days"}

def get_score(d, model, preproc):
    """Retourne (score_float, source_str). Ne lève JAMAIS d'exception."""
    # Toujours calculer l'heuristique d'abord (référence fiable)
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
        X     = engineer_features(d)
        X_p   = preproc.transform(X)
        score = float(model.predict_proba(X_p)[0, 1])
        return score, "modèle ML"
    except Exception:
        return h, "heuristique"

def classify_risk(p, thr):
    if p >= 0.80: return "🔴 CRITIQUE",  OCP_RED,    "alert-critical", "Arrêt immédiat + Maintenance d'urgence"
    if p >= 0.55: return "🟠 ÉLEVÉ",    OCP_ORANGE, "alert-high",     "Planifier maintenance sous 48h"
    if p >= thr:  return "🟡 MODÉRÉ",   OCP_YELLOW, "alert-moderate", "Surveillance renforcée + inspection préventive"
    return             "🟢 FAIBLE",   OCP_GREEN,  "alert-low",      "Fonctionnement normal — maintenance planifiée"

def make_gauge(proba, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(proba * 100, 1),
        number={"suffix":"%","font":{"size":38,"color":color}},
        gauge={
            "axis":{"range":[0,100],"tickfont":{"size":11}},
            "bar":{"color":color,"thickness":0.28},
            "bgcolor":"white",
            "steps":[{"range":[0,55],"color":"#E8F5E9"},
                     {"range":[55,80],"color":"#FFF3E0"},
                     {"range":[80,100],"color":"#FFEBEE"}],
            "threshold":{"line":{"color":OCP_RED,"width":3},"thickness":0.75,"value":80},
        },
        title={"text":"Score de Risque","font":{"size":15,"color":"#333"}},
    ))
    fig.update_layout(height=280, margin=dict(l=20,r=20,t=40,b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig

# ════════════════════════════════════════
model, preproc, DEFAULT_THRESHOLD, status = load_artifacts()
meta      = load_metadata()
threshold = DEFAULT_THRESHOLD or meta.get("optimal_threshold", 0.8817)
ml_active = (status == "ok")

st.markdown("""<div class="ocp-header">
  <h1>🏭 OCP — Maintenance Prédictive</h1>
  <p>Prédiction des pannes machines dans les 7 prochains jours · Office Chérifien des Phosphates</p>
</div>""", unsafe_allow_html=True)

if not ml_active:
    st.info(f"ℹ️ Mode estimation heuristique ({status}). Scores représentatifs basés sur 8 indicateurs capteurs.")

with st.sidebar:
    st.markdown(f"<div class='section-title'>📊 Modèle</div>", unsafe_allow_html=True)
    st.metric("Modèle", meta["model_name"])
    st.metric("ROC-AUC",      f"{meta['roc_auc_test']:.4f}")
    st.metric("F1-Score",     f"{meta['f1_test']:.3f}")
    st.metric("Seuil optimal",f"{threshold:.4f}")
    st.metric("Train samples",f"{meta['train_samples']:,}")
    st.divider()
    st.markdown("✅ Modèle ML" if ml_active else "⚙️ Mode heuristique")
    st.divider()
    page = st.radio("Navigation", ["🔬 Prédiction Individuelle","📋 Analyse par Lot","ℹ️ À propos"])

# ════ PAGE 1 ════
if page == "🔬 Prédiction Individuelle":
    st.markdown("<div class='section-title'>🔬 Saisie des données capteurs</div>", unsafe_allow_html=True)

    c0a, c0b, c0c, c0d = st.columns(4)
    machine_id     = c0a.text_input("Machine ID", "MC_OCP_0001")
    machine_type   = c0b.selectbox("Type de machine", list(MACHINE_TYPE_MAP.keys()), index=8)
    ai_supervision = c0c.checkbox("Supervision IA active", True)
    inst_year      = c0d.number_input("Année d'installation", 2000, 2040, 2025)

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
        oil     = st.slider("Niveau huile (%)",                   5.0, 100.0, 75.0, 1.0)
        coolant = st.slider("Niveau liquide refroidissement (%)", 5.0, 100.0, 80.0, 1.0)
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

    # ── Calcul EN TEMPS RÉEL (sans bouton) + bouton pour forcer le refresh visuel
    d = {
        "Machine_ID": machine_id, "Machine_Type": machine_type,
        "Installation_Year": inst_year, "Operational_Hours": op_hours,
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

    if st.button("🚀 Analyser le risque de panne", type="primary", use_container_width=True):
        proba, source = get_score(d, model, preproc)
        level, color, css_class, action = classify_risk(proba, threshold)

        r1, r2 = st.columns([1, 2])
        with r1:
            st.plotly_chart(make_gauge(proba, color), use_container_width=True)
            st.markdown(f"<div class='risk-label' style='color:{color}'>{level}</div>",
                        unsafe_allow_html=True)
            st.caption(f"Source : {source}")

        with r2:
            st.markdown(f"""
            <div class='{css_class}' style='margin-top:1rem;'>
              <h3 style='margin:0 0 8px 0;'>Machine : {machine_id} — {machine_type}</h3>
              <p style='margin:4px 0;'><strong>Score de risque :</strong> {proba:.1%}</p>
              <p style='margin:4px 0;'><strong>Niveau d'alerte :</strong> {level}</p>
              <p style='margin:4px 0;'><strong>Action recommandée :</strong> {action}</p>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>**Indicateurs clés**", unsafe_allow_html=True)
            k1,k2,k3,k4 = st.columns(4)
            k1.metric("Temp.",       f"{temp:.0f}°C",       delta="⚠️" if temp>80      else "✅")
            k2.metric("Vibration",   f"{vibration:.1f} mm/s",delta="⚠️" if vibration>15 else "✅")
            k3.metric("Huile",       f"{oil:.0f}%",          delta="⚠️" if oil<30       else "✅")
            k4.metric("Erreurs/30j", str(errors_30d),         delta="⚠️" if errors_30d>5 else "✅")

            st.markdown("<br>**Facteurs de risque**", unsafe_allow_html=True)
            factors = [
                ("🌡️ Température",      np.clip((temp-40)/60,  0,1)*0.25),
                ("📳 Vibration",         np.clip(vibration/30,  0,1)*0.20),
                ("🚨 Codes erreur",      np.clip(errors_30d/15, 0,1)*0.18),
                ("🛢️ Huile",            np.clip((100-oil)/90,  0,1)*0.15),
                ("❄️ Refroidissement",  np.clip((100-coolant)/90,0,1)*0.08),
                ("🔧 Retard maint.",    np.clip(last_maint/400,0,1)*0.08),
                ("💥 Pannes passées",   np.clip(fail_count/8,  0,1)*0.04),
                ("🤖 AI override",      np.clip(ai_events/5,   0,1)*0.02),
            ]
            for label, val in factors:
                pct = min(val / 0.25 * 100, 100)
                bar_col = OCP_RED if pct>70 else OCP_ORANGE if pct>40 else OCP_GREEN
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:8px;margin:3px 0;'>"
                    f"<span style='width:170px;font-size:.85rem;'>{label}</span>"
                    f"<div style='flex:1;background:#eee;border-radius:4px;height:13px;'>"
                    f"<div style='width:{pct:.0f}%;background:{bar_col};border-radius:4px;height:13px;'></div></div>"
                    f"<span style='width:36px;text-align:right;font-size:.8rem;color:#555;'>{val:.3f}</span>"
                    f"</div>", unsafe_allow_html=True)

# ════ PAGE 2 ════
elif page == "📋 Analyse par Lot":
    st.markdown("<div class='section-title'>📋 Simulation Fleet — Parc de machines OCP</div>",
                unsafe_allow_html=True)
    n_machines = st.slider("Nombre de machines", 5, 100, 30, 5)

    if st.button("🔄 Générer & Analyser", type="primary"):
        np.random.seed(42)
        mt = list(MACHINE_TYPE_MAP.keys())
        batch = [{
            "Machine_ID":                f"MC_OCP_{i:04d}",
            "Machine_Type":              np.random.choice(mt),
            "Operational_Hours":         int(np.random.randint(5000,100000)),
            "Temperature_C":             float(np.random.uniform(35,110)),
            "Vibration_mms":             float(np.random.uniform(1,30)),
            "Sound_dB":                  float(np.random.uniform(55,100)),
            "Oil_Level_pct":             float(np.random.uniform(5,100)),
            "Coolant_Level_pct":         float(np.random.uniform(10,100)),
            "Power_Consumption_kW":      float(np.random.uniform(30,250)),
            "Last_Maintenance_Days_Ago": int(np.random.randint(0,400)),
            "Maintenance_History_Count": int(np.random.randint(1,10)),
            "Failure_History_Count":     int(np.random.randint(0,8)),
            "AI_Supervision":            bool(np.random.choice([True,False])),
            "Error_Codes_Last_30_Days":  int(np.random.randint(0,15)),
            "AI_Override_Events":        int(np.random.randint(0,5)),
            "Installation_Year":         int(np.random.randint(2010,2038)),
        } for i in range(n_machines)]

        rows = []
        for d in batch:
            proba, _ = get_score(d, model, preproc)
            level, _, _, action = classify_risk(proba, threshold)
            rows.append({"Machine":d["Machine_ID"],"Type":d["Machine_Type"],
                         "Score":round(proba,4),"Niveau":level,"Action":action})

        res = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)

        k1,k2,k3,k4 = st.columns(4)
        k1.metric("🔴 Critiques", int((res["Score"]>=0.80).sum()))
        k2.metric("🟠 Élevés",   int(((res["Score"]>=0.55)&(res["Score"]<0.80)).sum()))
        k3.metric("🟡 Modérés",  int(((res["Score"]>=threshold)&(res["Score"]<0.55)).sum()))
        k4.metric("🟢 Faibles",  int((res["Score"]<threshold).sum()))

        colors = [OCP_RED if s>=0.80 else OCP_ORANGE if s>=0.55
                  else OCP_YELLOW if s>=threshold else OCP_GREEN for s in res["Score"]]
        fig = go.Figure(go.Bar(
            x=res["Score"], y=res["Machine"], orientation="h",
            marker_color=colors,
            text=[f"{s:.0%}" for s in res["Score"]], textposition="outside",
        ))
        fig.add_vline(x=threshold, line_dash="dash", line_color="gray",
                      annotation_text=f"Seuil {threshold:.2f}")
        fig.update_layout(height=max(350,len(res)*22),
                          xaxis=dict(range=[0,1.15],title="Score"),
                          margin=dict(l=120,r=60,t=30,b=40),
                          paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(res, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Télécharger CSV",
            res.to_csv(index=False).encode(), "ocp_risk_report.csv","text/csv")

# ════ PAGE 3 ════
elif page == "ℹ️ À propos":
    st.markdown("<div class='section-title'>ℹ️ À propos</div>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.markdown(f"""
**Objectif** : Prédire les pannes OCP dans les 7 prochains jours.

**Modèle** : {meta['model_name']}
- ROC-AUC : **{meta['roc_auc_test']:.4f}**
- F1-Score : **{meta['f1_test']:.3f}**
- Seuil optimal : **{threshold:.4f}**
- Train samples : **{meta['train_samples']:,}**

| Niveau | Score | Action |
|--------|-------|--------|
| 🔴 CRITIQUE | ≥ 80% | Arrêt immédiat |
| 🟠 ÉLEVÉ | 55–80% | Maintenance < 48h |
| 🟡 MODÉRÉ | seuil–55% | Surveillance renforcée |
| 🟢 FAIBLE | < seuil | Normal |
        """)
    with c2:
        st.markdown("""
**28 features** dont :
- `Thermal_Stress` = Temp × Vibration / 100
- `Maintenance_Urgency` = Jours / (Nb_maintenances + 1)
- `Fluid_Degradation` = dégradation huile + coolant
- `Failure_Density` = pannes / (heures / 1000)
- Flags : `Overheat`, `High_Vibration`, `Late_Maintenance`

**Sites** : Khouribga · Youssoufia · Gantour · Jorf Lasfar · Safi
        """)
