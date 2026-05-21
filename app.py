import streamlit as st
import pickle
import numpy as np
import pandas as pd
import base64
import io
import urllib.request
import urllib.parse
import json
from pathlib import Path
from datetime import datetime

# ── Resolve base directory
BASE_DIR = Path(__file__).parent

# ── Page config
st.set_page_config(
    page_title="OCP — Maintenance Prédictive",
    page_icon="🏭",
    layout="wide",
)

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
        width: 80px; height: 80px; flex-shrink: 0;
        border-radius: 50%;
        overflow: hidden;
        background: white;
        padding: 8px;
        box-sizing: border-box;
        box-shadow: 0 0 0 3px rgba(255,255,255,0.4);
        display: flex; align-items: center; justify-content: center;
    }}
    .main-header-logo img {{
        width: 100%; height: 100%;
        object-fit: contain;
        border-radius: 50%;
        display: block;
    }}
    .main-header-text h1 {{ margin: 0; font-size: 1.7rem; }}
    .main-header-text p  {{ margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.9rem; }}

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
</style>
""", unsafe_allow_html=True)

# ── Load model
@st.cache_resource
def load_model():
    with open(BASE_DIR / "predictive_maintenance_pipeline.pkl", "rb") as f:
        return pickle.load(f)

try:
    artifacts    = load_model()
    model        = artifacts["model"]
    preprocessor = artifacts["preprocessor"]
    FEATURE_COLS = artifacts["feature_cols"]
    THRESHOLD    = artifacts["threshold"]
    le           = artifacts["label_encoder"]
except Exception as e:
    st.error(f"❌ Erreur chargement modèle : {e}")
    st.stop()

MACHINE_TYPES = ['Conveyor_Belt', 'CNC_Lathe', 'Hydraulic_Press',
                 'Crusher', 'Flotation_Cell', 'Dryer', 'Mixer', 'Pump']

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
    X      = engineer_features(d)
    X_prep = preprocessor.transform(X)
    proba  = model.predict_proba(X_prep)[0, 1]
    if proba >= 0.80:
        level = "CRITIQUE";  css = "critique"; action = "Arret immediat + Maintenance d'urgence"
    elif proba >= 0.55:
        level = "ELEVE";     css = "eleve";    action = "Planifier maintenance sous 48h"
    elif proba >= THRESHOLD:
        level = "MODERE";    css = "modere";   action = "Surveillance renforcee + inspection preventive"
    else:
        level = "FAIBLE";    css = "faible";   action = "Fonctionnement normal - maintenance planifiee"
    return proba, level, css, action

# ════════════════════════════════════════════════
# PDF GENERATION
# ════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════
# SENDGRID EMAIL ALERT
# ═══════════════════════════════════════════════════════════
def send_alert_sendgrid(api_key: str, sender: str, sender_name: str,
                        recipients: list, machine_id: str, machine_type: str,
                        level: str, score_pct: float, action: str,
                        pdf_bytes: bytes = None, pdf_filename: str = None) -> tuple:
    """Send alert email via SendGrid API. No SMTP, no password needed."""

    level_colors = {"CRITIQUE": "#D32F2F", "ELEVE": "#E65100"}
    level_labels  = {"CRITIQUE": "🔴 CRITIQUE — Intervention Immédiate",
                     "ELEVE":    "🟠 ÉLEVÉ — Planifier Maintenance"}
    color  = level_colors.get(level, "#006633")
    label  = level_labels.get(level, level)
    now_str = datetime.now().strftime("%d/%m/%Y à %H:%M")

    html_body = f"""
    <html><body style="font-family:Arial,sans-serif;background:#f4f4f4;margin:0;padding:0">
    <div style="max-width:600px;margin:30px auto;background:#fff;border-radius:12px;overflow:hidden;box-shadow:0 2px 10px rgba(0,0,0,.12)">
      <div style="background:#006633;padding:24px 32px">
        <h1 style="color:#fff;margin:0;font-size:22px">⚠️ OCP — Alerte Maintenance Prédictive</h1>
        <p style="color:#a8d5b5;margin:6px 0 0;font-size:13px">Détection automatique · {now_str}</p>
      </div>
      <div style="background:{color};padding:18px 32px;text-align:center">
        <p style="color:#fff;margin:0;font-size:16px;font-weight:bold">{label}</p>
        <p style="color:#fff;margin:8px 0 0;font-size:36px;font-weight:bold;line-height:1">{score_pct:.1f}%</p>
        <p style="color:rgba(255,255,255,.8);margin:4px 0 0;font-size:12px">Score de risque de panne</p>
      </div>
      <div style="padding:28px 32px">
        <table style="width:100%;border-collapse:collapse;font-size:14px">
          <tr><td style="padding:10px 0;border-bottom:1px solid #eee;color:#555;width:40%"><b>🏭 Machine ID</b></td>
              <td style="padding:10px 0;border-bottom:1px solid #eee;font-weight:bold">{machine_id}</td></tr>
          <tr><td style="padding:10px 0;border-bottom:1px solid #eee;color:#555"><b>⚙️ Type</b></td>
              <td style="padding:10px 0;border-bottom:1px solid #eee">{machine_type}</td></tr>
          <tr><td style="padding:10px 0;border-bottom:1px solid #eee;color:#555"><b>📅 Détecté le</b></td>
              <td style="padding:10px 0;border-bottom:1px solid #eee">{now_str}</td></tr>
        </table>
        <div style="background:#fff8e1;border-left:4px solid {color};padding:16px 20px;margin:20px 0;border-radius:0 8px 8px 0">
          <p style="margin:0 0 4px;font-size:11px;color:#888;text-transform:uppercase;letter-spacing:1px">Action Recommandée</p>
          <p style="margin:0;font-size:15px;font-weight:bold;color:#333">{action}</p>
        </div>
        {"<p style='font-size:13px;color:#555'>📎 Rapport PDF joint à cet email.</p>" if pdf_bytes else ""}
        <hr style="border:none;border-top:1px solid #eee;margin:24px 0">
        <p style="font-size:11px;color:#aaa;margin:0">
          OCP Group — Système de Maintenance Prédictive · Message automatique, ne pas répondre.
        </p>
      </div>
    </div></body></html>"""

    to_list = [{"email": r} for r in recipients]
    payload = {
        "personalizations": [{"to": to_list}],
        "from": {"email": sender, "name": sender_name},
        "subject": f"[OCP ALERTE {level}] Machine {machine_id} — Score {score_pct:.1f}%",
        "content": [
            {"type": "text/plain",
             "value": (f"ALERTE {level} - Machine {machine_id} ({machine_type})\n"
                       f"Score de risque : {score_pct:.1f}%\n"
                       f"Action : {action}\n"
                       f"Detecte le {now_str}")},
            {"type": "text/html", "value": html_body},
        ],
    }

    if pdf_bytes and pdf_filename:
        payload["attachments"] = [{
            "content": base64.b64encode(pdf_bytes).decode(),
            "type": "application/pdf",
            "filename": pdf_filename,
            "disposition": "attachment",
        }]

    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        "https://api.sendgrid.com/v3/mail/send",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status in (200, 202):
                return True, "Email envoyé avec succès ✅"
            return False, f"SendGrid erreur HTTP {resp.status}"
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        try:
            errs = json.loads(body).get("errors", [])
            msg  = errs[0].get("message", body) if errs else body
        except Exception:
            msg = body[:200]
        return False, f"❌ SendGrid : {msg}"
    except Exception as e:
        return False, f"❌ Erreur réseau : {str(e)}"


def generate_pdf_single(data: dict, proba: float, level: str, action: str, flags: list) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, HRFlowable)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    # ── Styles
    styles = getSampleStyleSheet()
    ocp_green  = colors.HexColor("#006633")
    ocp_orange = colors.HexColor("#FF6600")
    level_colors = {
        "CRITIQUE": colors.HexColor("#D32F2F"),
        "ELEVE":    colors.HexColor("#FF6600"),
        "MODERE":   colors.HexColor("#FBC02D"),
        "FAIBLE":   colors.HexColor("#2E7D32"),
    }
    lc = level_colors.get(level, ocp_green)

    title_style = ParagraphStyle("title", fontSize=18, textColor=ocp_green,
                                  fontName="Helvetica-Bold", spaceAfter=4,
                                  alignment=TA_CENTER)
    sub_style   = ParagraphStyle("sub",   fontSize=10, textColor=colors.HexColor("#555555"),
                                  fontName="Helvetica", spaceAfter=2,
                                  alignment=TA_CENTER)
    section_style = ParagraphStyle("section", fontSize=12, textColor=ocp_green,
                                    fontName="Helvetica-Bold", spaceBefore=14, spaceAfter=6)
    normal_style  = ParagraphStyle("norm", fontSize=10, fontName="Helvetica",
                                    leading=14, textColor=colors.HexColor("#222222"))
    bold_style    = ParagraphStyle("bold", fontSize=10, fontName="Helvetica-Bold",
                                    textColor=colors.HexColor("#111111"))

    story = []

    # ── En-tête
    story.append(Paragraph("OCP GROUP", title_style))
    story.append(Paragraph("Rapport de Maintenance Predictive", sub_style))
    story.append(Paragraph(f"Genere le : {datetime.now().strftime('%d/%m/%Y a %H:%M')}", sub_style))
    story.append(HRFlowable(width="100%", thickness=2, color=ocp_green, spaceAfter=12))

    # ── Niveau de risque (bandeau coloré)
    pct = proba * 100
    risk_data = [[
        Paragraph(f"NIVEAU DE RISQUE : {level}", ParagraphStyle(
            "risk", fontSize=14, fontName="Helvetica-Bold",
            textColor=colors.white, alignment=TA_CENTER)),
        Paragraph(f"Score : {pct:.1f}%", ParagraphStyle(
            "score", fontSize=14, fontName="Helvetica-Bold",
            textColor=colors.white, alignment=TA_CENTER)),
    ]]
    risk_table = Table(risk_data, colWidths=["60%","40%"])
    risk_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), lc),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [lc]),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ("RIGHTPADDING",  (0,0), (-1,-1), 12),
        ("ROUNDEDCORNERS", (0,0), (-1,-1), [6, 6, 6, 6]),
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"Action recommandee : {action}", ParagraphStyle(
        "act", fontSize=11, fontName="Helvetica-Bold",
        textColor=lc, spaceBefore=4, spaceAfter=10)))

    # ── Identite machine
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#dddddd")))
    story.append(Paragraph("Identite de la Machine", section_style))
    id_data = [
        ["ID Machine",     data.get("Machine_ID","—"),
         "Type",           data.get("Machine_Type","—")],
        ["Age",            f"{data.get('machine_age','—')} ans",
         "Heures oper.",   f"{data.get('Operational_Hours',0):,} h"],
        ["Supervision IA", "Oui" if data.get("AI_Supervision") else "Non",
         "AI Overrides",   str(data.get("AI_Override_Events","—"))],
    ]
    t = Table(id_data, colWidths=["25%","25%","25%","25%"])
    t.setStyle(TableStyle([
        ("FONTNAME",    (0,0), (-1,-1), "Helvetica"),
        ("FONTNAME",    (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTNAME",    (2,0), (2,-1), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("TEXTCOLOR",   (0,0), (0,-1), ocp_green),
        ("TEXTCOLOR",   (2,0), (2,-1), ocp_green),
        ("BACKGROUND",  (0,0), (-1,-1), colors.HexColor("#F5F5F5")),
        ("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.HexColor("#F5F5F5"), colors.white]),
        ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#DDDDDD")),
        ("TOPPADDING",  (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0), (-1,-1), 6),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
    ]))
    story.append(t)

    # ── Capteurs
    story.append(Paragraph("Donnees Capteurs", section_style))
    sensor_rows = [
        ["Parametre", "Valeur", "Statut"],
        ["Temperature",       f"{data.get('Temperature_C',0):.1f} C",
         "ALERTE" if data.get('Temperature_C',0) > 80 else "Normal"],
        ["Vibration",         f"{data.get('Vibration_mms',0):.1f} mm/s",
         "ALERTE" if data.get('Vibration_mms',0) > 15 else "Normal"],
        ["Son",               f"{data.get('Sound_dB',0):.1f} dB",     "—"],
        ["Consommation",      f"{data.get('Power_Consumption_kW',0):.1f} kW", "—"],
        ["Niveau Huile",      f"{data.get('Oil_Level_pct',0):.1f} %",
         "ALERTE" if data.get('Oil_Level_pct',0) < 30 else "Normal"],
        ["Liquide Refroid.",  f"{data.get('Coolant_Level_pct',0):.1f} %",
         "ALERTE" if data.get('Coolant_Level_pct',0) < 30 else "Normal"],
    ]
    sensor_t = Table(sensor_rows, colWidths=["40%","30%","30%"])
    sensor_style = TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), ocp_green),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME",    (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#DDDDDD")),
        ("TOPPADDING",  (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.HexColor("#F9F9F9")]),
        ("ALIGN",       (1,0), (-1,-1), "CENTER"),
    ])
    # Color ALERTE cells red
    for i, row in enumerate(sensor_rows[1:], 1):
        if row[2] == "ALERTE":
            sensor_style.add("TEXTCOLOR",   (2,i), (2,i), colors.HexColor("#D32F2F"))
            sensor_style.add("FONTNAME",    (2,i), (2,i), "Helvetica-Bold")
    sensor_t.setStyle(sensor_style)
    story.append(sensor_t)

    # ── Maintenance
    story.append(Paragraph("Historique Maintenance", section_style))
    maint_data = [
        ["Parametre", "Valeur"],
        ["Jours depuis derniere maintenance", str(data.get("Last_Maintenance_Days_Ago","—"))],
        ["Nombre de maintenances",            str(data.get("Maintenance_History_Count","—"))],
        ["Nombre de pannes historique",       str(data.get("Failure_History_Count","—"))],
        ["Codes erreur (30 derniers jours)",  str(data.get("Error_Codes_Last_30_Days","—"))],
    ]
    mt = Table(maint_data, colWidths=["60%","40%"])
    mt.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), ocp_green),
        ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTNAME",    (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#DDDDDD")),
        ("TOPPADDING",  (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0),(-1,-1), 5),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.HexColor("#F9F9F9")]),
        ("ALIGN",       (1,0), (-1,-1), "CENTER"),
    ]))
    story.append(mt)

    # ── Alertes actives
    if flags:
        story.append(Paragraph("Alertes Actives", section_style))
        for flag in flags:
            # strip emoji for PDF (ReportLab can't render them)
            clean = flag.replace("🔴","[CRITIQUE]").replace("🟠","[ELEVE]").replace("🟡","[MODERE]")
            story.append(Paragraph(f"• {clean}", ParagraphStyle(
                "flag", fontSize=10, fontName="Helvetica", textColor=colors.HexColor("#B71C1C"),
                leftIndent=10, spaceAfter=3)))

    # ── Pied de page
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CCCCCC")))
    story.append(Paragraph(
        "OCP Group - Systeme de Maintenance Predictive | Document confidentiel",
        ParagraphStyle("footer", fontSize=8, textColor=colors.HexColor("#999999"),
                       alignment=TA_CENTER, spaceBefore=6)))

    doc.build(story)
    return buf.getvalue()


def generate_pdf_fleet(df_res: pd.DataFrame, n_total: int,
                        critiques: int, eleves: int, normaux: int) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Table, TableStyle, HRFlowable)
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=1.5*cm, rightMargin=1.5*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    ocp_green = colors.HexColor("#006633")
    styles_map = {
        "CRITIQUE": colors.HexColor("#FFCDD2"),
        "ELEVE":    colors.HexColor("#FFE0B2"),
        "MODERE":   colors.HexColor("#FFF9C4"),
        "FAIBLE":   colors.HexColor("#C8E6C9"),
    }
    fg_map = {
        "CRITIQUE": colors.HexColor("#7f0000"),
        "ELEVE":    colors.HexColor("#bf360c"),
        "MODERE":   colors.HexColor("#827717"),
        "FAIBLE":   colors.HexColor("#1b5e20"),
    }

    story = []

    story.append(Paragraph("OCP GROUP", ParagraphStyle(
        "t", fontSize=18, textColor=ocp_green, fontName="Helvetica-Bold",
        spaceAfter=4, alignment=TA_CENTER)))
    story.append(Paragraph("Rapport Flotte - Simulation Maintenance Predictive",
        ParagraphStyle("s", fontSize=11, textColor=colors.HexColor("#555"),
                       fontName="Helvetica", alignment=TA_CENTER, spaceAfter=2)))
    story.append(Paragraph(f"Genere le : {datetime.now().strftime('%d/%m/%Y a %H:%M')}",
        ParagraphStyle("d", fontSize=9, textColor=colors.HexColor("#888"),
                       fontName="Helvetica", alignment=TA_CENTER, spaceAfter=10)))
    story.append(HRFlowable(width="100%", thickness=2, color=ocp_green, spaceAfter=12))

    # KPI row
    kpi_data = [[
        Paragraph(f"<b>{n_total}</b><br/>Machines Analysees",
            ParagraphStyle("k", fontSize=11, fontName="Helvetica", alignment=TA_CENTER, textColor=colors.HexColor("#333"))),
        Paragraph(f"<b>{critiques}</b><br/>Critiques",
            ParagraphStyle("k2", fontSize=11, fontName="Helvetica", alignment=TA_CENTER, textColor=colors.HexColor("#D32F2F"))),
        Paragraph(f"<b>{eleves}</b><br/>Niveau Eleve",
            ParagraphStyle("k3", fontSize=11, fontName="Helvetica", alignment=TA_CENTER, textColor=colors.HexColor("#E65100"))),
        Paragraph(f"<b>{normaux}</b><br/>Normal",
            ParagraphStyle("k4", fontSize=11, fontName="Helvetica", alignment=TA_CENTER, textColor=colors.HexColor("#2E7D32"))),
    ]]
    kpi_t = Table(kpi_data, colWidths=["25%","25%","25%","25%"])
    kpi_t.setStyle(TableStyle([
        ("BACKGROUND",      (0,0),(-1,-1), colors.HexColor("#F5F5F5")),
        ("GRID",            (0,0),(-1,-1), 0.5, colors.HexColor("#DDDDDD")),
        ("TOPPADDING",      (0,0),(-1,-1), 12),
        ("BOTTOMPADDING",   (0,0),(-1,-1), 12),
    ]))
    story.append(kpi_t)
    story.append(Spacer(1, 14))

    # Machines table
    story.append(Paragraph("Detail par Machine", ParagraphStyle(
        "sec", fontSize=12, textColor=ocp_green, fontName="Helvetica-Bold",
        spaceBefore=6, spaceAfter=8)))

    header = [["Machine ID", "Type", "Score (%)", "Niveau", "Action Recommandee"]]
    rows = []
    for _, r in df_res.iterrows():
        niveau_clean = r["Niveau"].replace("🔴 ","").replace("🟠 ","").replace("🟡 ","").replace("🟢 ","")
        rows.append([r["Machine_ID"], r["Type"],
                     f"{r['Score (%)']:.1f}%", niveau_clean, r["Action"]])

    table_data = header + rows
    col_w = ["20%","18%","14%","14%","34%"]
    data_t = Table(table_data, colWidths=col_w, repeatRows=1)
    ts = TableStyle([
        ("BACKGROUND",      (0,0),(-1,0), ocp_green),
        ("TEXTCOLOR",       (0,0),(-1,0), colors.white),
        ("FONTNAME",        (0,0),(-1,0), "Helvetica-Bold"),
        ("FONTNAME",        (0,1),(-1,-1), "Helvetica"),
        ("FONTSIZE",        (0,0),(-1,-1), 8),
        ("GRID",            (0,0),(-1,-1), 0.3, colors.HexColor("#CCCCCC")),
        ("TOPPADDING",      (0,0),(-1,-1), 5),
        ("BOTTOMPADDING",   (0,0),(-1,-1), 5),
        ("LEFTPADDING",     (0,0),(-1,-1), 6),
        ("ALIGN",           (2,0),(3,-1), "CENTER"),
        ("VALIGN",          (0,0),(-1,-1), "MIDDLE"),
    ])
    # Color each data row by level
    level_order = {"CRITIQUE":0,"ELEVE":1,"MODERE":2,"FAIBLE":3}
    for i, r in enumerate(rows, 1):
        niv = r[3]
        bg = styles_map.get(niv, colors.white)
        fg = fg_map.get(niv, colors.black)
        ts.add("BACKGROUND", (0,i),(-1,i), bg)
        ts.add("TEXTCOLOR",  (3,i),(3,i), fg)
        ts.add("FONTNAME",   (3,i),(3,i), "Helvetica-Bold")
    data_t.setStyle(ts)
    story.append(data_t)

    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CCCCCC")))
    story.append(Paragraph(
        "OCP Group - Systeme de Maintenance Predictive | Document confidentiel",
        ParagraphStyle("footer", fontSize=8, textColor=colors.HexColor("#999"),
                       alignment=TA_CENTER, spaceBefore=6, fontName="Helvetica")))

    doc.build(story)
    return buf.getvalue()


# ── Logo
_logo_html = ""
_logo_path = BASE_DIR / "ocp_logo.png"
if _logo_path.exists():
    _logo_b64 = base64.b64encode(_logo_path.read_bytes()).decode()
    _logo_html = f'<div class="main-header-logo"><img src="data:image/png;base64,{_logo_b64}" alt="OCP"/></div>'

st.markdown(f"""
<div class="main-header">
  {_logo_html}
  <div class="main-header-text">
    <h1>OCP — Systeme de Maintenance Predictive</h1>
    <p>Prediction de pannes machines dans les 7 prochains jours · Groupe OCP</p>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# SIDEBAR — Configuration SendGrid
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.image(str(BASE_DIR / "ocp_logo.png"), width=90) if (BASE_DIR / "ocp_logo.png").exists() else None
    st.markdown("## 📧 Alertes Email")
    st.markdown("---")

    email_enabled = st.toggle("Activer les alertes email", value=False)

    if email_enabled:
        sg_api_key = st.text_input(
            "🔑 SendGrid API Key",
            type="password",
            placeholder="SG.xxxxxxxxxxxxxxxx",
            help="Créez une clé sur sendgrid.com → Settings → API Keys (gratuit, 100 emails/jour)"
        )
        sender_email = st.text_input(
            "📤 Email expéditeur",
            placeholder="alertes@ocp.ma",
            help="Doit être vérifié dans SendGrid (Sender Authentication)"
        )
        sender_name = st.text_input("Nom expéditeur", value="OCP Maintenance")
        recipients_raw = st.text_area(
            "📬 Destinataires (un par ligne)",
            placeholder="responsable@ocp.ma\nmaintenance@ocp.ma",
            height=90,
        )
        recipients = [r.strip() for r in recipients_raw.splitlines() if r.strip()]
        attach_pdf = st.checkbox("📎 Joindre le rapport PDF", value=True)

        st.markdown("**Déclenchement :** 🔴 CRITIQUE · 🟠 ÉLEVÉ")

        with st.expander("ℹ️ Comment configurer SendGrid ?"):
            st.markdown("""
1. Créez un compte sur [sendgrid.com](https://sendgrid.com) (gratuit)
2. **Settings → API Keys → Create API Key**
3. Permission : *Mail Send* suffit
4. **Settings → Sender Authentication** → vérifiez votre email expéditeur
5. Copiez la clé ici — aucun mot de passe Gmail requis
""")

        st.markdown("---")
        if st.button("🔌 Envoyer un email de test", use_container_width=True):
            if not sg_api_key or not sender_email or not recipients:
                st.error("Remplissez la clé API, l'expéditeur et au moins un destinataire.")
            else:
                with st.spinner("Envoi en cours..."):
                    ok, msg = send_alert_sendgrid(
                        sg_api_key, sender_email, sender_name, recipients[:1],
                        "MC_OCP_TEST", "Test", "CRITIQUE", 94.2,
                        "Ceci est un email de test — configuration réussie.",
                        None, None)
                st.success(msg) if ok else st.error(msg)
    else:
        sg_api_key = sender_email = sender_name = attach_pdf = None
        recipients = []
        st.caption("Activez le toggle pour configurer SendGrid.")

tab1, tab2 = st.tabs(["🔍 Analyse Machine", "📊 Simulation Flotte"])

# ═══════════════════════════════
# TAB 1
# ═══════════════════════════════
with tab1:
    col_left, col_right = st.columns([1.1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-title">⚙️ Données Machine</div>', unsafe_allow_html=True)
        machine_id   = st.text_input("ID Machine", "MC_OCP_4521")
        machine_type = st.selectbox("Type de Machine", MACHINE_TYPES, index=1)
        machine_age  = st.slider("Âge de la machine (années)", 1, 35, 15)
        install_year = 2040 - machine_age
        op_hours     = st.number_input("Heures opérationnelles", 0, 200000, 85000, step=500)

        st.markdown('<div class="section-title" style="margin-top:1.2rem">🌡️ Capteurs Temps Réel</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        temp      = c1.slider("Température (°C)",   20.0, 120.0, 87.5, 0.5)
        vibration = c2.slider("Vibration (mm/s)",    0.0,  40.0, 18.3, 0.1)
        sound     = c1.slider("Son (dB)",            40.0, 120.0, 91.0, 0.5)
        power     = c2.slider("Consommation (kW)",   10.0, 300.0,145.0, 1.0)
        oil       = c1.slider("Niveau Huile (%)",     0.0, 100.0, 22.0, 1.0)
        coolant   = c2.slider("Liquide Refroid. (%)", 0.0, 100.0, 31.0, 1.0)

        st.markdown('<div class="section-title" style="margin-top:1.2rem">🔧 Historique Maintenance</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        last_maint   = c3.number_input("Jours depuis maintenance", 0, 500, 210)
        maint_count  = c4.number_input("Nb maintenances",          0,  30,   3)
        fail_count   = c3.number_input("Nb pannes historique",     0,  20,   4)
        error_codes  = c4.number_input("Codes erreur (30j)",       0,  50,   7)
        ai_sup       = st.checkbox("Supervision IA active", value=True)
        ai_overrides = st.number_input("Événements AI Override", 0, 20, 3)

    with col_right:
        st.markdown('<div class="section-title">🎯 Résultat de Prédiction</div>', unsafe_allow_html=True)

        machine_data = {
            "Machine_ID": machine_id, "Machine_Type": machine_type,
            "Installation_Year": install_year, "Operational_Hours": op_hours,
            "machine_age": machine_age,
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

        angle    = -90 + 180 * proba
        needle_x = 100 + 75 * np.cos(np.radians(angle - 90))
        needle_y = 100 - 75 * np.sin(np.radians(angle - 90)) + 30

        st.markdown(f"""
        <svg viewBox="0 0 200 140" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:280px;display:block;margin:0 auto">
          <path d="M 25 130 A 75 75 0 0 1 62 57"  fill="none" stroke="#1B5E20" stroke-width="16" stroke-linecap="round"/>
          <path d="M 62 57  A 75 75 0 0 1 100 25" fill="none" stroke="#FBC02D" stroke-width="16" stroke-linecap="round"/>
          <path d="M 100 25 A 75 75 0 0 1 138 57" fill="none" stroke="#FF6600" stroke-width="16" stroke-linecap="round"/>
          <path d="M 138 57 A 75 75 0 0 1 175 130" fill="none" stroke="#D32F2F" stroke-width="16" stroke-linecap="round"/>
          <line x1="100" y1="130" x2="{needle_x:.1f}" y2="{needle_y:.1f}" stroke="#333" stroke-width="3" stroke-linecap="round"/>
          <circle cx="100" cy="130" r="6" fill="#333"/>
          <text x="100" y="118" text-anchor="middle" font-size="22" font-weight="bold" fill="#1a1a1a">{pct:.1f}%</text>
          <text x="100" y="138" text-anchor="middle" font-size="9" fill="#666">Score de Risque</text>
        </svg>
        """, unsafe_allow_html=True)

        emoji_map = {"CRITIQUE":"🔴","ELEVE":"🟠","MODERE":"🟡","FAIBLE":"🟢"}
        st.markdown(f"""
        <div class="alert-box alert-{css}" style="margin-top:1rem">
          <div style="font-size:1.3rem">{emoji_map[level]} {level}</div>
          <div style="margin-top:0.4rem;font-size:0.95rem">{action}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        km1, km2 = st.columns(2)
        km1.metric("🌡️ Température",    f"{temp}°C",         delta=f"+{temp-75:.0f}°C vs normal" if temp > 75 else None)
        km2.metric("📳 Vibration",      f"{vibration} mm/s", delta=f"+{vibration-15:.1f}" if vibration > 15 else None)
        km1.metric("🔧 Âge Machine",    f"{machine_age} ans")
        km2.metric("📅 Dernière maint.", f"{last_maint}j",   delta="Retard" if last_maint > 180 else None, delta_color="inverse")

        st.markdown('<div class="section-title" style="margin-top:1rem">⚠️ Indicateurs d\'Alerte</div>', unsafe_allow_html=True)
        flags = []
        if temp > 80:        flags.append("🔴 Surchauffe détectée")
        if vibration > 15:   flags.append("🔴 Vibration élevée")
        if last_maint > 180: flags.append("🟠 Maintenance en retard")
        if oil < 30:         flags.append("🟠 Niveau huile critique")
        if coolant < 30:     flags.append("🟠 Liquide refroidissement bas")
        if error_codes > 5:  flags.append("🟡 Codes erreur fréquents")
        if not flags:
            st.success("✅ Aucune alerte active — tous paramètres normaux")
        for f in flags:
            st.warning(f)

        # ── PDF machine : cached pour éviter régénération à chaque rerender
        st.markdown("---")
        _pdf_key = f"single_pdf_{machine_id}_{level}"
        if _pdf_key not in st.session_state:
            try:
                st.session_state[_pdf_key] = generate_pdf_single(
                    machine_data, proba, level, action, flags)
            except Exception as _e:
                st.error(f"❌ Erreur PDF : {_e}")

        if _pdf_key in st.session_state:
            st.download_button(
                label="📄 Télécharger le Rapport PDF",
                data=st.session_state[_pdf_key],
                file_name=f"rapport_OCP_{machine_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )

        # ── Alerte email automatique (CRITIQUE ou ELEVE)
        if email_enabled and sg_api_key and sender_email and recipients and level in ("CRITIQUE", "ELEVE"):
            _alert_key = f"alert_sent_{machine_id}_{level}"
            if _alert_key not in st.session_state:
                with st.sidebar:
                    with st.spinner(f"Envoi alerte pour {machine_id}..."):
                        _pdf_data  = st.session_state.get(_pdf_key) if attach_pdf else None
                        _pdf_fname = f"rapport_OCP_{machine_id}.pdf" if _pdf_data else None
                        ok_e, msg_e = send_alert_sendgrid(
                            sg_api_key, sender_email, sender_name, recipients,
                            machine_id, machine_type, level, pct, action,
                            _pdf_data, _pdf_fname)
                st.session_state[_alert_key] = (ok_e, msg_e)
            _ok_e, _msg_e = st.session_state[f"alert_sent_{machine_id}_{level}"]
            with st.sidebar:
                st.success(f"✅ Alerte envoyée — {machine_id}") if _ok_e else st.error(_msg_e)

# ═══════════════════════════════
# TAB 2
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
            proba2, level2, css2, action2 = predict(row.to_dict())
            emoji2 = {"CRITIQUE":"🔴","ELEVE":"🟠","MODERE":"🟡","FAIBLE":"🟢"}[level2]
            results.append({"Machine_ID": row["Machine_ID"],
                             "Type": row["Machine_Type"],
                             "Score (%)": round(proba2 * 100, 1),
                             "Niveau": f"{emoji2} {level2}",
                             "Action": action2,
                             "_level": level2})
        df_res = pd.DataFrame(results).sort_values("Score (%)", ascending=False)

        # ── Sauvegarde dans session_state pour survivre aux re-renders
        st.session_state["fleet_df"]       = df_res
        st.session_state["fleet_total"]    = len(df_res)
        st.session_state["fleet_critiques"]= int((df_res["_level"] == "CRITIQUE").sum())
        st.session_state["fleet_eleves"]   = int((df_res["_level"] == "ELEVE").sum())
        st.session_state["fleet_normaux"]  = int((df_res["_level"] == "FAIBLE").sum())

    # ── Affichage des résultats (persistent via session_state)
    if "fleet_df" in st.session_state:
        df_res    = st.session_state["fleet_df"]
        total     = st.session_state["fleet_total"]
        critiques = st.session_state["fleet_critiques"]
        eleves    = st.session_state["fleet_eleves"]
        normaux   = st.session_state["fleet_normaux"]

        k1,k2,k3,k4 = st.columns(4)
        k1.metric("🏭 Machines Analysées", total)
        k2.metric("🔴 CRITIQUES", critiques)
        k3.metric("🟠 ÉLEVÉ",     eleves)
        k4.metric("🟢 Normal",    normaux)

        st.markdown("---")

        def color_row(row):
            lv = row["_level"]
            maps = {"CRITIQUE":("#FFCDD2","#7f0000"),"ELEVE":("#FFE0B2","#bf360c"),
                    "MODERE":("#FFF9C4","#827717"),   "FAIBLE":("#C8E6C9","#1b5e20")}
            bg, fg = maps.get(lv, ("#ffffff","#111111"))
            return [f"background-color:{bg};color:{fg};font-weight:600;"] * len(row)

        display_cols = ["Machine_ID","Type","Score (%)","Niveau","Action"]
        st.dataframe(
            df_res[display_cols + ["_level"]].style.apply(color_row, axis=1)
                   .hide(axis="columns", subset=["_level"]),
            use_container_width=True,
            height=420,
        )

        urgents = df_res[df_res["_level"] == "CRITIQUE"]
        if not urgents.empty:
            st.error(f"🚨 {len(urgents)} machine(s) nécessitent une intervention immédiate !")
            st.dataframe(urgents[["Machine_ID","Type","Score (%)","Action"]],
                         use_container_width=True)
        else:
            st.success("✅ Aucune machine en état critique dans cet échantillon.")

        # ── PDF flotte : génération + download
        st.markdown("---")
        if "fleet_pdf" not in st.session_state or st.session_state.get("fleet_pdf_n") != total:
            with st.spinner("Génération du rapport PDF flotte..."):
                try:
                    _fleet_pdf = generate_pdf_fleet(
                        df_res[display_cols + ["_level"]],
                        total, critiques, eleves, normaux)
                    st.session_state["fleet_pdf"]       = _fleet_pdf
                    st.session_state["fleet_pdf_fname"] = f"rapport_OCP_flotte_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                    st.session_state["fleet_pdf_n"]     = total
                except Exception as _e:
                    st.error(f"❌ Erreur PDF flotte : {_e}")

        if "fleet_pdf" in st.session_state:
            st.download_button(
                label="📄 Télécharger le Rapport PDF Flotte",
                data=st.session_state["fleet_pdf"],
                file_name=st.session_state["fleet_pdf_fname"],
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )

        # ── Alertes email flotte (CRITIQUE + ELEVE)
        if email_enabled and sg_api_key and sender_email and recipients:
            _fleet_alert_key = f"fleet_alert_{total}"
            if _fleet_alert_key not in st.session_state:
                _targets = df_res[df_res["_level"].isin(["CRITIQUE", "ELEVE"])]
                if not _targets.empty:
                    with st.sidebar:
                        with st.spinner(f"Envoi de {len(_targets)} alertes email..."):
                            _sent, _failed = 0, []
                            for _, _row in _targets.iterrows():
                                _ok_f, _msg_f = send_alert_sendgrid(
                                    sg_api_key, sender_email, sender_name, recipients,
                                    _row["Machine_ID"], _row["Type"], _row["_level"],
                                    float(_row["Score (%)"]), _row["Action"],
                                    None, None)
                                if _ok_f:
                                    _sent += 1
                                else:
                                    _failed.append(_msg_f)
                    st.session_state[_fleet_alert_key] = (_sent, list(set(_failed)))

            if _fleet_alert_key in st.session_state:
                _s, _f = st.session_state[_fleet_alert_key]
                with st.sidebar:
                    if _s:
                        st.success(f"✅ {_s} alertes email envoyées")
                    for _err in _f:
                        st.error(_err)

# ── Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#999;font-size:0.8rem'>"
    "OCP Group — Système de Maintenance Prédictive · Modèle scikit-learn · ROC-AUC > 0.95"
    "</div>",
    unsafe_allow_html=True,
)