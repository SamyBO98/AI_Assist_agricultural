import io
import tempfile
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image
)
from reportlab.lib import colors

from config import PALETTE

# Raccourcis
VERT        = PALETTE["vert"]
VERT_CLAIR  = PALETTE["vert_clair"]
ROUGE       = PALETTE["rouge"]
ROUGE_CLAIR = PALETTE["rouge_clair"]
ORANGE      = PALETTE["orange"]
ORANGE_CLAIR= PALETTE["orange_clair"]
BLEU        = PALETTE["bleu"]
GRIS_FOND   = PALETTE["gris_fond"]
GRIS_BORD   = PALETTE["gris_bord"]
BLANC       = PALETTE["blanc"]
NOIR        = PALETTE["noir"]

W, H = A4



def get_styles():
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        "Titre",
        parent=styles["Title"],
        fontSize=22,
        textColor=VERT,
        spaceAfter=4,
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        "Sous_titre",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#757575"),
        spaceAfter=12,
        alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=VERT,
        spaceBefore=14,
        spaceAfter=6,
        fontName="Helvetica-Bold",
        borderPad=4,
    ))
    styles.add(ParagraphStyle(
        "Normal_custom",
        parent=styles["Normal"],
        fontSize=10,
        textColor=NOIR,
        spaceAfter=4,
        leading=14,
    ))
    styles.add(ParagraphStyle(
        "Alerte",
        parent=styles["Normal"],
        fontSize=10,
        textColor=ROUGE,
        spaceAfter=3,
        leftIndent=10,
    ))
    styles.add(ParagraphStyle(
        "Conseil",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#1b5e20"),
        spaceAfter=3,
        leftIndent=10,
    ))
    styles.add(ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#9e9e9e"),
        alignment=TA_CENTER,
        spaceAfter=0,
    ))
    return styles



def hr(color=GRIS_BORD):
    return HRFlowable(width="100%", thickness=1, color=color, spaceAfter=8, spaceBefore=4)


def table_indicateurs(data, col_widths=None):
    """data = liste de [label, valeur, unite]"""
    rows = [["Indicateur", "Valeur", "Unité"]]
    rows += [[d[0], d[1], d[2]] for d in data]
    col_widths = col_widths or [90*mm, 40*mm, 30*mm]
    t = Table(rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  VERT),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  BLANC),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0),  10),
        ("ALIGN",        (0, 0), (-1, -1), "LEFT"),
        ("ALIGN",        (1, 0), (1, -1),  "CENTER"),
        ("ALIGN",        (2, 0), (2, -1),  "CENTER"),
        ("ROWBACKGROUNDS",(0,1), (-1,-1),  [BLANC, GRIS_FOND]),
        ("FONTSIZE",     (0, 1), (-1, -1), 9),
        ("GRID",         (0, 0), (-1, -1), 0.5, GRIS_BORD),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
    ]))
    return t


def fig_to_image(fig, width=160*mm):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="#ffffff")
    buf.seek(0)
    img = Image(buf, width=width)
    img.hAlign = "CENTER"
    return img


def entete(story, styles, titre, sous_titre):
    story.append(Paragraph(titre, styles["Titre"]))
    story.append(Paragraph(sous_titre, styles["Sous_titre"]))
    story.append(hr(VERT))


def pied_de_page(story, styles):
    story.append(Spacer(1, 12))
    story.append(hr())
    story.append(Paragraph(
        "Rapport généré automatiquement par la Plateforme Agricole IA · "
        "Données synthétiques à titre démonstratif · "
        "Ne se substitue pas à un diagnostic terrain.",
        styles["Disclaimer"]
    ))


# Indice de risque agronomique (Culture)
def calcul_risque_agronomique(temperature, pluviometrie, azote, ph_sol, matiere_org, densite_semis):
    """
    Score 0-100 : 0 = aucun risque, 100 = risque maximal.
    Chaque paramètre contribue selon son écart à l'optimum.
    """
    def ecart_norm(val, opt, plage):
        return min(abs(val - opt) / plage, 1.0)

    scores = [
        ecart_norm(temperature,   14,  12),   # opt 14°C, plage ±12
        ecart_norm(pluviometrie,  600, 400),  # opt 600 mm, plage ±400
        ecart_norm(azote,         180, 120),  # opt 180 kg/ha
        ecart_norm(ph_sol,        6.8, 1.8),  # opt 6.8
        ecart_norm(matiere_org,   2.8, 2.2),  # opt 2.8%
        ecart_norm(densite_semis, 220, 180),  # opt 220 gr/m²
    ]
    risque = int(np.mean(scores) * 100)

    if risque < 25:
        label, couleur = "Faible", VERT
    elif risque < 50:
        label, couleur = "Modéré", colors.HexColor("#f57f17")
    elif risque < 75:
        label, couleur = "Élevé", ORANGE
    else:
        label, couleur = "Critique", ROUGE

    return risque, label, couleur


def badge_risque(risque, label, couleur):
    """Retourne un mini-tableau coloré avec le score de risque."""
    bg = VERT_CLAIR if label == "Faible" else (
         ORANGE_CLAIR if label in ("Modéré", "Élevé") else ROUGE_CLAIR)
    t = Table([[f"Indice de risque agronomique : {risque}/100 — {label}"]],
              colWidths=[170*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), bg),
        ("TEXTCOLOR",    (0,0), (-1,-1), couleur),
        ("FONTNAME",     (0,0), (-1,-1), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 11),
        ("ALIGN",        (0,0), (-1,-1), "CENTER"),
        ("TOPPADDING",   (0,0), (-1,-1), 8),
        ("BOTTOMPADDING",(0,0), (-1,-1), 8),
        ("BOX",          (0,0), (-1,-1), 1.5, couleur),
        ("ROUNDEDCORNERS",(0,0),(-1,-1), 4),
    ]))
    return t


# Priorité d'intervention (Vache)
def calcul_priorite(score_sante, ccs, temperature_v, bcs, production):
    """
    Croisse score IA + indicateurs critiques pour donner
    une priorité d'intervention Vert / Orange / Rouge.
    """
    critique = (
        ccs > 400 or
        temperature_v > 39.5 or
        bcs < 2.0 or
        production < 10
    )
    vigilance = (
        ccs > 200 or
        temperature_v > 38.8 or
        bcs < 2.5 or
        score_sante < 40
    )

    if critique or score_sante < 30:
        return "Intervention immédiate", ROUGE, ROUGE_CLAIR
    elif vigilance or score_sante < 60:
        return "Surveillance renforcée", ORANGE, ORANGE_CLAIR
    else:
        return "Suivi normal", VERT, VERT_CLAIR


def badge_priorite(label, couleur, bg):
    t = Table([[f"Priorité d'intervention : {label}"]],
              colWidths=[170*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), bg),
        ("TEXTCOLOR",    (0,0), (-1,-1), couleur),
        ("FONTNAME",     (0,0), (-1,-1), "Helvetica-Bold"),
        ("FONTSIZE",     (0,0), (-1,-1), 11),
        ("ALIGN",        (0,0), (-1,-1), "CENTER"),
        ("TOPPADDING",   (0,0), (-1,-1), 8),
        ("BOTTOMPADDING",(0,0), (-1,-1), 8),
        ("BOX",          (0,0), (-1,-1), 1.5, couleur),
    ]))
    return t


# Graphique importance (Culture) PDF
def fig_importance(importances):
    labels = ["Temp.", "Pluie", "Azote", "pH sol", "Mat. org.", "Densité", "Type sol"]
    colors_bar = ["#2e7d32" if i == np.argmax(importances) else "#81c784"
                  for i in range(len(importances))]
    fig, ax = plt.subplots(figsize=(5.5, 2.0))
    bars = ax.barh(labels, importances * 100, color=colors_bar)
    ax.set_xlabel("Importance (%)", fontsize=7)
    ax.set_title("Poids des facteurs sur le rendement", fontsize=8, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(labelsize=7)
    for bar, val in zip(bars, importances * 100):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=6)
    plt.tight_layout()
    return fig


def fig_comparaison(rendement_pred, rend_opt):
    categories = ["Votre rendement", "Optimum", "Moyenne FR"]
    values = [rendement_pred, rend_opt, 7.4]
    colors_bar = ["#1565c0", "#2e7d32", "#e65100"]
    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    bars = ax.bar(categories, values, color=colors_bar)
    ax.set_ylabel("t/ha", fontsize=7)
    ax.set_title("Comparaison des rendements", fontsize=8, fontweight="bold")
    ax.set_ylim(0, max(values) * 1.3)
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(labelsize=7)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", fontweight="bold", fontsize=7)
    plt.tight_layout()
    return fig


# Graphique radar santé (Vache) PDF
def fig_radar_vache(production, taux_tb, taux_tp, temperature_v, ccs, bcs):
    labels = ["Production", "TB", "TP", "Temp.", "CCS (inv.)", "BCS"]

    def norm(val, vmin, vmax):
        return np.clip((val - vmin) / (vmax - vmin), 0, 1)

    valeurs = [
        norm(production,   8,  45),
        norm(taux_tb,     28,  52),
        norm(taux_tp,     24,  42),
        1 - norm(abs(temperature_v - 38.5), 0, 2),
        1 - norm(ccs,     10, 600),
        norm(bcs,        1.5,   5),
    ]

    N = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    valeurs += valeurs[:1]

    fig, ax = plt.subplots(figsize=(2, 2), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=4)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=3, color="grey")
    ax.plot(angles, valeurs, color="#2e7d32", linewidth=1.5)
    ax.fill(angles, valeurs, color="#2e7d32", alpha=0.25)
    ax.set_title("Profil de santé", fontsize=5, fontweight="bold", pad=10)
    plt.tight_layout()
    return fig



def export_pdf_culture(
    temperature, pluviometrie, azote, ph_sol,
    matiere_org, densite_semis, type_sol,
    rendement_pred, rend_opt, ecart, conseils, importances,
    risque_score=None, risque_label=None, **_
):
    styles = get_styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=20*mm, bottomMargin=20*mm,
        title="Rapport d'analyse parcelle",
        author="Plateforme Agricole IA",
        subject="Analyse culture : prédiction de rendement",
    )
    story = []
    date_str = datetime.now().strftime("%d/%m/%Y à %H:%M")

    # En-tête
    entete(story, styles,
           "Rapport d'analyse parcelle",
           f"Généré le {date_str}")

    # Badge risque
    risque, label_r, couleur_r = calcul_risque_agronomique(
        temperature, pluviometrie, azote, ph_sol, matiere_org, densite_semis)
    story.append(badge_risque(risque, label_r, couleur_r))
    story.append(Spacer(1, 10))

    # Résultats clés
    story.append(Paragraph("Résultats", styles["Section"]))
    ecart_str = f"+{ecart:.1f}%" if ecart >= 0 else f"{ecart:.1f}%"
    resultats = [
        ["Rendement estimé",         f"{rendement_pred:.2f} t/ha"],
        ["Rendement optimum modèle", f"{rend_opt:.2f} t/ha"],
        ["Moyenne nationale (blé)",  "7.40 t/ha"],
        ["Écart vs moyenne FR",      ecart_str],
    ]
    t = Table(resultats, colWidths=[100*mm, 70*mm])
    t.setStyle(TableStyle([
        ("FONTNAME",      (0,0), (0,-1),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 10),
        ("ROWBACKGROUNDS",(0,0), (-1,-1), [BLANC, GRIS_FOND]),
        ("GRID",          (0,0), (-1,-1), 0.5, GRIS_BORD),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("TEXTCOLOR",     (1, 3),(1, 3),
         VERT if ecart >= 0 else ROUGE),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    # Paramètres saisis
    story.append(Paragraph("Paramètres saisis", styles["Section"]))
    indicateurs = [
        ["Température",       f"{temperature:.1f}",    "°C"],
        ["Pluviométrie",      f"{pluviometrie:.0f}",   "mm/an"],
        ["Azote",             f"{azote:.0f}",          "kg N/ha"],
        ["pH sol",            f"{ph_sol:.1f}",         ""],
        ["Matière organique", f"{matiere_org:.1f}",    "%"],
        ["Densité semis",     f"{densite_semis:.0f}",  "gr/m²"],
        ["Type de sol",       type_sol,                ""],
    ]
    story.append(table_indicateurs(indicateurs))
    story.append(Spacer(1, 10))

    # Graphiques
    story.append(Paragraph("Analyse graphique", styles["Section"]))
    fig_imp  = fig_importance(importances)
    fig_comp = fig_comparaison(rendement_pred, rend_opt)

    row = Table(
        [[fig_to_image(fig_imp, width=95*mm), fig_to_image(fig_comp, width=70*mm)]],
        colWidths=[100*mm, 75*mm]
    )
    row.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP")]))
    story.append(row)
    plt.close("all")
    story.append(Spacer(1, 10))

    # Recommandations
    story.append(Paragraph("Recommandations agronomiques", styles["Section"]))
    for c in conseils:
        story.append(Paragraph(f"→ {c}", styles["Conseil"]))

    pied_de_page(story, styles)
    doc.build(story)
    buf.seek(0)
    return buf



def export_pdf_vache(
    production, taux_tb, taux_tp, temperature_v,
    ccs, bcs, age_mois, lactation_j,
    alertes, prediction, score_sante,
    statut=None, priorite=None, **_
):
    styles = get_styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=20*mm, bottomMargin=20*mm,
        title="Rapport d'analyse animal",
        author="Plateforme Agricole IA",
        subject="Analyse vache laitière : détection d'anomalies",
    )
    story = []
    date_str = datetime.now().strftime("%d/%m/%Y à %H:%M")

    # En-tête
    entete(story, styles,
           "Rapport d'analyse animal",
           f"Généré le {date_str}")

    # Badge priorité
    label_p, couleur_p, bg_p = calcul_priorite(
        score_sante, ccs, temperature_v, bcs, production)
    story.append(badge_priorite(label_p, couleur_p, bg_p))
    story.append(Spacer(1, 10))

    # Score santé + statut fusionné (pire signal entre modèle IA et règles métier)
    story.append(Paragraph("Résultats", styles["Section"]))
    if prediction != 1 or label_p == "Intervention immédiate":
        statut      = "ANOMALIE DÉTECTÉE"
        coul_statut = ROUGE
    elif label_p == "Surveillance renforcée":
        statut      = "SURVEILLANCE RENFORCÉE"
        coul_statut = ORANGE
    else:
        statut      = "NORMAL"
        coul_statut = VERT

    resultats = [
        ["Score de santé IA",  f"{score_sante}/100"],
        ["Statut",             statut],
    ]
    t = Table(resultats, colWidths=[100*mm, 70*mm])
    t.setStyle(TableStyle([
        ("FONTNAME",      (0,0), (0,-1),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 10),
        ("ROWBACKGROUNDS",(0,0), (-1,-1), [BLANC, GRIS_FOND]),
        ("GRID",          (0,0), (-1,-1), 0.5, GRIS_BORD),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("TEXTCOLOR",     (1,1), (1,1),   coul_statut),
        ("FONTNAME",      (1,1), (1,1),   "Helvetica-Bold"),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    # Indicateurs saisis
    story.append(Paragraph("Indicateurs saisis", styles["Section"]))
    indicateurs = [
        ["Production lait",    f"{production:.1f}",    "L/j"],
        ["Taux Butyreux (TB)", f"{taux_tb:.1f}",       "g/kg"],
        ["Taux Protéique (TP)",f"{taux_tp:.1f}",       "g/kg"],
        ["Température",        f"{temperature_v:.1f}", "°C"],
        ["CCS",                f"{ccs:.0f}",           "k/mL"],
        ["BCS",                f"{bcs:.1f}",           "/5"],
        ["Âge",                f"{age_mois:.0f}",      "mois"],
        ["Lactation",          f"{lactation_j:.0f}",   "jours"],
    ]
    story.append(table_indicateurs(indicateurs))
    story.append(Spacer(1, 10))

    # Radar
    story.append(Paragraph("Profil de santé", styles["Section"]))
    fig_radar = fig_radar_vache(production, taux_tb, taux_tp, temperature_v, ccs, bcs)
    story.append(fig_to_image(fig_radar, width=85*mm))
    plt.close("all")
    story.append(Spacer(1, 10))

    # Alertes
    story.append(Paragraph("Alertes vétérinaires", styles["Section"]))
    anomalie_reelle = [a for a in alertes if "Aucune anomalie" not in a]
    if anomalie_reelle:
        for a in anomalie_reelle:
            story.append(Paragraph(f"{a}", styles["Alerte"]))
    else:
        story.append(Paragraph("Aucune anomalie détectée : animal en bonne santé apparente.",
                               styles["Conseil"]))

    pied_de_page(story, styles)
    doc.build(story)
    buf.seek(0)
    return buf


# Sauvegarde temporaire (pour Gradio)
def buf_to_tempfile(buf, suffix=".pdf"):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(buf.read())
    tmp.flush()
    return tmp.name