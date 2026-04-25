import numpy as np
import matplotlib.pyplot as plt
import warnings
from services.vache_service import analyser_vache
warnings.filterwarnings("ignore")


def creer_graphs(prediction, score_sante, alertes,
                 production, taux_tb, taux_tp, temperature_v, ccs, bcs):

    labels_radar = ["Production", "TB", "TP", "Température", "CCS\n(inversé)", "BCS"]

    norm_prod  = np.clip((production - 8) / (50 - 8), 0, 1)
    norm_tb    = np.clip((taux_tb - 28) / (52 - 28), 0, 1)
    norm_tp    = np.clip((taux_tp - 24) / (42 - 24), 0, 1)
    norm_temp  = np.clip(1 - abs(temperature_v - 38.5) / 1.5, 0, 1)
    norm_ccs   = np.clip(1 - (ccs - 10) / 1990, 0, 1)
    norm_bcs   = np.clip(1 - abs(bcs - 3.0) / 1.5, 0, 1)

    values_radar = [norm_prod, norm_tb, norm_tp, norm_temp, norm_ccs, norm_bcs]
    values_radar += [values_radar[0]]

    angles = np.linspace(0, 2 * np.pi, len(labels_radar), endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.patch.set_facecolor("#1a1a2e")

    # ===== RADAR =====
    ax_radar = fig.add_subplot(121, polar=True)
    ax_radar.set_facecolor("#16213e")

    color_radar = "#4CAF50" if prediction == 1 else "#f44336"

    ax_radar.plot(angles, values_radar, color=color_radar, linewidth=2)
    ax_radar.fill(angles, values_radar, alpha=0.25, color=color_radar)

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(labels_radar, color="#ccc", size=9)

    ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax_radar.set_yticklabels(["25%", "50%", "75%", "100%"], color="#555", size=7)

    ax_radar.set_title("Profil santé", color="white", fontsize=11, fontweight="bold")
    ax_radar.grid(color="#333", linestyle="--", alpha=0.5)

    # ===== JAUGE =====
    ax2 = axes[1]
    ax2.set_facecolor("#16213e")

    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 1)

    ax2.barh(0.5, 100, height=0.3, color="#333")

    bar_color = "#4CAF50" if score_sante >= 70 else ("#FF9800" if score_sante >= 45 else "#f44336")

    ax2.barh(0.5, score_sante, height=0.3, color=bar_color)

    ax2.text(score_sante / 2, 0.5, f"{score_sante}/100",
             ha="center", va="center", color="white", fontsize=14, fontweight="bold")

    zones = [
        (0, 45, "#f44336", "Risque élevé"),
        (45, 70, "#FF9800", "Surveillance"),
        (70, 100, "#4CAF50", "Sain")
    ]

    for start, end, c, lbl in zones:
        ax2.text((start + end) / 2, 0.2, lbl, ha="center", color=c, fontsize=8)

    ax2.axvline(score_sante, color="white", linestyle="--", alpha=0.5)

    ax2.set_title("Score de santé global", color="white", fontsize=11)
    ax2.set_xlabel("Score (0 = critique, 100 = excellent)", color="#aaa")
    ax2.set_yticks([])
    ax2.spines[:].set_visible(False)

    statut = "NORMAL" if prediction == 1 else "ANOMALIE DÉTECTÉE"

    texte = (
        f"{statut} — Score : {score_sante}/100\n\n"
        f"Modèle : Isolation Forest\n\n"
        f"Alertes :\n"
        + "\n".join(f"- {a}" for a in alertes)
    )

    plt.tight_layout()

    return fig, texte

def pipeline_vache(model, scaler, score_min, score_max,
                   production, taux_tb, taux_tp,
                   temperature_v, ccs, bcs,
                   age_mois, lactation_j):

    alertes, prediction, score_sante = analyser_vache(
        model, scaler, score_min, score_max,
        production, taux_tb, taux_tp,
        temperature_v, ccs, bcs,
        age_mois, lactation_j
    )

    fig, texte = creer_graphs(
        prediction, score_sante, alertes,
        production, taux_tb, taux_tp,
        temperature_v, ccs, bcs
    )

    return fig, texte