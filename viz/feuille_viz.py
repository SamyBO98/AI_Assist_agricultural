import matplotlib.pyplot as plt
import numpy as np

from PIL import Image


def pipeline_feuille(image: Image.Image, resultat: dict):
    """
    Génère la figure de visualisation feuille et le texte résumé.

    Paramètres :
        image    : image PIL uploadée par l'utilisateur
        resultat : dict retourné par analyser_feuille()

    Retourne :
        fig  : matplotlib Figure
        texte: str résumé lisible pour Gradio
    """
    if not resultat.get("success"):
        fig, ax = plt.subplots(figsize=(8, 3))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        ax.text(
            0.5,
            0.5,
            f"Erreur : {resultat.get('error', 'inconnue')}",
            color="#f44336",
            fontsize=13,
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.axis("off")
        return fig, "Analyse impossible."

    plante = resultat["plante"]
    etat = resultat["etat"]
    confiance = resultat["confiance"]
    sain = resultat["sain"]
    top_k = resultat["top_k"]

    couleur_principale = "#4CAF50" if sain else "#f44336"
    if etat == "Incertain":
        couleur_principale = "#FFA726"

    # Figure
    fig = plt.figure(figsize=(12, 5))
    fig.patch.set_facecolor("#1a1a2e")

    # Panneau gauche : image uploadée
    ax_img = fig.add_subplot(1, 2, 1)
    ax_img.imshow(image)
    ax_img.axis("off")
    ax_img.set_title(
        "Image analysée", color="white", fontsize=11, fontweight="bold", pad=10
    )

    # Cadre coloré autour de l'image selon le résultat
    for spine in ax_img.spines.values():
        spine.set_edgecolor(couleur_principale)
        spine.set_linewidth(3)
        spine.set_visible(True)

    # ── Panneau droit : top-k barres horizontales ─────────────────────────────
    ax_bar = fig.add_subplot(1, 2, 2)
    ax_bar.set_facecolor("#16213e")

    labels = [f"{r['plante']}\n{r['etat']}" for r in top_k]
    valeurs = [r["confiance"] for r in top_k]
    conf_max = valeurs[0] if valeurs[0] > 0 else 1
    couleurs = [
        (
            couleur_principale
            if i == 0
            else (
                *plt.matplotlib.colors.to_rgb("#546e7a"),
                max(0.3, r["confiance"] / conf_max),
            )
        )
        for i, r in enumerate(top_k)
    ]

    y_pos = np.arange(len(labels))
    bars = ax_bar.barh(y_pos, valeurs, color=couleurs, height=0.5, edgecolor="none")

    # Valeurs sur les barres
    for bar, val in zip(bars, valeurs):
        ax_bar.text(
            min(val + 1.5, 102),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            color="white",
            fontsize=9,
            fontweight="bold",
        )

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels, color="#ccc", fontsize=9)
    ax_bar.set_xlim(0, 110)
    ax_bar.set_xlabel("Confiance (%)", color="#aaa", fontsize=9)
    ax_bar.set_title("Top prédictions", color="white", fontsize=11, fontweight="bold")
    ax_bar.tick_params(axis="x", colors="#aaa")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["left"].set_color("#333")
    ax_bar.spines["bottom"].set_color("#333")
    ax_bar.xaxis.label.set_color("#aaa")

    # Badge statut en bas du graphique
    statut_txt = (
        "SAIN" if sain else ("INCERTAIN" if etat == "Incertain" else "MALADIE DETECTEE")
    )
    ax_bar.text(
        0.5,
        1.08,
        f"Confiance : {confiance:.1f}%",
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color=couleur_principale,
        transform=ax_bar.transAxes,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    # ── Texte résumé ─────────────────────────────────────────────────────────
    if etat == "Incertain":
        texte = (
            f"Plante détectée : {plante}\n"
            f"Resultat : Incertain (confiance {confiance:.1f}% < seuil)\n"
            f"Veuillez soumettre une image plus nette ou mieux cadree."
        )
    elif sain:
        texte = (
            f"Plante détectée : {plante}\n"
            f"Etat : Sain\n"
            f"Confiance : {confiance:.1f}%"
        )
    else:
        texte = (
            f"Plante détectée : {plante}\n"
            f"Maladie détectée : {etat}\n"
            f"Confiance : {confiance:.1f}%"
        )

    return fig, texte
