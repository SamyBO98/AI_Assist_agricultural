import numpy as np
import matplotlib.pyplot as plt
from services.culture_service import predire_rendement
from sklearn.inspection import permutation_importance
import warnings

warnings.filterwarnings("ignore")


def creer_graphs(importances, rendement_pred, rend_opt):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax in axes:
        ax.set_facecolor("#16213e")

    labels = ["Temp.", "Pluie", "Azote", "pH sol", "Mat. org.", "Densité", "Type sol"]

    colors = [
        "#4CAF50" if i == np.argmax(importances) else "#81C784"
        for i in range(len(importances))
    ]
    bars = axes[0].barh(labels, importances * 100, color=colors)
    axes[0].set_title("Poids des facteurs", color="white")
    axes[0].set_xlabel("Importance (%)", color="#aaa")
    axes[0].tick_params(colors="#ccc")
    axes[0].spines[:].set_visible(False)
    for bar, val in zip(bars, importances * 100):
        axes[0].text(
            val + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            color="#ccc",
            fontsize=8,
        )

    categories = ["Votre rendement", "Optimum", "Moyenne FR"]
    values = [rendement_pred, rend_opt, 7.4]
    colors_bar = ["#2196F3", "#4CAF50", "#FF9800"]

    b = axes[1].bar(categories, values, color=colors_bar)

    axes[1].set_title("Comparaison rendements", color="white")
    axes[1].set_ylabel("t/ha", color="#aaa")

    axes[1].set_ylim(0, max(values) * 1.25)
    axes[1].tick_params(colors="#ccc")
    axes[1].spines[:].set_visible(False)

    for bar, val in zip(b, values):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.1f}",
            ha="center",
            color="white",
            fontweight="bold",
        )

    plt.tight_layout()
    return fig


def pipeline(
    model,
    scaler,
    importances,
    temperature,
    pluviometrie,
    azote,
    ph_sol,
    matiere_org,
    densite_semis,
    type_sol,
):

    rend_pred, rend_opt, ecart, conseils = predire_rendement(
        model,
        scaler,
        temperature,
        pluviometrie,
        azote,
        ph_sol,
        matiere_org,
        densite_semis,
        type_sol,
    )

    fig = creer_graphs(importances, rend_pred, rend_opt)

    texte = (
        f"Rendement estimé : {rend_pred:.2f} t/ha\n"
        f"Écart : {ecart:+.1f}%\n\n"
        "Recommandations :\n" + "\n".join("- " + c for c in conseils)
    )

    return fig, texte
