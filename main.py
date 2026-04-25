import gradio as gr
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest,GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from generate_data import generer_donnees_cultures,generer_donnees_troupeau
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")



SOLS = {0: "Limoneux", 1: "Argileux", 2: "Sableux", 3: "Calcaire"}
SOL_IDX = {v: k for k, v in SOLS.items()}

#First model
def train_model_culture():
    df_cult = generer_donnees_cultures()
    #print(df_cult)

    #We want to predict efficiency
    features_cult = ["temperature", "pluviometrie", "azote", "ph_sol", "matiere_org", "densite_semis", "type_sol"]
    X_c = df_cult[features_cult]
    y_c = df_cult["rendement"]
    #Split in train and test dataset and then apply normalization -> no data leakage
    X_train, X_test, y_train, y_test = train_test_split(X_c, y_c, test_size=0.2, random_state=42)
    #Not that usefull here since we will not compare with KNN or linear model after and since trees make relative comparison not distance
    scaler_c = StandardScaler()
    #Learn on train dataset + apply on train dataset
    X_train_scaled = scaler_c.fit_transform(X_train)
    #Just apply on test dataset
    X_test_scaled = scaler_c.transform(X_test)

    modele_rendement = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.08,
        max_depth=4,
        random_state=42
    )
    modele_rendement.fit(X_train_scaled, y_train)

    #En moyenne de combien je me trompe
    #Calcul de l'écart moyen entre prédictions et valeurs réelles
    mae_cult = mean_absolute_error(y_test, modele_rendement.predict(X_test_scaled))
    #Coeff de détermination
    #Est ce que mon modèle fait mieux que la moyenne
    r2_cult  = r2_score(y_test, modele_rendement.predict(X_test_scaled))
    print(mae_cult)
    print(r2_cult)
    return modele_rendement, scaler_c, mae_cult, r2_cult



#Second model
def train_model_troupeau():
    df_trp = generer_donnees_troupeau()
    #print(df_trp)
    #Detect anomaly
    features_trp = ["production", "taux_tb", "taux_tp", "temperature_v", "ccs", "bcs"]
    X_t = df_trp[features_trp]
    scaler_t = StandardScaler()
    X_t_scaled = scaler_t.fit_transform(X_t)
    #Detection anomalie sans labels
    modele_anomalie = IsolationForest(
        n_estimators=200, contamination=0.08, random_state=42
    )
    modele_anomalie.fit(X_t_scaled)
    return modele_anomalie, scaler_t


def predire_rendement(model, scaler, temperature, pluviometrie, azote, ph_sol, matiere_org, densite_semis, type_sol_nom):
    sol_idx = SOL_IDX.get(type_sol_nom, 0)
    X_input = np.array([[temperature, pluviometrie, azote, ph_sol, matiere_org, densite_semis, sol_idx]])
    X_scaled = scaler.transform(X_input)
    rendement_pred = model.predict(X_scaled)[0]
    # Comparaison avec cas "optimal" pour calculer les marges
    X_opt = np.array([[14, 600, 180, 6.8, 2.8, 220, 0]])
    rend_opt = model.predict(scaler.transform(X_opt))[0]

    conseils = []
    if azote < 120:
        conseils.append(f"Azote faible ({azote:.0f} kg/ha) envisager un apport complémentaire")
    elif azote > 220:
        conseils.append(f"Azote excessif ({azote:.0f} kg/ha) risque de verse et pollution nitrique")
    if ph_sol < 6.0:
        conseils.append(f"pH acide ({ph_sol:.1f}) chaulage recommandé (objectif 6.5–7.0)")
    if matiere_org < 1.5:
        conseils.append("Matière organique très faible intégrer des couverts végétaux")
    if pluviometrie < 400:
        conseils.append("Faible pluviométrie explorer l'irrigation d'appoint")
    if not conseils:
        conseils.append("Paramètres globalement favorables maintenir les pratiques actuelles")

    ecart = ((rendement_pred - 7.4) / 7.4) * 100
    return rendement_pred, rend_opt, ecart, conseils


def creer_graphs(model, rendement_pred, rend_opt):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax in axes:
        ax.set_facecolor("#16213e")
    #Graph 1 : Importance des variables
    labels = [
        "Temp.", "Pluie", "Azote",
        "pH sol", "Mat. org.", "Densité", "Type sol"
    ]
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        importances = np.ones(len(labels)) / len(labels)
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
            fontsize=8
        )
    #Graph 2 : Comparaison 
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
            fontweight="bold"
        )

    plt.tight_layout()

    return fig

def pipeline(model, scaler,temperature, pluviometrie, azote,ph_sol, matiere_org, densite_semis,type_sol):

    rend_pred, rend_opt, ecart, conseils = predire_rendement(model, scaler,temperature, pluviometrie, azote,ph_sol, matiere_org, densite_semis,type_sol)

    fig = creer_graphs(model, rend_pred, rend_opt)

    texte = (
        f"Rendement estimé : {rend_pred:.2f} t/ha\n"
        f"Écart : {ecart:+.1f}%\n\n"
        "Recommandations :\n"
        + "\n".join("- " + c for c in conseils)
    )

    return fig, texte

def load_model():
    return train_model_culture()

model, scaler, mae, r2 = load_model()

def pipeline_wrapper(temperature, pluviometrie, azote,
                     ph_sol, matiere_org, densite_semis,
                     type_sol):

    return pipeline(
        model, scaler,
        temperature, pluviometrie, azote,
        ph_sol, matiere_org, densite_semis,
        type_sol
    )

interface = gr.Interface(
    fn=pipeline_wrapper,
    inputs=[
        gr.Number(label="Température"),
        gr.Number(label="Pluviométrie"),
        gr.Number(label="Azote"),
        gr.Number(label="pH sol"),
        gr.Number(label="Matière organique"),
        gr.Number(label="Densité semis"),
        gr.Dropdown(list(SOLS.values()), label="Type de sol")
    ],
    outputs=[
        gr.Plot(label="Graphiques"),
        gr.Textbox(label="Analyse")
    ],
    title="Modèle agricole : prédiction rendement"
)


if __name__ == '__main__':
    interface.launch()

