import gradio as gr
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest,HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
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

    modele_rendement = HistGradientBoostingRegressor(
        max_iter=200,
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


def creer_graph(model, rendement_pred, rend_opt):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    


if __name__ == '__main__':
    model, scaler, mae, r2 = train_model_culture()

