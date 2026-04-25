from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from data.generate_data import generer_donnees_troupeau
import joblib
import os
import warnings
warnings.filterwarnings("ignore")


MODEL_PATH = "models/saved/troupeau_model.pkl"

def train_model_troupeau():
    df_trp = generer_donnees_troupeau()

    features_trp = ["production", "taux_tb", "taux_tp", "temperature_v", "ccs", "bcs", "age_mois", "lactation_j"]
    X_t = df_trp[features_trp]
    scaler_t = StandardScaler()
    X_t_scaled = scaler_t.fit_transform(X_t)


    modele_anomalie = IsolationForest(
        n_estimators=200, contamination=0.08, random_state=42
    )
    modele_anomalie.fit(X_t_scaled)

    scores_train = modele_anomalie.score_samples(X_t_scaled)
    score_min = scores_train.min()
    score_max = scores_train.max()

        # Sauvegarde
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({
        "model": modele_anomalie,
        "scaler": scaler_t,
        "score_min": score_min,
        "score_max": score_max,
    }, MODEL_PATH)
    print(f"[troupeau] Modèle sauvegardé → {MODEL_PATH}")
    return modele_anomalie, scaler_t, score_min, score_max

def load_or_train_troupeau():
    if os.path.exists(MODEL_PATH):
        print(f"[troupeau] Chargement depuis {MODEL_PATH}")
        data = joblib.load(MODEL_PATH)
        return data["model"], data["scaler"], data["score_min"], data["score_max"]
    return train_model_troupeau()