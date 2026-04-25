from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from data.generate_data import generer_donnees_troupeau
import warnings
warnings.filterwarnings("ignore")

def train_model_troupeau():
    df_trp = generer_donnees_troupeau()

    features_trp = ["production", "taux_tb", "taux_tp", "temperature_v", "ccs", "bcs"]
    X_t = df_trp[features_trp]
    scaler_t = StandardScaler()
    X_t_scaled = scaler_t.fit_transform(X_t)

    modele_anomalie = IsolationForest(
        n_estimators=200, contamination=0.08, random_state=42
    )
    modele_anomalie.fit(X_t_scaled)
    return modele_anomalie, scaler_t