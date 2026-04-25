from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from data.generate_data import generer_donnees_cultures
import warnings
warnings.filterwarnings("ignore")

def train_model_culture():
    df_cult = generer_donnees_cultures()

    features_cult = ["temperature", "pluviometrie", "azote", "ph_sol", "matiere_org", "densite_semis", "type_sol"]
    X_c = df_cult[features_cult]
    y_c = df_cult["rendement"]

    X_train, X_test, y_train, y_test = train_test_split(X_c, y_c, test_size=0.2, random_state=42)

    scaler_c = StandardScaler()
    X_train_scaled = scaler_c.fit_transform(X_train)
    X_test_scaled = scaler_c.transform(X_test)

    modele_rendement = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.08,
        max_depth=4,
        random_state=42
    )
    modele_rendement.fit(X_train_scaled, y_train)

    mae_cult = mean_absolute_error(y_test, modele_rendement.predict(X_test_scaled))
    r2_cult  = r2_score(y_test, modele_rendement.predict(X_test_scaled))
    print(mae_cult)
    print(r2_cult)
    return modele_rendement, scaler_c, mae_cult, r2_cult