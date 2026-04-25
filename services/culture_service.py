import numpy as np
from config import SOL_IDX
import warnings
warnings.filterwarnings("ignore")


def predire_rendement(model, scaler, temperature, pluviometrie, azote, ph_sol, matiere_org, densite_semis, type_sol_nom):
    sol_idx = SOL_IDX.get(type_sol_nom, 0)
    X_input = np.array([[temperature, pluviometrie, azote, ph_sol, matiere_org, densite_semis, sol_idx]])
    X_scaled = scaler.transform(X_input)
    rendement_pred = model.predict(X_scaled)[0]

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