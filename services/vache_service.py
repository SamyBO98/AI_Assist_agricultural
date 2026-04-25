import numpy as np
import warnings
warnings.filterwarnings("ignore")

def analyser_vache(model, scaler, score_min, score_max, production, taux_tb, taux_tp, temperature_v, ccs, bcs, age_mois, lactation_j):
    
    X_input = np.array([[production, taux_tb, taux_tp, temperature_v, ccs, bcs, age_mois, lactation_j]])
    X_scaled = scaler.transform(X_input)

    # plus négatif = plus anormal
    score = model.score_samples(X_scaled)[0]

    # -1 = anomalie, 1 = normal
    prediction = model.predict(X_scaled)[0]

    # Score normalisé 0-100
    score = model.score_samples(X_scaled)[0]
    score_sante = np.interp(score, [score_min, score_max], [0, 100])
    score_sante = int(np.clip(score_sante, 0, 100))

    # Alertes métier
    alertes = []

    if ccs > 400:
        alertes.append(f"CCS très élevé ({ccs:.0f} k/mL) suspicion mammite, prélèvement urgent")
    elif ccs > 200:
        alertes.append(f"CCS élevé ({ccs:.0f} k/mL) surveiller l'évolution, hygiène traite")

    if temperature_v > 39.5:
        alertes.append(f"Hyperthermie ({temperature_v:.1f}°C) fièvre possible, appeler vétérinaire")

    if production < 12:
        alertes.append(f"Production faible ({production:.1f} L/j) vérifier alimentation et confort")

    if bcs < 2.0:
        alertes.append("BCS très bas animal trop maigre, risque cétose")
    elif bcs > 4.0:
        alertes.append("BCS élevé risque surpoids, adapter ration")

    if not alertes:
        alertes.append("Aucune anomalie détectée animal en bonne santé apparente")

    return alertes, prediction, score_sante