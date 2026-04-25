import numpy as np
import pandas as pd

rng = np.random.default_rng(42)


def generer_donnees_cultures(n=1200):
    temperature   = rng.normal(14, 4, n).clip(2, 30)        # °C moyenne saison
    pluviometrie  = rng.normal(550, 120, n).clip(200, 900)  # mm/an
    azote         = rng.normal(160, 30, n).clip(60, 260)    # kg N/ha
    ph_sol        = rng.normal(6.8, 0.5, n).clip(5.0, 8.0)
    matiere_org   = rng.normal(2.5, 0.6, n).clip(0.8, 5.0) # %
    densite_semis = rng.normal(220, 30, n).clip(120, 320)   # grains/m²
    type_sol      = rng.integers(0, 4, n)                    # 0=limoneux,1=argileux,2=sableux,3=calcaire
 
    rendement = (
        4.5
        + 0.08 * (temperature - 14)
        - 0.003 * (temperature - 14)**2
        + 0.004 * (pluviometrie - 550)
        + 0.018 * (azote - 100)
        - 0.00005 * azote**2
        + 1.2 * (ph_sol - 6.0)
        - 0.8 * (ph_sol - 7.2)**2
        + 0.4 * matiere_org
        + type_sol * 0.15
        + rng.normal(0, 0.4, n)
    ).clip(1.5, 12.0)
 
    return pd.DataFrame({
        "temperature": temperature,
        "pluviometrie": pluviometrie,
        "azote": azote,
        "ph_sol": ph_sol,
        "matiere_org": matiere_org,
        "densite_semis": densite_semis,
        "type_sol": type_sol,
        "rendement": rendement,
    })



def generer_donnees_troupeau(n=800):
    age_mois      = rng.integers(24, 120, n).astype(float)
    lactation_j   = rng.integers(1, 305, n).astype(float)
    production    = rng.normal(28, 6, n).clip(8, 50)   # L/jour
    taux_tb       = rng.normal(38, 3, n).clip(28, 52)  # g/kg TB pourcentage matière grasse dans le lait
    taux_tp       = rng.normal(32, 2, n).clip(24, 42)  # g/kg TP pourcentage protéine dans le lait
    temperature_v = rng.normal(38.5, 0.3, n).clip(37.5, 40.5)  # °C
    ccs           = rng.lognormal(4.8, 0.8, n).clip(10, 2000)  # cellules somatiques (k)
    bcs           = rng.normal(3.0, 0.4, n).clip(1.5, 5.0)     # Body Condition Score
 
    return pd.DataFrame({
        "age_mois": age_mois,
        "lactation_j": lactation_j,
        "production": production,
        "taux_tb": taux_tb,
        "taux_tp": taux_tp,
        "temperature_v": temperature_v,
        "ccs": ccs,
        "bcs": bcs,
    })