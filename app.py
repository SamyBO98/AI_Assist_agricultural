import gradio as gr
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")

from models.culture import load_or_train_culture
from viz.culture_viz import pipeline
from config import SOLS

from models.troupeau import load_or_train_troupeau
from viz.vache_viz import pipeline_vache
from reports.pdf_report import export_pdf_culture, export_pdf_vache, buf_to_tempfile
from services.culture_service import predire_rendement


model, scaler, mae, r2, importances= load_or_train_culture()
model_vache, scaler_vache, score_min, score_max = load_or_train_troupeau()





# Bornes physiques extrêmes par champ
# (min, max, unité, justification)
BORNES_CULTURE = {
    "Température (°C)"      : (-15,  50,   "°C",      "de -15°C (gel extrême) à 50°C (record absolu au sol)"),
    "Pluviométrie (mm/an)"  : (  0,  12000,"mm/an",   "de 0 (sécheresse totale) à 12 000 mm (Meghalaya, Inde)"),
    "Azote (kg N/ha)"       : (  0,  400,  "kg N/ha", "de 0 (zéro intrant) à 400 kg/ha (au-delà = toxique)"),
    "pH sol"                : (  2,  11,   "",         "de 2.0 (sol volcanique acide) à 11.0 (sol sodique)"),
    "Matière organique (%)" : (  0,  100,  "%",        "de 0% (sable pur) à 100% (tourbe pure)"),
    "Densité semis (gr/m²)" : (  1,  600,  "gr/m²",   "de 1 grain/m² à 600 gr/m² (au-delà = étouffement)"),
}

BORNES_VACHE = {
    "Production lait (L/j)" : (  0,   90,  "L/j",   "de 0 (vache tarie) à 90 L/j (record mondial Holstein)"),
    "TB (g/kg)"             : ( 20,   80,  "g/kg",  "de 20 g/kg (pathologique) à 80 g/kg (race à viande)"),
    "TP (g/kg)"             : ( 20,   60,  "g/kg",  "de 20 g/kg (pathologique) à 60 g/kg (extrême)"),
    "Température (°C)"      : ( 35,   42,  "°C",    "de 35°C (hypothermie sévère) à 42°C (hyperthermie létale)"),
    "CCS (k/mL)"            : ( 10, 10000, "k/mL",  "de 10 k/mL (minimum analytique) à 10 000 k/mL (mammite aiguë)"),
    "BCS (1–5)"             : (  1,    5,  "",      "de 1.0 (cachexie) à 5.0 (obésité maximale)"),
    "Âge (mois)"            : ( 12,  240,  "mois",  "de 12 mois (génisse primipare) à 240 mois (20 ans)"),
    "Lactation (jours)"     : (  1,  500,  "jours", "de 1 jour à 500 jours (lactation prolongée)"),
}


def li(msg):
    return f"<li style='display:list-item; margin:2px 0;'>{msg}</li>"


def valider_champs(valeurs, bornes):
    """
    valeurs : dict { label: valeur }
    bornes  : dict { label: (min, max, unité, justification) }
    Retourne une liste de <li> d'erreurs.
    """
    erreurs = []
    for label, valeur in valeurs.items():
        vmin, vmax, unite, justif = bornes[label]
        if valeur is None:
            erreurs.append(li(f"{label} : champ obligatoire"))
        elif valeur < vmin or valeur > vmax:
            plage = f"{vmin}–{vmax} {unite}".strip()
            erreurs.append(li(f"{label} : valeur hors limites physiques ({plage}) — {justif}"))
    return erreurs


def bandeau_erreur(erreurs):
    if not erreurs:
        return ""
    lignes = "\n".join(erreurs)
    return f"""
        <div style="
            background-color: #fff0f0;
            border: 1px solid #e53935;
            border-left: 4px solid #e53935;
            border-radius: 6px;
            padding: 10px 14px;
            margin: 8px 0;
            color: #b71c1c;
            font-size: 0.85em;
        ">
            <strong>Veuillez corriger les erreurs suivantes :</strong>
            <ul style="margin: 6px 0 0 0; padding-left: 20px; list-style-type: disc;">
                {lignes}
            </ul>
        </div>
    """


# Stockage du dernier résultat pour l'export PDF
last_culture = {}
last_vache   = {}


def pipeline_wrapper(temperature, pluviometrie, azote,
                     ph_sol, matiere_org, densite_semis,
                     type_sol):

    erreurs = valider_champs(
        valeurs={
            "Température (°C)"      : temperature,
            "Pluviométrie (mm/an)"  : pluviometrie,
            "Azote (kg N/ha)"       : azote,
            "pH sol"                : ph_sol,
            "Matière organique (%)" : matiere_org,
            "Densité semis (gr/m²)" : densite_semis,
        },
        bornes=BORNES_CULTURE
    )
    if type_sol is None or type_sol == "":
        erreurs.append(li("Type de sol : champ obligatoire"))
    if erreurs:
        return None, "", bandeau_erreur(erreurs)

    fig, texte = pipeline(
        model, scaler, importances,
        temperature, pluviometrie, azote,
        ph_sol, matiere_org, densite_semis,
        type_sol
    )

    rend_pred, rend_opt, ecart, conseils = predire_rendement(
        model, scaler, temperature, pluviometrie, azote,
        ph_sol, matiere_org, densite_semis, type_sol
    )
    last_culture.update(dict(
        temperature=temperature, pluviometrie=pluviometrie, azote=azote,
        ph_sol=ph_sol, matiere_org=matiere_org, densite_semis=densite_semis,
        type_sol=type_sol, rendement_pred=rend_pred, rend_opt=rend_opt,
        ecart=ecart, conseils=conseils, importances=importances
    ))
    return fig, texte, ""


def export_culture_pdf():
    if not last_culture:
        return None
    buf = export_pdf_culture(**last_culture)
    return buf_to_tempfile(buf)


def pipeline_vache_wrapper(production, taux_tb, taux_tp,
                           temperature_v, ccs, bcs,
                           age_mois, lactation_j):

    erreurs = valider_champs(
        valeurs={
            "Production lait (L/j)" : production,
            "TB (g/kg)"             : taux_tb,
            "TP (g/kg)"             : taux_tp,
            "Température (°C)"      : temperature_v,
            "CCS (k/mL)"            : ccs,
            "BCS (1–5)"             : bcs,
            "Âge (mois)"            : age_mois,
            "Lactation (jours)"     : lactation_j,
        },
        bornes=BORNES_VACHE
    )
    if erreurs:
        return None, "", bandeau_erreur(erreurs)

    fig, texte, alertes, prediction, score_sante = pipeline_vache(
        model_vache, scaler_vache, score_min, score_max,
        production, taux_tb, taux_tp,
        temperature_v, ccs, bcs,
        age_mois, lactation_j
    )

    last_vache.update(dict(
        production=production, taux_tb=taux_tb, taux_tp=taux_tp,
        temperature_v=temperature_v, ccs=ccs, bcs=bcs,
        age_mois=age_mois, lactation_j=lactation_j,
        alertes=alertes, prediction=prediction, score_sante=score_sante
    ))
    return fig, texte, ""


def export_vache_pdf():
    if not last_vache:
        return None
    buf = export_pdf_vache(**last_vache)
    return buf_to_tempfile(buf)








def scenario_bon():
    return 15, 600, 160, 6.7, 2.5, 210, "Limoneux"

def scenario_moyen():
    return 18, 200, 140, 6.0, 2.0, 200, "Argileux"

def scenario_mauvais():
    return 35, 200, 300, 5.0, 0.8, 400, "Sableux"

def clear_inputs():
    return None, None, None, None, None, None, None


def scenario_vache_bon():
    return (
        28,     # production (L/j)
        40,     # TB (bon lait gras)
        33,     # TP (bon protéine)
        38.3,   # température normale
        80,     # CCS très bas (excellent)
        3.0,    # BCS idéal
        60,     # âge (mois)
        120     # lactation
    )

def scenario_vache_moyen():
    return (
        20,     # production moyenne
        37,
        30,
        39,     # hausse température
        250,    # CCS moyen
        2.7,    # BCS un peu bas
        80,
        180
    )

def scenario_vache_mauvais():
    return (
        10,     # faible production
        32,     # lait pauvre
        26,
        40.0,   # fièvre
        600,    # CCS très élevé (mammite)
        1.8,    # maigreur
        90,
        220
    )

def clear_inputs_vache():
    return (
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None
    )



APROPOS_MD = """
# Guide des indicateurs
 
Cette plateforme utilise des modèles de machine learning pour aider à l'analyse agricole.
Elle comporte deux modules : **prédiction de rendement céréalier** et **détection d'anomalies sur vaches laitières**.
 
---
 
## Module Culture : indicateurs
 
| Indicateur | Unité | Valeur typique | Rôle |
|---|---|---|---|
| **Température** | °C | 12–18 °C | Température moyenne de la saison de croissance |
| **Pluviométrie** | mm/an | 400–700 mm | Précipitations totales annuelles |
| **Azote (N)** | kg N/ha | 120–200 kg/ha | Fertilisation azotée apportée à la culture |
| **pH sol** | — | 6.0–7.5 | Acidité du sol (6.5–7.0 = idéal pour céréales) |
| **Matière organique** | % | 1.5–4 % | Teneur en humus du sol — indicateur de fertilité |
| **Densité semis** | grains/m² | 180–260 | Nombre de graines semées par m² |
| **Type de sol** | — | Limoneux, Argileux… | Nature dominante du sol |
 
**Rendement de référence :** la moyenne nationale française pour le blé est d'environ **7,4 t/ha**.
 
---
 
## Module Vache : indicateurs
 
### Lait
| Indicateur | Unité | Plage normale | Signification |
|---|---|---|---|
| **Production** | L/jour | 20–35 L/j | Quantité de lait produite par vache par jour |
| **TB** (Taux Butyreux) | g/kg | 35–45 g/kg | Teneur en matières grasses du lait. Un TB < 30 peut indiquer une acidose |
| **TP** (Taux Protéique) | g/kg | 30–38 g/kg | Teneur en protéines du lait. Un TP bas peut signaler un déficit énergétique |
 
### Santé
| Indicateur | Unité | Plage normale | Signification |
|---|---|---|---|
| **CCS** (Comptage Cellules Somatiques) | milliers/mL | < 200 k/mL | Indicateur d'infection mammaire. **> 400 k/mL = suspicion mammite**. Plus le chiffre est bas, mieux c'est |
| **BCS** (Body Condition Score) | score 1–5 | 2.5–3.5 | État d'engraissement de l'animal. < 2 = maigreur, > 4 = surpoids |
| **Température** | °C | 38.0–39.5 °C | Température corporelle. > 39.5 °C = fièvre potentielle |
 
### Profil animal
| Indicateur | Unité | Info |
|---|---|---|
| **Âge** | mois | Influence la production et la sensibilité aux maladies |
| **Stade de lactation** | jours | Cycle de 305 jours — la production varie fortement selon le stade |
 
---
 
## Modèles utilisés
 
- **Culture** : `HistGradientBoostingRegressor` (scikit-learn) pour la régression sur données simulées
- **Vache** : `IsolationForest` pour la détection non supervisée d'anomalies
 
> **Note** : les modèles sont entraînés sur des données synthétiques à des fins de démonstration.
> Pour un usage en production, ils doivent être réentraînés sur de vraies données terrain.
"""


with gr.Blocks() as interface:

    gr.Markdown("# Plateforme agricole IA")

    gr.HTML("""
        <div style="
            background-color: #fff8e1;
            border: 1px solid #f9a825;
            border-left: 4px solid #f9a825;
            border-radius: 6px;
            padding: 8px 14px;
            margin: 4px 0 10px 0;
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 0.85em;
            color: #5d4037;
        ">
            <span style="font-size: 1.1em;">⚠️</span>
            <span>
                <strong>Mode démonstration</strong> — Les modèles sont entraînés sur des données synthétiques.
                Les résultats sont à titre illustratif et ne se substituent pas à un diagnostic terrain.
            </span>
        </div>
    """)

    with gr.Tabs():

        # ===================== CULTURE =====================
        with gr.Tab("Culture"):

            with gr.Row():

                with gr.Column(scale=1):
                    temp = gr.Number(label="Température")
                    pluie = gr.Number(label="Pluviométrie")
                    azote = gr.Number(label="Azote")
                    ph = gr.Number(label="pH sol")
                    org = gr.Number(label="Matière organique")
                    densite = gr.Number(label="Densité semis")
                    sol = gr.Dropdown(list(SOLS.values()), label="Type de sol")

                    clear_btn = gr.Button("Clear")

                with gr.Column(scale=2):
                    with gr.Row():
                        gr.Button("Bon").click(
                            scenario_bon,
                            outputs=[temp, pluie, azote, ph, org, densite, sol]
                        )

                        gr.Button("Moyen").click(
                            scenario_moyen,
                            outputs=[temp, pluie, azote, ph, org, densite, sol]
                        )

                        gr.Button("Mauvais").click(
                            scenario_mauvais,
                            outputs=[temp, pluie, azote, ph, org, densite, sol]
                        )

                    btn = gr.Button("Prédire culture", variant="primary")

                    err1  = gr.HTML()
                    out1  = gr.Plot()
                    out2  = gr.Textbox(label="Recommandations")
                    pdf1  = gr.File(label="-----Rapport PDF-----", visible=False)
                    btn_pdf1 = gr.Button("Exporter en PDF", variant="secondary", visible=False)

            clear_btn.click(
                fn=clear_inputs,
                inputs=[],
                outputs=[temp, pluie, azote, ph, org, densite, sol]
            )

            btn.click(
                fn=pipeline_wrapper,
                inputs=[temp, pluie, azote, ph, org, densite, sol],
                outputs=[out1, out2, err1]
            ).then(
                fn=lambda fig, *_: (gr.update(visible=fig is not None), gr.update(visible=fig is not None)),
                inputs=[out1],
                outputs=[btn_pdf1, pdf1]
            )

            btn_pdf1.click(
                fn=export_culture_pdf,
                inputs=[],
                outputs=[pdf1]
            )

        # ===================== VACHE =====================
        with gr.Tab("Vache"):

            with gr.Row():

                with gr.Column(scale=1):
                    production = gr.Number(label="Production lait")
                    tb = gr.Number(label="TB")
                    tp = gr.Number(label="TP")
                    temp_v = gr.Number(label="Température")
                    ccs = gr.Number(label="CCS")
                    bcs = gr.Number(label="BCS")
                    age = gr.Number(label="Âge (mois)")
                    lactation = gr.Number(label="Lactation (jours)")

                    clear_btn = gr.Button("Clear")

                with gr.Column(scale=2):
                    with gr.Row():
                        gr.Button("Bon").click(
                            scenario_vache_bon,
                            outputs=[production, tb, tp, temp_v, ccs, bcs, age, lactation]
                        )

                        gr.Button("Moyen").click(
                            scenario_vache_moyen,
                            outputs=[production, tb, tp, temp_v, ccs, bcs, age, lactation]
                        )

                        gr.Button("Mauvais").click(
                            scenario_vache_mauvais,
                            outputs=[production, tb, tp, temp_v, ccs, bcs, age, lactation]
                        )
                    btn_v = gr.Button("Analyser vache", variant="primary")

                    err_v    = gr.HTML()
                    out_v1   = gr.Plot()
                    out_v2   = gr.Textbox(label="Analyse")
                    pdf_v    = gr.File(label="----Rapport PDF----", visible=False)
                    btn_pdf_v = gr.Button("Exporter en PDF", variant="secondary", visible=False)

            clear_btn.click(
                fn=clear_inputs_vache,
                inputs=[],
                outputs=[production, tb, tp, temp_v, ccs, bcs, age, lactation]
            )
            btn_v.click(
                fn=pipeline_vache_wrapper,
                inputs=[production, tb, tp, temp_v, ccs, bcs, age, lactation],
                outputs=[out_v1, out_v2, err_v]
            ).then(
                fn=lambda fig, *_: (gr.update(visible=fig is not None), gr.update(visible=fig is not None)),
                inputs=[out_v1],
                outputs=[btn_pdf_v, pdf_v]
            )

            btn_pdf_v.click(
                fn=export_vache_pdf,
                inputs=[],
                outputs=[pdf_v]
            )

        with gr.Tab("À propos"):
            gr.Markdown(APROPOS_MD)


if __name__ == '__main__':
    interface.launch()