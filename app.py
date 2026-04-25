import gradio as gr
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")

from models.culture import train_model_culture
from viz.culture_viz import pipeline
from config import SOLS

from models.troupeau import train_model_troupeau
from viz.vache_viz import pipeline_vache


def load_model():
    return train_model_culture()

def load_model2():
    return train_model_troupeau()

model, scaler, mae, r2 = load_model()
model_vache, scaler_vache, score_min, score_max = train_model_troupeau()

def pipeline_wrapper(temperature, pluviometrie, azote,
                     ph_sol, matiere_org, densite_semis,
                     type_sol):
    return pipeline(
        model, scaler,
        temperature, pluviometrie, azote,
        ph_sol, matiere_org, densite_semis,
        type_sol
    )

def pipeline_vache_wrapper(production, taux_tb, taux_tp,
                           temperature_v, ccs, bcs,
                           age_mois, lactation_j):

    return pipeline_vache(
        model_vache, scaler_vache,score_min, score_max,
        production, taux_tb, taux_tp,
        temperature_v, ccs, bcs,
        age_mois, lactation_j
    )




def scenario_bon():
    return 15, 600, 160, 6.7, 2.5, 210, "Limoneux"

def scenario_moyen():
    return 18, 500, 140, 6.2, 2.0, 200, "Argileux"

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
        38.8,   # légère hausse température
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


with gr.Blocks() as interface:

    gr.Markdown("# 🌾 Plateforme agricole IA")

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

                    btn = gr.Button("Prédire culture")

                    out1 = gr.Plot()
                    out2 = gr.Textbox()

            clear_btn.click(
                fn=clear_inputs,
                inputs=[],
                outputs=[temp, pluie, azote, ph, org, densite, sol]
            )

            btn.click(
                fn=pipeline_wrapper,
                inputs=[temp, pluie, azote, ph, org, densite, sol],
                outputs=[out1, out2]
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
                    btn_v = gr.Button("Analyser vache")

                    out_v1 = gr.Plot()
                    out_v2 = gr.Textbox()

            btn_v.click(
                fn=pipeline_vache_wrapper,
                inputs=[production, tb, tp, temp_v, ccs, bcs, age, lactation],
                outputs=[out_v1, out_v2]
            )


if __name__ == '__main__':
    interface.launch()