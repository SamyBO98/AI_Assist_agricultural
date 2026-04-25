import gradio as gr
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")

from models.culture import train_model_culture
from viz.culture_viz import pipeline
from config import SOLS


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

def scenario_bon():
    return 15, 600, 160, 6.7, 2.5, 210, "Limoneux"

def scenario_moyen():
    return 18, 500, 140, 6.2, 2.0, 200, "Argileux"

def scenario_mauvais():
    return 35, 200, 300, 5.0, 0.8, 400, "Sableux"

def clear_inputs():
    return None, None, None, None, None, None, None


with gr.Blocks() as interface:

    gr.Markdown("# Modèle agricole : Prédiction rendement")

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
            clear_btn.click(
                fn=clear_inputs,
                inputs=[],
                outputs=[temp, pluie, azote, ph, org, densite, sol]
            )

        with gr.Column(scale=2):
            with gr.Row():
                gr.Button("Bon", scale=1).click(
                    scenario_bon,
                    outputs=[temp, pluie, azote, ph, org, densite, sol]
                )
                gr.Button("Moyen", scale=1).click(
                    scenario_moyen,
                    outputs=[temp, pluie, azote, ph, org, densite, sol]
                )
                gr.Button("Mauvais", scale=1).click(
                    scenario_mauvais,
                    outputs=[temp, pluie, azote, ph, org, densite, sol]
                )
            btn = gr.Button("Prédire")
            out1 = gr.Plot(label="Graphiques")
            out2 = gr.Textbox(label="Analyse")

    btn.click(
        fn=pipeline_wrapper,
        inputs=[temp, pluie, azote, ph, org, densite, sol],
        outputs=[out1, out2]
    )


if __name__ == '__main__':
    interface.launch()