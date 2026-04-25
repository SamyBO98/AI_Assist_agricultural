import gradio as gr
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from generate_data import generer_donnees_cultures,generer_donnees_troupeau
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")


#First model

def create_dataSet_culture():
    df_cult = generer_donnees_cultures()
    #print(df_cult)
    #We want to predict efficiency
    features_cult = ["temperature", "pluviometrie", "azote", "ph_sol", "matiere_org", "densite_semis", "type_sol"]
    X_c = df_cult[features_cult]
    y_c = df_cult["rendement"]


#Second model
def create_dataSet_troupeau():
    df_trp = generer_donnees_troupeau()
    #print(df_trp)
    #Detect anomaly
    features_trp = ["production", "taux_tb", "taux_tp", "temperature_v", "ccs", "bcs"]
    X_t = df_trp[features_trp]