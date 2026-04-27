"""
API REST — AI Assist Agricultural
Endpoints : /culture  /vache  /feuille
"""

import io
import warnings
warnings.filterwarnings("ignore")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from database import init_db

init_db()

from models.culture import load_or_train_culture
from models.troupeau import load_or_train_troupeau

model_culture, scaler_culture, _, _, _ = load_or_train_culture()
model_vache, scaler_vache, score_min, score_max = load_or_train_troupeau()

app = FastAPI(
    title="AI Assist Agricultural",
    description="API de prédiction agricole : Culture, Vache laitière, Maladie foliaire",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)