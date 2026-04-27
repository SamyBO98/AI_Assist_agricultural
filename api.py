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
from pydantic import BaseModel, Field


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


class CultureInput(BaseModel):
    temperature: float = Field(..., ge=-15, le=50, description="Température (°C)")
    pluviometrie: float = Field(..., ge=0, le=12000, description="Pluviométrie (mm/an)")
    azote: float = Field(..., ge=0, le=400, description="Azote (kg N/ha)")
    ph_sol: float = Field(..., ge=2, le=11, description="pH du sol")
    matiere_org: float = Field(..., ge=0, le=100, description="Matière organique (%)")
    densite_semis: float = Field(
        ..., ge=1, le=600, description="Densité de semis (gr/m²)"
    )
    type_sol: str = Field(..., description="Type de sol (ex: Limoneux)")
