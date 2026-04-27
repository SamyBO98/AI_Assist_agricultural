"""
API REST — AI Assist Agricultural
Endpoints : /culture  /vache  /feuille
"""

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from database import init_db, save_analyse_culture
from services.culture_service import predire_rendement
from reports.pdf_report import calcul_risque_agronomique
from contextlib import asynccontextmanager
from typing import Literal
from models.culture import load_or_train_culture
from models.troupeau import load_or_train_troupeau
from pydantic import BaseModel, Field
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    app.state.model_culture, app.state.scaler_culture, _, _, _ = load_or_train_culture()
    (
        app.state.model_vache,
        app.state.scaler_vache,
        app.state.score_min,
        app.state.score_max,
    ) = load_or_train_troupeau()
    yield


app = FastAPI(
    title="AI Assist Agricultural",
    description="API de prédiction agricole : Culture, Vache laitière, Maladie foliaire",
    version="1.0.0",
    lifespan=lifespan,
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
        ..., ge=1, le=600, description="Densité de semis (g/m²)"
    )
    type_sol: Literal["Limoneux", "Argileux", "Sableux", "Calcaire"] = Field(
        ..., description="Type de sol"
    )


class VacheInput(BaseModel):
    production: float = Field(..., ge=0, le=90, description="Production lait (L/j)")
    taux_tb: float = Field(..., ge=20, le=80, description="Taux butyreux (g/kg)")
    taux_tp: float = Field(..., ge=20, le=60, description="Taux protéique (g/kg)")
    temperature_v: float = Field(
        ..., ge=35, le=42, description="Température corporelle (°C)"
    )
    ccs: float = Field(..., ge=10, le=10000, description="Cellules somatiques (k/mL)")
    bcs: float = Field(..., ge=1, le=5, description="Body Condition Score")
    age_mois: float = Field(..., ge=12, le=240, description="Âge (mois)")
    lactation_j: float = Field(
        ..., ge=1, le=500, description="Stade de lactation (jours)"
    )


@app.get("/", tags=["Santé"])
def root():
    return {"status": "ok", "modules": ["/culture", "/vache", "/feuille"]}


@app.post("/culture", tags=["Culture"])
async def analyse_culture(data: CultureInput, request: Request):
    logger.info(
        "Analyse culture lancée : sol=%s temp=%.1f", data.type_sol, data.temperature
    )
    try:
        rendement_pred, rend_opt, ecart, conseils = predire_rendement(
            request.app.state.model_culture,
            request.app.state.scaler_culture,
            data.temperature,
            data.pluviometrie,
            data.azote,
            data.ph_sol,
            data.matiere_org,
            data.densite_semis,
            data.type_sol,
        )

        risque_score, risque_label, _, _ = calcul_risque_agronomique(
            data.temperature,
            data.pluviometrie,
            data.azote,
            data.ph_sol,
            data.matiere_org,
            data.densite_semis,
        )

        row = {
            "temperature": data.temperature,
            "pluviometrie": data.pluviometrie,
            "azote": data.azote,
            "ph_sol": data.ph_sol,
            "matiere_org": data.matiere_org,
            "densite_semis": data.densite_semis,
            "type_sol": data.type_sol,
            "rendement_pred": round(rendement_pred, 2),
            "rend_opt": round(rend_opt, 2),
            "ecart": round(ecart, 2),
            "risque_score": risque_score,
            "risque_label": risque_label,
            "conseils": conseils,
        }

        row["id"] = save_analyse_culture(row)
        logger.info(
            "Analyse culture OK : id=%s rendement=%.2f",
            row["id"],
            row["rendement_pred"],
        )
        return row

    except Exception as e:
        logger.exception("Erreur critique sur /culture")
        raise HTTPException(status_code=500, detail=str(e))
