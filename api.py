"""
API REST — AI Assist Agricultural
Endpoints : /culture  /vache  /feuille
"""

import warnings
import io
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

from database import (
    init_db,
    save_analyse_culture,
    save_analyse_vache,
    save_analyse_feuille,
)
from models.culture import load_or_train_culture
from models.troupeau import load_or_train_troupeau
from services.culture_service import predire_rendement
from services.vache_service import analyser_vache
from services.feuille_service import analyser_feuille
from reports.pdf_report import calcul_risque_agronomique, calcul_priorite

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
        ..., ge=1, le=600, description="Densité de semis (gr/m²)"
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
        "event=culture_analysis status=start temp=%.1f sol=%s",
        data.temperature,
        data.type_sol,
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

        risque_score, risque_label, _= calcul_risque_agronomique(
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
            "event=culture_analysis status=success id=%s metrics={score=%.2f, optimal=%.2f, gap=%.2f}",
            row["id"],
            row["rendement_pred"],
            rend_opt,
            ecart,
        )
        return row

    except Exception as e:
        logger.exception("event=culture_analysis status=error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vache", tags=["Vache"])
async def analyse_vache(data: VacheInput, request: Request):
    logger.info(
        "event=vache_analysis status=start production=%.1f ccs=%.0f",
        data.production,
        data.ccs,
    )
    try:
        alertes, prediction, score_sante = analyser_vache(
            request.app.state.model_vache,
            request.app.state.scaler_vache,
            request.app.state.score_min,
            request.app.state.score_max,
            data.production,
            data.taux_tb,
            data.taux_tp,
            data.temperature_v,
            data.ccs,
            data.bcs,
            data.age_mois,
            data.lactation_j,
        )

        statut = "Normal" if prediction == 1 else "Anomalie détectée"
        priorite, _, _ = calcul_priorite(
            score_sante,
            data.ccs,
            data.temperature_v,
            data.bcs,
            data.production,
        )

        row = {
            "production": data.production,
            "taux_tb": data.taux_tb,
            "taux_tp": data.taux_tp,
            "temperature_v": data.temperature_v,
            "ccs": data.ccs,
            "bcs": data.bcs,
            "age_mois": data.age_mois,
            "lactation_j": data.lactation_j,
            "score_sante": score_sante,
            "prediction": int(prediction),
            "statut": statut,
            "priorite": priorite,
            "alertes": alertes,
        }

        row["id"] = save_analyse_vache(row)
        logger.info(
            "event=vache_analysis status=success id=%s metrics={score=%.3f, prediction=%d, priority=%s}",
            row["id"],
            row["score_sante"],
            prediction,
            priorite,
        )
        return row

    except Exception as e:
        logger.exception("event=vache_analysis status=error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feuille", tags=["Feuille"])
async def analyse_feuille(file: UploadFile = File(..., description="Photo de feuille (jpg/png)")):
    logger.info(
        "event=feuille_analysis status=start fichier=%s",
        file.filename,
    )
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=422, detail="Le fichier doit être une image (jpg, png...)")
    contents = await file.read()
    if len(contents) > 5_000_000:
        raise HTTPException(status_code=413, detail="Image trop lourde (max 5 Mo)")
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        logger.exception("event=feuille_analysis status=error reason=unreadable_image")
        raise HTTPException(status_code=422, detail="Image illisible ou format non supporté")
    try:
        resultat = analyser_feuille(image)
    except RuntimeError as e:
        logger.exception("event=feuille_analysis status=error reason=model_failure")
        raise HTTPException(status_code=503, detail=str(e))
    if not resultat.get("success"):
        logger.error(
            "event=feuille_analysis status=error reason=analyse_failure detail=%s",
            resultat.get("error"),
        )
        raise HTTPException(status_code=500, detail=resultat.get("error", "Erreur analyse"))
 
    resultat["id"] = save_analyse_feuille(resultat)
    logger.info(
        "event=feuille_analysis status=success id=%s metrics={plante=%s, etat=%s, confiance=%.2f}",
        resultat["id"],
        resultat["plante"],
        resultat["etat"],
        resultat["confiance"],
    )
    resultat.pop("success")
    return resultat
    