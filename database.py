from sqlalchemy import (
    create_engine, Column, Integer, Float, String, DateTime, Text
)
from sqlalchemy.orm import DeclarativeBase, Session
from datetime import datetime, timezone

import os
os.makedirs("db", exist_ok=True)

DATABASE_URL = "sqlite:///db/agricole.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})


class Base(DeclarativeBase):
    pass


class AnalyseCulture(Base):
    __tablename__ = "analyses_culture"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    created_at      = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    # Paramètres saisis
    temperature     = Column(Float, nullable=False)
    pluviometrie    = Column(Float, nullable=False)
    azote           = Column(Float, nullable=False)
    ph_sol          = Column(Float, nullable=False)
    matiere_org     = Column(Float, nullable=False)
    densite_semis   = Column(Float, nullable=False)
    type_sol        = Column(String(20), nullable=False)

    # Résultats
    rendement_pred  = Column(Float, nullable=False)
    rend_opt        = Column(Float, nullable=False)
    ecart_pct       = Column(Float, nullable=False)   # % vs moyenne FR
    risque_score    = Column(Integer, nullable=False)  # 0-100
    risque_label    = Column(String(20), nullable=False)  # Faible / Modéré / Élevé / Critique
    conseils        = Column(Text, nullable=False)     # JSON liste


class AnalyseVache(Base):
    __tablename__ = "analyses_vache"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    created_at      = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    # Paramètres saisis
    production      = Column(Float, nullable=False)
    taux_tb         = Column(Float, nullable=False)
    taux_tp         = Column(Float, nullable=False)
    temperature_v   = Column(Float, nullable=False)
    ccs             = Column(Float, nullable=False)
    bcs             = Column(Float, nullable=False)
    age_mois        = Column(Float, nullable=False)
    lactation_j     = Column(Float, nullable=False)

    # Résultats
    score_sante     = Column(Integer, nullable=False)   # 0-100
    prediction      = Column(Integer, nullable=False)   # 1=normal, -1=anomalie
    statut          = Column(String(30), nullable=False) # NORMAL / SURVEILLANCE / ANOMALIE
    priorite        = Column(String(30), nullable=False) # Suivi normal / Surveillance / Intervention
    alertes         = Column(Text, nullable=False)       # JSON liste


def init_db():
    """Crée les tables si elles n'existent pas encore."""
    Base.metadata.create_all(engine)


def save_analyse_culture(data: dict) -> int:
    """Persiste une analyse culture. Retourne l'id inséré."""
    import json
    with Session(engine) as session:
        row = AnalyseCulture(
            temperature=data["temperature"],
            pluviometrie=data["pluviometrie"],
            azote=data["azote"],
            ph_sol=data["ph_sol"],
            matiere_org=data["matiere_org"],
            densite_semis=data["densite_semis"],
            type_sol=data["type_sol"],
            rendement_pred=data["rendement_pred"],
            rend_opt=data["rend_opt"],
            ecart_pct=data["ecart"],
            risque_score=data["risque_score"],
            risque_label=data["risque_label"],
            conseils=json.dumps(data["conseils"], ensure_ascii=False),
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return row.id


def save_analyse_vache(data: dict) -> int:
    """Persiste une analyse vache. Retourne l'id inséré."""
    import json
    with Session(engine) as session:
        row = AnalyseVache(
            production=data["production"],
            taux_tb=data["taux_tb"],
            taux_tp=data["taux_tp"],
            temperature_v=data["temperature_v"],
            ccs=data["ccs"],
            bcs=data["bcs"],
            age_mois=data["age_mois"],
            lactation_j=data["lactation_j"],
            score_sante=data["score_sante"],
            prediction=data["prediction"],
            statut=data["statut"],
            priorite=data["priorite"],
            alertes=json.dumps(data["alertes"], ensure_ascii=False),
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        return row.id