import os
from datetime import datetime, timezone

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    String,
    DateTime,
    JSON
)
from sqlalchemy.orm import DeclarativeBase,sessionmaker 



os.makedirs("db", exist_ok=True)

DATABASE_URL = "sqlite:///db/agricole.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(bind=engine)


class Base(DeclarativeBase):
    pass



class AnalyseCulture(Base):
    __tablename__ = "analyses_culture"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    # Inputs
    temperature = Column(Float, nullable=False)
    pluviometrie = Column(Float, nullable=False)
    azote = Column(Float, nullable=False)
    ph_sol = Column(Float, nullable=False)
    matiere_org = Column(Float, nullable=False)
    densite_semis = Column(Float, nullable=False)
    type_sol = Column(String(20), nullable=False)

    # Outputs
    rendement_pred = Column(Float, nullable=False)
    rend_opt = Column(Float, nullable=False)
    ecart_pct = Column(Float, nullable=False)
    risque_score = Column(Integer, nullable=False)
    risque_label = Column(String(20), nullable=False)
    conseils = Column(JSON, nullable=False)   # JSON list


class AnalyseVache(Base):
    __tablename__ = "analyses_vache"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    # Inputs
    production = Column(Float, nullable=False)
    taux_tb = Column(Float, nullable=False)
    taux_tp = Column(Float, nullable=False)
    temperature_v = Column(Float, nullable=False)
    ccs = Column(Float, nullable=False)
    bcs = Column(Float, nullable=False)
    age_mois = Column(Float, nullable=False)
    lactation_j = Column(Float, nullable=False)

    # Outputs
    score_sante = Column(Integer, nullable=False)
    prediction = Column(Integer, nullable=False)
    statut = Column(String(30), nullable=False)
    priorite = Column(String(30), nullable=False)
    alertes = Column(JSON, nullable=False)    # JSON list


class AnalyseFeuille(Base):
    __tablename__ = "analyses_feuille"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    # Results
    plante = Column(String(50), nullable=False)
    etat = Column(String(50), nullable=False)
    confiance = Column(Float, nullable=False)
    sain = Column(Integer, nullable=False)
    top_k = Column(JSON, nullable=False)  # JSON list



def init_db():
    """Crée les tables si elles n'existent pas."""
    Base.metadata.create_all(engine)


def get_session():
    """Retourne une nouvelle session SQLAlchemy."""
    return SessionLocal()


def save_analyse_culture(data: dict) -> int:
    """Persiste une analyse culture et retourne l'id."""
    with get_session() as session:
        try:
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
                conseils=data["conseils"],
            )

            session.add(row)
            session.commit()
            session.refresh(row)
            return row.id
        except Exception:
            session.rollback()
            raise


def save_analyse_vache(data: dict) -> int:
    """Persiste une analyse vache et retourne l'id."""
    with get_session() as session:
        try:
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
                alertes=data["alertes"],
            )

            session.add(row)
            session.commit()
            session.refresh(row)
            return row.id
        except Exception:
            session.rollback()
            raise
    
def save_analyse_feuille(data: dict) -> int:
    """Persiste une analyse feuille. Retourne l'id inséré."""
    with get_session() as session:
        try:
            row = AnalyseFeuille(
                plante=data["plante"],
                etat=data["etat"],
                confiance=data["confiance"],
                sain=int(data["sain"]),
                top_k=data["top_k"],
            )
            session.add(row)
            session.commit()
            session.refresh(row)
            return row.id
        except Exception:
            session.rollback()
            raise
