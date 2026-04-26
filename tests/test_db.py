"""
Tests basiques pour vérifier le bon fonctionnement de la base de données.
Lancer depuis la racine du projet : python -m pytest tests/ -v
ou directement                    : python tests/test_db.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database import init_db, save_analyse_culture, save_analyse_vache, engine
from sqlalchemy.orm import Session
from database import AnalyseCulture, AnalyseVache


def test_save_culture():
    init_db()
    data = dict(
        temperature=15.0, pluviometrie=600.0, azote=160.0,
        ph_sol=6.7, matiere_org=2.5, densite_semis=210.0,
        type_sol="Limoneux", rendement_pred=8.2, rend_opt=9.1,
        ecart=10.8, conseils=["Paramètres globalement favorables"],
        risque_score=18, risque_label="Faible",
    )
    row_id = save_analyse_culture(data)
    assert isinstance(row_id, int) and row_id > 0

    with Session(engine) as session:
        row = session.get(AnalyseCulture, row_id)
        assert row is not None
        assert row.type_sol == "Limoneux"
        assert row.risque_label == "Faible"
        assert row.created_at is not None
    print(f"[culture] OK — id={row_id}")


def test_save_vache():
    init_db()
    data = dict(
        production=28.0, taux_tb=40.0, taux_tp=33.0,
        temperature_v=38.3, ccs=80.0, bcs=3.0,
        age_mois=60.0, lactation_j=120.0,
        score_sante=85, prediction=1,
        statut="NORMAL", priorite="Suivi normal",
        alertes=["Aucune anomalie détectée"],
    )
    row_id = save_analyse_vache(data)
    assert isinstance(row_id, int) and row_id > 0

    with Session(engine) as session:
        row = session.get(AnalyseVache, row_id)
        assert row is not None
        assert row.statut == "NORMAL"
        assert row.score_sante == 85
        assert row.created_at is not None
    print(f"[vache]   OK — id={row_id}")


def test_lecture_toutes_lignes():
    init_db()
    with Session(engine) as session:
        cultures = session.query(AnalyseCulture).all()
        vaches   = session.query(AnalyseVache).all()
    print(f"[lecture] {len(cultures)} analyse(s) culture, {len(vaches)} analyse(s) vache en base")
    assert len(cultures) >= 1
    assert len(vaches)   >= 1


if __name__ == "__main__":
    test_save_culture()
    test_save_vache()
    test_lecture_toutes_lignes()
    print("\nTous les tests sont passés ✓")