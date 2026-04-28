"""
Tests des endpoints de l'API REST.
Lancer depuis la racine du projet : python -m pytest tests/ -v
ou directement                    : python -m pytest tests/test_api.py -v
"""

import io
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from fastapi.testclient import TestClient
from api import app


# déclenche le lifespan (chargement modèles + init_db)


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


CULTURE_OK = dict(
    temperature=18.0,
    pluviometrie=700.0,
    azote=120.0,
    ph_sol=6.5,
    matiere_org=3.0,
    densite_semis=180.0,
    type_sol="Limoneux",
)

VACHE_OK = dict(
    production=28.0,
    taux_tb=40.0,
    taux_tp=33.0,
    temperature_v=38.3,
    ccs=80.0,
    bcs=3.0,
    age_mois=60.0,
    lactation_j=120.0,
)


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    print("[root] OK")


def test_culture_ok(client):
    r = client.post("/culture", json=CULTURE_OK)
    assert r.status_code == 200
    body = r.json()
    for key in (
        "id",
        "rendement_pred",
        "rend_opt",
        "ecart",
        "risque_score",
        "risque_label",
        "conseils",
    ):
        assert key in body, f"Clé manquante : {key}"
    assert isinstance(body["id"], int) and body["id"] > 0
    assert isinstance(body["conseils"], list)
    print(f"[culture] OK : id={body['id']} rendement={body['rendement_pred']}")


def test_culture_type_sol_invalide(client):
    payload = {**CULTURE_OK, "type_sol": "Humifère"}
    r = client.post("/culture", json=payload)
    assert r.status_code == 422
    print("[culture] type_sol invalide → 422 OK")


def test_culture_temperature_hors_bornes(client):
    payload = {**CULTURE_OK, "temperature": 99.0}
    r = client.post("/culture", json=payload)
    assert r.status_code == 422
    print("[culture] température hors bornes → 422 OK")


def test_culture_champ_manquant(client):
    payload = {k: v for k, v in CULTURE_OK.items() if k != "azote"}
    r = client.post("/culture", json=payload)
    assert r.status_code == 422
    print("[culture] champ manquant → 422 OK")


def test_vache_ok(client):
    r = client.post("/vache", json=VACHE_OK)
    assert r.status_code == 200
    body = r.json()
    for key in ("id", "score_sante", "prediction", "statut", "priorite", "alertes"):
        assert key in body, f"Clé manquante : {key}"
    assert isinstance(body["id"], int) and body["id"] > 0
    assert body["prediction"] in (0, 1)
    assert isinstance(body["alertes"], list)
    print(
        f"[vache] OK : id={body['id']} score={body['score_sante']} statut={body['statut']}"
    )


def test_vache_ccs_hors_bornes(client):
    payload = {**VACHE_OK, "ccs": 99999.0}
    r = client.post("/vache", json=payload)
    assert r.status_code == 422
    print("[vache] ccs hors bornes → 422 OK")


def test_vache_champ_manquant(client):
    payload = {k: v for k, v in VACHE_OK.items() if k != "bcs"}
    r = client.post("/vache", json=payload)
    assert r.status_code == 422
    print("[vache] champ manquant → 422 OK")


def _image_png_1x1() -> bytes:
    """Génère un PNG 1x1 pixel en mémoire sans dépendance externe."""
    from PIL import Image as PILImage

    buf = io.BytesIO()
    PILImage.new("RGB", (1, 1), color=(100, 150, 80)).save(buf, format="PNG")
    return buf.getvalue()


def test_feuille_ok(client):
    img_bytes = _image_png_1x1()
    r = client.post(
        "/feuille",
        files={"file": ("feuille.png", img_bytes, "image/png")},
    )
    assert r.status_code == 200
    body = r.json()
    for key in ("id", "plante", "etat", "confiance", "top_k"):
        assert key in body, f"Clé manquante : {key}"
    assert isinstance(body["id"], int) and body["id"] > 0
    assert isinstance(body["top_k"], list)
    assert 0.0 <= body["confiance"] <= 100.0
    print(f"[feuille] OK : id={body['id']} plante={body['plante']} etat={body['etat']}")


def test_feuille_pas_image(client):
    r = client.post(
        "/feuille",
        files={"file": ("data.csv", b"a,b,c", "text/csv")},
    )
    assert r.status_code == 422
    print("[feuille] fichier non-image → 422 OK")


def test_feuille_image_trop_lourde(client):
    big = b"0" * 6_000_000
    r = client.post(
        "/feuille",
        files={"file": ("big.jpg", big, "image/jpeg")},
    )
    assert r.status_code == 413
    print("[feuille] image > 5 Mo → 413 OK")


def test_feuille_sans_fichier(client):
    r = client.post("/feuille")
    assert r.status_code == 422
    print("[feuille] sans fichier → 422 OK")
