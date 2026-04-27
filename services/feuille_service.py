import json
import torch
from torchvision import transforms
import logging
from PIL import Image

from config import (
    FEUILLE_MODEL_PATH,
    FEUILLE_CLASSES_PATH,
    FEUILLE_IMG_SIZE,
    FEUILLE_DEVICE,
    FEUILLE_CONF_MIN,
)
from models.feuille import build_model

DEVICE = FEUILLE_DEVICE

logger = logging.getLogger(__name__)


# Noms lisibles par classe (plante, état)
CLASS_LABELS = {
    "Apple___Apple_scab": ("Pommier", "Tavelure"),
    "Apple___Black_rot": ("Pommier", "Pourriture noire"),
    "Apple___Cedar_apple_rust": ("Pommier", "Rouille gymnosporange"),
    "Apple___healthy": ("Pommier", "Sain"),
    "Blueberry___healthy": ("Myrtille", "Sain"),
    "Cherry_(including_sour)___Powdery_mildew": ("Cerisier", "Oïdium"),
    "Cherry_(including_sour)___healthy": ("Cerisier", "Sain"),
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": ("Maïs", "Cercosporiose"),
    "Corn_(maize)___Common_rust_": ("Maïs", "Rouille commune"),
    "Corn_(maize)___Northern_Leaf_Blight": ("Maïs", "Helminthosporiose"),
    "Corn_(maize)___healthy": ("Maïs", "Sain"),
    "Grape___Black_rot": ("Vigne", "Pourriture noire"),
    "Grape___Esca_(Black_Measles)": ("Vigne", "Esca (rougeot parasitaire)"),
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": ("Vigne", "Brûlure foliaire"),
    "Grape___healthy": ("Vigne", "Sain"),
    "Orange___Haunglongbing_(Citrus_greening)": ("Oranger", "Huanglongbing (greening)"),
    "Peach___Bacterial_spot": ("Pêcher", "Tache bactérienne"),
    "Peach___healthy": ("Pêcher", "Sain"),
    "Pepper,_bell___Bacterial_spot": ("Poivron", "Tache bactérienne"),
    "Pepper,_bell___healthy": ("Poivron", "Sain"),
    "Potato___Early_blight": ("Pomme de terre", "Alternariose précoce"),
    "Potato___Late_blight": ("Pomme de terre", "Mildiou"),
    "Potato___healthy": ("Pomme de terre", "Sain"),
    "Raspberry___healthy": ("Framboisier", "Sain"),
    "Soybean___healthy": ("Soja", "Sain"),
    "Squash___Powdery_mildew": ("Courge", "Oïdium"),
    "Strawberry___Leaf_scorch": ("Fraisier", "Brûlure foliaire"),
    "Strawberry___healthy": ("Fraisier", "Sain"),
    "Tomato___Bacterial_spot": ("Tomate", "Tache bactérienne"),
    "Tomato___Early_blight": ("Tomate", "Alternariose précoce"),
    "Tomato___Late_blight": ("Tomate", "Mildiou"),
    "Tomato___Leaf_Mold": ("Tomate", "Moisissure foliaire"),
    "Tomato___Septoria_leaf_spot": ("Tomate", "Septoriose"),
    "Tomato___Spider_mites Two-spotted_spider_mite": (
        "Tomate",
        "Acariens (tétranyques)",
    ),
    "Tomato___Target_Spot": ("Tomate", "Tache cible"),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ("Tomate", "Virus enroulement jaune"),
    "Tomato___Tomato_mosaic_virus": ("Tomate", "Virus mosaïque"),
    "Tomato___healthy": ("Tomate", "Sain"),
}

from functools import lru_cache


@lru_cache(maxsize=1)
def _load_model():
    try:
        with open(FEUILLE_CLASSES_PATH) as f:
            classes = json.load(f)
        checkpoint = torch.load(
            FEUILLE_MODEL_PATH, map_location=DEVICE, weights_only=True
        )
        model = build_model(len(classes))  # build_model gère le .to(DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, classes
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Modèle ou classes introuvable : {e}. "
            "Lancez d'abord : python -m models.feuille"
        ) from e
    except Exception as e:
        logger.exception("Erreur inattendue lors du chargement du modèle")
        raise RuntimeError(f"Erreur lors du chargement du modèle : {e}") from e


_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((FEUILLE_IMG_SIZE, FEUILLE_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def _preprocess(image: Image.Image) -> torch.Tensor:
    return _TRANSFORM(image.convert("RGB")).unsqueeze(0).to(DEVICE)


def analyser_feuille(image: Image.Image, top_k: int = 3) -> dict:
    """
    Analyse une image de feuille et retourne la prédiction principale + top-k.

    Retourne un dict :
        plante: str
        etat: str
        confiance: float (0-100)
        sain: bool
        top_k: list of dict
    """
    try:
        model, classes = _load_model()
        top_k = min(top_k, len(classes))

        tensor = _preprocess(image)

        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1)[0]

        top_indices = probs.argsort(descending=True)[:top_k].tolist()

        top_resultats = []
        for idx in top_indices:
            classe = classes[idx]
            plante, etat = CLASS_LABELS.get(classe, (classe, "Inconnu"))

            top_resultats.append(
                {
                    "classe": classe,
                    "plante": plante,
                    "etat": etat,
                    "confiance": round(probs[idx].item() * 100, 2),
                }
            )

        meilleur = top_resultats[0]
        etat_final = (
            "Incertain"
            if meilleur["confiance"] < FEUILLE_CONF_MIN
            else meilleur["etat"]
        )
        is_sain = meilleur["etat"] == "Sain"
        return {
            "success": True,
            "plante": meilleur["plante"],
            "etat": etat_final,
            "confiance": meilleur["confiance"],
            "sain": is_sain,
            "top_k": top_resultats,
        }
    except RuntimeError:
        raise
    except (OSError, ValueError, TypeError) as e:
        return {"error": f"Image invalide ou format non supporté : {e}"}
    except Exception as e:
        logger.exception("Erreur inattendue lors de l'analyse de l'image")
        raise RuntimeError(f"Erreur inattendue lors de l'analyse : {e}") from e
