import json
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
 
from config import (
    FEUILLE_MODEL_PATH, FEUILLE_CLASSES_PATH,
    FEUILLE_IMG_SIZE, FEUILLE_DEVICE,
)
from models.feuille import build_model
 
DEVICE = FEUILLE_DEVICE


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
    "Tomato___Spider_mites Two-spotted_spider_mite": ("Tomate", "Acariens (tétranyques)"),
    "Tomato___Target_Spot": ("Tomate", "Tache cible"),
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ("Tomate", "Virus enroulement jaune"),
    "Tomato___Tomato_mosaic_virus": ("Tomate", "Virus mosaïque"),
    "Tomato___healthy": ("Tomate", "Sain"),
}



