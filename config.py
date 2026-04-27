from reportlab.lib import colors
import torch

SOLS = {0: "Limoneux", 1: "Argileux", 2: "Sableux", 3: "Calcaire"}
SOL_IDX = {v: k for k, v in SOLS.items()}


# Palette couleurs utilisée dans pdf_report.py et viz/
PALETTE = {
    "vert"         : colors.HexColor("#2e7d32"),
    "vert_clair"   : colors.HexColor("#e8f5e9"),
    "rouge"        : colors.HexColor("#c62828"),
    "rouge_clair"  : colors.HexColor("#ffebee"),
    "orange"       : colors.HexColor("#e65100"),
    "orange_clair" : colors.HexColor("#fff3e0"),
    "bleu"         : colors.HexColor("#1565c0"),
    "gris_fond"    : colors.HexColor("#f5f5f5"),
    "gris_bord"    : colors.HexColor("#bdbdbd"),
    "blanc"        : colors.white,
    "noir"         : colors.HexColor("#212121"),
}


PALETTE_HEX = {
    "vert"         : "#2e7d32",
    "vert_clair"   : "#81c784",
    "rouge"        : "#c62828",
    "orange"       : "#e65100",
    "bleu"         : "#1565c0",
    "gris"         : "#bdbdbd",
}

#Module Feuille
FEUILLE_DATA_DIR = "data/plantvillage/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"
FEUILLE_TRAIN_DIR = f"{FEUILLE_DATA_DIR}/train"
FEUILLE_VALID_DIR = f"{FEUILLE_DATA_DIR}/valid"
FEUILLE_MODEL_PATH = "models/saved/feuille_model.pth"
FEUILLE_CLASSES_PATH = "models/saved/feuille_classes.json"

FEUILLE_BATCH_SIZE = 32
FEUILLE_EPOCHS = 25
FEUILLE_LR = 1e-3
FEUILLE_IMG_SIZE = 224
FEUILLE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEUILLE_CONF_MIN     = 50.0   # % en dessous duquel le résultat est "Incertain"