from reportlab.lib import colors

SOLS = {0: "Limoneux", 1: "Argileux", 2: "Sableux", 3: "Calcaire"}
SOL_IDX = {v: k for k, v in SOLS.items()}


# Palette couleurs — utilisée dans pdf_report.py et viz/
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