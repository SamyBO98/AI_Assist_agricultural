# AI Assist Agricultural

Plateforme d'aide à la décision agricole basée sur le machine learning.  
Deux modules indépendants : **prédiction de rendement céréalier** et **détection d'anomalies sur vaches laitières**.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-ff6b6b?logo=gradio&logoColor=white)
![License](https://img.shields.io/badge/licence-MIT-green)

## Aperçu

| Module Culture | Module Vache |
|---|---|
| Prédiction de rendement + importance des facteurs | Radar de santé + score d'anomalie |

## Fonctionnalités

- **Prédiction de rendement** : estime le rendement en t/ha à partir de paramètres agronomiques (température, pluviométrie, azote, pH, matière organique, densité de semis, type de sol)
- **Détection d'anomalies** : analyse le profil d'une vache laitière et détecte les situations à risque (mammite, fièvre, cétose...)
- **Visualisations interactives** : graphiques d'importance des variables, comparaison aux références nationales, radar de santé, jauge de score
- **Recommandations automatiques** : conseils agronomiques et alertes vétérinaires contextualisés
- **Export PDF** : rapport d'analyse complet avec graphiques, indice de risque agronomique (Culture) et priorité d'intervention (Vache)
- **Validation des inputs** : vérification des bornes physiques extrêmes pour chaque indicateur, avec message d'erreur contextuel
- **Persistance des modèles** : les modèles sont sauvegardés au premier lancement et rechargés directement ensuite, pas de réentraînement inutile
- **Scénarios de démonstration** : boutons Bon / Moyen / Mauvais pour tester l'application rapidement

## Structure du projet

```
AI_Assist_agricultural/
│
├── app.py                      # Interface Gradio, point d'entrée
├── config.py                   # Configuration globale (types de sol, palette couleurs)
├── requirements.txt            # Dépendances Python
│
├── data/
│   └── generate_data.py        # Génération des données d'entraînement
│
├── models/
│   ├── culture.py              # Entraînement / chargement modèle culture
│   ├── troupeau.py             # Entraînement / chargement modèle vache
│   └── saved/                  # Modèles sérialisés (joblib), auto-généré
│
├── services/
│   ├── culture_service.py      # Logique métier culture (prédiction + conseils)
│   └── vache_service.py        # Logique métier vache (analyse + alertes)
│
├── reports/
│   └── pdf_report.py           # Génération des rapports PDF (reportlab)
│
└── viz/
    ├── culture_viz.py          # Graphiques culture
    └── vache_viz.py            # Graphiques vache
```

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/votre-utilisateur/AI_Assist_agricultural.git
cd AI_Assist_agricultural
```

### 2. Créer un environnement virtuel (recommandé)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Lancer l'application

```bash
python app.py
```

L'interface est accessible à l'adresse **http://localhost:7860**

> Au premier lancement, les modèles s'entraînent et se sauvegardent dans `models/saved/`.  
> Les lancements suivants chargent directement les fichiers, démarrage quasi-instantané.

## Modèles

### Culture - `HistGradientBoostingRegressor`

Régression supervisée pour prédire le rendement en t/ha.

| Paramètre | Valeur |
|---|---|
| Algorithme | `HistGradientBoostingRegressor` |
| `max_iter` | 200 |
| `learning_rate` | 0.08 |
| `max_depth` | 4 |
| Métrique principale | R² ≈ 0.86, MAE ≈ 0.38 t/ha |

**Features :** température, pluviométrie, azote, pH sol, matière organique, densité semis, type de sol

> Les importances par permutation sont précalculées à l'entraînement et sauvegardées dans le pkl -> pas de recalcul à chaque prédiction.

### Vache - `IsolationForest`

Détection non supervisée d'anomalies sur le profil d'une vache laitière.

| Paramètre | Valeur |
|---|---|
| Algorithme | `IsolationForest` |
| `n_estimators` | 200 |
| `contamination` | 0.08 (8 % d'anomalies attendues) |

**Features :** production, TB, TP, température corporelle, CCS, BCS, âge, stade de lactation

## Référence des indicateurs

### Culture

| Indicateur | Unité | Valeur typique | Bornes physiques |
|---|---|---|---|
| Température | °C | 12-18 °C | -15 à 50 °C |
| Pluviométrie | mm/an | 400-700 mm | 0 à 12 000 mm |
| Azote | kg N/ha | 120-200 kg/ha | 0 à 400 kg/ha |
| pH sol | - | 6.5-7.0 (idéal céréales) | 2 à 11 |
| Matière organique | % | 1.5-4 % | 0 à 100 % |
| Densité semis | grains/m² | 180-260 | 1 à 600 |

### Vache laitière

| Indicateur | Signification | Seuil d'alerte | Bornes physiques |
|---|---|---|---|
| **TB** (Taux Butyreux) | % matières grasses du lait | < 30 g/kg : acidose possible | 20 à 80 g/kg |
| **TP** (Taux Protéique) | % protéines du lait | < 28 g/kg : déficit énergétique | 20 à 60 g/kg |
| **CCS** (Cellules Somatiques) | Indicateur d'infection mammaire | > 200 k/mL : surveillance, > 400 k/mL : mammite | 10 à 10 000 k/mL |
| **BCS** (Body Condition Score) | État d'engraissement (1-5) | < 2.0 : maigreur, > 4.0 : surpoids | 1 à 5 |
| Température | Température corporelle | > 39.5 °C : fièvre | 35 à 42 °C |

## Limites connues

- Les données d'entraînement sont **synthétiques** : le modèle illustre le fonctionnement mais ne reflète pas la réalité agronomique terrain
- Pour un usage en production, réentraîner les modèles sur de vraies données (ex. [Agreste](https://agreste.agriculture.gouv.fr/))
- L'indice de risque agronomique utilise une normalisation symétrique —> en réalité les effets sont asymétriques (ex. excès d'azote ≠ manque d'azote)

## Pistes d'évolution

- [ ] Intégration de données réelles open-data
- [ ] Module de détection de maladies foliaires par image (PlantVillage)
- [ ] API REST pour intégration dans d'autres outils
- [ ] Historique des analyses par exploitation

