# AI Assist Agricultural

Plateforme d'aide à la décision agricole basée sur le machine learning.
Trois modules indépendants : **prédiction de rendement céréalier**, **détection d'anomalies sur vaches laitières** et **détection de maladies foliaires par image**.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-ff6b6b?logo=gradio&logoColor=white)
![License](https://img.shields.io/badge/licence-MIT-green)

## Aperçu

| Module Culture | Module Vache | Module Feuille |
|---|---|---|
| Prédiction de rendement + importance des facteurs | Radar de santé + score d'anomalie | Détection de maladies foliaires par image |

## Fonctionnalités

- **Prédiction de rendement** : estime le rendement en t/ha à partir de paramètres agronomiques (température, pluviométrie, azote, pH, matière organique, densité de semis, type de sol)
- **Détection d'anomalies** : analyse le profil d'une vache laitière et détecte les situations à risque (mammite, fièvre, cétose...)
- **Détection de maladies foliaires** : classification d'une photo de feuille parmi 38 classes (plante saine ou maladie identifiée) — intégration UI en cours
- **Visualisations interactives** : graphiques d'importance des variables, comparaison aux références nationales, radar de santé, jauge de score
- **Recommandations automatiques** : conseils agronomiques et alertes vétérinaires contextualisés
- **Export PDF** : rapport d'analyse complet avec graphiques, indice de risque agronomique (Culture) et priorité d'intervention (Vache)
- **Validation des inputs** : vérification des bornes physiques extrêmes pour chaque indicateur, avec message d'erreur contextuel
- **Persistance des modèles** : les modèles sont sauvegardés au premier lancement et rechargés directement ensuite, pas de réentraînement inutile
- **Scénarios de démonstration** : boutons Bon / Moyen / Mauvais pour tester l'application rapidement

## Structure du projet

```
AI_Assist_agricultural/
|
+-- app.py                      # Interface Gradio, point d'entrée
+-- config.py                   # Configuration globale (types de sol, palette couleurs, hyperparamètres feuille)
+-- database.py                 # Modèles SQLAlchemy + fonctions de persistance
+-- requirements.txt            # Dépendances Python
+-- requirements-dev.txt        # Dépendances de développement (pytest, kaggle)
|
+-- data/
|   +-- generate_data.py        # Génération des données d'entraînement (culture/vache)
|   +-- plantvillage/           # Dataset PlantVillage 38 classes (non versionné)
|
+-- db/                         # Base de données SQLite (auto-générée, non versionnée)
|   +-- agricole.db
|
+-- models/
|   +-- culture.py              # Entraînement / chargement modèle culture
|   +-- feuille.py              # Entraînement / chargement modèle EfficientNetB0
|   +-- troupeau.py             # Entraînement / chargement modèle vache
|   +-- saved/                  # Modèles sérialisés (joblib / pth), auto-généré
|
+-- services/
|   +-- culture_service.py      # Logique métier culture (prédiction + conseils)
|   +-- feuille_service.py      # Logique métier feuille (inférence + top-3) — en cours
|   +-- vache_service.py        # Logique métier vache (analyse + alertes)
|
+-- reports/
|   +-- pdf_report.py           # Génération des rapports PDF (reportlab)
|
+-- tests/
|   +-- test_db.py              # Tests de la base de données
|
+-- viz/
    +-- culture_viz.py          # Graphiques culture
    +-- feuille_viz.py          # Graphiques feuille (top-3 confiance) — en cours
    +-- vache_viz.py            # Graphiques vache
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
# Production
pip install -r requirements.txt

# PyTorch avec support CUDA (NVIDIA GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Développement (inclut pytest + kaggle)
pip install -r requirements-dev.txt
```

### 4. Lancer l'application

```bash
python app.py
```

L'interface est accessible à l'adresse **http://localhost:7860**

> Au premier lancement, les modèles culture et vache s'entraînent et se sauvegardent dans `models/saved/`.
> Les lancements suivants chargent directement les fichiers, démarrage quasi-instantané.

## Modèles

### Culture - HistGradientBoostingRegressor

Régression supervisée pour prédire le rendement en t/ha.

| Paramètre | Valeur |
|---|---|
| Algorithme | HistGradientBoostingRegressor |
| max_iter | 200 |
| learning_rate | 0.08 |
| max_depth | 4 |
| Métrique principale | R² ≈ 0.86, MAE ≈ 0.38 t/ha |

Features : température, pluviométrie, azote, pH sol, matière organique, densité semis, type de sol

> Les importances par permutation sont précalculées à l'entraînement et sauvegardées dans le pkl, pas de recalcul à chaque prédiction.

### Vache - IsolationForest

Détection non supervisée d'anomalies sur le profil d'une vache laitière.

| Paramètre | Valeur |
|---|---|
| Algorithme | IsolationForest |
| n_estimators | 200 |
| contamination | 0.08 (8% d'anomalies attendues) |

Features : production, TB, TP, température corporelle, CCS, BCS, âge, stade de lactation

### Feuille - EfficientNetB0 (Transfer Learning)

Classification d'image par deep learning pour identifier 38 classes de maladies foliaires.

| Paramètre | Valeur |
|---|---|
| Architecture | EfficientNetB0 (pré-entraîné ImageNet) |
| Dataset | PlantVillage — 70 295 images train / 17 572 valid |
| Epochs | 25 (backbone gelé puis fine-tuning 5 dernières couches) |
| Optimizer | Adam + CosineAnnealingLR |
| Meilleure valid_acc | 99.9% |

Classes : 38 classes couvrant pommier, myrtille, cerisier, maïs, raisin, orange, pêcher, poivron, pomme de terre, framboise, soja, courge, fraisier, tomate.

#### Entraîner le modèle feuille

Le modèle n'est pas versionné et doit être entraîné localement avant de lancer l'application.

**1. Télécharger le dataset PlantVillage**

Créer un compte sur [kaggle.com](https://www.kaggle.com) puis générer un token API dans Settings > API > Create New Token. Créer le fichier de credentials :

```
Windows   : C:\Users\<utilisateur>\.kaggle\kaggle.json
Linux/Mac : ~/.kaggle/kaggle.json
```

Contenu du fichier :

```json
{"username": "votre_username", "key": "votre_token"}
```

Télécharger le dataset :

```bash
kaggle datasets download -d vipoooool/new-plant-diseases-dataset
```

Extraire dans `data/plantvillage/` de sorte que la structure soit :

```
data/plantvillage/
+-- New Plant Diseases Dataset(Augmented)/
    +-- New Plant Diseases Dataset(Augmented)/
        +-- train/
        +-- valid/
```

**2. Lancer l'entraînement**

```bash
python -m models.feuille
```

L'entraînement dure environ 20-30 minutes sur GPU NVIDIA (CUDA). Le meilleur checkpoint est sauvegardé automatiquement dans `models/saved/feuille_model.pth` et la liste des classes dans `models/saved/feuille_classes.json`.

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
| TB (Taux Butyreux) | % matières grasses du lait | < 30 g/kg : acidose possible | 20 à 80 g/kg |
| TP (Taux Protéique) | % protéines du lait | < 28 g/kg : déficit énergétique | 20 à 60 g/kg |
| CCS (Cellules Somatiques) | Indicateur d'infection mammaire | > 200 k/mL : surveillance, > 400 k/mL : mammite | 10 à 10 000 k/mL |
| BCS (Body Condition Score) | État d'engraissement (1-5) | < 2.0 : maigreur, > 4.0 : surpoids | 1 à 5 |
| Température | Température corporelle | > 39.5 °C : fièvre | 35 à 42 °C |

## Base de données

Chaque export PDF déclenche une sauvegarde automatique de l'analyse dans une base SQLite locale (`db/agricole.db`), gérée via SQLAlchemy.

Deux tables : `analyses_culture` et `analyses_vache`. Chaque ligne contient les paramètres saisis, les résultats (score, statut, conseils) et un timestamp UTC.

La base est créée automatiquement au premier lancement, le dossier `db/` n'est pas versionné.

Pour lancer les tests :

```bash
python -m pytest tests/ -v
```

## Limites connues

- Les données d'entraînement culture/vache sont **synthétiques** : le modèle illustre le fonctionnement mais ne reflète pas la réalité agronomique terrain
- Pour un usage en production, réentraîner les modèles sur de vraies données (ex. [Agreste](https://agreste.agriculture.gouv.fr/))
- L'indice de risque agronomique utilise une normalisation symétrique, en réalité les effets sont asymétriques (ex. excès d'azote != manque d'azote)
- Le modèle feuille atteint 99.9% sur PlantVillage mais ce dataset est en conditions contrôlées, les performances sur photos terrain (éclairage variable, angle, fond) peuvent être inférieures

## Pistes d'évolution

- [ ] Intégration de données réelles open-data (culture/vache)
- [x] Module de détection de maladies foliaires par image (PlantVillage) — modèle entraîné, intégration UI en cours
- [ ] API REST pour intégration dans d'autres outils
- [ ] Historique des analyses par exploitation
