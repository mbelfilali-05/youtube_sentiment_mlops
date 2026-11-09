# YouTube Sentiment Analyzer — Projet MLOps (TP3)

Système complet d'analyse de sentiment des commentaires YouTube : modèle ML (TF-IDF + Régression Logistique), API REST FastAPI et extension Chrome.

> **ENSAM Rabat — Filière INDIA — Cloud Computing / MLOps — Prof. Maria Zemzami**

![Popup de l'extension](docs/screenshots/popup_all.png)

## Description

Les créateurs YouTube reçoivent des centaines de commentaires par vidéo. L'extension extrait les commentaires visibles de la page, les envoie par lots à l'API, et affiche la répartition des sentiments (positif / neutre / négatif), la liste des commentaires avec leur score de confiance, des filtres, un mode sombre et des fonctions de copie/export.

## Architecture

```
┌────────────────────┐  JSON (batch)   ┌─────────────────────┐   ┌──────────────────────┐
│ Extension Chrome   │ ──────────────► │ API FastAPI         │──►│ Pipeline scikit-learn│
│ content.js (DOM)   │ POST            │ /health             │   │ nettoyage → TF-IDF   │
│ popup (UI, charts) │ /predict_batch  │ /predict_batch      │   │ (mots + caractères)  │
│                    │ ◄────────────── │ Pydantic + CORS     │   │ → LogisticRegression │
└────────────────────┘  sentiments +   └─────────────────────┘   └──────────────────────┘
                        statistiques
```

| Composant | Technologie |
|---|---|
| Frontend | Extension Chrome Manifest V3 (JavaScript, HTML, CSS) |
| Backend | FastAPI + Pydantic + Uvicorn |
| ML | scikit-learn 1.7.2 : TF-IDF (mots + n-grammes de caractères) + Logistic Regression |
| Conteneurisation | Docker (python:3.10-slim, port 7860) |
| Versioning | Git + GitHub |

## Structure du dépôt

```
├── chrome_extension/     # extension (manifest, content script, popup)
├── data/                 # raw/ et processed/ (non versionnés, régénérables)
├── docs/                 # documentation API, guide utilisateur, captures, rapport
├── models/               # sentiment_pipeline.joblib (modèle final), tfidf_vectorizer.joblib + sentiment_model.joblib (baseline)
├── reports/              # métriques (json) et figures
├── src/
│   ├── data/             # download_data, prepare_data, eda
│   ├── models/           # train_model, tune_negative, make_report_figures
│   ├── api/              # main.py, schemas.py
│   └── utils/            # text_cleaning.py (partagé entraînement/API)
├── tests/                # pytest (modèle + API)
├── app_api.py            # point d'entrée de déploiement
├── Dockerfile
├── requirements.txt      # développement
└── requirements-prod.txt # production (API seule)
```

## Installation

Prérequis : Python 3.10+, Git, Google Chrome.

```bash
git clone https://github.com/mbelfilali-05/youtube_sentiment_mlops.git
cd youtube_sentiment_mlops
python -m venv venv
# Windows : venv\Scripts\activate      Linux/macOS : source venv/bin/activate
pip install -r requirements.txt
```

## Utilisation

### 1. Pipeline de données et entraînement (optionnel : les modèles `.joblib` sont déjà fournis)

```bash
python -m src.data.download_data      # télécharge data/raw/reddit.csv
python -m src.data.prepare_data       # nettoyage + split train/test (80/20, seed 42)
python -m src.data.eda                # figures dans reports/figures
python -m src.models.train_model      # comparaison d'algorithmes + GridSearchCV
python -m src.models.tune_negative    # modèle final (TF-IDF mots+caractères)
```

### 2. Lancer l'API

```bash
uvicorn src.api.main:app --port 8000
```
Documentation interactive : http://localhost:8000/docs — voir aussi [docs/API.md](docs/API.md).

### 3. Installer l'extension Chrome

1. Ouvrir `chrome://extensions` et activer le **Mode développeur**.
2. **Charger l'extension non empaquetée** → sélectionner le dossier `chrome_extension/`.
3. Ouvrir une vidéo YouTube (`youtube.com/watch?v=...`), **descendre** jusqu'aux commentaires, cliquer sur l'icône de l'extension puis **Analyze comments**.

Guide détaillé avec captures : [docs/GUIDE_UTILISATEUR.md](docs/GUIDE_UTILISATEUR.md).

### 4. Tests

```bash
pip install -r requirements.txt
python -m pytest -q
```

### 5. Docker

```bash
docker build -t youtube-sentiment-api .
docker run -p 7860:7860 youtube-sentiment-api
curl http://localhost:7860/health
```

## Résultats

Jeu de test (7 287 commentaires), modèle final `sentiment_pipeline.joblib` :

| Critère du TP | Objectif | Résultat |
|---|---|---|
| Accuracy | ≥ 0,80 | **0,844** |
| F1 Négatif | > 0,75 | **0,754** |
| F1 Neutre | > 0,75 | **0,887** |
| F1 Positif | > 0,75 | **0,857** |
| Inférence (batch de 50) | < 100 ms | **≈ 40 ms** |

Détails, comparaison des algorithmes et analyse : [docs/Rapport_Technique.pdf](docs/Rapport_Technique.pdf).

## État du déploiement cloud

Le `Dockerfile`, `app_api.py` et `requirements-prod.txt` sont fournis (port 7860, standard Hugging Face Spaces). **Le déploiement final sur Hugging Face Spaces n'a pas été réalisé** faute de temps (voir la section « Difficultés » du rapport). L'extension fonctionne avec l'API locale (`API_URL` dans `chrome_extension/popup.js`).

## Limites connues

- L'extension analyse les commentaires **présents dans le DOM** (YouTube charge à la demande) ; le bouton « Load more » fait défiler la page pour en charger davantage.
- Le modèle est entraîné sur des commentaires Reddit en anglais : les commentaires dans d'autres langues et les emojis sont mal traités.
- Les sélecteurs CSS de YouTube peuvent changer et nécessiter une mise à jour de `content.js`.
