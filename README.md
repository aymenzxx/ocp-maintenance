# 🏭 OCP — Maintenance Prédictive · Streamlit App

Application de déploiement du modèle de maintenance prédictive OCP.

## Structure

```
ocp_streamlit/
├── app.py                              ← Application principale
├── requirements.txt                    ← Dépendances Python
├── predictive_maintenance_pipeline.pkl ← Modèle (à copier ici)
├── model_metadata.json                 ← Métadonnées (à copier ici)
└── README.md
```

## Lancement local

```bash
# 1. Copier les fichiers modèle
cp /chemin/vers/predictive_maintenance_pipeline.pkl .
cp /chemin/vers/model_metadata.json .

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'app
streamlit run app.py
```

L'app sera disponible sur http://localhost:8501

## Déploiement sur Streamlit Cloud

1. Créez un repo GitHub avec les fichiers ci-dessus
2. Allez sur https://share.streamlit.io
3. Connectez votre repo → sélectionnez `app.py`
4. ⚠️ Le fichier `.pkl` doit être commité dans le repo (< 100 MB)

## ⚠️ Note sur le modèle pickle

Le modèle a été entraîné avec **scikit-learn 1.6.1**.  
Le `requirements.txt` épingle cette version pour éviter les incompatibilités.

Si vous re-générez le modèle avec une version plus récente, mettez à jour `requirements.txt`.

## Fonctionnalités

- **Prédiction individuelle** : saisie manuelle des capteurs + gauge de risque
- **Analyse par lot** : simulation d'un parc de machines avec dashboard
- **Export CSV** des résultats de lot
- **Mode dégradé** : si le fichier `.pkl` est absent, une estimation heuristique est utilisée
