# 🏭 OCP — Maintenance Prédictive Industrielle

Application Streamlit complète pour la prédiction de pannes des équipements industriels OCP Group.

---

## 📦 Structure du projet

```
ocp_app/
├── app.py              ← Application principale Streamlit
├── requirements.txt    ← Dépendances Python
├── .streamlit/
│   └── config.toml     ← Configuration thème sombre OCP
└── README.md
```

---

## 🚀 Déploiement en 4 étapes

### Étape 1 — Préparer GitHub

```bash
git init
git add .
git commit -m "OCP Maintenance Prédictive v2.0"
```

Créez un repo sur [github.com](https://github.com/new) puis :

```bash
git remote add origin https://github.com/VOTRE_USERNAME/ocp-maintenance.git
git push -u origin main
```

### Étape 2 — Déployer sur Streamlit Cloud

1. Allez sur **[share.streamlit.io](https://share.streamlit.io)**
2. Connectez votre compte GitHub
3. Cliquez **"New app"**
4. Sélectionnez votre repo `ocp-maintenance`
5. Fichier principal : `app.py`
6. Cliquez **"Deploy!"**

> ✅ Votre app sera live en ~2 minutes à l'adresse :
> `https://VOTRE_USERNAME-ocp-maintenance-app-XXXXX.streamlit.app`

### Étape 3 — Test en local (optionnel)

```bash
pip install -r requirements.txt
streamlit run app.py
```

L'app s'ouvre sur `http://localhost:8501`

### Étape 4 — Ajouter vos modèles .pkl (optionnel)

Si vous voulez utiliser les vrais modèles entraînés de votre notebook :

```bash
# Copiez vos fichiers dans le dossier
cp ocp_best_model.pkl  ocp_app/
cp ocp_scaler.pkl      ocp_app/
cp ocp_label_encoder.pkl ocp_app/
```

Puis dans `app.py`, remplacez la fonction `predict_failure()` par :

```python
import joblib

@st.cache_resource
def load_model():
    model  = joblib.load("ocp_best_model.pkl")
    scaler = joblib.load("ocp_scaler.pkl")
    le     = joblib.load("ocp_label_encoder.pkl")
    return model, scaler, le
```

---

## 🖥️ Fonctionnalités

| Page | Contenu |
|------|---------|
| 🏠 Accueil | KPIs, pipeline, types de pannes |
| 📊 Dashboard | EDA interactive (distributions, corrélations, boxplots) |
| 🤖 Simulateur | Prédiction temps réel avec gauge d'alerte |
| 📈 Modèles | Comparaison 6 algo, SHAP, matrice de confusion |
| 📋 Rapport | Recommandations & impact économique OCP |

---

## ⚙️ Configuration

Le fichier `.streamlit/config.toml` configure le thème sombre OCP automatiquement.

---

## 📊 Dataset

**AI4I 2020 Predictive Maintenance Dataset**  
10 000 enregistrements | 5 capteurs IoT | 6 types de pannes

---

*Projet OCP Group — Maintenance Prédictive Industrielle v2.0*
