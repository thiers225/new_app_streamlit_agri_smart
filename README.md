# 🌽 AGRI-SMART – Assistant Intelligent pour le Maïs  
Application d’intelligence artificielle permettant la **détection automatique des maladies du maïs** et la **prédiction du rendement**.  
Développée dans le cadre du projet AGRI-SMART.

---

## 📌 Description

AGRI-SMART est une application Streamlit composée de deux modules principaux :

---

### 🦠 1. Détection automatique des maladies du maïs

À partir d'une photo de feuille de maïs, l’IA basée sur **MobileNetV2** identifie :

- **Helminthosporiose (Blight)**  
- **Rouille commune (Common Rust)**  
- **Tache grise (Gray Leaf Spot)**  
- **Feuille saine (Healthy)**  

L’application fournit :

- La classe détectée  
- Le niveau de confiance (%)  
- Un graphique détaillant les probabilités  
- Une interprétation agronomique pour faciliter la prise de décision sur le terrain  

Ce module est un prototype. Sa généralisation à des photos de terrain prises par smartphone reste à évaluer.

---

### 🌾 2. Prédiction du rendement (kg/ha)

Un modèle Machine Learning (basé sur Scikit-Learn) estime le rendement à partir des caractéristiques suivantes :

| Variable | Description |
|---------|-------------|
| **PL_HT** | Hauteur de la plante |
| **E_HT** | Hauteur de l’épi |
| **DY_SK** | Jours jusqu’à l’apparition des soies |
| **AEZONE** | Zone agro-écologique |
| **RUST** | Score de rouille |
| **BLIGHT** | Score d’helminthosporiose |

Après saisie des données agronomiques, l'application retourne une estimation du rendement en **kg/ha**.

---

## 🎯 Objectifs du projet

- Fournir un **outil intelligent** aux agriculteurs et techniciens agricoles  
- Réduire les pertes dues aux maladies foliaires  
- Améliorer la **prise de décision agronomique**  
- Faciliter l'accès à des diagnostics rapides via un **smartphone**  
- Soutenir la digitalisation du secteur agricole en Afrique

---

## 🧠 Technologies utilisées

| Domaine | Outils |
|--------|--------|
| **Deep Learning** | TensorFlow 2.19, Keras, MobileNetV2 |
| **Machine Learning** | Scikit-Learn, Joblib |
| **Développement Web** | Streamlit |
| **Visualisation** | Matplotlib, Pandas, Seaborn |

---

## 📌 Limitations & Perspectives

### 🔸 Limitations actuelles
- Performances dépendantes de la qualité des images (floues ou sombres).
- Pas encore de détection multi-maladies sur une même feuille.
- Données limitées à **4 classes**, extensibles à d'autres maladies.

### 🔸 Perspec​tives d’amélioration
- Conversion du modèle en **TensorFlow Lite** pour application mobile offline.  
- Ajout de nouvelles maladies et ravageurs du maïs.  
- Géolocalisation des parcelles et suivi des symptômes dans le temps.  
- Intégration d’un module de recommandations agronomiques personnalisées.  

---

## 👨🏽‍💻 Auteur

**Thierry N'DRI**  
Projet AGRI-SMART — Module d’assistance agricole intelligente basée sur l’IA.

## Installation et lancement

```bash
git clone https://github.com/thiers225/new_app_streamlit_agri_smart.git
cd new_app_streamlit_agri_smart
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell : .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Les dépendances de référence sont dans [requirements.txt](requirements.txt). Lancez l'application depuis la racine du dépôt.

## Architecture du prototype

| Élément | Rôle |
| --- | --- |
| [app.py](app.py) | Interface Streamlit, préparation des entrées et inférence |
| `models/maize_mobilenetv2_model_v2_final.keras` | Modèle de classification attendu par l'application |
| `models/yield_prediction_model.pkl` | Modèle de rendement attendu |
| `models/model_input_columns.pkl` | Colonnes attendues par le modèle de rendement |

Les images sont converties en RGB, redimensionnées à 224 × 224 et normalisées par division par 255 avant la classification.

**Comportement du rendement :** si le modèle ou ses colonnes ne sont pas disponibles ou ne chargent pas, l'application affiche une estimation heuristique explicitement marquée « démo ». Cette formule ne constitue pas une prédiction issue d'un modèle entraîné. Une erreur d'inférence avec un modèle chargé affiche un message d'erreur.

## Évaluation et limites

Le score de confiance affiché pour une image n'est pas une mesure de précision globale et n'est pas nécessairement calibré.

Les métriques et la provenance des données ne sont pas publiées dans ce README. Pour permettre une évaluation reproductible, la prochaine documentation devra préciser :

- Sources, licences et effectifs des données ; séparation entraînement, validation et test.
- Classification : précision, rappel et F1-score par classe, matrice de confusion.
- Rendement : MAE, RMSE, unités des cibles et comparaison à une référence simple.
- Tests sur des données indépendantes et photos de terrain.

## Autres travaux AGRI-SMART

[agri_smart_streamlit_app](https://github.com/thiers225/agri_smart_streamlit_app) contient une autre implémentation et des guides de compatibilité des modèles. Les deux dépôts restent distincts ; le présent dépôt est celui présenté dans le README du profil.
