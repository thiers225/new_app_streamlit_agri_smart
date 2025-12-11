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

Ce module est conçu pour fonctionner en conditions réelles, même avec des images prises par smartphone.

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
