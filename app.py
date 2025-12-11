# app_simple_fixed.py
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import joblib
import time
import os
import matplotlib.pyplot as plt

# -------------------------
# Minimal configuration
# -------------------------
MODEL_PATH = "models/maize_mobilenetv2_model_v2_final.keras"
YIELD_MODEL_PATH = "models/yield_prediction_model.pkl"
YIELD_INPUT_COLS_PATH = "models/model_input_columns.pkl"  # expected pickle of columns list
IMG_SIZE = (224, 224)

CLASS_NAMES = ['Blight', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
CLASS_TRANSLATIONS = {
    'Blight': 'Helminthosporiose (Blight)',
    'Common_Rust': 'Rouille commune',
    'Gray_Leaf_Spot': 'Tache grise (Gray Leaf Spot)',
    'Healthy': 'Saine'
}

# Page config
st.set_page_config(page_title="Assistant Intelligent Maïs", page_icon="🌽", layout="centered")

# Simple CSS
st.markdown("""
    <style>
    .main { background-color: #f8faf7; }
    h1 { color: #1b5e20; text-align: center; }
    .prediction-box {
        padding: 16px;
        border-radius: 8px;
        background-color: white;
        box-shadow: 0 3px 6px rgba(0,0,0,0.08);
        text-align: center;
        margin-top: 12px;
    }
    .small { font-size: 0.9rem; color: #555; }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🌽 Assistant Intelligent Maïs")
st.markdown("Application simple pour détecter des maladies foliaires du maïs et estimer le rendement.")

# Load disease model (cached)
@st.cache_resource
def load_disease_model(path=MODEL_PATH):
    if not os.path.exists(path):
        return None
    try:
        model = tf.keras.models.load_model(path, compile=False)
        return model
    except Exception:
        return None

disease_model = load_disease_model()

# Tabs: Disease detection | Yield estimation
tab1, tab2 = st.tabs(["🦠 Détection de maladies", "📈 Estimation du rendement"])

# ---- TAB 1: Disease detection ----
with tab1:
    st.markdown("**Téléversez une photo claire d'une feuille de maïs.** L'application retournera la classe prédite et la confiance associée.")
    if disease_model is None:
        st.error("Modèle introuvable. Placez le fichier de modèle au chemin : `models/maize_mobilenetv2_model_v2_final.keras`.")
    else:
        st.success("Modèle chargé.")

    uploaded_file = st.file_uploader("Choisissez une image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        # use_container_width to avoid deprecated warning
        st.image(image, caption="Image téléchargée", use_container_width=True)

        if st.button("Analyser la feuille"):
            if disease_model is None:
                st.error("Impossible d'analyser : modèle non chargé.")
            else:
                with st.spinner("Analyse en cours..."):
                    # preprocess
                    if image.mode != "RGB":
                        image = image.convert("RGB")
                    img = image.resize(IMG_SIZE)
                    arr = np.array(img).astype("float32") / 255.0
                    arr = np.expand_dims(arr, axis=0)

                    preds = disease_model.predict(arr, verbose=0)[0]
                    idx = int(np.argmax(preds))
                    prob = float(np.max(preds)) * 100.0

                    time.sleep(0.5)  # léger délai UX

                predicted_en = CLASS_NAMES[idx]
                predicted_fr = CLASS_TRANSLATIONS.get(predicted_en, predicted_en)

                # Display result
                st.markdown(f"""
                <div class="prediction-box">
                  <h2 style="color:#1b5e20; margin:0;">Résultat : <strong>{predicted_fr}</strong></h2>
                  <p class="small">Confiance : <strong>{prob:.2f}%</strong></p>
                </div>
                """, unsafe_allow_html=True)

                # Short advice
                if predicted_en == 'Healthy':
                    st.info(f"Feuille saine détectée (confiance {prob:.2f}%).")
                elif predicted_en == 'Blight':
                    st.warning(f"Helminthosporiose détectée (confiance {prob:.2f}%). Considérez des mesures phytosanitaires.")
                elif predicted_en == 'Common_Rust':
                    st.warning(f"Rouille commune détectée (confiance {prob:.2f}%). Inspectez les pustules sur la face inférieure des feuilles.")
                elif predicted_en == 'Gray_Leaf_Spot':
                    st.warning(f"Tache grise détectée (confiance {prob:.2f}%). Surveillez l'évolution des symptômes.")

        # show probabilities (expandable)
        if disease_model is not None and uploaded_file is not None:
            with st.expander("Voir la distribution des probabilités"):
                preds_for_display = disease_model.predict(
                    np.expand_dims(np.array(image.resize(IMG_SIZE)).astype("float32")/255.0, axis=0),
                    verbose=0
                )[0]
                df = pd.DataFrame({
                    "Classe": [CLASS_TRANSLATIONS.get(c, c) for c in CLASS_NAMES],
                    "Confiance (%)": (preds_for_display * 100).round(2)
                }).sort_values("Confiance (%)", ascending=False)
                st.table(df.set_index("Classe"))

                # bar chart via st.bar_chart
                st.bar_chart(df.set_index("Classe")["Confiance (%)"])

                # matplotlib version (optional have nicer ticks)
                fig, ax = plt.subplots(figsize=(6,3))
                df.set_index("Classe")["Confiance (%)"].plot(kind="bar", ax=ax)
                ax.set_ylabel("Confiance (%)")
                ax.set_ylim(0, 100)
                ax.grid(axis='y', linestyle='--', alpha=0.5)
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig, use_container_width=True)

# ---- TAB 2: Yield prediction ----
with tab2:
    st.markdown("**Estimation simple du rendement (kg/ha)** — si le modèle de rendement est disponible, il sera utilisé ; sinon une estimation heuristique est affichée.")
    # load yield model
    @st.cache_resource
    def load_yield_model(path=YIELD_MODEL_PATH, cols_path=YIELD_INPUT_COLS_PATH):
        if not os.path.exists(path) or not os.path.exists(cols_path):
            return None, None
        try:
            m = joblib.load(path)
            cols = joblib.load(cols_path)
            return m, cols
        except Exception:
            return None, None

    yield_model, input_cols = load_yield_model()

    st.markdown("Entrez les caractéristiques ci-dessous :")
    col1, col2 = st.columns(2)
    with col1:
        pl_ht = st.number_input("Hauteur de la plante (PL_HT) — cm", min_value=50, max_value=300, value=180)
        e_ht = st.number_input("Hauteur de l'épi (E_HT) — cm", min_value=20, max_value=200, value=90)
        dy_sk = st.number_input("Jours jusqu'à l'apparition des soies (DY_SK) — jours", min_value=40, max_value=100, value=60)
    with col2:
        aezone = st.selectbox("Zone agro-écologique (AEZONE)", ["Forest/Transitional", "Moist Savanna"])
        rust_score = st.slider("Score de Rouille (RUST) — 1 (léger) à 5 (fort)", 1, 5, 2)
        blight_score = st.slider("Score d'Helminthosporiose (BLIGHT) — 1 (léger) à 5 (fort)", 1, 5, 2)

    if st.button("Prédire le rendement"):
        if yield_model is not None and input_cols is not None:
            # Prepare input dataframe with zeros
            X = pd.DataFrame(0, index=[0], columns=input_cols)
            # Fill values if columns exist
            if 'PL_HT' in input_cols: X['PL_HT'] = pl_ht
            if 'E_HT' in input_cols: X['E_HT'] = e_ht
            if 'DY_SK' in input_cols: X['DY_SK'] = dy_sk
            # For categorical AEZONE, the pipeline that produced input_cols should define how to fill it.
            # We'll set AEZONE if present, otherwise try to set one-hot columns
            if 'AEZONE' in input_cols:
                X['AEZONE'] = aezone
            else:
                # try to fill one-hot style e.g. AEZONE_Forest/Transitional or AEZONE_Moist Savanna
                for col in input_cols:
                    if col.startswith("AEZONE_") and aezone.replace(" ", "_") in col:
                        X[col] = 1
            if 'RUST' in input_cols: X['RUST'] = rust_score
            if 'BLIGHT' in input_cols: X['BLIGHT'] = blight_score

            try:
                pred = yield_model.predict(X)[0]
                st.markdown(f"""
                <div class="prediction-box">
                  <h2 style="color:#1b5e20; margin:0;">Rendement prédit</h2>
                  <h1 style="color:#2e7d32; margin:0;">{pred:,.2f} kg/ha</h1>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error("Erreur lors de la prédiction du rendement. Vérifiez la configuration du modèle de rendement.")
        else:
            # heuristic fallback
            sim = (pl_ht * 10) + (e_ht * 5) - (dy_sk * 2) + 3000 - (rust_score * 50) - (blight_score * 50)
            st.markdown(f"""
            <div class="prediction-box">
              <h2 style="color:#1b5e20; margin:0;">Rendement estimé (démo)</h2>
              <h1 style="color:#2e7d32; margin:0;">{sim:,.2f} kg/ha</h1>
              <p class="small">Modèle de rendement non disponible — estimation heuristique</p>
            </div>
            """, unsafe_allow_html=True)

# footer
st.markdown("---")
st.markdown("Développé pour le projet AGRI-SMART • Thierry N'DRI")
st.markdown("© 2025 Tous droits réservés.")