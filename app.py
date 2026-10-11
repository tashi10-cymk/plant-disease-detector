import streamlit as st
import tensorflow as tf
import numpy as np
import json
import sqlite3
import os
from datetime import datetime
from PIL import Image

# Initialize SQLite Database
def init_db():
    conn = sqlite3.connect("history.db", check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  timestamp TEXT, 
                  disease TEXT, 
                  confidence REAL)''')
    conn.commit()
    return conn

conn = init_db()

st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="centered")

@st.cache_resource
def load_resources():
    with open("class_names.json", "r", encoding="utf-8") as f:
        class_names = json.load(f)
    with open("treatments.json", "r", encoding="utf-8") as f:
        treatments = json.load(f)
    model = tf.keras.models.load_model("model.keras")
    return model, class_names, treatments

model, class_names, treatments = load_resources()

st.title("🌿 Plant Disease Detector")
lang = st.radio("Language / भाषा चुनें:", ("English", "Hindi"), horizontal=True)

uploaded_file = st.file_uploader("Upload a leaf photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, width=300)
    
    # Preprocess and Predict
    arr = np.array(img.resize((224, 224)), dtype="float32")[None]
    probs = model.predict(arr)[0]
    
    top_3_idx = np.argsort(probs)[-3:][::-1]
    best_idx = top_3_idx[0]
    conf = float(probs[best_idx])
    predicted_class = class_names[best_idx]
    
    if conf < 0.6:
        st.warning("⚠️ Not sure. Please upload a clearer, close-up photo of a single leaf.")
    else:
        st.success(f"**Prediction:** {predicted_class.replace('___', ' - ')} ({conf:.0%} confidence)")
        
        # Save to SQLite
        c = conn.cursor()
        c.execute("INSERT INTO predictions (timestamp, disease, confidence) VALUES (?, ?, ?)",
                  (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), predicted_class, conf))
        conn.commit()
        
        # Treatment Advice
        if predicted_class in treatments:
            advice_key = "en" if lang == "English" else "hi"
            advice = treatments[predicted_class][advice_key]
            st.subheader("💊 Treatment Advice / उपचार सलाह:")
            st.warning(advice)
            
        # Top-3 Breakdown
        st.subheader("📊 Top Predictions:")
        for idx in top_3_idx:
            c_name = class_names[idx].replace('___', ' - ')
            c_score = float(probs[idx])
            st.progress(c_score, text=f"{c_name}: {c_score:.1%}")

# Sidebar Analytics & TFLite Info
st.sidebar.subheader("⚙️ Model & Performance")
if os.path.exists("model.keras"):
    size_mb = os.path.getsize("model.keras") / (1024 * 1024)
    st.sidebar.text(f"Model Size (Keras): {size_mb:.2f} MB")

# Optional TFLite conversion display check
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open("model.tflite", "wb") as f:
        f.write(tflite_model)
    tflite_size = os.path.getsize("model.tflite") / (1024 * 1024)
    st.sidebar.text(f"Optimized TFLite Size: {tflite_size:.2f} MB")
except Exception as e:
    pass

# History Viewer
if st.sidebar.checkbox("📜 View Prediction History"):
    st.sidebar.subheader("Recent Scans")
    c = conn.cursor()
    c.execute("SELECT timestamp, disease, confidence FROM predictions ORDER BY id DESC LIMIT 10")
    for row in c.fetchall():
        st.sidebar.text(f"{row[0]}\n{row[1]} ({row[2]:.0%})")
        st.sidebar.markdown("---")