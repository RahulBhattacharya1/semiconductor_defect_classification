from pathlib import Path
import os, sys

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- Make repo root & app dir importable ----
ROOT = Path(__file__).resolve().parents[1]      # repo/
APP_DIR = Path(__file__).resolve().parent       # repo/app/
for p in (ROOT, APP_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Imports from our packages (now resolvable)
from src.generate_data import synth_wafer, CLASSES
from src.predict import (
    predict_one,
    prepare_img_from_csv_bytes,
    prepare_img_from_png_bytes,
    load_model as _load_model,
)
from components.wafer_plot import wafer_imshow

MODEL_PATH = ROOT / "models" / "trained" / "model.pkl"

# ---- Page header ----
st.set_page_config(page_title="Wafer Map Defect Classifier", layout="wide")
st.title("Wafer Map Defect Classifier")
st.caption("Synthetic demo for yield engineering: center, edge_ring, scratch, donut, random")

# ---- Sidebar controls ----
with st.sidebar:
    st.header("Input")
    mode = st.radio("Input Mode", ["Generate", "Upload"])
    default_kind = st.selectbox("Default class", CLASSES, index=0)
    seed = st.number_input("Random seed", value=42, step=1)
    model_path = str(MODEL_PATH)

# ---- Get an image (generated or uploaded) ----
if mode == "Generate":
    img = synth_wafer(kind=default_kind, seed=int(seed))
else:
    uploaded = st.file_uploader("Upload wafer map", type=["csv", "png"])
    if uploaded is not None:
        if uploaded.type.endswith("csv"):
            img = prepare_img_from_csv_bytes(uploaded.getvalue())
        else:
            img = prepare_img_from_png_bytes(uploaded.getvalue())
    else:
        st.info("Upload a CSV (28×28) or PNG to run inference, or switch to Generate.")
        img = synth_wafer(kind=default_kind, seed=int(seed))

# ---- Two-column layout ----
col1, col2 = st.columns([1, 1])

# Left: wafer preview
with col1:
    st.subheader("Wafer Map")
    fig, ax = plt.subplots(figsize=(4, 4))
    wafer_imshow(img, ax=ax, title="Input")
    st.pyplot(fig, width="content")  # (replaces deprecated use_container_width)

# --- Right: prediction ---
with col2:
    st.subheader("Prediction")

    @st.cache_resource(show_spinner=False)
    def _build_demo_model():
        # Train a small model in this environment (no pickles)
        from src.generate_data import make_dataset, CLASSES as _CLASSES
        from src.features import batch_features
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier

        Ximgs, y = make_dataset(n=600, seed=7, classes=_CLASSES)
        X = batch_features(Ximgs)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300, random_state=7, class_weight="balanced_subsample"
            ))
        ])
        pipe.fit(X, y)
        classes = sorted(set(y.tolist()))
        return pipe, classes

    try:
        pipe, classes = _build_demo_model()   # ← always use the in-memory model
        pred, proba = predict_one(img, pipe=pipe, classes=classes)
        st.write(f"Predicted class: **{pred}**")
        st.bar_chart(pd.DataFrame.from_dict(proba, orient="index", columns=["probability"]))
    except Exception as e:
        st.error(f"Model not available or failed to predict: {e}")

