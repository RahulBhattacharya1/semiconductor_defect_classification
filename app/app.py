from pathlib import Path
import os, sys

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Make repo root & app dir importable ---
ROOT = Path(__file__).resolve().parents[1]      # repo/
APP_DIR = Path(__file__).resolve().parent       # repo/app/
for p in (ROOT, APP_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.generate_data import synth_wafer, CLASSES
from src.predict import (
    predict_one,
    prepare_img_from_csv_bytes,
    prepare_img_from_png_bytes,
)
# IMPORTANT: import the sibling package directly to avoid "app is not a package"
from components.wafer_plot import wafer_imshow

MODEL_PATH = ROOT / "models" / "trained" / "model.pkl"

st.set_page_config(page_title="Wafer Map Defect Classifier", layout="wide")
st.title("Wafer Map Defect Classifier")
st.caption("Synthetic demo for yield engineering: center, edge_ring, scratch, donut, random")

with st.sidebar:
    st.header("Input")
    mode = st.radio("Input Mode", ["Generate", "Upload"])
    default_kind = st.selectbox("Default class", CLASSES, index=0)
    seed = st.number_input("Random seed", value=42, step=1)
    model_path = str(MODEL_PATH)

col1, col2 = st.columns([1, 1])

# --- Input image ---
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

# --- Left: wafer preview ---
with col1:
    st.subheader("Wafer Map")
    fig, ax = plt.subplots(figsize=(4, 4))
    wafer_imshow(img, ax=ax, title="Input")
    st.pyplot(fig, width="content")  # replaces deprecated use_container_width

# --- Right: prediction ---
with col2:
    st.subheader("Prediction")
    try:
        from src.predict import load_model  # add near your other imports
        pipe, classes = load_model(model_path)
        pred, proba = predict_one(img, pipe=pipe, classes=classes)
        st.write(f"Predicted class: **{pred}**")
        st.bar_chart(pd.DataFrame.from_dict(proba, orient="index", columns=["probability"]))
    except Exception as e:
        st.error(f"Model not available or failed to predict: {e}")

st.divider()
st.write("Use the pages on the left for a dashboard and model analysis.")
