from pathlib import Path
import os, sys

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# repo root & app dir
ROOT = Path(__file__).resolve().parents[2]   # repo/
APP_DIR = Path(__file__).resolve().parents[1]  # repo/app/
for p in (ROOT, APP_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.generate_data import make_dataset, CLASSES
from src.features import batch_features
from src.predict import load_model
from components.wafer_plot import wafer_imshow

st.title("📈 Dashboard")

with st.sidebar:
    n = st.slider("Synthetic samples", min_value=100, max_value=1000, value=300, step=50)
    seed = st.number_input("Seed", value=7, step=1)

model_path = str(ROOT / "models" / "trained" / "model.pkl")
pipe, classes = load_model(path=model_path)  # use correct kwarg

Ximgs, y = make_dataset(n=int(n), seed=int(seed), classes=CLASSES)
X = batch_features(Ximgs)
yhat = pipe.predict(X)

# KPIs
st.subheader("KPIs")
col1, col2, col3 = st.columns(3)
acc = float((yhat == y).mean())
col1.metric("Accuracy", f"{acc*100:.2f}%")
counts = pd.Series(y).value_counts().reindex(classes, fill_value=0)
col2.metric("Classes", len(classes))
col3.metric("Samples", len(y))

# Distribution by class
st.subheader("Class distribution")
st.bar_chart(counts)

# Sample gallery
st.subheader("Sample wafer maps")
cols = st.columns(5)
for i in range(min(10, len(Ximgs))):
    c = cols[i % 5]
    with c:
        fig, ax = plt.subplots(figsize=(2, 2))
        wafer_imshow(Ximgs[i], ax=ax, title=f"y={y[i]} / ŷ={yhat[i]}")
        st.pyplot(fig, width="content")
