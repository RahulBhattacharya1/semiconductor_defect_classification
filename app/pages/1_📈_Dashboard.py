import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from src.generate_data import make_dataset, CLASSES
from src.features import batch_features
from src.predict import load_model
from app.components.wafer_plot import wafer_imshow

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.generate_data import make_dataset, CLASSES
from src.features import batch_features
from src.predict import load_model
from app.components.wafer_plot import wafer_imshow

st.title("📈 Dashboard")

with st.sidebar:
    n = st.slider("Synthetic samples", min_value=100, max_value=1000, value=300, step=50)
    seed = st.number_input("Seed", value=7, step=1)
    model_path = "models/trained/model.pkl"

Ximgs, y = make_dataset(n=n, seed=int(seed), classes=CLASSES)
X = batch_features(Ximgs)

pipe, classes = load_model(model_path)
yhat = pipe.predict(X)

st.subheader("KPIs")
col1, col2, col3 = st.columns(3)
acc = (yhat == y).mean()
col1.metric("Accuracy", f"{acc*100:.2f}%")
counts = pd.Series(y).value_counts().reindex(classes, fill_value=0)
col2.metric("Classes", len(classes))
col3.metric("Samples", len(y))

st.subheader("Class distribution")
st.bar_chart(counts)

st.subheader("Sample wafer maps")
cols = st.columns(5)
for i in range(10):
    c = cols[i % 5]
    with c:
        fig, ax = plt.subplots(figsize=(2,2))
        wafer_imshow(Ximgs[i], ax=ax, title=f"y={y[i]} / ŷ={yhat[i]}")
        st.pyplot(fig)
