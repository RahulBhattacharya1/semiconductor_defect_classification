from pathlib import Path
import os, sys

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# repo root & app dir
ROOT = Path(__file__).resolve().parents[2]   # repo/
APP_DIR = Path(__file__).resolve().parents[1]  # repo/app/
for p in (ROOT, APP_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.generate_data import make_dataset, CLASSES
from src.features import batch_features
from src.predict import load_model

st.title("🔎 Model Analysis")

with st.sidebar:
    n = st.slider("Synthetic samples", min_value=200, max_value=1200, value=600, step=100)
    seed = st.number_input("Seed", value=11, step=1)

model_path = str(ROOT / "models" / "trained" / "model.pkl")
pipe, classes = load_model(path=model_path)  # correct kwarg

Ximgs, y = make_dataset(n=int(n), seed=int(seed), classes=CLASSES)
X = batch_features(Ximgs)
yhat = pipe.predict(X)

st.subheader("Confusion matrix")
cm = confusion_matrix(y, yhat, labels=classes)
df_cm = pd.DataFrame(cm, index=classes, columns=classes)
st.dataframe(df_cm.style.background_gradient(cmap="Blues"), width="stretch")

st.subheader("Classification report")
report = classification_report(y, yhat, labels=classes, output_dict=True, digits=4)
st.dataframe(pd.DataFrame(report).T, width="stretch")

st.subheader("Feature importances (from RF)")
try:
    importances = pipe.named_steps["clf"].feature_importances_
    st.line_chart(pd.Series(importances))
except Exception as e:
    st.info(f"Feature importances not available: {e}")
