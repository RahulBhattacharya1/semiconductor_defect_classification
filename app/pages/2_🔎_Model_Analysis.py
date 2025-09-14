import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

from src.generate_data import make_dataset, CLASSES
from src.features import batch_features
from src.predict import load_model

st.title("🔎 Model Analysis")

with st.sidebar:
    n = st.slider("Synthetic samples", min_value=200, max_value=1200, value=600, step=100)
    seed = st.number_input("Seed", value=11, step=1)
    model_path = "models/trained/model.pkl"

Ximgs, y = make_dataset(n=n, seed=int(seed), classes=CLASSES)
X = batch_features(Ximgs)

pipe, classes = load_model(model_path=model_path)
yhat = pipe.predict(X)

st.subheader("Confusion matrix")
cm = confusion_matrix(y, yhat, labels=classes)
df_cm = pd.DataFrame(cm, index=classes, columns=classes)
st.dataframe(df_cm.style.background_gradient(cmap="Blues"), use_container_width=True)

st.subheader("Classification report")
report = classification_report(y, yhat, labels=classes, output_dict=True, digits=4)
st.dataframe(pd.DataFrame(report).T, use_container_width=True)

st.subheader("Feature importances (from RF)")
try:
    importances = pipe.named_steps["clf"].feature_importances_
    st.line_chart(pd.Series(importances))
except Exception as e:
    st.info(f"Feature importances not available: {e}")