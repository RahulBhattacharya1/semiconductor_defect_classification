import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.generate_data import synth_wafer, CLASSES
from src.predict import predict_one, prepare_img_from_csv_bytes, prepare_img_from_png_bytes
from app.components.wafer_plot import wafer_imshow

st.set_page_config(page_title="Wafer Map Defect Classifier", layout="wide")

st.title("Wafer Map Defect Classifier")
st.caption("Synthetic demo for yield engineering: center, edge_ring, scratch, donut, random")

with st.sidebar:
    st.header("Input")
    mode = st.radio("Input Mode", ["Generate", "Upload"])
    default_kind = st.selectbox("Default class", CLASSES, index=0)
    seed = st.number_input("Random seed", value=42, step=1)
    model_path = "models/trained/model.pkl"

col1, col2 = st.columns([1,1])

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

with col1:
    st.subheader("Wafer Map")
    fig, ax = plt.subplots(figsize=(4,4))
    wafer_imshow(img, ax=ax, title="Input")
    st.pyplot(fig, use_container_width=False)

with col2:
    st.subheader("Prediction")
    try:
        pred, proba = predict_one(img, model_path=model_path)
        st.write(f"Predicted class: **{pred}**")
        st.bar_chart(pd.DataFrame.from_dict(proba, orient="index", columns=["probability"]))
    except Exception as e:
        st.error(f"Model not available or failed to predict: {e}")
        st.stop()

st.divider()
st.write("Use the pages on the left for a dashboard and model analysis.")