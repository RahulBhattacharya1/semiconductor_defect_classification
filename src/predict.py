import io
import pickle
import numpy as np
from PIL import Image

from .features import features_from_image

def load_model(path="models/trained/model.pkl"):
    with open(path, "rb") as f:
        blob = pickle.load(f)
    return blob["pipe"], blob["classes"]

def prepare_img_from_csv_bytes(b):
    import pandas as pd
    df = pd.read_csv(io.BytesIO(b), header=None)
    arr = df.values.astype(float)
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr

def prepare_img_from_png_bytes(b):
    img = Image.open(io.BytesIO(b)).convert("L")
    img = img.resize((28, 28), Image.BILINEAR)
    arr = np.array(img).astype(float)
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr

def predict_one(arr, model_path="models/trained/model.pkl"):
    pipe, classes = load_model(model_path)
    feat = features_from_image(arr).reshape(1, -1)
    proba = pipe.predict_proba(feat)[0]
    idx = int(np.argmax(proba))
    return classes[idx], dict(zip(classes, proba.tolist()))
