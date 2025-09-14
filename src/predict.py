import io
import pickle
from pathlib import Path
import numpy as np
from PIL import Image

from .features import features_from_image

def load_model(path="models/trained/model.pkl"):
    """Load model; if unpickling fails (version/ABI mismatch), retrain once and load."""
    p = Path(path)
    try:
        with open(p, "rb") as f:
            blob = pickle.load(f)
        return blob["pipe"], blob["classes"]
    except Exception:
        # Fallback: retrain a small model in-place so the app keeps working
        from .train import train_model
        p.parent.mkdir(parents=True, exist_ok=True)
        train_model(n=400, seed=7, save_path=str(p), csv_out=None)
        with open(p, "rb") as f:
            blob = pickle.load(f)
        return blob["pipe"], blob["classes"]

def prepare_img_from_csv_bytes(b):
    import pandas as pd
    arr = pd.read_csv(io.BytesIO(b), header=None).values.astype(float)
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr

def prepare_img_from_png_bytes(b):
    img = Image.open(io.BytesIO(b)).convert("L").resize((28, 28), Image.BILINEAR)
    arr = np.array(img).astype(float)
    arr = arr - arr.min()
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr

def predict_one(arr, model_path="models/trained/model.pkl", pipe=None, classes=None):
    """Predict from an image. If no model provided, load (and auto-retrain if needed)."""
    if pipe is None or classes is None:
        pipe, classes = load_model(model_path)
    feat = features_from_image(arr).reshape(1, -1)
    proba = pipe.predict_proba(feat)[0]
    idx = int(np.argmax(proba))
    return classes[idx], dict(zip(classes, proba.tolist()))
