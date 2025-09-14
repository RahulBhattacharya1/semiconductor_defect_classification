import io
import pickle
from pathlib import Path
import numpy as np
from PIL import Image

from .features import features_from_image

def _retrain(path: Path):
    """Train a small RF so the app always has a working model."""
    from .train import train_model
    path.parent.mkdir(parents=True, exist_ok=True)
    # small, fast training set; adjusts to whatever sklearn/numpy is installed
    train_model(n=400, seed=7, save_path=str(path), csv_out=None)

def load_model(path: str = "models/trained/model.pkl"):
    """
    Load a model. If unpickling fails due to version/ABI changes or file is missing,
    retrain once and then load.
    """
    p = Path(path)
    try:
        with open(p, "rb") as f:
            blob = pickle.load(f)
        return blob["pipe"], blob["classes"]
    except Exception:
        _retrain(p)
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

def predict_one(arr, model_path: str = "models/trained/model.pkl", pipe=None, classes=None):
    """Predict using an already-loaded pipe/classes, or load (and retrain if needed)."""
    if pipe is None or classes is None:
        pipe, classes = load_model(model_path)
    feat = features_from_image(arr).reshape(1, -1)
    proba = pipe.predict_proba(feat)[0]
    idx = int(np.argmax(proba))
    return classes[idx], dict(zip(classes, proba.tolist()))
