# app/app.py
# Wafer Map Defect Classifier – Streamlit demo
# - Fixes st.pyplot misuse (uses use_container_width=True)
# - Adds robust error handling and a light heuristic fallback “model”
# - Works in two modes: Generate (synthetic) and Upload (.npy or image)

from __future__ import annotations

import io
import os
import sys
import json
import math
from typing import Tuple, Optional

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# -------------------------------
# App / page config
# -------------------------------
st.set_page_config(
    page_title="Wafer Map Defect Classifier",
    page_icon="🧪",
    layout="wide",
)

TITLE = "Wafer Map Defect Classifier"
CLASSES = ["center", "edge_ring", "scratch", "donut", "random"]

# -------------------------------
# Utilities
# -------------------------------

def show_versions_banner(expected: dict[str, str] | None = None) -> None:
    """Optionally warn if critical packages differ from what you pinned."""
    try:
        import numpy as _np
        import matplotlib as _mpl
        vers = {"numpy": _np.__version__, "matplotlib": _mpl.__version__, "python": ".".join(map(str, sys.version_info[:3]))}
        if expected:
            mismatch = {k: (vers.get(k, "?"), v) for k, v in expected.items() if vers.get(k) != v}
            if mismatch:
                st.info(f"Environment versions: {vers}. Expected: {expected}.")
        else:
            st.caption(f"Env: numpy {vers['numpy']} • matplotlib {vers['matplotlib']} • Python {vers['python']}")
    except Exception:
        pass


def as_uint8(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32)
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return (x * 255).astype(np.uint8)


def draw_wafer(ax: plt.Axes, arr: np.ndarray, title: str = "Input") -> None:
    ax.imshow(arr, cmap="viridis", interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])


def gaussian_2d(h: int, w: int, cx: float, cy: float, sx: float, sy: float) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    return np.exp(-(((x - cx) ** 2) / (2 * sx ** 2) + ((y - cy) ** 2) / (2 * sy ** 2)))


def synthesize_wafer(label: str, size: int = 64, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 0.15, (size, size)).astype(np.float32)

    if label == "center":
        base += 1.8 * gaussian_2d(size, size, size / 2, size / 2, size / 6, size / 6)

    elif label == "edge_ring":
        g_outer = gaussian_2d(size, size, size / 2, size / 2, size / 1.9, size / 1.9)
        g_inner = gaussian_2d(size, size, size / 2, size / 2, size / 3.2, size / 3.2)
        base += 1.4 * (g_outer - g_inner)

    elif label == "scratch":
        # thin diagonal line
        for i in range(size):
            j = int(0.2 * size + 0.6 * i) % size
            base[i, max(0, j - 1):min(size, j + 2)] += 1.8
        base = (base + rng.normal(0, 0.05, base.shape)).astype(np.float32)

    elif label == "donut":
        g_outer = gaussian_2d(size, size, size / 2, size / 2, size / 5, size / 5)
        g_center = gaussian_2d(size, size, size / 2, size / 2, size / 10, size / 10)
        base += 2.0 * (g_outer - g_center)

    elif label == "random":
        base += rng.normal(0.0, 0.6, base.shape).astype(np.float32)

    # clip + scale
    base = (base - base.min()) / (base.max() - base.min() + 1e-8)
    return base


# -------------------------------
# Lightweight heuristic classifier
# -------------------------------

def radial_profile(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    y, x = np.mgrid[0:h, 0:w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    rbin = np.clip(r.astype(int), 0, max(h, w))
    prof = np.bincount(rbin.ravel(), weights=img.ravel(), minlength=rbin.max() + 1)
    cnts = np.bincount(rbin.ravel(), minlength=rbin.max() + 1) + 1e-8
    return (prof / cnts)[: (min(h, w) // 2 + 1)]


def simple_features(img: np.ndarray) -> dict:
    rp = radial_profile(img)
    center_mean = img[img.shape[0]//2 - 4:img.shape[0]//2 + 4, img.shape[1]//2 - 4:img.shape[1]//2 + 4].mean()
    edge_mean = np.mean([rp[-3:], rp[-6:-3]])
    ringness = (rp[int(len(rp)*0.75)] - rp[int(len(rp)*0.45)])
    donut_gap = rp[int(len(rp)*0.35)] - rp[int(len(rp)*0.15)]
    var = img.var()
    # scratch detector via Sobel-like gradient energy
    gy, gx = np.gradient(img)
    diag_energy = np.mean((gx + gy) ** 2)
    return {
        "center_mean": float(center_mean),
        "edge_mean": float(edge_mean),
        "ringness": float(ringness),
        "donut_gap": float(donut_gap),
        "var": float(var),
        "diag_energy": float(diag_energy),
    }


def heuristic_predict(img: np.ndarray) -> Tuple[str, dict]:
    f = simple_features(img)
    scores = {k: 0.0 for k in CLASSES}
    # center: high center_mean relative to edge
    scores["center"] = f["center_mean"] * 3.0 - f["edge_mean"]
    # edge_ring: high edge relative to mid + some variance
    scores["edge_ring"] = f["ringness"] * 4.0 + 0.3 * f["var"]
    # scratch: strong diagonal gradient energy
    scores["scratch"] = f["diag_energy"] * 5.0 + 0.2 * f["var"]
    # donut: strong outer vs inner contrast with inner dip
    scores["donut"] = (f["ringness"] * 3.0 + max(0.0, -f["donut_gap"]) * 2.0)
    # random: high variance but weak structure → baseline on var
    scores["random"] = 0.8 * f["var"]

    # softmax-ish
    arr = np.array(list(scores.values()), dtype=np.float32)
    exp = np.exp(arr - arr.max())
    probs = exp / (exp.sum() + 1e-8)
    pred_idx = int(np.argmax(probs))
    pred = CLASSES[pred_idx]
    out = {c: float(p) for c, p in zip(CLASSES, probs)}
    return pred, out


# -------------------------------
# Optional: load a real model if present
# -------------------------------

def try_load_model(path: str = "model.pkl"):
    """Return (predict_fn, label) where predict_fn(img)->(pred, probs)."""
    if not os.path.isfile(path):
        return None, "fallback-heuristic (no model found)"
    try:
        import joblib
        model = joblib.load(path)

        def _predict(img: np.ndarray) -> Tuple[str, dict]:
            x = img.astype(np.float32).reshape(1, -1)
            prob = model.predict_proba(x)[0]
            idx = int(np.argmax(prob))
            return CLASSES[idx], {c: float(p) for c, p in zip(CLASSES, prob)}
        return _predict, "joblib-model"
    except Exception as e:
        st.warning(f"Model load failed ({e}). Falling back to heuristic.")
        return None, "fallback-heuristic (load error)"


# -------------------------------
# Sidebar controls
# -------------------------------
st.sidebar.header("Input")

mode = st.sidebar.radio("Input Mode", ["Generate", "Upload"], index=0)
default_class = st.sidebar.selectbox("Default class", CLASSES, index=0)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1)

# -------------------------------
# Main title
# -------------------------------
st.title(TITLE)
st.caption("Synthetic demo for yield engineering: center, edge_ring, scratch, donut, random")

show_versions_banner()

# -------------------------------
# Prepare input wafer map
# -------------------------------
img: Optional[np.ndarray] = None

col_input, col_pred = st.columns([1, 1])

with col_input:
    st.subheader("Wafer Map")

    if mode == "Generate":
        img = synthesize_wafer(default_class, size=64, seed=int(seed))
    else:
        up = st.file_uploader("Upload wafer (.npy grayscale 2D) or image (PNG/JPG)", type=["npy", "png", "jpg", "jpeg"])
        if up is not None:
            try:
                if up.name.lower().endswith(".npy"):
                    arr = np.load(io.BytesIO(up.read()))
                    if arr.ndim == 3:
                        arr = arr.mean(axis=2)  # to grayscale
                    img = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
                else:
                    from PIL import Image
                    im = Image.open(up).convert("L").resize((64, 64))
                    img = np.asarray(im, dtype=np.float32) / 255.0
            except Exception as e:
                st.error(f"Failed to read upload: {e}")

    # Always draw something (placeholder if none)
    fig, ax = plt.subplots(layout="constrained", figsize=(4, 4))
    if img is None:
        placeholder = np.zeros((64, 64), dtype=np.float32)
        draw_wafer(ax, placeholder, title="Waiting for input…")
    else:
        draw_wafer(ax, img, title="Input")

    # ✅ FIX: use_container_width instead of width="content"
    st.pyplot(fig, use_container_width=True)

# -------------------------------
# Prediction
# -------------------------------
with col_pred:
    st.subheader("Prediction")

    if img is None:
        st.info("Provide an input to see predictions.")
    else:
        predict_fn, model_label = try_load_model()
        if predict_fn is None:
            pred, probs = heuristic_predict(img)
        else:
            pred, probs = predict_fn(img)

        st.markdown(f"**Predicted class:** `{pred}`  \n_Model:_ {model_label}")

        # bar chart of probabilities
        fig2, ax2 = plt.subplots(layout="constrained", figsize=(5, 3))
        ax2.bar(list(probs.keys()), list(probs.values()))
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("probability")
        ax2.set_title("Class probabilities")
        for i, (k, v) in enumerate(probs.items()):
            ax2.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        st.pyplot(fig2, use_container_width=True)

        # show raw features to help debug (optional)
        with st.expander("Debug: features (heuristic)"):
            st.json(simple_features(img))
