# app/pages/2_🔎_Model_Analysis.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Model Analysis • Wafer Map", layout="wide")
st.title("🔎 Model Analysis")
st.caption("Synthetic, self-contained analysis to keep the page working even without a trained model.")

CLASSES = ["center", "edge_ring", "scratch", "donut", "random"]

# -------------------------------
# Minimal synth + heuristic (match main app behavior)
# -------------------------------
def gaussian_2d(h, w, cx, cy, sx, sy):
    y, x = np.mgrid[0:h, 0:w]
    return np.exp(-(((x - cx) ** 2) / (2 * sx ** 2) + ((y - cy) ** 2) / (2 * sy ** 2)))

def synthesize_wafer(label: str, size: int = 64, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = np.random.default_rng() if rng is None else rng
    base = rng.normal(0.0, 0.15, (size, size)).astype(np.float32)

    if label == "center":
        base += 1.8 * gaussian_2d(size, size, size / 2, size / 2, size / 6, size / 6)
    elif label == "edge_ring":
        g_outer = gaussian_2d(size, size, size / 2, size / 2, size / 1.9, size / 1.9)
        g_inner = gaussian_2d(size, size, size / 2, size / 2, size / 3.2, size / 3.2)
        base += 1.4 * (g_outer - g_inner)
    elif label == "scratch":
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

    base = (base - base.min()) / (base.max() - base.min() + 1e-8)
    return base

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

def heuristic_predict(img: np.ndarray) -> str:
    f = simple_features(img)
    scores = {k: 0.0 for k in CLASSES}
    scores["center"] = f["center_mean"] * 3.0 - f["edge_mean"]
    scores["edge_ring"] = f["ringness"] * 4.0 + 0.3 * f["var"]
    scores["scratch"] = f["diag_energy"] * 5.0 + 0.2 * f["var"]
    scores["donut"] = (f["ringness"] * 3.0 + max(0.0, -f["donut_gap"]) * 2.0)
    scores["random"] = 0.8 * f["var"]
    arr = np.array(list(scores.values()), dtype=np.float32)
    return CLASSES[int(np.argmax(arr))]

# -------------------------------
# Controls
# -------------------------------
left, right = st.columns([1, 1])
with left:
    per_class = st.slider("Samples per class", 20, 500, 120, step=20)
with right:
    seed = st.number_input("Seed", 0, 99999, 7, step=1)

rng = np.random.default_rng(int(seed))

# -------------------------------
# Run synthetic evaluation
# -------------------------------
y_true, y_pred = [], []
for c in CLASSES:
    for _ in range(per_class):
        img = synthesize_wafer(c, size=64, rng=rng)
        y_true.append(c)
        y_pred.append(heuristic_predict(img))

# Confusion matrix
cm = pd.crosstab(pd.Series(y_true, name="true"),
                 pd.Series(y_pred, name="pred"),
                 dropna=False).reindex(index=CLASSES, columns=CLASSES, fill_value=0)

# Normalize rows to rates for visualization
cm_rates = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)

# -------------------------------
# Plots / Tables
# -------------------------------
c1, c2 = st.columns([1, 1])

with c1:
    st.subheader("Confusion Matrix (counts)")
    fig1, ax1 = plt.subplots(layout="constrained", figsize=(6, 6))
    im = ax1.imshow(cm.values, interpolation="nearest", cmap="Blues")
    ax1.set_xticks(range(len(CLASSES))); ax1.set_xticklabels(CLASSES, rotation=30, ha="right")
    ax1.set_yticks(range(len(CLASSES))); ax1.set_yticklabels(CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax1.text(j, i, str(cm.values[i, j]), ha="center", va="center", fontsize=9)
    fig1.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    st.pyplot(fig1, use_container_width=True)  # ✅ correct

with c2:
    st.subheader("Confusion Matrix (row-normalized)")
    fig2, ax2 = plt.subplots(layout="constrained", figsize=(6, 6))
    im2 = ax2.imshow(cm_rates.values, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax2.set_xticks(range(len(CLASSES))); ax2.set_xticklabels(CLASSES, rotation=30, ha="right")
    ax2.set_yticks(range(len(CLASSES))); ax2.set_yticklabels(CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax2.text(j, i, f"{cm_rates.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig2.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    st.pyplot(fig2, use_container_width=True)  # ✅ correct

st.subheader("Confusion Matrix (table)")
# ✅ FIX: use_container_width instead of width="stretch"
st.dataframe(cm.style.background_gradient(cmap="Blues"), use_container_width=True)
