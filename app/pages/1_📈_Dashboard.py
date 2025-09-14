from __future__ import annotations

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard • Wafer Map Defect Classifier", layout="wide")

st.title("Dashboard")
st.caption("Model-free demo dashboard with synthetic metrics for the wafer classes.")

CLASSES = ["center", "edge_ring", "scratch", "donut", "random"]

# Controls
colA, colB, colC = st.columns(3)
with colA:
    n_samples = st.slider("Samples", 100, 5000, 1000, step=100)
with colB:
    rng_seed = st.number_input("Seed", 0, 99999, 42, step=1)
with colC:
    show_conf = st.checkbox("Show confusion matrix", True)

rng = np.random.default_rng(int(rng_seed))

# Fake per-class support and accuracy just to render charts without a trained model.
support = rng.integers(low=max(10, n_samples // 40), high=max(20, n_samples // 10), size=len(CLASSES))
support = (support / support.sum() * n_samples).astype(int)
per_class_acc = np.clip(rng.normal(loc=0.82, scale=0.08, size=len(CLASSES)), 0.4, 0.98)

# Summary metrics
overall_acc = float(np.average(per_class_acc, weights=support))
st.metric("Overall accuracy (demo)", f"{overall_acc*100:.1f}%")

# Bar: per-class accuracy
fig1, ax1 = plt.subplots(layout="constrained", figsize=(6, 3))
ax1.bar(CLASSES, per_class_acc)
ax1.set_ylim(0, 1)
ax1.set_ylabel("accuracy")
ax1.set_title("Per-class accuracy (demo)")
for i, v in enumerate(per_class_acc):
    ax1.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
st.pyplot(fig1, use_container_width=True)  # ✅ correct usage

# Support chart
fig2, ax2 = plt.subplots(layout="constrained", figsize=(6, 3))
ax2.bar(CLASSES, support)
ax2.set_ylabel("samples")
ax2.set_title("Per-class support (demo)")
st.pyplot(fig2, use_container_width=True)  # ✅ correct usage

# Confusion matrix (demo)
if show_conf:
    # Construct a plausible confusion matrix consistent with the supports and accuracies
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=float)
    for i, s in enumerate(support):
        tp = int(round(s * per_class_acc[i]))
        fp = s - tp
        cm[i, i] = tp
        if fp > 0:
            off = rng.dirichlet(np.ones(len(CLASSES) - 1)) * fp
            cm[i, np.arange(len(CLASSES)) != i] = off
    # Normalize row-wise to show rates
    row_sum = cm.sum(axis=1, keepdims=True) + 1e-9
    cm_norm = cm / row_sum

    fig3, ax3 = plt.subplots(layout="constrained", figsize=(6, 6))
    im = ax3.imshow(cm_norm, interpolation="nearest")
    ax3.set_title("Confusion matrix (row-normalized, demo)")
    ax3.set_xticks(range(len(CLASSES))); ax3.set_xticklabels(CLASSES, rotation=30, ha="right")
    ax3.set_yticks(range(len(CLASSES))); ax3.set_yticklabels(CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax3.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", fontsize=9)
    st.pyplot(fig3, use_container_width=True)  # ✅ correct usage
