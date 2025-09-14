import matplotlib.pyplot as plt

def wafer_imshow(img, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(3,3))
    cax = ax.imshow(img, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)
    return ax, cax