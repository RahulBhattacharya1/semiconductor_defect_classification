import numpy as np

CLASSES = ["center", "edge_ring", "scratch", "donut", "random"]

def _disk(h, w, cx, cy, r):
    Y, X = np.ogrid[:h, :w]
    return (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2

def _ring(h, w, cx, cy, r1, r2):
    Y, X = np.ogrid[:h, :w]
    rr = (X - cx) ** 2 + (Y - cy) ** 2
    return (rr >= r1 ** 2) & (rr <= r2 ** 2)

def _line(h, w, angle_deg=0, thickness=2):
    img = np.zeros((h, w), dtype=bool)
    angle = np.deg2rad(angle_deg)
    cx, cy = w // 2, h // 2
    for t in range(-w, w):
        x = int(cx + t * np.cos(angle))
        y = int(cy + t * np.sin(angle))
        for k in range(-thickness, thickness + 1):
            xx = x - int(k * np.sin(angle))
            yy = y + int(k * np.cos(angle))
            if 0 <= xx < w and 0 <= yy < h:
                img[yy, xx] = True
    return img

def synth_wafer(h=28, w=28, kind="center", seed=None):
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w), dtype=float)
    img += rng.normal(0, 0.03, size=(h, w))

    cx, cy = w // 2, h // 2
    if kind == "center":
        mask = _disk(h, w, cx, cy, r=rng.integers(h//8, h//4))
        img[mask] += rng.uniform(0.55, 0.9)
    elif kind == "edge_ring":
        r1 = rng.integers(h//3, h//2 - 3)
        r2 = r1 + rng.integers(1, 3)
        mask = _ring(h, w, cx, cy, r1, r2)
        img[mask] += rng.uniform(0.55, 0.9)
    elif kind == "scratch":
        ang = rng.uniform(0, 180)
        mask = _line(h, w, angle_deg=ang, thickness=rng.integers(1, 3))
        img[mask] += rng.uniform(0.55, 0.9)
    elif kind == "donut":
        r1 = rng.integers(h//6, h//4)
        r2 = r1 + rng.integers(2, 4)
        mask = _ring(h, w, cx, cy, r1, r2)
        img[mask] += rng.uniform(0.55, 0.9)
        hole = _disk(h, w, cx, cy, r=r1 - 1)
        img[hole] -= rng.uniform(0.2, 0.4)
    elif kind == "random":
        for _ in range(rng.integers(2, 5)):
            rr = rng.integers(h//10, h//4)
            mx = _disk(h, w, rng.integers(rr, w-rr), rng.integers(rr, h-rr), rr)
            img[mx] += rng.uniform(0.45, 0.85)
    else:
        raise ValueError(f"Unknown kind: {kind}")

    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    return img

def make_dataset(n=500, seed=0, classes=CLASSES):
    rng = np.random.default_rng(seed)
    imgs, ys = [], []
    for i in range(n):
        cls = classes[i % len(classes)]
        img = synth_wafer(kind=cls, seed=int(rng.integers(0, 1e9)))
        imgs.append(img)
        ys.append(cls)
    return np.stack(imgs, axis=0), np.array(ys)

if __name__ == "__main__":
    X, y = make_dataset(n=20, seed=1)
    print("Demo shapes:", X.shape, y.shape, "classes:", sorted(set(y)))