import numpy as np

def radial_profile(img):
    h, w = img.shape
    cx, cy = w // 2, h // 2
    Y, X = np.indices((h, w))
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
    max_r = int(r.max()) + 1
    prof = np.zeros(max_r, dtype=float)
    counts = np.zeros(max_r, dtype=float)
    for i in range(h):
        for j in range(w):
            rr = r[i, j]
            prof[rr] += img[i, j]
            counts[rr] += 1.0
    counts[counts == 0] = 1.0
    prof = prof / counts
    bins = np.linspace(0, len(prof)-1, 14).astype(int)
    out = []
    for k in range(len(bins)-1):
        out.append(prof[bins[k]:bins[k+1]].mean())
    out.append(prof[bins[-1]:].mean())
    return np.array(out, dtype=float)

def ring_ratio(img):
    h, w = img.shape
    cx, cy = w // 2, h // 2
    Y, X = np.indices((h, w))
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    rnorm = r / r.max()
    inner = img[rnorm < 0.3].mean()
    middle = img[(rnorm >= 0.3) & (rnorm < 0.6)].mean()
    outer = img[rnorm >= 0.6].mean()
    return np.array([inner, middle, outer, (outer + middle) / (inner + 1e-6)])

def line_energy(img):
    gy, gx = np.gradient(img)
    gmag = np.hypot(gx, gy)
    orientation = np.arctan2(gy, gx)
    bins = np.linspace(-np.pi, np.pi, 9)
    hist, _ = np.histogram(orientation, bins=bins, weights=gmag, density=True)
    return np.concatenate(([gmag.mean(), gmag.std()], hist))

def moments(img):
    m = img.mean()
    s = img.std()
    sk = ((img - m) ** 3).mean() / (s**3 + 1e-6)
    ku = ((img - m) ** 4).mean() / (s**4 + 1e-6)
    return np.array([m, s, sk, ku])

def features_from_image(img):
    rp = radial_profile(img)
    rr = ring_ratio(img)
    le = line_energy(img)
    mo = moments(img)
    return np.concatenate([rp, rr, le, mo])

def batch_features(imgs):
    return np.stack([features_from_image(x) for x in imgs], axis=0)