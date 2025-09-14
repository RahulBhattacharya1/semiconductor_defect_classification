import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle

from .generate_data import make_dataset, CLASSES
from .features import batch_features

def train_model(n=600, seed=7, save_path="models/trained/model.pkl", csv_out=None):
    Ximgs, y = make_dataset(n=n, seed=seed, classes=CLASSES)
    X = batch_features(Ximgs)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=seed, class_weight="balanced_subsample"
        ))
    ])
    pipe.fit(Xtr, ytr)

    yhat = pipe.predict(Xte)
    labels = pipe.classes_
    cm = confusion_matrix(yte, yhat, labels=labels)

    print("Classes:", labels.tolist())
    print(classification_report(yte, yhat, digits=4))
    print("Confusion matrix:\n", cm)

    with open(save_path, "wb") as f:
        pickle.dump({"pipe": pipe, "classes": labels.tolist()}, f)

    if csv_out:
        nsave = min(len(Ximgs), 400)
        df = pd.DataFrame(Ximgs[:nsave].reshape(nsave, -1))
        df["label"] = y[:nsave]
        df.to_csv(csv_out, index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--save", type=str, default="models/trained/model.pkl")
    ap.add_argument("--csv", type=str, default="data/sample/wafer_samples.csv")
    args = ap.parse_args()
    train_model(n=args.n, seed=args.seed, save_path=args.save, csv_out=args.csv)