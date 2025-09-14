# --- Right: prediction ---
with col2:
    st.subheader("Prediction")

    # Build a tiny demo model in memory if loading fails (no pickle dependency)
    @st.cache_resource(show_spinner=False)
    def _build_demo_model():
        from src.generate_data import make_dataset, CLASSES as _CLASSES
        from src.features import batch_features
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        import numpy as _np

        Ximgs, y = make_dataset(n=600, seed=7, classes=_CLASSES)
        X = batch_features(Ximgs)

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300, random_state=7, class_weight="balanced_subsample"
            ))
        ])
        pipe.fit(X, y)
        # keep class order consistent with training labels
        classes = sorted(set(y.tolist()))
        return pipe, classes

    try:
        # 1) Try the on-disk model (may be missing or binary-incompatible)
        from src.predict import load_model as _load_model
        try:
            pipe, classes = _load_model(model_path)
        except Exception:
            # 2) Fall back to in-memory demo model (version-proof)
            pipe, classes = _build_demo_model()

        # Predict using the already-loaded model
        pred, proba = predict_one(img, pipe=pipe, classes=classes)
        st.write(f"Predicted class: **{pred}**")
        st.bar_chart(pd.DataFrame.from_dict(proba, orient="index", columns=["probability"]))

    except Exception as e:
        st.error(f"Model not available or failed to predict: {e}")
