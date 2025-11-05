Inference recipe:
1) Load final_model_sklearn.pkl
2) Ensure input columns match feature_names.json order
3) Convert input to float32 ndarray with same column order
4) p = model.predict_proba(X)[:,1]
5) y_hat = (p >= 0.290000).astype(int)
