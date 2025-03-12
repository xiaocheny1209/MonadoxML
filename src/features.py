import pandas as pd
from sklearn.decomposition import PCA


def extract_features(df):
    """Extract new features (e.g., PCA, FFT)."""
    pca = PCA(n_components=10)
    return pd.DataFrame(pca.fit_transform(df))


def select_features(X, y, k=10):
    """Select best features(e.g., Random Forest)"""
    return X


def normalize_features(X):
    return X
