import os
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
import pickle as pkl
from preprocessing.STEW import load_stew_data
from feature_extraction import FeatureExtraction
#from models.svm import train_svm, evaluate_svm (uncomment to use SVM model)
from feature_extraction_stew import FeatureExtractionSTEW
###Random Forest Tree Imports:
from models.svm import evaluate_svm
from sklearn.ensemble import RandomForestClassifier



if __name__ == "__main__":
    # Load STEW Dataset 
    print("\nLoading STEW dataset...")
    X, y = load_stew_data(data_dir="data/STEW", mode="task", threshold=5)
    print(f"Loaded {len(X)} samples")

    # Feature Extraction 
    freq = 128  # Emotiv EPOC sampling rate
    freq_bands = [(1, 4), (4, 8), (8, 14), (14, 30), (30, 47)]

    all_features = []
    all_labels = []

    for i in tqdm(range(len(X))):
        signal = X[i].T  # shape (14, T)
        extractor = FeatureExtractionSTEW(signal, freq_bands)
        features = extractor.extract_psd_feature(1, freq)  # shape: (windows, bands, channels)
        features = np.transpose(features, (0, 2, 1))  # shape: (windows, channels, bands)

        all_features.append(features)
        all_labels.append(np.full(features.shape[0], y[i]))  # one label per second

    all_features = np.concatenate(all_features, axis=0)  # shape: (N, 14, 5)
    all_labels = np.concatenate(all_labels, axis=0)      # shape: (N,)

    # Training
    print("\nStarting 10-fold cross-validation...")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    best_accuracy = 0
    best_fold = -1

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_labels)):
        print(f"\n  Fold {fold + 1}")

        X_train = all_features[train_idx].reshape(len(train_idx), -1)
        X_val = all_features[val_idx].reshape(len(val_idx), -1)
        y_train = all_labels[train_idx]
        y_val = all_labels[val_idx]

        #SVM Section
        # svm = train_svm(X_train, y_train)
        # acc = evaluate_svm(svm, X_val, y_val)
        #SVM Section end
        ###Random Forest Tree: 
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        acc = evaluate_svm(rf, X_val, y_val)
        ###Comment this section out and uncomment the SVM section to switch between models
        print(f"    Validation Accuracy: {acc:.4f}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_fold = fold + 1

    print(f"\nBest fold: {best_fold} with Validation Accuracy: {best_accuracy:.4f}")
