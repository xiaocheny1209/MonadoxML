import numpy as np
from data import load_data, clean_data
from features import extract_features, select_features, normalize_features
from models.svm import train_svm, evaluate_svm
from models.lstm import LSTMModel, train_lstm, evaluate_lstm
from sklearn.model_selection import train_test_split

# Load and preprocess data
df = load_data("data/eeg_signals.csv")
df = clean_data(df)

# Feature extraction and selection
X = extract_features(df.iloc[:, :-1])
y = df.iloc[:, -1]
X_selected = select_features(X, y)
X_normalized = normalize_features(X_selected)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_normalized, y, test_size=0.2, random_state=42
)

# Convert X_train and X_test to the shape [samples, timesteps, features] for LSTM
# Assuming each sample is a 1D array, we need to reshape it to 2D or 3D
# e.g., for a single time step per sample
X_train = np.expand_dims(X_train, axis=1)  # Shape: (samples, timesteps, features)
X_test = np.expand_dims(X_test, axis=1)  # Shape: (samples, timesteps, features)

# Train and evaluate SVM model
svm_model = train_svm(X_train, y_train)
svm_accuracy = evaluate_svm(svm_model, X_test, y_test)
print(f"SVM Model Accuracy: {svm_accuracy * 100:.2f}%")

# Initialize and train LSTM model
input_size = X_train.shape[2]  # Number of features
lstm_model = LSTMModel(input_size)
# optional: device to GPU
lstm_model = train_lstm(lstm_model, X_train, y_train, epochs=20, batch_size=32)

# Evaluate the LSTM model
lstm_accuracy = evaluate_lstm(lstm_model, X_test, y_test)
print(f"LSTM Model Accuracy: {lstm_accuracy * 100:.2f}%")
