from preprocessing import Preprocessing
from feature_extraction import FeatureExtraction
from models.svm import train_svm, evaluate_svm
from models.lstm import LSTMModel, train_lstm, evaluate_lstm
from sklearn.model_selection import train_test_split

# Load and preprocess data
print("Preprocessing data ...")
data_path = "data/FACED/sub000/data.bdf"  # replace this with your dataset(s)
preprocessor = Preprocessing(data_path)
# Optional: describe / plot data
preprocessor.filter_data(l_freq=1, h_freq=50)
preprocessor.remove_bad_channels()
preprocessor.downsample_data(new_sfreq=200)
preprocessed_data = preprocessor.data

# Extract features
print("Extracting features ...")
feature_extractor = FeatureExtraction(preprocessed_data)
all_features = feature_extractor.extract_all_features()
print("Extracted features: ", all_features)

# Normalize and select features
normalized_features = FeatureExtraction.normalize_features(all_features)
reduced_features = FeatureExtraction.apply_pca(normalized_features)
# Optional: Apply feature selection if labels are provided, supervised feature selection
# if labels is not None:
#     reduced_features = FeatureExtraction.select_best_features(reduced_features, labels)
print("Reduced features: ", reduced_features)


# TODO
# # Split data
# X_train, X_test, y_train, y_test = train_test_split(
#     X_normalized, y, test_size=0.2, random_state=42
# )

# # Convert X_train and X_test to the shape [samples, timesteps, features] for LSTM
# # Assuming each sample is a 1D array, we need to reshape it to 2D or 3D
# # e.g., for a single time step per sample
# X_train = np.expand_dims(X_train, axis=1)  # Shape: (samples, timesteps, features)
# X_test = np.expand_dims(X_test, axis=1)  # Shape: (samples, timesteps, features)

# # Train and evaluate SVM model
# svm_model = train_svm(X_train, y_train)
# svm_accuracy = evaluate_svm(svm_model, X_test, y_test)
# print(f"SVM Model Accuracy: {svm_accuracy * 100:.2f}%")

# # Initialize and train LSTM model
# input_size = X_train.shape[2]  # Number of features
# lstm_model = LSTMModel(input_size)
# # optional: device to GPU
# lstm_model = train_lstm(lstm_model, X_train, y_train, epochs=20, batch_size=32)

# # Evaluate the LSTM model
# lstm_accuracy = evaluate_lstm(lstm_model, X_test, y_test)
# print(f"LSTM Model Accuracy: {lstm_accuracy * 100:.2f}%")
