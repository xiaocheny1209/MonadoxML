from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def train_svm(X_train, y_train):
    """Train a Support Vector Machine (SVM) classifier."""
    # Create a pipeline with scaling and SVM model
    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1, gamma="scale"))
    model.fit(X_train, y_train)
    return model


def evaluate_svm(model, X_test, y_test):
    """Evaluate the SVM model performance."""
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    return accuracy
