from sklearn.svm import SVC


def train_svm(x_train, y_train):
    """Train a Support Vector Machine (SVM) classifier."""
    model = SVC(kernel="rbf", C=1, gamma="scale")
    model.fit(x_train, y_train)
    return model


def evaluate_svm(model, x_test, y_test):
    """Evaluate the SVM model performance."""
    y_pred = model.predict(x_test)
    accuracy = (y_pred == y_test).mean()
    return accuracy
