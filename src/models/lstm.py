import torch
import torch.nn as nn
import torch.optim as optim


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()  # For binary classification

    def forward(self, x):
        # LSTM forward pass
        out, _ = self.lstm(x)
        # Only take the output from the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.sigmoid(out)
        return out


def train_lstm(model, X_train, y_train, epochs=20, batch_size=32, lr=0.001):
    """Train the LSTM model with PyTorch."""
    criterion = nn.BCELoss()  # Binary Cross-Entropy for binary classification
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    # Training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()  # Backpropagation
        optimizer.step()  # Optimizer step
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    return model


def evaluate_lstm(model, X_test, y_test):
    """Evaluate the LSTM model performance with PyTorch."""
    model.eval()
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    with torch.no_grad():  # No need to track gradients during evaluation
        outputs = model(X_test)
        predicted = (outputs.squeeze() > 0.5).float()  # Convert output to 0 or 1
        accuracy = (predicted == y_test).float().mean()

    return accuracy.item()
