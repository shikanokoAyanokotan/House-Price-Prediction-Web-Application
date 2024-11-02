import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib


# R-squared
def R_squared(y_pred, y_actual):
    with torch.no_grad():
        ss_res = np.sum((y_pred - y_actual) ** 2)
        ss_tot = np.sum((y_pred - np.mean(y_actual)) ** 2)
        return 1 - ss_res / ss_tot


# Define the model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


def train_linear_regression(X_train, X_test, y_train, y_test):
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    
    # Define loss and optimizer
    input_dim = X_train_tensor.shape[1]
    model = LinearRegressionModel(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)


    # Training
    for epoch in range(1000):
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()

        # Training loss and accuracy
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1} / 1000], Loss = {loss.item():.4f}, R-squared = {R_squared(output.detach().numpy(), y_train_tensor.detach().numpy()):.4f}")

    predict = model(X_test_tensor).detach().numpy()
    actual = y_test_tensor.detach().numpy()
    print(f'Test R-squared: {R_squared(predict, actual):.4f}')


    # Save the model
    joblib.dump(model, 'predictor/linearRegressionModel.joblib')

    return model
