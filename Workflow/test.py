import torch
from torch import nn
import matplotlib.pyplot as plt
import pathlib as path

# Testing out with Linear Regression: y = B1*x + B0
weight = 0.7  # B1
bias = 0.3  # B0

# Generate some random data
X = torch.arange(1,10)  # 100 samples, 1 feature
y = weight * X + bias

# Split data into training and testing
train_split = int(0.8 * len(X))  # 80% of data used for training set, 20% for testing
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]


class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, x):
        return x * self.weight + self.bias

torch.manual_seed(42)
model = Linear()

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

loops = 20
train_loss_values = []
test_loss_values = []
epoch_count = []
for times in range(loops):
    model.train()  # Set the model save to training mode

    y_pred = model.forward(X_train)  # Forward pass

    l = loss(y_pred, y_train)  # Calculate loss

    optimizer.zero_grad()  # Reset gradients to zero before backpropagation

    l.backward()  # Backpropagation

    optimizer.step()  # Update weights using the optimizer and gradients

    model.eval()

    with torch.inference_mode():  # Disable gradient calculation for efficiency
        test_pred = model.forward(X_test)
        test_loss = loss(test_pred, y_test)

    if times % 10 == 0:
        epoch_count.append(times)
        train_loss_values.append(l.detach().numpy())
        test_loss_values.append(test_loss.detach().numpy())
        print(f"Epoch: {times} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")


model.eval()

with torch.inference_mode():
    y_preds = model.forward(X_test)

# Visualize the data
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, label="Testing data", c='b')
plt.scatter(X_train, y_train, label="Training data", c='g')
plt.plot(X_test, y_preds, label="Model prediction", c='r')

plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()

model_path = path.Path('model save')