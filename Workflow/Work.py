import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
X = torch.tensor(data.data, dtype=torch.float32)
y = torch.tensor(data.target, dtype=torch.float32).unsqueeze(1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(8, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegression()
optimizer = optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

epochs = 100
train_losses = []
for epoch in range(epochs):
    model.train()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")

model.eval()
with torch.inference_mode():
    y_pred_true = model(X_test)
    test_loss = loss_fn(y_pred_true, y_test)
print(f"Test Loss: {test_loss.item()}")

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_true, label="Predicted vs True", color='b')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label="Ideal Prediction")
plt.xlabel("True Prices")
plt.ylabel("Predicted Prices")
plt.legend()
plt.title("Boston Housing Price Prediction")
plt.show()