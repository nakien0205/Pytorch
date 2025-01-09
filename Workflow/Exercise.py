import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

weight = 0.3
bias = 0.9

X = torch.arange(1, 100)
y = weight * X + bias

train_test_split = int(0.8 * len(X))
X_train, y_train = X[:train_test_split], y[:train_test_split]
X_test, y_test = X[train_test_split:], y[train_test_split:]

class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(1), requires_grad=True)
        self.bias = nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(self, x):
        return self.weight * x + self.bias


model = Linear()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss = nn.MSELoss()

epochs = 100
train_loss_values = []
test_loss_values = []
epoch_count = []
for i in range(epochs):
    model.train()
    y_pred = model(X_train)
    l = loss(y_pred, y_train)
    optimizer.zero_grad()
    l.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_pred = model.forward(X_test)
        test_loss = loss(test_pred, y_test)

    if i % 10 == 0:
        epoch_count.append(i)
        train_loss_values.append(l.detach().numpy())
        test_loss_values.append(test_loss.detach().numpy())
        print(f"Epoch: {i} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")


model.eval()
with torch.inference_mode():
    y_preds = model.forward(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, label="Testing data", c='b')
plt.scatter(X_train, y_train, label="Training data", c='g')
plt.plot(X_test, y_preds, label="Model prediction", c='r')
plt.show()

plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()