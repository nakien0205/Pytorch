from helper_functions import plot_decision_boundary
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as op


classes = 4
features = 2

X_blob, y_blob = make_blobs(n_samples=1000, centers=classes, n_features=features, random_state=42)
X_blob = torch.from_numpy(X_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.long)

X_train, X_test, y_train, y_test = train_test_split(X_blob, y_blob, test_size=0.2, random_state=42)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
X_train, X_test, y_train, y_test = X_train.to(device), X_test.to(device), y_train.to(device), y_test.to(device)

class BlobClassification(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(features, 20)
        self.linear2 = nn.Linear(20, 10)
        self.linear3 = nn.Linear(10, 4)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x

model = BlobClassification().to(device)
optimizer = op.Adam(model.parameters(), lr=0.01)
loss_fun = nn.CrossEntropyLoss()

torch.manual_seed(42)

epochs = 100
for i in range(epochs):
    model.train()

    y_logit = model(X_train.to(device))
    y_pred = torch.softmax(y_logit, dim=1)

    loss = loss_fun(y_logit, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_logits = model(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fun(test_logits, y_test)
    if i % 10 == 0:
        print(f"Epoch: {epochs} | Loss: {loss:.5f} | Test loss: {test_loss:.5f}")

# Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model, X_train, y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model, X_test, y_test)
plt.show()