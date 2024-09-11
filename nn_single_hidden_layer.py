import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim

torch.manual_seed(1)

# Sample dataset
X = torch.arange(-5, 5, 0.5).view(-1, 1)
y = torch.zeros(X.shape[0])
y[(X[:, 0] > -2) & (X[:, 0] < 2)] = 1

# Define the model
class NN(nn.Module):
    # Constructor
    def __init__(self, input_dim, hidden, output_dim):
        super().__init__()
        # Hidden layer
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, output_dim)
        # Attriibutes
        self.l1 = None
        self.a1 = None
        self.l2 = None

    # Forward pass
    def forward(self, x):
        self.f1 = self.linear1(x)
        self.a1 = torch.sigmoid(self.f1)
        self.f2 = self.linear2(self.a1)
        yhat = torch.sigmoid(self.f2)
        return yhat

# Define criterion function -->  cross entropy loss
def criterion(yhat, y):
    return -1 * torch.mean(y * torch.log(yhat) + (1-y) * torch.log(1-yhat))

input_dim = 1
hidden = 2
output_dim = 1

# Instantiate the model
model = NN(input_dim, hidden, output_dim)
# Define optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training function
def train_model(model, X, Y, criterion, optimizer, epochs=1000):
    loss_list = []
    for epoch in range(epochs):
        total_loss = 0
        for x, y in zip(X, Y):
            yhat = model(x)
            loss = criterion(yhat, y)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_list.append(total_loss)

        if epoch % 200 == 0:
            # Plot predictions vs actual
            plt.plot(X.numpy(), model(X).detach().numpy(), 'r', label="Predictions")
            plt.plot(X.numpy(), Y.numpy(), 'b', label="GroundTruth")
            plt.legend()
            plt.title(f"Epoch: {epoch}")
            plt.show()

            # Plot activations
            model(X)
            plt.scatter(
                model.a1.detach().numpy()[:, 0],
                model.a1.detach().numpy()[:, 1],
                c=Y.numpy().reshape(-1),
                )
            plt.title("Activations")
            plt.show()
    return loss_list

cost = train_model(model, X, y, criterion, optimizer)

# Plot training loss
plt.plot(cost)
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.title("Training Loss")
plt.show()

