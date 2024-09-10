import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(1)

class Dataset(Dataset):
    def __init__(self, train=True):
        self.x = torch.arange(-3, 3, 0.1).view(-1, 1)
        self.f = self.x * 3 + 1
        self.y =  self.f + torch.randn(self.x.size())
        self.len = self.x.shape[0]

        if train == True:
            self.y[0] = 0
            self.y[50: 55] = 20

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.len
    
# Create training dataset and validation dataset
train_dataset = Dataset()
validation_dataset = Dataset(train=False)  

# Create train dataloader
trainloader = DataLoader(dataset=train_dataset, batch_size=1)

# Create linear regression model
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        yhat = self.linear(x)
        return yhat
    
# Citerion function
criterion = nn.MSELoss()

# Different learning rates
learning_rates = [0.0001, 0.001, 0.01, 0.1]

# Error vectors
train_errors = torch.zeros(len(learning_rates))
val_errors = torch.zeros(len(learning_rates))

# Models
MODELS = []

# Training porcess
def train_model(learning_rates, iters):
    for i, lr in enumerate(learning_rates):
        model = LinearRegression(1, 1)
        optimizer = optim.SGD(model.parameters(), lr=lr)

        for epoch in range(iters):
            for x, y in trainloader:
                y_hat = model(x)
                loss = criterion(y_hat, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Add test error
        Y_hat = model(train_dataset.x)
        loss = criterion(Y_hat, train_dataset.y)
        train_errors[i] = loss.item()

        # Add validation error
        Y_hat = model(validation_dataset.x)
        loss = criterion(Y_hat, validation_dataset.y)
        val_errors[i] = loss.item()

        # Store model
        MODELS.append(model)

# Train model
train_model(learning_rates, 10)

# Plot errors
plt.semilogx(np.array(learning_rates), train_errors.numpy(), label = 'Training loss')
plt.semilogx(np.array(learning_rates), val_errors.numpy(), label = 'Validation loss')
plt.xlabel("Learning rate")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plot prediction lines
i = 0
for i, model in enumerate(MODELS):
    plt.plot(validation_dataset.x.numpy(), model(validation_dataset.x).detach().numpy(), 
             label = 'Learning rate = ' + str(learning_rates[i]))
plt.plot(validation_dataset.x.numpy(), validation_dataset.y.numpy(), 'or', label = 'Ground truth')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()