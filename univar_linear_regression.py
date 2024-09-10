import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from plotter_utils.plotter import plot_error_surfaces
import matplotlib.pyplot as plt

torch.manual_seed(0)

# define dataset
class DataSet(Dataset):
    def __init__(self):
        self.x = torch.arange(-3, 3, 0.1).view(-1, 1)
        self.y = self.x * 3 + 1 + torch.randn(self.x.size())
        self.len = self.x.shape[0]

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# create dataloader
dataset = DataSet()
train_loader = DataLoader(dataset=dataset, batch_size=1)

# define model
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        yhat = self.linear(x)
        return yhat
    
# criterion function
criterion = nn.MSELoss()

# model
model = LinearRegression(1, 1)

# optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Get the optimizer state
# optimizer.state_dict()

# initialize weights
model.state_dict()['linear.weight'][0] = -15.0
model.state_dict()['linear.bias'][0] = -10.0

# Create plot surface object
get_surface = plot_error_surfaces(15, 13, dataset.x, dataset.y, 30, go = False)

# store loss
LOSS = []

# train model
def train_model(iters):
    for epoch in range(iters):
        total = 0
        for x, y in train_loader:
            y_hat = model(x)
            loss = criterion(y_hat, y)
            get_surface.set_para_loss(model, loss.tolist())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        get_surface.plot_ps()
        LOSS.append(total)

train_model(10)

# Get the weights and biases of the model
# model.state_dict()

# Plot the loss
plt.plot(LOSS,label = "Batch Gradient Descent")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.show()