import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from plotter_utils.plotter2d import Plot2DPlane

torch.manual_seed(1)

# Define dataset
class DataSet2D(Dataset):
    def __init__(self):
        self.x = torch.zeros(20, 2)
        self.x[:, 0] = torch.arange(-5, 5, 0.5)
        self.x[:, 1] = torch.arange(-1, 1, 0.1)
        self.w = torch.tensor([[1.0], [1.0]])
        self.b = 1
        self.f = torch.matmul(self.x, self.w) + self.b
        self.y = self.f + 0.1 * torch.randn((self.x.shape[0], 1))
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return (self.x[index], self.y[index])
    
    def __len__(self):
        return self.len
    

    
# Insatance of dataset
dataset = DataSet2D()

# Create dataloader
trainloader = DataLoader(dataset=dataset, batch_size=20)

# Define model
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        yhat = self.linear(x)
        return yhat

# Define criterion function
criterion = nn.MSELoss()

# Create model instance
model = LinearRegression(2, 1)

# Create optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1)

LOSS = []
epochs = 100
print("Before training: \n")
print(model.state_dict())
# Plot the error surface
Plot2DPlane(model, dataset)

def train_model(epochs):
    for epoch in range(epochs):
        for x, y in trainloader:
            yhat = model(x)
            loss = criterion(yhat, y)
            LOSS.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

train_model(epochs)

print("After training: \n")
print(model.state_dict())
# Plot the error surface
Plot2DPlane(model, dataset, epochs)

