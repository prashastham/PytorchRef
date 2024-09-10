import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from plotter_utils.plotter2d import Plot2DPlane

torch.manual_seed(1)

# # Define dataset
# class DataSet(Dataset):
#     def __init__(self):
#         self.x = torch.zeros(20, 2)
#         self.x[:, 0] = torch.arange(-5, 5, 0.5)
#         self.x[:, 1] = torch.arange(-1, 1, 0.1)

#         self.w = torch.tensor([[1.5], [1.5]])
#         self.b = torch.tensor([[1.0], [1.0]])

#         self.f = torch.zeros(20, 2)
#         self.f[:, 0] = self.x[:, 0] * self.w[0] + self.b[0]
#         self.f[:, 1] = self.x[:, 1] * self.w[1] + self.b[1]

#         self.y = torch.zeros(20, 2)
#         self.y[:, 0] = self.f[:, 0] + 0.1 * torch.randn((self.x.shape[0]))
#         self.y[:, 1] = self.f[:, 1] + 0.1 * torch.randn((self.x.shape[0]))
        
#         self.len = self.x.shape[0]

#     def __getitem__(self, idx):
#         return self.x[idx], self.y[idx]
    
#     def __len__(self):
#         return self.len

class DataSet2D(Dataset):
    def __init__(self):
        self.x = torch.zeros(100, 2)
        self.x[:, 0] = torch.arange(-5, 5, 0.1)
        self.x[:, 1] = torch.arange(-1, 1, 0.02)

        self.w = torch.tensor([[-1.5, 1.5], [-1.0, 1.0]])
        self.b = torch.tensor([[1.0, 1.0]])

        self.f = torch.matmul(self.x, self.w) + self.b
        self.y = self.f + 0.1 * torch.randn((self.x.shape[0], 1))

        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len
    
# Instance of daraset
dataset = DataSet2D()

# Trainig dataloader
trainloader = DataLoader(dataset=dataset, batch_size=10)

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

# Insttance of model
model = LinearRegression(2, 2)

# Define optimizer
optimizer = optim.SGD(model.parameters(), lr=0.1)

LOSS = []
epochs = 10

# Training function
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

# Plot loss
plt.plot(LOSS)
plt.xlabel("iterations")
plt.ylabel("Cost/total loss")
plt.show()

for i in range(5):
    print(f"Prediction: {model(dataset.x[i].data)}, Actual: {dataset.y[i].data}")