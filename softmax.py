import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from plotter_utils.plotter_data import plot_data
import matplotlib.pyplot as plt

class DataSet(Dataset):
    
    # Constructor
    def __init__(self):
        self.x = torch.arange(-2, 2, 0.1).view(-1, 1)
        self.y = torch.zeros(self.x.shape[0])
        self.y[(self.x > -1.0)[:, 0] * (self.x < 1.0)[:, 0]] = 1
        self.y[(self.x >= 1.0)[:, 0]] = 2

        # Important: 
        self.y = self.y.type(torch.LongTensor)
        
        self.len = self.x.shape[0]
        
    # Getter
    def __getitem__(self,index):      
        return self.x[index], self.y[index]
    
    # Get Length
    def __len__(self):
        return self.len

dataset = DataSet()
dataset.x
plot_data(dataset)

trainloader = DataLoader(dataset=dataset, batch_size=5)

# class Softmax(nn.Module):
#     def __init__(self, input_size, output_size):
#         super().__init__()
#         self.linear = nn.Linear(input_size, output_size)
    
#     def forward(self, x):
#         yhat = self.linear(x)
#         return yhat

# model = Softmax(1, 3)

# Alternate
model = nn.Sequential(nn.Linear(1, 3))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

LOSS = []
def train_model(iters):
    for epoch in range(iters):
        if epoch % 50 == 0:
            pass
            plot_data(dataset, model)
        for x, y in trainloader:
            yhat = model(x)
            loss = criterion(yhat, y)
            LOSS.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

train_model(300)

# Make prediction
z = model(dataset.x)
_, yhat = z.max(1)
print(f"Prdictions: {yhat}")

# Accuracy
correct =  (yhat == dataset.y).sum().item()
accuracy = correct / len(dataset)
print(f"Accuracy: {accuracy}")