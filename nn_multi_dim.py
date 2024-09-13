import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_regions_2class(model,data_set):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#00AAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#00AAFF'])
    X = data_set.x.numpy()
    y = data_set.y.numpy()
    h = .02
    x_min, x_max = X[:, 0].min() - 0.1 , X[:, 0].max() + 0.1 
    y_min, y_max = X[:, 1].min() - 0.1 , X[:, 1].max() + 0.1 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
    XX = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])

    yhat = np.logical_not((model(XX)[:, 0] > 0.5).numpy()).reshape(xx.shape)
    plt.pcolormesh(xx, yy, yhat, cmap=cmap_light)
    plt.plot(X[y[:, 0] == 0, 0], X[y[:, 0] == 0, 1], 'o', label='y=0')
    plt.plot(X[y[:, 0] == 1, 0], X[y[:, 0] == 1, 1], 'ro', label='y=1')
    plt.title("decision region")
    plt.legend()

# Define dataset
class DataSet(Dataset):
    # Constructor
    def __init__(self, N_s = 100):
        self.x = torch.zeros(N_s, 2)
        self.y = torch.zeros(N_s, 1)
        for i in range(N_s//4):
            self.x[i, :] = torch.Tensor([[1.0], [1.0]])
            self.y[i, :] = torch.Tensor([[0.0]])

            self.x[i + N_s//4, :] = torch.Tensor([[1.0], [-1.0]])
            self.y[i + N_s//4, :] = torch.Tensor([[1.0]])

            self.x[i + N_s//2, :] = torch.Tensor([[-1.0], [-1.0]])
            self.y[i + N_s//2, :] = torch.Tensor([[0.0]])

            self.x[i + 3*N_s//4, :] = torch.Tensor([[-1.0], [1.0]])
            self.y[i + 3*N_s//4, :] = torch.Tensor([[1.0]])
            self.x = self.x + 0.1 * torch.randn((N_s, 2))
        self.len = self.x.shape[0]

    # Getter
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    # Get length
    def __len__(self):
        return self.len
    
# Define train loader
data_set = DataSet()
train_loader = DataLoader(dataset=data_set, batch_size=1)
    
# Define mdodel
class Net(nn.Module):
    # Constructor
    def __init__(self, input_dim, hidden, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, output_dim)

        # Prediction
        def forward(self, x):
            x = torch.sigmoid(self.linear1(x))
            x = torch.sigmoid(self.linear2(x))
            return x
        
# Define criterion function
criterion = nn.BCELoss()

# Define accuracy function
def accuracy(data_set, model):
    return np.mean(data_set.y.view(-1).numpy() == (model(data_set.x)[:, 0] > 0.5).numpy())

# Training function
def train_model(data_set, model, train_loader, criterion, optimizer, epochs=1000):
    LOSS = []
    ACC = []

    for epoch in range(epochs):
        loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
            loss += loss.item()

        LOSS.append(loss)
        ACC.append(accuracy(data_set, model))

    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.plot(LOSS, color=color)
    ax1.set_xlabel('epoch', color=color)
    ax1.set_ylabel('total loss', color=color)
    ax1.tick_params(axis='y', color=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.plot(ACC, color=color)
    ax2.set_ylabel('accuracy', color=color)
    ax2.tick_params(axis='y', color=color)
    fig.tight_layout()
    plt.show()

    return LOSS

# Triain model --> single hidden neuron
model = Net(2, 1, 1)
optimizer = optim.SGD(model.parameters(), lr=0.01)
cost = train_model(data_set, model, train_loader, criterion, optimizer)
# Plot decision boundary
plot_decision_regions_2class(model, data_set)