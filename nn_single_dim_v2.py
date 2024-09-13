import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

def plot_data(X, Y, epoch="Pre-training", model=None) -> None:
    Y[(X[:, 0] > -5) & (X[:, 0] < 0)] = 1
    Y[(X[:, 0] > 5) & (X[:, 0] < 10)] = 1
    plt.plot(X[Y==0].numpy(), Y[Y==0].numpy(), 'ro', label='training points y=0' )
    plt.plot(X[Y==1].numpy(), Y[Y==1].numpy(), 'bo', label='training points y=1' )
    if model is not None:
        plt.plot(X.numpy(), model(X).detach().numpy(), 'g-', label='neural network')
    plt.title(f"Epoch: {epoch}") 
    plt.show()

torch.manual_seed(1)

# Sample dataser
class DataSet(Dataset):
    # Constructor
    def __init__(self):
        self.x = torch.arange(-10, 15, 0.1).view(-1, 1)
        self.y = torch.zeros(self.x.shape[0])
        self.y[(self.x[:, 0] > -5) & (self.x[:, 0] < 0)] = 1
        self.y[(self.x[:, 0] > 5) & (self.x[:, 0] < 10)] = 1
        self.y = self.y.view(-1, 1)
        self.len = self.x.shape[0]
        
    # Getter
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    # Length
    def __len__(self):
        return self.len

# Define model
class NN(nn.Module):
    # Constructor
    def __init__(self, input_dim, hidden, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden)
        self.linear2 = nn.Linear(hidden, output_dim)

    # Forward pass
    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        yhat = torch.sigmoid(self.linear2(x))
        return yhat

# Criterion function
criterion = nn.BCELoss()

# Training function
def train_model(data_set, model, criterion, train_loader, optimizer, 
    epochs=1000, plot_frq=200)-> list:
    
    cost_list = []
    for epoch in range(epochs):
        cost = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
            cost += loss.item()

        if epoch % plot_frq == 0:
            plot_data(data_set.x, data_set.y, epoch, model)

        cost_list.append(cost)
    return cost_list

# Model parameters
input_dim = 1
hidden = 9
output_dim = 1
learning_rate = 0.1

# Create data loader
data_set = DataSet()
train_loader = DataLoader(dataset=data_set, batch_size=100)

# Create model
model = NN(input_dim, hidden, output_dim)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Plot initial data
plot_data(data_set.x, data_set.y)

# Train model
cost_list = train_model(data_set, model, criterion, train_loader, optimizer, epochs=2000)

# Plot cost
plt.plot(cost_list)
plt.xlabel("epoch")
plt.ylabel("cost")
plt.title("Cost per epoch")
plt.show()