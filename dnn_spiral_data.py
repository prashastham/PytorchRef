import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

torch.manual_seed(1)

# Define the function to plot decision regions
def plot_decision_regions_3class(model, data_set):
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#00AAFF'])
    #cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#00AAFF'])
    X = data_set.x.numpy()
    y = data_set.y.numpy()
    h = .02
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1 
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    XX = torch.Tensor(np.c_[xx.ravel(), yy.ravel()])
    _, yhat = torch.max(model(XX), 1)
    yhat = yhat.numpy().reshape(xx.shape)
    plt.pcolormesh(xx, yy, yhat, cmap=cmap_light)
    plt.plot(X[y[:] == 0, 0], X[y[:] == 0, 1], 'ro', label = 'y=0')
    plt.plot(X[y[:] == 1, 0], X[y[:] == 1, 1], 'go', label = 'y=1')
    plt.plot(X[y[:] == 2, 0], X[y[:] == 2, 1], 'o', label = 'y=2')
    plt.title("decision region")
    plt.legend()

# Create spiral dataset
class Spiral2D(Dataset):
    '''
        N: number of points per class
        D: dimensionality
        K: number of classes   
    '''
    def __init__(self, K=3, N=500):
        self.D = 2 # dimensionality set to 2
        self.K = K # number of classes
        self.N = N # number of points per class
        X = np.zeros((self.N*self.K, self.D)) # data matrix (each row = single example)
        y = np.zeros(self.N*self.K, dtype='uint8') # class labels
        for j in range(self.K):
            ix = range(self.N*j, self.N*(j+1))
            r = np.linspace(0.0,1,self.N) # radius
            t = np.linspace(j*4,(j+1)*4,self.N) + np.random.randn(self.N)*0.2 # theta
            X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
            y[ix] = j

        self.x = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.len
    
    def plot_data(self):
        plt.plot(self.x[self.y==0, 0], self.x[self.y==0, 1], 'o', label="y=0")
        plt.plot(self.x[self.y==1, 0], self.x[self.y==1, 1], 'ro', label="y=1")
        plt.plot(self.x[self.y==2, 0], self.x[self.y==2, 1], 'go', label="y=2")
        plt.legend()
        plt.show()

# Define model
class NNet(nn.Module):
    def __init__(self, Layers):
        super().__init__()
        self.hidden = nn.ModuleList()
        for input_size, output_size in zip(Layers, Layers[1:]):
                self.hidden.append(nn.Linear(input_size, output_size))

    # Define the forward function
    def forward(self, x):
        L = len(self.hidden)
        for (l, linear_layer) in zip(range(L), self.hidden):
            if l < L-1:
                x = torch.relu(linear_layer(x))
            else:
                x = linear_layer(x)
        return x
    
# Defne training function
def train(data_set, model, criterion, train_loader, optimizer, epochs=100):
    training_info = {
        'loss': [],
        'accuracy': []
    }

    for epoch in range(epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            yhat = model(x)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
            training_info['loss'].append(loss.data.item())
        
        predictions = model(data_set.x)
        _, yhat = torch.max(predictions, dim=1)
        accuracy = 100 * (yhat == data_set.y).numpy().mean()
        training_info['accuracy'].append(accuracy)

    return training_info

# Define model parameters
layers = [2, 20, 10, 3]
# Instantiate model, loss function and optimizer
model = NNet(layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
# Create dataset
data_set = Spiral2D()
# Create data loader
train_loader = DataLoader(dataset=data_set, batch_size=100)
# Train model
training_info = train(data_set, model, criterion, train_loader, optimizer, epochs=1000)
# Plot training loss
plt.subplot(2, 1, 1)
plt.plot(training_info['loss'], 'r')
plt.ylabel('loss')
# Plot training accuracy
plt.subplot(2, 1, 2)
plt.plot(training_info['accuracy'], 'b')
plt.ylabel('accuracy')
plt.xlabel('iterations')
plt.show()
# Plot decision regions
plot_decision_regions_3class(model, data_set)
