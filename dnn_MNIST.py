import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

torch.manual_seed(1)

# Import MNIST dataset
training_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=False, # Set to true if you want to download the dataset
    transform=ToTensor()
)

validation_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=False, # Set to true if you want to download the dataset
    transform=ToTensor()
)

# Create data loaders
train_loader = DataLoader(dataset=training_dataset, batch_size=5000, shuffle=True)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=2000, shuffle=False)

# Define model
class NNet(nn.Module):
    def __init__(self, input_dim, h1_dim, h2_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, h1_dim)
        self.linear2 = nn.Linear(h1_dim, h2_dim)
        self.linear3 = nn.Linear(h2_dim, output_dim)

    def forward(self, x):
        # Try different activation functions, tanh worked best for this problem
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        x = self.linear3(x)
        return x

# Define training function
def train_model(model, criterion, train_loader, validation_loader, optimizer, epochs=100):
    training_info = {'training_loss':[], 'validation_accuracy':[]}
    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            z = model(x.view(-1, 28*28))
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            training_info['training_loss'].append(loss.data.item())
        
        correct_count = 0
        for x_val, y_val in validation_loader:
            z_val = model(x_val.view(-1, 28*28))
            _, yhat = torch.max(z_val, dim=1)
            correct_count += (yhat == y_val).sum().item()
        accuracy = 100 * correct_count / len(validation_dataset)
        training_info['validation_accuracy'].append(accuracy)

    return training_info

# Define model parameters
input_dim = 28*28
h1_dim = 50
h2_dim = 50
output_dim = 10

# Define model and optimizer
model = NNet(input_dim, h1_dim, h2_dim, output_dim)
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Define criterion function
criterion = nn.CrossEntropyLoss()

# Train model
training_info = train_model(model, criterion, train_loader, validation_loader, optimizer, epochs=50)

# Plot loss and accuracy
plt.subplot(2, 1, 1)
plt.plot(training_info['training_loss'], 'r')
plt.ylabel('Training loss')
plt.subplot(2, 1, 2)
plt.plot(training_info['validation_accuracy'], 'b')
plt.ylabel('Validation accuracy')
plt.xlabel('Iterations')
plt.show()