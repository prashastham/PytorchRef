import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# Define a function to display data
def show_data(data_sample):
    plt.imshow(data_sample[0].numpy().reshape(28, 28), cmap='gray')
    plt.title(f"y = {data_sample[1]}")
    plt.show()

torch.manual_seed(1)

# Dataset
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
                x = torch.tanh(linear_layer(x))
            else:
                x = linear_layer(x)
        return x
    
# Training function
def train_model(model, criterion, train_loader, validation_loader, optimizer, epochs=100):
    training_info = {
        'training_loss':[],
        'validation_accuracy':[]
    }
    for epoch in range(epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            z = model(x.view(-1, 28*28))
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            training_info['training_loss'].append(loss.data.item())
        correct_count = 0
        for x_val, y_val in validation_loader:
            z_val = model(x_val.view(-1, 28*28))
            _, yhat = torch.max(z_val.data, dim=1)
            correct_count += (yhat == y_val).sum().item()
        accuracy = 100 * (correct_count / len(validation_dataset))
        training_info['validation_accuracy'].append(accuracy)
    return training_info

# Define model parameters
layers = [28*28, 128, 64, 10]

# Instantiate model, loss function and optimizer
model = NNet(layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train model
training_info = train_model(model, criterion, train_loader, validation_loader, optimizer, epochs=20)

# Plot training info
plt.subplot(2, 1, 1)
plt.plot(training_info['training_loss'], 'r')
plt.ylabel('Training Loss')
plt.subplot(2, 1, 2)
plt.plot(training_info['validation_accuracy'], 'b')
plt.ylabel('Validation Accuracy')
plt.xlabel('Iterations')
plt.show()

# Print misclassified samples
Softmax_fn=nn.Softmax(dim=-1)
count = 0
for x, y in validation_dataset:
    z = model(x.reshape(-1, 28 * 28))
    _, yhat = torch.max(z, 1)
    if yhat != y:
        show_data((x, y))
        print("yhat:", yhat)
        print("probability of class ", torch.max(Softmax_fn(z)).item())
        count += 1
        if count >= 5:
            break