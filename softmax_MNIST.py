import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn, optim
import matplotlib.pyplot as plt

from plotter_utils.plotter_data import plot_parameters, show_data

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

train_loader = DataLoader(dataset=training_dataset, batch_size=100, shuffle=True)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=100, shuffle=False)

# Define the model
class Softmax(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        z = self.linear(x)
        return z

# Defie input and output dims
input_dim = 28*28
output_dim = 10

# Create model
model = Softmax(input_dim, output_dim)
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Print parameters
print('W: ',list(model.parameters())[0].size())
print('b: ',list(model.parameters())[1].size())

# Plot parameters
plot_parameters(model=model)

n_epochs = 15
accuracy_list = []
loss_list = []
n_test = len(validation_dataset)

# Define trainging function
def train_model(epochs):
    for epoch in range(epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            # Flatten X --> from [100, 1, 28, 28] to [100, 784]
            z = model(x.view(-1, 28*28))
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()

        correct_count = 0
        for x_val, y_val in validation_loader:
            z_val = model(x_val.view(-1, 28*28))
            _, yhat = torch.max(z_val.data, dim=1)
            correct_count += (yhat == y_val).sum().item()
        accuracy = correct_count/n_test
        loss_list.append(loss.data)
        accuracy_list.append(accuracy)

train_model(n_epochs)

# Plot loss and accuracy
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(loss_list, color=color)
ax1.set_xlabel('epoch', color=color)
ax1.set_ylabel('loss', color=color)
ax1.tick_params(axis='y', color=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.plot(accuracy_list, color=color)
ax2.set_ylabel('accuracy', color=color)
ax2.tick_params(axis='y', color=color)
fig.tight_layout()
fig.show()

# Plot parameters after training
plot_parameters(model=model)

# Plot the misclassified samples
Softmax_fn=nn.Softmax(dim=-1)
count = 0
for x, y in validation_dataset:
    z = model(x.reshape(-1, 28 * 28))
    _, yhat = torch.max(z, 1)
    if yhat != y:
        show_data((x, y))
        plt.show()
        print("yhat:", yhat)
        print("probability of class ", torch.max(Softmax_fn(z)).item())
        count += 1
    if count >= 5:
        break