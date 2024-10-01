import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

def plot_training_info(training_info):
    plt.subplot(2, 1, 1)
    plt.plot(training_info['training_loss'], 'r')
    plt.ylabel('Loss')
    plt.subplot(2, 1, 2)
    plt.plot(training_info['validation_accuracy'], 'b')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.show()

# Define a function to plot model parameters
def print_model_parameters(model):
    count = 0
    for ele in model.state_dict():
        count += 1
        if count % 2 != 0:
            print ("The following are the parameters for the layer ", count // 2 + 1)
        if ele.find("bias") != -1:
            print("The size of bias: ", model.state_dict()[ele].size())
        else:
            print("The size of weights: ", model.state_dict()[ele].size())

# Define a function to display data
def show_data(data_sample):
    plt.imshow(data_sample[0].numpy().reshape(28, 28), cmap='gray')
    plt.title(f"y = {data_sample[1]}")
    plt.show()

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

# Define dataloaders
train_loader = DataLoader(dataset=training_dataset, batch_size=2000, shuffle=True)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=5000, shuffle=False)

# Define the model
class NNet(nn.Module):
    def __init__(self, input_dim, h_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, h_dim)
        self.linear2 = nn.Linear(h_dim, output_dim)

    def forward(self, x):
        h_out = torch.sigmoid(self.linear1(x))
        yhat = self.linear2(h_out)
        return yhat

# Define criterion function
criterion = nn.CrossEntropyLoss()

# Define model parameters
input_dim = 28*28
h_dim = 100
ouput_dim = 10

# Define model and optimiizer
model = NNet(input_dim, h_dim, ouput_dim)
optimizer = optim.SGD(model.parameters(), lr=0.01)

def train_model(model, criterion, train_loader, validation_loader, optimizer, epochs=100):
    training_info = {'training_loss':[], 'validation_accuracy':[]}
    for epoch in range(epochs):
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            yhat = model(x.view(-1, 28*28))
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
            training_info['training_loss'].append(loss.data.item())

        correct = 0
        for x_val, y_val in validation_loader:
            z = model(x.view(-1, 28 * 28))
            _, label = torch.max(z, 1)
            correct += (label == y).sum().item()
        accuracy = 100 * (correct / len(validation_dataset))
        training_info['validation_accuracy'].append(accuracy)

    return training_info

training_info = train_model(model, criterion, train_loader, validation_loader, optimizer, epochs=30)

# Plot trainging info
plot_training_info(training_info)

# print model parameters
print_model_parameters(model)

# Plot the first five misclassified samples

# Plot the misclassified samples
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