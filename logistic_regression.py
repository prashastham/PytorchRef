import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from plotter_utils.plotter_logistic import plot_error_surfaces

torch.manual_seed(1)

# Define dataset
class DataSet(Dataset):
    def __init__(self):
        self.x = torch.arange(-1, 1, 0.1).view(-1, 1)
        self.y = torch.zeros(self.x.shape[0], 1)
        self.y[self.x[:, 0] > 0.2] = 1
        self.len = self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.len
    
dataset = DataSet()
dataloader = DataLoader(dataset=dataset, batch_size=5)


class logistic_regression(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        yhat = torch.sigmoid(self.linear(x))
        return yhat
    
# Alternate
# logistic_regression = nn.Sequential(
#     nn.Linear(1, 1),
#     nn.Sigmoid()
# )
    
criterion = nn.MSELoss()
# Custom
# def criterion(yhat,y):
#     out = -1 * torch.mean(y * torch.log(yhat) + (1 - y) * torch.log(1 - yhat))
#     return out

model = logistic_regression(1)
model.state_dict() ['linear.weight'].data[0] = torch.tensor([[-5]])
model.state_dict() ['linear.bias'].data[0] = torch.tensor([[-10]])

optimizer = torch.optim.SGD(model.parameters(), lr=2.5)

get_surface = plot_error_surfaces(15, 13, dataset[:][0], dataset[:][1], 30)

LOSS = []
def train_model(iters):
    for epoch in range(iters):
        for x, y in dataloader:
            yhat = model(x)
            loss = criterion(yhat, y)
            LOSS.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            get_surface.set_para_loss(model, loss.tolist())
        if epoch % 20 == 0:
            get_surface.plot_ps()

train_model(100)

# plt.plot(LOSS)
# plt.xlabel("Iterations")
# plt.ylabel("Loss")
# plt.show()

# Make the Prediction

yhat = model(dataset.x)
label = yhat > 0.5
print("The accuracy: ", torch.mean((
    label == dataset.y.type(torch.ByteTensor))
    .type(torch.float)))