import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch import nn, optim

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