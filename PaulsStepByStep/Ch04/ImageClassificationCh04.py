import random
import numpy as np
from PIL import Image

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler, SubsetRandomSampler
from torchvision.transforms.v2 import Compose, ToTensor, Normalize, ToPILImage, RandomHorizontalFlip, Resize, ToImage, ToDtype

import matplotlib.pyplot as plt

from DataGenerationHelpersCh04 import generate_dataset
from StepByStepV0 import StepByStep
from PlotHelpersCh04 import *

import DataHelpersCh04 as dh

#########################
# Data Generation

def GeneratData00(img_size= 5, n_images=300, seed=13, showDataset=False):
    
    images, labels = generate_dataset(img_size=5, n_images=300, binary=True, seed=13)

    if showDataset:
        fig = plot_images(images, labels,  n_plot=30)
        plt.show()

    return images, labels

#########################
# Data Preparation

class TransformedTensorDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform
        
    def __getitem__(self, index):
        x = self.x[index]
        
        if self.transform:
            x = self.transform(x)
        
        return x, self.y[index]
        
    def __len__(self):
        return len(self.x)

def PrepareData00(images, labels):

    # Builds tensors from numpy arrays BEFORE split
    # Modifies the scale of pixel values from [0, 255] to [0, 1]
    x_tensor = torch.as_tensor(images / 255).float()
    y_tensor = torch.as_tensor(labels.reshape(-1, 1)).float()

    # Uses index_splitter to generate indices for training and
    # validation sets
    train_idx, val_idx = dh.index_splitter(len(x_tensor), [80, 20])

    # Uses indices to perform the split
    x_train_tensor = x_tensor[train_idx]
    y_train_tensor = y_tensor[train_idx]
    x_val_tensor = x_tensor[val_idx]
    y_val_tensor = y_tensor[val_idx]

    # Builds different composers because of data augmentation on training set
    train_composer = Compose([RandomHorizontalFlip(p=.5), Normalize(mean=(.5,), std=(.5,))])
    val_composer = Compose([Normalize(mean=(.5,), std=(.5,))])

    # Uses custom dataset to apply composed transforms to each set
    train_dataset = TransformedTensorDataset(x_train_tensor, y_train_tensor, transform=train_composer)
    val_dataset = TransformedTensorDataset(x_val_tensor, y_val_tensor, transform=val_composer)

    # Builds a weighted random sampler to handle imbalanced classes
    sampler = dh.make_balanced_sampler(y_train_tensor)

    # Uses sampler in the training set to get a balanced data loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, sampler=sampler)
    val_loader = DataLoader(dataset=val_dataset, batch_size=16)

    return train_loader, val_loader

def ConfigureModel00(use_bias=False):

    # Sets learning rate - this is "eta" ~ the "n" like Greek letter
    lr = 0.1

    torch.manual_seed(17)
    
    # Now we can create a model
    model_logistic = nn.Sequential()
    model_logistic.add_module('flatten', nn.Flatten())
    model_logistic.add_module('output', nn.Linear(25, 1, bias=use_bias))
    model_logistic.add_module('sigmoid', nn.Sigmoid())

    # Defines a SGD optimizer to update the parameters 
    optimizer_logistic = optim.SGD(model_logistic.parameters(), lr=lr)

    # Defines a binary cross entropy loss function
    binary_loss_fn = nn.BCELoss()

    return model_logistic, binary_loss_fn, optimizer_logistic

def ConfigureModel01(use_bias=False):

    # Sets learning rate - this is "eta" ~ the "n" like Greek letter
    lr = 0.1

    torch.manual_seed(17)
    # Now we can create a model
    model = nn.Sequential()
    model.add_module('flatten', nn.Flatten())
    model.add_module('hidden0', nn.Linear(25, 5, bias=False))
    model.add_module('hidden1', nn.Linear(5, 3, bias=False))
    model.add_module('output', nn.Linear(3, 1, bias=False))
    model.add_module('sigmoid', nn.Sigmoid())

    # Defines a SGD optimizer to update the parameters 
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Defines a binary cross entropy loss function
    loss_fn = nn.BCELoss()

    return model, loss_fn, optimizer

def ConfigureModel02(use_bias=False):

    # Sets learning rate - this is "eta" ~ the "n" like Greek letter
    lr = 0.1

    torch.manual_seed(17)
    # Now we can create a model
    model_relu = nn.Sequential()
    model_relu.add_module('flatten', nn.Flatten())
    model_relu.add_module('hidden0', nn.Linear(25, 5, bias=False))
    model_relu.add_module('activation0', nn.ReLU())
    model_relu.add_module('hidden1', nn.Linear(5, 3, bias=False))
    model_relu.add_module('activation1', nn.ReLU())
    model_relu.add_module('output', nn.Linear(3, 1, bias=False))
    model_relu.add_module('sigmoid', nn.Sigmoid())

    # Defines a SGD optimizer to update the parameters
    optimizer_relu = optim.SGD(model_relu.parameters(), lr=lr)

    # Defines a binary cross entropy loss function
    binary_loss_fn = nn.BCELoss()

    return model_relu, binary_loss_fn, optimizer_relu


def TrainModel00(model, loss_fn, optimizer, train_loader, val_loader, n_epochs = 100):

    sbs = StepByStep(model, loss_fn, optimizer)
    sbs.set_loaders(train_loader, val_loader)
    sbs.train(n_epochs)

    return sbs

def RunAll00(n_epochs = 100, use_bias=False, showDataset=True, showLosses=True):

    images, labels = GeneratData00(img_size=5, n_images=300, seed=13, showDataset=showDataset)

    train_loader, val_loader = PrepareData00(images, labels)

    model, loss_fn, optimizer = ConfigureModel00(use_bias=use_bias)

    sbs = TrainModel00(model, loss_fn, optimizer, train_loader, val_loader, n_epochs=n_epochs)

    if showLosses:
        sbs.plot_losses()
        plt.show()
    
def RunAll01(n_epochs = 100, use_bias=False, showDataset=True, showLosses=True):

    images, labels = GeneratData00(img_size=5, n_images=300, seed=13, showDataset=showDataset)

    train_loader, val_loader = PrepareData00(images, labels)

    model, loss_fn, optimizer = ConfigureModel01(use_bias=use_bias)

    sbs = TrainModel00(model, loss_fn, optimizer, train_loader, val_loader, n_epochs=n_epochs)

    if showLosses:
        sbs.plot_losses()
        plt.show()

def RunAll02(n_epochs = 100, use_bias=False, showDataset=True, showLosses=True):

    images, labels = GeneratData00(img_size=5, n_images=300, seed=13, showDataset=showDataset)

    train_loader, val_loader = PrepareData00(images, labels)

    model, loss_fn, optimizer = ConfigureModel02(use_bias=use_bias)

    sbs = TrainModel00(model, loss_fn, optimizer, train_loader, val_loader, n_epochs=n_epochs)

    if showLosses:
        sbs.plot_losses()
        plt.show()
    
    