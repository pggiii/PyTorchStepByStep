import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.v2 import Compose, Normalize

from DataGenerationHelpersCh05 import generate_dataset
from StepByStepV0 import StepByStep
from PlotHelpersCh05 import *

import random
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

import DataHelpersCh05 as dh

#########################
# Data Generation

def GeneratData00(img_size=5, n_images=300, seed=13, binary=True, showDataset=False):
    
    images, labels = generate_dataset(img_size=img_size, n_images=n_images, binary=binary, seed=seed)

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
    y_tensor = torch.as_tensor(labels).long()

    # Uses index_splitter to generate indices for training and
    # validation sets
    train_idx, val_idx = dh.index_splitter(len(x_tensor), [80, 20])
    # Uses indices to perform the split
    x_train_tensor = x_tensor[train_idx]
    y_train_tensor = y_tensor[train_idx]
    x_val_tensor = x_tensor[val_idx]
    y_val_tensor = y_tensor[val_idx]

    # We're not doing any data augmentation now
    train_composer = Compose([Normalize(mean=(.5,), std=(.5,))])
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

#############
# Configure Model

def ConfigureModel00(use_bias=False):

    torch.manual_seed(13)
    model_cnn1 = nn.Sequential()

    # Featurizer
    # Block 1: 1@10x10 -> n_channels@8x8 -> n_channels@4x4
    n_channels = 1
    model_cnn1.add_module('conv1', nn.Conv2d(in_channels=1, out_channels=n_channels, kernel_size=3,bias=use_bias))
    model_cnn1.add_module('relu1', nn.ReLU())
    model_cnn1.add_module('maxp1', nn.MaxPool2d(kernel_size=2))
    # Flattening: n_channels * 4 * 4
    model_cnn1.add_module('flatten', nn.Flatten())
    # Classification
    # Hidden Layer
    model_cnn1.add_module('fc1', nn.Linear(in_features=n_channels*4*4, out_features=10,bias=use_bias))
    model_cnn1.add_module('relu2', nn.ReLU())
    # Output Layer
    model_cnn1.add_module('fc2', nn.Linear(in_features=10, out_features=3, bias=use_bias))

    lr = 0.1
    multi_loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optimizer_cnn1 = optim.SGD(model_cnn1.parameters(), lr=lr) 

    return model_cnn1, multi_loss_fn, optimizer_cnn1

######################
# Train Model

def TrainModel00(model, loss_fn, optimizer, train_loader, val_loader, n_epochs = 100):

    sbs = StepByStep(model, loss_fn, optimizer)
    sbs.set_loaders(train_loader, val_loader)
    sbs.train(n_epochs)

    return sbs

#########################
# Run All

def RunAll00(n_epochs = 100, use_bias=False, showDataset=True, showLosses=True):

    images, labels = GeneratData00(img_size=10, n_images=1000, binary=False, seed=17, showDataset=showDataset)

    train_loader, val_loader = PrepareData00(images, labels)

    model, loss_fn, optimizer = ConfigureModel00(use_bias=use_bias)

    sbs = TrainModel00(model, loss_fn, optimizer, train_loader, val_loader, n_epochs=n_epochs)

    if showLosses:
        sbs.plot_losses()
        plt.show()