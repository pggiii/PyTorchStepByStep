import numpy as np
from PIL import Image
from copy import deepcopy

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.transforms.v2 import Compose, ToImage, Normalize, \
ToPILImage, Resize, ToDtype
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, \
MultiStepLR, CyclicLR, LambdaLR

from StepByStepV3 import StepByStep
import matplotlib.pyplot as plt

import chapter6 as ch6figs

#import DataHelpersCh05 as dh

#########################
# Data Preparation

def PreparData(rpsTrainFolder, rpsValFolder, showFig2=False):

    # Calculate statistics of dataset.
    temp_transform = Compose([Resize(28), ToImage(), ToDtype(torch.float32, scale=True)])
    temp_dataset = ImageFolder(root=rpsTrainFolder, transform=temp_transform)
    print(temp_dataset[0][0].shape, temp_dataset[0][1])

    temp_loader = DataLoader(temp_dataset, batch_size=16)

    first_images, first_labels = next(iter(temp_loader))
    print(StepByStep.statistics_per_channel(first_images, first_labels))
    
    results = StepByStep.loader_apply(temp_loader, StepByStep.statistics_per_channel)
    print(results)

    normalizer = StepByStep.make_normalizer(temp_loader)
    print(normalizer)

    # Make the actual training set using the previously computed statistics to statardize.
    composed_transform = Compose([Resize(28),
                        ToImage(),
                        ToDtype(torch.float32, scale=True),
                        normalizer])
    
    composed_target_transform = Compose([ToDtype(torch.float32, scale=True)])
    
    train_data = ImageFolder(root=rpsTrainFolder, transform=composed_transform)
    val_data = ImageFolder(root=rpsValFolder, transform=composed_transform)

    # Builds a loader of each set
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16)

    if showFig2:
        torch.manual_seed(88)
        first_images, first_labels = next(iter(train_loader))
        fig = ch6figs.figure2(first_images, first_labels)

    return train_loader, val_loader

#########################
# Fancier Model CNN2

class CNN2(nn.Module):
    def __init__(self, n_feature, p=0.0):
        super(CNN2, self).__init__()
        self.n_feature = n_feature
        self.p = p
        # Creates the convolution layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=n_feature, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=n_feature, out_channels=n_feature, kernel_size=3)
        # Creates the linear layers
        # Where do this 5 * 5 come from?! Check it below
        self.fc1 = nn.Linear(n_feature * 5 * 5, 50) 
        self.fc2 = nn.Linear(50, 3)
        # Creates dropout layers
        self.drop = nn.Dropout(self.p)
        
    def featurizer(self, x):
        # Featurizer
        # First convolutional block
        # 3@28x28 -> n_feature@26x26 -> n_feature@13x13
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        # Second convolutional block
        # n_feature * @13x13 -> n_feature@11x11 -> n_feature@5x5
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        # Input dimension (n_feature@5x5)
        # Output dimension (n_feature * 5 * 5)
        x = nn.Flatten()(x)
        return x
    
    def classifier(self, x):
        # Classifier
        # Hidden Layer
        # Input dimension (n_feature * 5 * 5)
        # Output dimension (50)
        if self.p > 0:
            x = self.drop(x)
        x = self.fc1(x)
        x = F.relu(x)
        # Output Layer
        # Input dimension (50)
        # Output dimension (3)
        if self.p > 0:
            x = self.drop(x)
        x = self.fc2(x)
        return x
                
    def forward(self, x):
        x = self.featurizer(x)
        x = self.classifier(x)
        return x

#########################
# Model Configuration

def ConfigureModel(n_features=5, dropProb=.3, seed=13):
    torch.manual_seed(seed)
    model_cnn2 = CNN2(n_feature=n_features, p=dropProb)
    multi_loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optimizer_cnn2 = optim.Adam(model_cnn2.parameters(), lr=.001, betas=(0.9,0.999), eps=1e-8, weight_decay=0)
    return model_cnn2, multi_loss_fn, optimizer_cnn2

def TrainModel(train_loader, val_loader, model, loss_fn, optimizer, n_epochs=10, seed=42):
    sbs = StepByStep(model, loss_fn, optimizer)
    sbs.set_loaders(train_loader, val_loader)
    sbs.train(n_epochs=10,seed=seed)
    return sbs

#########################
# Run All

def RunAll(n_epochs = 10, n_features = 5, dropProb=.5, rpsTrainFolder=r"C:\Users\pgott\Desktop\Rock-Paper-Scissors\train", rpsValFolder=r"C:\Users\pgott\Desktop\Rock-Paper-Scissors\test", showFig2=False, showLosses=False):

    train_loader, val_loader = PreparData(rpsTrainFolder, rpsValFolder, showFig2=showFig2)

    model, loss_fn, optimizer = ConfigureModel(n_features=n_features,dropProb=dropProb)

    sbs = TrainModel(train_loader, val_loader, model, loss_fn, optimizer, n_epochs=n_epochs, seed=88)

    if showLosses:
        fig = sbs.plot_losses()
        plt.show()

    return sbs