import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

import StepByStep

def RunModelTraining(model, loss_fn, optimizer, train_loader, val_loader, showLossPlot=False):
    """Does the training."""

    n_epochs = 100

    sbs = StepByStep.StepByStep(model, loss_fn, optimizer)
    sbs.set_loaders(train_loader, val_loader)
    sbs.train(n_epochs)

    print(model.state_dict())

    if showLossPlot:
        fig = sbs.plot_losses()

    return sbs