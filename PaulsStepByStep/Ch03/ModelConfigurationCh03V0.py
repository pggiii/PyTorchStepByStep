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

def RunModelConfiguration():

    # Sets learning rate - this is "eta" ~ the "n" like Greek letter
    lr = 0.1

    torch.manual_seed(42)
    model = nn.Sequential()
    model.add_module('linear', nn.Linear(2, 1))

    # Defines a SGD optimizer to update the parameters
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Defines a BCE loss function
    loss_fn = nn.BCEWithLogitsLoss()

    return model, loss_fn, optimizer