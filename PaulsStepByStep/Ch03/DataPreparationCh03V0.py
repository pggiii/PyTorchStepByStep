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

def RunDataPreparation(X_train, y_train, X_val, y_val):
    torch.manual_seed(13)

    # Builds tensors from numpy arrays
    x_train_tensor = torch.as_tensor(X_train).float()
    y_train_tensor = torch.as_tensor(y_train.reshape(-1, 1)).float()

    x_val_tensor = torch.as_tensor(X_val).float()
    y_val_tensor = torch.as_tensor(y_val.reshape(-1, 1)).float()

    # Builds dataset containing ALL data points
    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(x_val_tensor, y_val_tensor)

    # Builds a loader of each set
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=16)

    return train_loader, val_loader