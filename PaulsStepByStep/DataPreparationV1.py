import numpy as np
from sklearn.linear_model import LinearRegression
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
from tensorboardX import SummaryWriter
from torchviz import make_dot
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def RunDataPreparation(x_train, y_train):
    """Does the data prep"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Our data was in Numpy arrays, but we need to transform them into PyTorch's Tensors
    x_train_tensor = torch.from_numpy(x_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()

    # Builds Dataset
    train_data = TensorDataset(x_train_tensor, y_train_tensor)

    # Builds DataLoader
    train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True)

    return device, train_loader