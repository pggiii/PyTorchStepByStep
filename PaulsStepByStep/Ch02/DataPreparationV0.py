import numpy as np
from sklearn.linear_model import LinearRegression
import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def RunDataPreparation(x_train, y_train):
    """Does the data prep"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Our data was in Numpy arrays, but we need to transform them
    # into PyTorch's Tensors and then we send them to the 
    # chosen device
    x_train_tensor = torch.as_tensor(x_train).float().to(device)
    y_train_tensor = torch.as_tensor(y_train).float().to(device)

    return device, x_train_tensor, y_train_tensor