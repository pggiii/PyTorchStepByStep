import numpy as np
from sklearn.linear_model import LinearRegression
import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def RunModelTraining(x_train_tensor, y_train_tensor, train_step_fn, get_model_fn):
    """Do training using the data, model, optimizer, and loss_fn passed in."""

    # Defines number of epochs
    n_epochs = 1000

    losses = []

    for epoch in range(n_epochs):
         # Performs one train step and returns the corresponding loss
        loss = train_step_fn(x_train_tensor, y_train_tensor)
        losses.append(loss)

    model = get_model_fn()
    print(model.state_dict())