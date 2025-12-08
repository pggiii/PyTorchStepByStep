import numpy as np
from sklearn.linear_model import LinearRegression
import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import HelpersV0 as helpers

def RunModelTraining(device, train_loader, train_step_fn, get_model_fn):
    """Do training using the data, model, optimizer, and loss_fn passed in."""

   # Defines number of epochs
    n_epochs = 200

    losses = []

    for epoch in range(n_epochs):
        # inner loop
        loss = helpers.mini_batch(device, train_loader, train_step_fn)
        losses.append(loss)

    model = get_model_fn()
    print(model.state_dict())