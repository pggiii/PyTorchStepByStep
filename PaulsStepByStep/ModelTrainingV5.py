import numpy as np
from sklearn.linear_model import LinearRegression
import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import HelpersV0 as helpers
import StepByStep

def RunModelTraining(model, loss_fn, optimizer, train_loader, val_loader, plot_losses = False):
    """Do training using the data, model, optimizer, and loss_fn passed in."""

    n_epochs = 200

    sbs = StepByStep.StepByStep(model, loss_fn, optimizer)
    sbs.set_loaders(train_loader, val_loader)
    sbs.set_tensorboard('classy')
    sbs.train(n_epochs=n_epochs)

    print(model.state_dict())
    print(sbs.total_epochs)

    if plot_losses:
        fig = sbs.plot_losses()

    return sbs
