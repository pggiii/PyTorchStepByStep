import numpy as np
from sklearn.linear_model import LinearRegression
import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def RunModelTraining(x_train_tensor, y_train_tensor, model, optimizer, loss_fn):
    """Do training using the data, model, optimizer, and loss_fn passed in."""

    # Defines number of epochs
    n_epochs = 1000

    for epoch in range(n_epochs):
        # Sets model to TRAIN mode
        model.train()

        # Step 1 - Computes model's predicted output - forward pass
        yhat = model(x_train_tensor)
        
        # Step 2 - Computes the loss
        loss = loss_fn(yhat, y_train_tensor)

        # Step 3 - Computes gradients for both "b" and "w" parameters
        loss.backward()
        
        # Step 4 - Updates parameters using gradients and 
        # the learning rate
        optimizer.step()
        optimizer.zero_grad()

        print(model.state_dict())