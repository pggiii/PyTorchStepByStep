import numpy as np
from sklearn.linear_model import LinearRegression
import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def RunModelTraining(device, train_loader, train_step_fn, get_model_fn):
    """Do training using the data, model, optimizer, and loss_fn passed in."""

   # Defines number of epochs
    n_epochs = 1000

    losses = []

    # For each epoch...
    for epoch in range(n_epochs):
        # inner loop
        mini_batch_losses = []
        for x_batch, y_batch in train_loader:
            # the dataset "lives" in the CPU, so do our mini-batches
            # therefore, we need to send those mini-batches to the
            # device where the model "lives"
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # Performs one train step and returns the corresponding loss 
            # for this mini-batch
            mini_batch_loss = train_step_fn(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)

        # Computes average loss over all mini-batches - that's the epoch loss
        loss = np.mean(mini_batch_losses)
        losses.append(loss)

    model = get_model_fn()
    print(model.state_dict())