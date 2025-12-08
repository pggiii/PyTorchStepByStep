import numpy as np
from sklearn.linear_model import LinearRegression
import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import HelpersV0 as helpers

def RunModelTraining(device, train_loader, val_loader, train_step_fn, val_step_fn, get_model_fn, writer):
    """Do training using the data, model, optimizer, and loss_fn passed in."""

   # Defines number of epochs
    n_epochs = 200

    losses = []
    val_losses = []

    for epoch in range(n_epochs):
        # inner loop
        loss = helpers.mini_batch(device, train_loader, train_step_fn)
        losses.append(loss)

        # VALIDATION
        # no gradients in validation!
        with torch.no_grad():
            val_loss = helpers.mini_batch(device, val_loader, val_step_fn)
            val_losses.append(val_loss)  

        # Records both losses for each epoch under the main tag "loss"
        writer.add_scalars(main_tag='loss', tag_scalar_dict={'training': loss, 'validation': val_loss}, global_step=epoch)

    # Closes the writer
    writer.close()

    # See the models state
    model = get_model_fn()
    print(model.state_dict())

    return losses, val_losses