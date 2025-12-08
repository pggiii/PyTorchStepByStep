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

import HelpersV0 as helpers

def RunModelConfiguration(train_loader):
    """Runs model configuration"""
    
    # This is redundant now, but it won't be when we introduce
    # Datasets...
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Sets learning rate - this is "eta" ~ the "n"-like Greek letter
    lr = 0.1

    torch.manual_seed(42)

    # Now we can create a model and send it at once to the device
    model = nn.Sequential(nn.Linear(1, 1)).to(device)

    # Defines a SGD optimizer to update the parameters 
    # (now retrieved directly from the model)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Defines a MSE loss function
    loss_fn = nn.MSELoss(reduction='mean')

    train_step_fn, get_model_fn = helpers.make_train_step_fn(model, loss_fn, optimizer)

    val_step_fn = helpers.make_val_step_fn(model, loss_fn)

    # Creates a Summary Writer to interface with TensorBoard
    #
    writer = SummaryWriter('PyTorchStepByStep/PaulsStepByStep/runs/simple_linear_regression')

    # Fetches a single mini-batch so we can use add_graph
    x_sample, y_sample = next(iter(train_loader))
    writer.add_graph(model, x_sample.to(device))
    
    return device, train_step_fn, val_step_fn, get_model_fn, writer