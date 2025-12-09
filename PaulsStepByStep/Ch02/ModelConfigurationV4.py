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

def RunModelConfiguration():
    """Runs model configuration"""

    # Sets learning rate - this is "eta" ~ the "n" like Greek letter
    lr = 0.1

    torch.manual_seed(42)
    # Now we can create a model and send it at once to the device
    model = nn.Sequential(nn.Linear(1, 1))

    # Defines a SGD optimizer to update the parameters
    # (now retrieved directly from the model)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Defines a MSE loss function
    loss_fn = nn.MSELoss(reduction='mean')

    # Pring the model's state
    print(model.state_dict())

    return model, optimizer, loss_fn