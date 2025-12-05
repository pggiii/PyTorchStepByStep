import numpy as np
from sklearn.linear_model import LinearRegression
import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot
import matplotlib.pyplot as plt
import matplotlib.colors as colors

showfigs = True
showfig1 = True

# Synthetic data generation
#

true_b = 1
true_w = 2
N = 100

# Data Generation
np.random.seed(42)
x = np.random.rand(N, 1)
epsilon = (.1 * np.random.randn(N, 1))
y = true_b + true_w * x + epsilon

# Shuffles the indices
idx = np.arange(N)
np.random.shuffle(idx)

# Uses first 80 random indices for train
train_idx = idx[:int(N*.8)]
# Uses the remaining indices for validation
val_idx = idx[int(N*.8):]

# Generates train and validation sets
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

# Display the data.
if showfigs and showfig1:
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8,4))
    ax1.scatter(x_train,y_train, color='blue')
    ax1.set_title('Generated Data - Train')
    ax2.scatter(x_val,y_val,color='red')
    ax2.set_title('Generated Data = Validation')
    plt.show()

    i = 0