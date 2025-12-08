import numpy as np
from sklearn.linear_model import LinearRegression
import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def RunDataGeneration():
    """Runs data generation code"""
    showfigs = True
    showfig1 = False

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

    return x, y