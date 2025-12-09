import numpy as np
from sklearn.linear_model import LinearRegression
import torch
import torch.optim as optim
import torch.nn as nn
from torchviz import make_dot
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def RunDataGeneration(showPlot = False):
    """Runs data generation code"""
    
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

    if showPlot:
        fig, ax = plt.subplots()
        ax.scatter(x, y, color='blue')
        #ax.scatter(x_train, yhat, color='red', label="Model")
        ax.set_title('Generated Data')
        ax.set_xlabel('Years of Experience')
        ax.set_ylabel('Salary')
        #ax.legend()
        plt.show()

    return x, y