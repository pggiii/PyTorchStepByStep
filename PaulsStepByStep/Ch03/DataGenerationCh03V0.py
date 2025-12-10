import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader, TensorDataset


from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

import matplotlib.pyplot as plt

from StepByStep import StepByStep

import HelpersCh03V0 as h

def RunDataGeneration(n_data_points, showPlot = False, noise=0.3):

    X, y = make_moons(n_samples=n_data_points, noise=noise, random_state=0)
    X_train_orig, X_val_orig, y_train, y_val = train_test_split(X, y, test_size=.2, random_state=13)

    sc = StandardScaler()
    sc.fit(X_train_orig)

    X_train = sc.transform(X_train_orig)
    X_val = sc.transform(X_val_orig)

    h.PlotData(X_train, y_train, X_val, y_val, show=showPlot)

    return X_train, y_train, X_val, y_val