import random
import numpy as np
from PIL import Image

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler, SubsetRandomSampler
from torchvision.transforms.v2 import Compose, ToTensor, Normalize, ToPILImage, RandomHorizontalFlip, Resize, ToImage, ToDtype

import matplotlib.pyplot as plt

from DataGenerationHelpersCh05 import generate_dataset
from StepByStepV3 import StepByStep
from PlotHelpersCh05 import *

import ImageClassificationCh06 as ic

plt.style.use('fivethirtyeight')

sbs = ic.RunAll(n_epochs = 10, n_features=5, dropProb=.5, showFig2=False, showLosses=True, rpsTrainFolder=r"C:\Users\pgott\Desktop\Rock-Paper-Scissors\train", rpsValFolder=r"C:\Users\pgott\Desktop\Rock-Paper-Scissors\test")

print(sbs.model.conv1.weight.shape)
sbs.visualize_filters('conv1')
sbs.visualize_filters('conv2')
 
i = 0