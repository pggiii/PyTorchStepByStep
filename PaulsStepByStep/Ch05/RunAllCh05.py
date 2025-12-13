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
from StepByStepV0 import StepByStep
from PlotHelpersCh05 import *

import ImageClassificationCh05 as ic

plt.style.use('fivethirtyeight')

ic.RunAll00(n_epochs = 20, use_bias=True, showDataset=True, showLosses=True)



i = 0