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

from DataGenerationHelpersCh04 import generate_dataset
from StepByStepV0 import StepByStep
from PlotHelpersCh04 import *

import ImageClassificationCh04 as ic

plt.style.use('fivethirtyeight')

#ic.RunAll00(n_epochs=100, use_bias=False, showDataset=True, showLosses=True)

#ic.RunAll01(n_epochs=100, use_bias=False, showDataset=True, showLosses=True)

ic.RunAll02(n_epochs=50, use_bias=False, showDataset=True, showLosses=True)

i=0
