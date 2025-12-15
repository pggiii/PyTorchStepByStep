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

import ImageClassificationCh05 as ic

plt.style.use('fivethirtyeight')

sbs = ic.RunAll00(n_images=1000, n_epochs = 20, n_channels=1, use_bias=True, showDataset=False, showLosses=True)

layers_to_hook = ['conv1', 'relu1', 'maxp1', 'flatten', 'fc1', 'relu2', 'fc2']
sbs.attach_hooks(layers_to_hook)
images_batch, labels_batch = next(iter(sbs.val_loader))
logits = sbs.predict(images_batch)
sbs.remove_hooks()
predicted = np.argmax(logits, 1)
print(labels_batch)
print(predicted)

featurizer_layers = ['conv1', 'relu1', 'maxp1', 'flatten']
fig = sbs.visualize_outputs(featurizer_layers)

classifier_layers = ['fc1', 'relu2', 'fc2']
fig = sbs.visualize_outputs(classifier_layers, y=labels_batch, yhat=predicted)

print(sbs.correct(images_batch, labels_batch))
print(StepByStep.loader_apply(sbs.val_loader,sbs.correct))

i = 0