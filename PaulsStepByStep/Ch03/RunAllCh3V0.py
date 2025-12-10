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

import DataGenerationCh03V0 as dg
import DataPreparationCh03V0 as dp
import ModelConfigurationCh03V0 as mc
import ModelTrainingCh03V0 as mt

import HelpersCh03V0 as h

n_data_points = 100

X_train, y_train, X_val, y_val = dg.RunDataGeneration(n_data_points, showPlot=True, noise=0.3)

train_loader, val_loader = dp.RunDataPreparation(X_train, y_train, X_val, y_val)

model, loss_fn, optimizer = mc.RunModelConfiguration()

sbs = mt.RunModelTraining(model, loss_fn, optimizer, train_loader, val_loader, showLossPlot=True)

# Make predictions

logit_predictions = sbs.predict(X_train[:4])
print(logit_predictions)

# Go from logit predictions to probabilities
probabilities = torch.sigmoid(torch.as_tensor(logit_predictions).float())
print(probabilities)

# Now to classes by thresholding the probabilities, or equivalently, the threshold mapped into logit space on the logit predications.
# for pthresh=.5, this is logitthresh = 0.
classes = (logit_predictions >=0).astype(int)
print(classes)

# Now lets do confusion matrix
#
logit_val_predictions = sbs.predict(X_val)
probabilities_val = torch.sigmoid(torch.as_tensor(logit_val_predictions).float())

cm_thresh50 = confusion_matrix(y_val, (probabilities_val >= .5))
print(cm_thresh50)

# And the ROC and Precision-Recall curves
#
all_logit_predictions_val = sbs.predict(X_val)
all_probabilities_val = torch.sigmoid(torch.as_tensor(all_logit_predictions_val).float())
h.plot_roc_and_prc(y_val, all_probabilities_val)

auroc, auprc = h.area_under_roc_and_prc(y_val, all_probabilities_val)
print(auroc, auprc)

i = 0