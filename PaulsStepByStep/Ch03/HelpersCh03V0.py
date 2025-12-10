import matplotlib.pyplot as plt

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


def PlotData(X_train, y_train, X_val, y_val, show=True):
    """Produces scatterplots of training and validation data for Chapter 3."""

    if show:

        # Training data
        X_train_blue = X_train[y_train == 1]
        X_train_red = X_train[y_train == 0]
        
        x1_train_blue = X_train_blue[:,0]
        x2_train_blue = X_train_blue[:,1]

        x1_train_red = X_train_red[:,0]
        x2_train_red = X_train_red[:,1]

        # Validation data
        X_val_blue = X_val[y_val == 1]
        X_val_red = X_val[y_val == 0] 

        x1_val_blue = X_val_blue[:,0]
        x2_val_blue = X_val_blue[:,1]

        x1_val_red = X_val_red[:,0]
        x2_val_red = X_val_red[:,1]

        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,6))

        ax1.set_xlabel('X1')
        ax1.set_ylabel('X2')
        ax1.scatter(x1_train_blue, x2_train_blue, color='blue')
        ax1.scatter(x1_train_red, x2_train_red, color='red')
        ax1.set_title('Generated Data - Train')

        ax2.set_xlabel('X1')
        ax2.set_ylabel('X2')
        ax2.scatter(x1_val_blue,x2_val_blue,color='blue')
        ax2.scatter(x1_val_red,x2_val_red,color='red')
        ax2.set_title('Generated Data = Validation')

        plt.show()

def make_cm(labels, label_probabilites_from_model, threshold):
    """
    Docstring for make_cm
    
    :param y_val: The labels, 0 or 1
    :param predictions: The predications, 0 or 1.
    """

    return confusion_matrix(labels, label_probabilites_from_model >= threshold)

def split_cm(cm):
    # Actual negatives go in the top row, 
    # above the probability line
    actual_negative = cm[0]
    # Predicted negatives go in the first column
    tn = actual_negative[0]
    # Predicted positives go in the second column
    fp = actual_negative[1]

    # Actual positives go in the bottow row, 
    # below the probability line
    actual_positive = cm[1]
    # Predicted negatives go in the first column
    fn = actual_positive[0]
    # Predicted positives go in the second column
    tp = actual_positive[1]
    
    return tn, fp, fn, tp

def tpr_fpr(cm):
    tn, fp, fn, tp = split_cm(cm)
    
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    
    return tpr, fpr

def precision_recall(cm):
    tn, fp, fn, tp = split_cm(cm)
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    return precision, recall

def accuracy(cm):
    tn, fp, fn, tp = split_cm(cm)

    acc = (tp + tn)/(tp+tn+fp+fn)

    return acc

def make_roc_data(labels, predicted_label_probabilities):

    fpr, tpr, thresholds_roc = roc_curve(labels, predicted_label_probabilities)
    
    return fpr, tpr, thresholds_roc

def make_prc_data(labels, predicted_label_probabilities):

    precision, recall, thesholds_prec_recall = precision_recall_curve(labels, predicted_label_probabilities)

    return  precision, recall, thesholds_prec_recall

def plot_roc_and_prc(labels, predicted_label_probabilities, show=True):

    fpr, tpr, thresholds_roc = make_roc_data(labels, predicted_label_probabilities)
    precision, recall, thesholds_prec_recall = make_prc_data(labels, predicted_label_probabilities)

    if show:
            fig, (ax1, ax2) = plt.subplots(1,2,figsize=(12,6))

            ax1.set_xlabel('False Positive Rate')
            ax1.set_ylabel('True Positive Rate')
            ax1.plot(fpr, tpr, color='red', marker='o', linestyle='-')
            ax1.set_title('ROC Curve')

            ax2.set_xlabel('Precision')
            ax2.set_ylabel('Recall')
            ax2.plot(precision,recall, color='red', marker='o', linestyle='-')
            ax2.set_title('Precision-Recall Curve')

            plt.show()

def area_under_roc_and_prc(labels, predicted_label_probabilities):

    fpr, tpr, thresholds_roc = make_roc_data(labels, predicted_label_probabilities)
    precision, recall, thesholds_prec_recall = make_prc_data(labels, predicted_label_probabilities)
    auroc = auc(fpr, tpr)
    auprc = auc(recall,precision)

    return auroc, auprc