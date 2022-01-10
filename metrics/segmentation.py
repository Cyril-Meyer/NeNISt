import numpy as np


def IoU(X, Y):
    X = X.astype(np.bool)
    Y = Y.astype(np.bool)
    I = np.sum(X * Y)
    U = np.sum(X + Y)
    return I/U


def F1(X, Y):
    X = X.astype(np.bool)
    Y = Y.astype(np.bool)
    I = np.sum(X * Y)
    U = np.sum(X + Y)
    return (2*I)/(U+I)

iou = IoU
f1 = F1
jaccard = IoU
dice = F1