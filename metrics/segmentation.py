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


def Phi(X, Y):
    X = X.astype(np.bool)
    Y = Y.astype(np.bool)
    TP = np.sum(X * Y)
    FP = np.sum(X) - TP
    FN = np.sum(Y) - TP
    TN = np.sum((1-X) * (1-Y))

    return (TP * TN + FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))


iou = IoU
f1 = F1
jaccard = IoU
dice = F1
MCC = Phi
