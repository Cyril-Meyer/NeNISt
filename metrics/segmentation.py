import numpy as np


def precision(TRUE, PRED):
    TRUE = TRUE.astype(np.bool)
    PRED = PRED.astype(np.bool)

    TP = np.sum(TRUE * PRED)
    # TN = np.sum((1-TRUE) * (1-PRED))
    FP = np.sum(PRED) - TP
    # FN = np.sum(TRUE) - TP

    return TP / (TP + FP)


def recall(TRUE, PRED):
    TRUE = TRUE.astype(np.bool)
    PRED = PRED.astype(np.bool)

    TP = np.sum(TRUE * PRED)
    # TN = np.sum((1-TRUE) * (1-PRED))
    # FP = np.sum(PRED) - TP
    FN = np.sum(TRUE) - TP

    return TP / (TP + FN)


def iou(TRUE, PRED):
    TRUE = TRUE.astype(np.bool)
    PRED = PRED.astype(np.bool)
    I = np.sum(TRUE * PRED)
    U = np.sum(TRUE + PRED)
    return I/U


def f1(TRUE, PRED):
    TRUE = TRUE.astype(np.bool)
    PRED = PRED.astype(np.bool)
    I = np.sum(TRUE * PRED)
    U = np.sum(TRUE + PRED)
    return (2*I)/(U+I)


def phi(TRUE, PRED):
    TRUE = TRUE.astype(np.bool)
    PRED = PRED.astype(np.bool)
    TP = np.sum(TRUE * PRED)
    TN = np.sum((1-TRUE) * (1-PRED))
    FP = np.sum(PRED) - TP
    FN = np.sum(TRUE) - TP

    return (TP * TN + FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))


IoU = iou
F1 = f1
jaccard = iou
dice = f1
MCC = phi
phi = phi
Phi = phi