import numpy as np
import scipy.ndimage
import edt

def label_dt(label, anisotropy=None, normalize=None):
    ndim = len(label.shape)
    # default anisotropy: no anisotropy
    if anisotropy == None:
        anisotropy = (1.0,) * ndim
    # check anisotropy shape
    if not len(anisotropy) == ndim:
        print("ERROR: label_dt invalid anisotropy tuple length")

    # compute label border
    border = 1.0 - (scipy.ndimage.morphology.binary_dilation(label, structure=scipy.ndimage.generate_binary_structure(ndim, 1)) - label)
    # compute distance to label border
    distance = edt.edt(border)
    # invert distance for the outside
    distance[label == 0] = - distance[label == 0]
    # normalize function
    if normalize is not None:
        distance = normalize(distance)
        
    return distance.astype(np.float32)


def normalize_tanh(distance, scale=4.0):
    return np.tanh(distance / scale)


def normalize_sign(distance, scale=2.0):
    return 1/(1 + np.exp(-(distance/scale)))


def normalize_linear_clip_0_1(distance, scale=8.0):
    return np.clip((distance/scale), 0, 1)
