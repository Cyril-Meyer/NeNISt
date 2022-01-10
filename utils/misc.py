import numpy as np

# source : https://github.com/scaelles/DEXTR-PyTorch
def make_gaussian_2d(size, sigma=10, center=None, d_type=np.float64):
    """ Make a square gaussian kernel.
    size: is the dimensions of the output gaussian
    sigma: is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size[1], 1, float)
    y = np.arange(0, size[0], 1, float)
    y = y[:, np.newaxis]

    if center is None:
        x0 = y0 = size[0] // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2).astype(d_type)


def make_gaussian_3d(size, sigma=10, center=None, d_type=np.float64):
    x = np.arange(0, size[2], 1, float)
    y = np.arange(0, size[1], 1, float)
    z = np.arange(0, size[0], 1, float)
    
    y = y[:, np.newaxis]
    z = z[:, np.newaxis, np.newaxis]

    if center is None:
        raise NotImplementedError
    else:
        z0 = center[0]
        y0 = center[1]
        x0 = center[2]
        
    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) / sigma ** 2).astype(d_type)


def make_multiple_gaussian_3d(size, sigma=10, centers=None, d_type=np.float64):
    x = np.arange(0, size[2], 1, float)
    y = np.arange(0, size[1], 1, float)
    z = np.arange(0, size[0], 1, float)
    
    y = y[:, np.newaxis]
    z = z[:, np.newaxis, np.newaxis]
    
    g = np.zeros(size, dtype=d_type)

    for center in centers:
        z0 = center[0]
        y0 = center[1]
        x0 = center[2]
        
        g = np.maximum(g, np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2) / sigma ** 2).astype(d_type))
        
    return g
