import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv3D, Cropping2D, Conv3DTranspose, concatenate
from tensorflow.keras.layers import Dropout, MaxPool3D, UpSampling3D


def conv_block(input, filters, kernel_size=3, conv_per_block=2, dropout=0, batch_normalization=False):
    # X represent the output of the previous layer
    X = input

    for i in range(conv_per_block):
        X = Conv3D(filters=filters, kernel_size=kernel_size,
                   activation='relu', kernel_initializer="he_normal", padding='same')(X)
        if batch_normalization:
            X = BatchNormalization()(X)

    if dropout > 0:
        X = Dropout(dropout)(X)

    return X


def UNet(input_shape=(None, None, None, 1), output_classes=2, output_activation='default',
         filters=64, depth=5, pool_size=(2, 2, 2), conv_per_block=2,
         dropouts=0.50, batch_normalization=True):
    """
    :param input_shape: input shape tuple
    :param output_classes: number of output classes (single output for output_classes <= 2)
    :param output_activation: the activation function of the last layer
    :param filters: number of filters per conv, integer with initial value or array of size depth * 2 - 1
    :param depth: number of conv block in contracting path and expansive path
    :param pool_size: tuple with pool_size for each contracting block, or array of size depth * 2 - 1
    :param conv_per_block: number of convolution per level
    :param dropouts: dropout per conv, integer with middle value or array of size depth * 2 - 1
    :param batch_normalization: add batch normalization after convolution or not
    :return: a unet-like keras model
    """

    if not type(input_shape) is tuple:
        print("WARNING: input_shape parameters invalid, set as default")
        input_shape = (None, None, None)

    if output_classes < 2:
        print("WARNING: output_classes parameters invalid, set as default")
        output_classes = 2

    if type(filters) is int:
        filters = [filters * 2**i for i in range(depth-1)] + [filters * 2**i for i in range(depth-1, -1, -1)]

    if (not type(filters) is list) or (len(filters) != 2 * depth - 1):
        print("WARNING: filters parameters invalid, set as default")
        filters = [64 * 2**i for i in range(depth-1)] + [64 * 2**i for i in range(depth-1, 0, -1)]
        
    
    if type(pool_size) is tuple:
        pool_size = [(pool_size) for i in range(depth-1)] + [(pool_size) for i in range(depth-1, -1, -1)]

    if (not type(pool_size) is list) or (len(pool_size) != 2 * depth - 1):
        print("WARNING: filters parameters invalid, set as default")
        pool_size = [(2, 2, 2) for i in range(depth-1)] + [(2, 2, 2) for i in range(depth-1, -1, -1)]

    if dropouts is None:
        dropouts = np.zeros((2 * depth - 1), dtype=np.float32)

    if type(dropouts) is float:
        d = dropouts
        dropouts = np.zeros((2 * depth - 1), dtype=np.float32)
        dropouts[depth-1] = d

    if (not type(dropouts) is list) and (not type(dropouts) is np.ndarray) or (len(dropouts) != 2 * depth - 1):
        print("WARNING: dropouts parameters invalid, set as default")
        dropouts = np.zeros((2 * depth - 1), dtype=np.float32)
        dropouts[depth-1] = 0.50
    
    inputs = tf.keras.Input(input_shape, name="input")

    # output of each block
    block_out = []
    # X represent the previous output
    X = inputs

    # Contracting path
    for i in range(depth-1):
        X = conv_block(X, filters[i], 3, conv_per_block, dropouts[i], batch_normalization)
        
        block_out.append(X)
        
        X = MaxPool3D(pool_size=pool_size[i])(X)

    X = conv_block(X, filters[depth-1], 3, conv_per_block, dropouts[depth-1], batch_normalization)

    # Expansive path
    crop = 2
    for i in range(depth-1):
        X = UpSampling3D(pool_size[depth+i])(X)

        X = concatenate([block_out[depth - i - 2], X], axis=4)

        X = conv_block(X, filters[depth+i], 3, conv_per_block, dropouts[depth+i], batch_normalization)

    if output_activation == 'default':
        if output_classes > 2:
            tmp = Conv3D(output_classes, 1, activation='softmax', name="output")(X)
        else:
            tmp = Conv3D(1, 1, activation='sigmoid', name="output")(X)
    else:
        tmp = Conv3D(output_classes, 1, activation=output_activation, name="output")(X)

    model = tf.keras.Model(inputs=inputs, outputs=tmp)

    return model
