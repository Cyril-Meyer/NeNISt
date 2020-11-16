import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv3D, Cropping2D, Conv3DTranspose, concatenate
from tensorflow.keras.layers import Dropout, MaxPooling3D


def conv_block(input, filters, kernel_size=3, conv_per_block=2, padding='same', dropout=0, batch_normalization=False, groups=1):
    # X represent the output of the previous layer
    X = input

    for i in range(conv_per_block):
        X = Conv3D(filters=filters, kernel_size=kernel_size,
                   activation='relu', kernel_initializer="he_normal", padding=padding, groups=groups)(X)
        if batch_normalization:
            X = BatchNormalization()(X)

    if dropout > 0:
        X = Dropout(dropout)(X)

    return X


def UNet(input_shape=(None, None, None, None), output_classes=2, filters=64, depth=5, conv_per_block=2,
         padding='same', dropouts=0.50, batch_normalization=False, groups=1):
    """
    :param input_shape: input shape tuple
    :param output_classes: number of output classes (single output for output_classes <= 2)
    :param filters: number of filters per conv, integer with initial value or array of size depth * 2 - 1
    :param depth: number of conv block in contracting path and expansive path
    :param conv_per_block: number of convolution per level
    :param padding: type of padding "same" or "valid". If valid, output is smaller than input
    :param dropouts: dropout per conv, integer with middle value or array of size depth * 2 - 1
    :param batch_normalization: add batch normalization after convolution or not
    :param groups: groups per conv, integer with first value or array of size depth * 2 - 1
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
    
    if groups is None:
        groups = np.ones((2 * depth - 1), dtype=np.uint8)

    if type(groups) is float or type(groups) is int:
        g = groups
        groups = np.ones((2 * depth - 1), dtype=np.uint8)
        groups[0] = g
    
    if (not type(groups) is list) and (not type(groups) is np.ndarray) or (len(groups) != 2 * depth - 1):
        print(type(groups))
        print("WARNING: groups parameters invalid, set as default")
        groups = np.ones((2 * depth - 1), dtype=np.uint8)

    inputs = tf.keras.Input(input_shape, name="input")

    # output of each block
    block_out = []
    # X represent the previous output
    X = inputs
    
    X = Conv2D(filters=3, kernel_size=filters[0], activation='relu', kernel_initializer="he_normal", padding='same')(X)

    # Contracting path
    for i in range(depth-1):
        X = conv_block(X, filters[i], 3, conv_per_block, padding, dropouts[i], batch_normalization, groups=groups[i])
        block_out.append(X)

        X = MaxPooling3D(pool_size=(2, 2, 1))(X)

    X = conv_block(X, filters[depth-1], 3, conv_per_block, padding, dropouts[depth-1], batch_normalization, groups=groups[depth-1])

    # Expansive path
    crop = 2
    for i in range(depth-1):
        # X = tf.keras.layers.UpSampling2D()(X)
        X = Conv3DTranspose(filters[depth+i], 2, 2, padding='valid')(X)

        if padding == 'valid':
            block_out[depth - i - 2] = Cropping2D(crop * conv_per_block)(block_out[depth - i - 2])
            # crop = crop * 2 + 2 * conv_per_block
            crop = 2 * (crop +  conv_per_block)

        X = concatenate([block_out[depth - i - 2], X], axis=3)

        X = conv_block(X, filters[depth+i], 3, conv_per_block, padding, dropouts[depth+i], batch_normalization, groups=groups[depth+i])

    if output_classes > 2:
        tmp = Conv3D(output_classes, 1, activation='softmax', name="output")(X)
    else:
        tmp = Conv3D(1, 1, activation='sigmoid', name="output")(X)

    model = tf.keras.Model(inputs=inputs, outputs=tmp)

    return model
