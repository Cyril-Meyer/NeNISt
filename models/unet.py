import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Conv2D, Cropping2D, Conv2DTranspose, MaxPooling2D
from tensorflow.keras.layers import Conv3D, Cropping3D, Conv3DTranspose, MaxPooling3D
from tensorflow.keras.layers import Activation, BatchNormalization, Dropout, Concatenate, Add


def block_2d(X, filters, batch_normalization=True, residual=True, dropout=0):
    if residual:
        res = Conv2D(filters=2*filters, kernel_size=1)(X)

    X = Conv2D(filters=filters, kernel_size=3, kernel_initializer="he_normal", padding='same')(X)
    if batch_normalization:
        X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv2D(filters=2*filters, kernel_size=3, kernel_initializer="he_normal", padding='same')(X)
    if batch_normalization:
        X = BatchNormalization()(X)
    if residual:
        X = Add()([res, X])
    X = Activation('relu')(X)

    if dropout > 0:
        X = Dropout(dropout)(X)

    return X


def block_3d(X, filters, batch_normalization=True, residual=True, dropout=0):
    if residual:
        res = Conv3D(filters=2*filters, kernel_size=1)(X)

    X = Conv3D(filters=filters, kernel_size=3, kernel_initializer="he_normal", padding='same')(X)
    if batch_normalization:
        X = BatchNormalization()(X)
    X = Activation('relu')(X)

    X = Conv3D(filters=2*filters, kernel_size=3, kernel_initializer="he_normal", padding='same')(X)
    if batch_normalization:
        X = BatchNormalization()(X)
    if residual:
        X = Add()([res, X])
    X = Activation('relu')(X)

    if dropout > 0:
        X = Dropout(dropout)(X)

    return X


def get(input_shape=(None, None, 1), output_classes=2, output_activation='sigmoid', filters=32, depth=5,
        batch_normalization=True, residual=True, multiple_outputs=False):

    assert len(input_shape) == 3 or len(input_shape) == 4

    if len(input_shape) == 3:
        block = block_2d
        Conv = Conv2D
        MaxPooling = MaxPooling2D
        ConvTranspose = Conv2DTranspose
    else:
        block = block_3d
        Conv = Conv3D
        MaxPooling = MaxPooling3D
        ConvTranspose = Conv3DTranspose

    inputs = tf.keras.Input(input_shape, name="input")
    block_out = []
    X = inputs

    for i in range(depth-1):
        X = block(X, filters*(2**i), batch_normalization, residual)
        block_out.append(X)
        X = MaxPooling(pool_size=2)(X)

    X = block(X, filters*(2**depth), batch_normalization, residual)

    print()

    # Expansive path
    crop = 2
    for i in range(depth-1):
        X = ConvTranspose(filters*(2**(depth-1-i)), 2, 2, padding='valid')(X)
        X = Concatenate(axis=-1)([block_out[::-1][i], X])
        X = block(X, filters*(2**i), batch_normalization)

    if multiple_outputs:
        outputs = []
        for c in range(output_classes):
            outputs.append(Conv(1, 1, activation=output_activation, name="output_" + str(c + 1))(X))
    else:
        outputs = Conv(output_classes, 1, activation=output_activation, name="output")(X)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


