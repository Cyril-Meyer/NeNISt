import tensorflow as tf


class Dilation2D(tf.keras.layers.Layer):
    def __init__(self, shape=(4, 4), **kwargs):
        super(Dilation2D, self).__init__(**kwargs)
        self.shape = shape

    def build(self, input_shape):
        self.w = self.add_weight(shape=self.shape + (input_shape[-1],),
                                 dtype=tf.float32,
                                 initializer='zero',
                                 trainable=True)

    def call(self, inputs):
        return tf.nn.dilation2d(input=inputs, filters=self.w, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], padding="SAME", data_format="NHWC")


class Erosion2D(tf.keras.layers.Layer):
    def __init__(self, shape=(4, 4), **kwargs):
        super(Erosion2D, self).__init__(**kwargs)
        self.shape = shape

    def build(self, input_shape):
        self.w = self.add_weight(shape=self.shape + (input_shape[-1],),
                                 dtype=tf.float32,
                                 initializer='zero',
                                 trainable=True)

    def call(self, inputs):
        return tf.nn.erosion2d(value=inputs, filters=self.w, strides=[1, 1, 1, 1], dilations=[1, 1, 1, 1], padding="SAME", data_format="NHWC")
