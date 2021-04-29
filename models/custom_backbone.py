import tensorflow as tf
import models.layers.morphology


class MM_Alpha:
    def __init__(self, backbone):
        self.backbone = backbone
        self.conv = tf.keras.layers.Conv2D

    def set3D(self):
        raise NotImplementedError

    def __call__(self, X, level):
        m_01 = models.layers.morphology.Dilation2D(shape=(5, 5))(X)
        m_02 = models.layers.morphology.Erosion2D(shape=(5, 5))(X)
        
        X = tf.keras.layers.Add()([m_01, m_02])
        
        X, block_depth = self.backbone(X, level)

        return X, block_depth


class MM_Beta:
    def __init__(self, backbone):
        self.backbone = backbone
        self.conv = tf.keras.layers.Conv2D

    def set3D(self):
        raise NotImplementedError

    def __call__(self, X, level):
        out, block_depth = self.backbone(X, level)
        
        res_01 = models.layers.morphology.Dilation2D(shape=(5, 5))(X)
        res_02 = models.layers.morphology.Erosion2D(shape=(5, 5))(X)
        
        res = tf.keras.layers.Add()([res_01, res_02])
        res = self.conv(filters=block_depth, kernel_size=1)(res)

        X = tf.keras.layers.Add()([res, out])

        return X, block_depth


class MM_Gamma:
    def __init__(self, backbone):
        self.backbone = backbone
        self.conv = tf.keras.layers.Conv2D

    def set3D(self):
        raise NotImplementedError

    def __call__(self, X, level):
        out, block_depth = self.backbone(X, level)
        
        m_01 = models.layers.morphology.Dilation2D(shape=(5, 5))(X)
        m_02 = models.layers.morphology.Erosion2D(shape=(5, 5))(X)
        res = tf.keras.layers.Concatenate()([m_01, m_02])
        res = tf.keras.layers.Conv2D(block_depth, 1, 1)(res)
        
        X = tf.keras.layers.Add()([res, out])
        
        return X, block_depth
