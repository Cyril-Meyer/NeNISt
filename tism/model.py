import tensorflow as tf
import architecture as a
import backbone as b


def get(architecture=a.UNet(), backbone_encoder=b.VGG(), backbone_decoder=b.VGG(initial_block_length=1)):
    return architecture(backbone_encoder, backbone_decoder)
