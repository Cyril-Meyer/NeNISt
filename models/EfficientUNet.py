import tensorflow as tf


def EfficientNet_to_EfficientUNet(model, classes, weight_name='imagenet'):
    # input 8x8, output 16x16
    output_level_6 = tf.keras.Model.get_layer(model, 'top_activation').output
    decoder_level_6_up = tf.keras.layers.UpSampling2D()(output_level_6)
    decoder_level_6_out = decoder_level_6_up
    
    # input AxA, output 2*Ax2*A
    output_level_5 = tf.keras.Model.get_layer(model, 'block5a_expand_activation').output
    decoder_level_5_concat = tf.keras.layers.concatenate([decoder_level_6_out, output_level_5], axis=3)
    decoder_level_5_conv_1 = tf.keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', activation='relu')(decoder_level_5_concat)
    decoder_level_5_up = tf.keras.layers.UpSampling2D()(decoder_level_5_conv_1)
    decoder_level_5_out = decoder_level_5_up
    
    # input BxB, output 2*Bx2*B, B=2*A
    output_level_4 = tf.keras.Model.get_layer(model, 'block4a_expand_activation').output
    decoder_level_4_concat = tf.keras.layers.concatenate([decoder_level_5_out, output_level_4], axis=3)
    decoder_level_4_conv_1 = tf.keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', activation='relu')(decoder_level_4_concat)
    decoder_level_4_up = tf.keras.layers.UpSampling2D()(decoder_level_4_conv_1)
    decoder_level_4_out = decoder_level_4_up
    
    # input CxC, output 2*Cx2*C, C=2*B
    output_level_3 = tf.keras.Model.get_layer(model, 'block3a_expand_activation').output
    decoder_level_3_concat = tf.keras.layers.concatenate([decoder_level_4_out, output_level_3], axis=3)
    decoder_level_3_conv_1 = tf.keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', activation='relu')(decoder_level_3_concat)
    decoder_level_3_up = tf.keras.layers.UpSampling2D()(decoder_level_3_conv_1)
    decoder_level_3_out = decoder_level_3_up
    
    # input DxD, output 2*Dx2*D, D=2*C
    output_level_2 = tf.keras.Model.get_layer(model, 'block2a_expand_activation').output
    decoder_level_2_concat = tf.keras.layers.concatenate([decoder_level_3_out, output_level_2], axis=3)
    decoder_level_2_conv_1 = tf.keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', activation='relu')(decoder_level_2_concat)
    decoder_level_2_up = tf.keras.layers.UpSampling2D()(decoder_level_2_conv_1)
    decoder_level_2_out = decoder_level_2_up
    
    # input ExE, output ExE, E=D*A
    # output_level_1 = tf.keras.Model.get_layer(model, 'normalization_').output
    output_level_1 = model.layers[2].output
    decoder_level_1_concat = tf.keras.layers.concatenate([decoder_level_2_out, output_level_1], axis=3)
    decoder_level_1_conv_1 = tf.keras.layers.Conv2D(64, kernel_size=(3,3), padding='same', activation='relu')(decoder_level_1_concat)
    decoder_level_1_out = tf.keras.layers.Conv2D(classes, (1), padding='same', activation='softmax')(decoder_level_1_conv_1)
    
    return tf.keras.Model(model.input, decoder_level_1_out)


def EfficientUNetB0(input_shape, classes, weight_name='imagenet'):
    if len(input_shape) != 3:
        return
    
    EfficientNetB0 = tf.keras.applications.EfficientNetB0(
    include_top=None,
    weights=weight_name,
    input_tensor=None,
    input_shape=(input_shape[0], input_shape[1], 3),
    pooling=None,
    classes=None,
    classifier_activation=None
    )
    
    EfficientUNetB0 = EfficientNet_to_EfficientUNet(EfficientNetB0, classes, weight_name='imagenet')
    
    if input_shape[-1] != 3:
            model_input = tf.keras.Input(shape=input_shape)
            model_out = tf.keras.layers.Conv2D(3, (1, 1))(model_input)
            model_out = EfficientUNetB0(model_out)
            EfficientUNetB0 = tf.keras.Model(model_input, model_out, name=EfficientUNetB0.name)
    
    return EfficientUNetB0


def EfficientUNetB1(input_shape, classes, weight_name='imagenet'):
    if len(input_shape) != 3:
        return
    
    EfficientNetB1 = tf.keras.applications.EfficientNetB1(
    include_top=None,
    weights=weight_name,
    input_tensor=None,
    input_shape=input_shape,
    pooling=None,
    classes=None,
    classifier_activation=None
    )
    
    EfficientUNetB1 = EfficientNet_to_EfficientUNet(EfficientNetB1, classes, weight_name='imagenet')
    
    if input_shape[-1] != 3:
            model_input = tf.keras.Input(shape=input_shape)
            model_out = tf.keras.layers.Conv2D(3, (1, 1))(model_input)
            model_out = EfficientUNetB1(model_out)
            EfficientUNetB1 = tf.keras.Model(model_input, model_out, name=EfficientUNetB1.name)
    
    return EfficientUNetB1


def EfficientUNetB2(input_shape, classes, weight_name='imagenet'):
    if len(input_shape) != 3:
        return
    
    EfficientNetB2 = tf.keras.applications.EfficientNetB2(
    include_top=None,
    weights=weight_name,
    input_tensor=None,
    input_shape=input_shape,
    pooling=None,
    classes=None,
    classifier_activation=None
    )
    
    EfficientUNetB2 = EfficientNet_to_EfficientUNet(EfficientNetB2, classes, weight_name='imagenet')
    
    if input_shape[-1] != 3:
            model_input = tf.keras.Input(shape=input_shape)
            model_out = tf.keras.layers.Conv2D(3, (1, 1))(model_input)
            model_out = EfficientUNetB2(model_out)
            EfficientUNetB2 = tf.keras.Model(model_input, model_out, name=EfficientUNetB2.name)
    
    return EfficientUNetB2


def EfficientUNetB3(input_shape, classes, weight_name='imagenet'):
    if len(input_shape) != 3:
        return
    
    EfficientNetB3 = tf.keras.applications.EfficientNetB3(
    include_top=None,
    weights=weight_name,
    input_tensor=None,
    input_shape=input_shape,
    pooling=None,
    classes=None,
    classifier_activation=None
    )
    
    EfficientUNetB3 = EfficientNet_to_EfficientUNet(EfficientNetB3, classes, weight_name='imagenet')
    
    if input_shape[-1] != 3:
            model_input = tf.keras.Input(shape=input_shape)
            model_out = tf.keras.layers.Conv2D(3, (1, 1))(model_input)
            model_out = EfficientUNetB3(model_out)
            EfficientUNetB3 = tf.keras.Model(model_input, model_out, name=EfficientUNetB3.name)
    
    return EfficientUNetB3


def EfficientUNetB4(input_shape, classes, weight_name='imagenet'):
    if len(input_shape) != 3:
        return
    
    EfficientNetB4 = tf.keras.applications.EfficientNetB4(
    include_top=None,
    weights=weight_name,
    input_tensor=None,
    input_shape=input_shape,
    pooling=None,
    classes=None,
    classifier_activation=None
    )
    
    EfficientUNetB4 = EfficientNet_to_EfficientUNet(EfficientNetB4, classes, weight_name='imagenet')
    
    if input_shape[-1] != 3:
            model_input = tf.keras.Input(shape=input_shape)
            model_out = tf.keras.layers.Conv2D(3, (1, 1))(model_input)
            model_out = EfficientUNetB4(model_out)
            EfficientUNetB4 = tf.keras.Model(model_input, model_out, name=EfficientUNetB4.name)
    
    return EfficientUNetB4


def EfficientUNetB5(input_shape, classes, weight_name='imagenet'):
    if len(input_shape) != 3:
        return
    
    EfficientNetB5 = tf.keras.applications.EfficientNetB5(
    include_top=None,
    weights=weight_name,
    input_tensor=None,
    input_shape=input_shape,
    pooling=None,
    classes=None,
    classifier_activation=None
    )
    
    EfficientUNetB5 = EfficientNet_to_EfficientUNet(EfficientNetB5, classes, weight_name='imagenet')
    
    if input_shape[-1] != 3:
            model_input = tf.keras.Input(shape=input_shape)
            model_out = tf.keras.layers.Conv2D(3, (1, 1))(model_input)
            model_out = EfficientUNetB5(model_out)
            EfficientUNetB5 = tf.keras.Model(model_input, model_out, name=EfficientUNetB5.name)
    
    return EfficientUNetB5


def EfficientUNetB6(input_shape, classes, weight_name='imagenet'):
    if len(input_shape) != 3:
        return
    
    EfficientNetB6 = tf.keras.applications.EfficientNetB6(
    include_top=None,
    weights=weight_name,
    input_tensor=None,
    input_shape=input_shape,
    pooling=None,
    classes=None,
    classifier_activation=None
    )
    
    EfficientUNetB6 = EfficientNet_to_EfficientUNet(EfficientNetB6, classes, weight_name='imagenet')
    
    if input_shape[-1] != 3:
            model_input = tf.keras.Input(shape=input_shape)
            model_out = tf.keras.layers.Conv2D(3, (1, 1))(model_input)
            model_out = EfficientUNetB6(model_out)
            EfficientUNetB6 = tf.keras.Model(model_input, model_out, name=EfficientUNetB6.name)
    
    return EfficientUNetB6


def EfficientUNetB7(input_shape, classes, weight_name='imagenet'):
    if len(input_shape) != 3:
        return
    
    EfficientNetB7 = tf.keras.applications.EfficientNetB7(
    include_top=None,
    weights=weight_name,
    input_tensor=None,
    input_shape=input_shape,
    pooling=None,
    classes=None,
    classifier_activation=None
    )
    
    EfficientUNetB7 = EfficientNet_to_EfficientUNet(EfficientNetB7, classes, weight_name='imagenet')

    if input_shape[-1] != 3:
            model_input = tf.keras.Input(shape=input_shape)
            model_out = tf.keras.layers.Conv2D(3, (1, 1))(model_input)
            model_out = EfficientUNetB7(model_out)
            EfficientUNetB7 = tf.keras.Model(model_input, model_out, name=EfficientUNetB7.name)
    
    return EfficientUNetB7

