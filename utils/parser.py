import argparse
import tensorflow as tf
import models.unet
import loss.segmentation


def args_train():
    parser = argparse.ArgumentParser()

    parser.add_argument("--images",
                        help="add image",
                        type=str,
                        action="append",
                        required=True)

    parser.add_argument("--labels",
                        help="add labels",
                        type=str,
                        nargs="+",
                        action="append",
                        required=True)

    parser.add_argument("--images-validation",
                        help="add image for validation",
                        type=str,
                        action="append",
                        default=[])

    parser.add_argument("--labels-validation",
                        help="add labels for validation",
                        type=str,
                        nargs="+",
                        action="append",
                        default=[])

    parser.add_argument("--save-folder",
                        help="save location",
                        type=str,
                        required=True)

    parser.add_argument("--patch-size",
                        help="patch size",
                        type=int,
                        nargs="+",
                        required=True)

    parser.add_argument("--batch-size",
                        help="batch size",
                        type=int,
                        default=1,
                        choices=range(1, 4096))

    parser.add_argument("--steps-per-epoch",
                        help="number of batch per epoch",
                        type=int,
                        default=512,
                        choices=range(16, 4096))

    parser.add_argument("--steps-per-epoch-validation",
                        help="number of validation batch per epoch",
                        type=int,
                        default=0,
                        choices=range(0, 4096))

    parser.add_argument("--epochs",
                        help="number of epochs",
                        type=int,
                        default=128,
                        choices=range(1, 4096))

    parser.add_argument("--loss",
                        help="loss",
                        type=str,
                        default='Dice')

    parser.add_argument("--activation",
                        help="activation",
                        type=str,
                        default='sigmoid')

    parser.add_argument("--model",
                        help="model codename",
                        type=str)

    parser.add_argument("--create-background",
                        help="create-background",
                        type=bool,
                        default=False)

    parser.add_argument("--multiple-outputs",
                        help="multiple outputs",
                        type=bool,
                        default=False)

    parser.add_argument("--verbose",
                        help="verbose",
                        type=bool,
                        default=False)

    args = parser.parse_args()
    assert len(args.images) == len(args.labels) > 0
    assert len(args.images_validation) == len(args.labels_validation) >= 0
    return args


def args_eval():
    parser = argparse.ArgumentParser()

    parser.add_argument("--images",
                        help="add image",
                        type=str,
                        action="append",
                        required=True)

    parser.add_argument("--labels",
                        help="add labels",
                        type=str,
                        nargs="+",
                        action="append",
                        required=True)

    parser.add_argument("--save-folder",
                        help="save location",
                        type=str,
                        required=True)

    parser.add_argument("--name-prefix",
                        help="name prefix for results",
                        type=str,
                        default=None)

    parser.add_argument("--load-model",
                        help="model file",
                        type=str,
                        required=True)

    parser.add_argument("--patch-size",
                        help="patch size",
                        type=int,
                        nargs="+",
                        default=None)

    parser.add_argument("--binary-threshold",
                        help="binary threshold",
                        type=float,
                        default=0.5)

    parser.add_argument("--create-background",
                        help="create-background",
                        type=bool,
                        default=False)

    parser.add_argument("--multiple-outputs",
                        help="multiple outputs",
                        type=bool,
                        default=False)

    parser.add_argument("--verbose",
                        help="verbose",
                        type=bool,
                        default=False)

    args = parser.parse_args()
    assert len(args.images) == len(args.labels) > 0
    return args


def get_model(args_model, output_classes, output_activation, name):
    if args_model == 'unet2d-5-32':
        return models.unet.get(input_shape=(      None, None, 1),
                               output_classes=output_classes, output_activation=output_activation,
                               filters=32, depth=5, name=name)
    elif args_model == 'unet3d-4-38':
        return models.unet.get(input_shape=(None, None, None, 1),
                               output_classes=output_classes, output_activation=output_activation,
                               filters=38, depth=4, name=name)
    elif args_model == 'unet3d-5-19':
        return models.unet.get(input_shape=(None, None, None, 1),
                               output_classes=output_classes, output_activation=output_activation,
                               filters=19, depth=5, name=name)
    else:
        raise NotImplementedError


def get_loss(args_loss, args_activation):
    if args_loss == 'Dice' and args_activation == 'softmax':
        return loss.segmentation.dice_coef_multi_tf_meyer
    elif args_loss == 'Dice' and args_activation == 'sigmoid':
        return loss.segmentation.dice_coef_tf_meyer
    elif args_loss == 'BinaryDice':
        return loss.segmentation.dice_coef_tf_meyer
    elif args_loss == 'bce' or args_loss == 'BinaryCrossentropy':
        return tf.keras.losses.BinaryCrossentropy()
    elif args_loss == 'cce' or args_loss == 'CategoricalCrossentropy':
        return tf.keras.losses.CategoricalCrossentropy()
    elif args_loss == 'MeanSquaredErrorsBoundaryDistance' and args_activation == 'tanh':
        return tf.keras.losses.MeanSquaredError()
    else:
        raise NotImplementedError
    '''
    elif args_loss == 'cce':
        loss_func = tf.keras.losses.CategoricalCrossentropy
        activation = 'softmax'
        multiple_outputs = False
        threshold = 0.5
    elif args_loss == 'bce':
        loss_func = tf.keras.losses.BinaryCrossentropy
        activation = 'softmax'
        multiple_outputs = True
        threshold = 0.5
    elif args_loss == 'dice-bce':
        loss_func = loss.segmentation.DiceBCELoss
        activation = 'softmax'
        multiple_outputs = True
        threshold = 0.5
    elif args_loss == 'iou':
        loss_func = loss.segmentation.IoULoss
        activation = 'softmax'
        multiple_outputs = True
        threshold = 0.5
    elif args_loss == 'mse-dt':
        loss_func = tf.keras.losses.MeanSquaredError
        activation = 'tanh'
        multiple_outputs = True
        threshold = 0.0
    '''

