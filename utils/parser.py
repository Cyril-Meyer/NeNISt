import argparse


def args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--images",
                        help="add image",
                        type=str,
                        action='append',
                        required=True)

    parser.add_argument("--labels",
                        help="add labels",
                        type=str,
                        nargs='+',
                        action='append',
                        required=True)

    parser.add_argument("--images-validation",
                        help="add image for validation",
                        type=str,
                        action='append',
                        default=[])

    parser.add_argument("--labels-validation",
                        help="add labels for validation",
                        type=str,
                        nargs='+',
                        action='append',
                        default=[])

    parser.add_argument("--save",
                        help="save location",
                        type=str,
                        required=True)

    parser.add_argument("--patch-size",
                        help="patch size",
                        type=int,
                        nargs='+',
                        required=True,
                        choices=range(1, 4096))

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
                        default='cross-entropy',
                        choices=['cross-entropy', 'dice', 'iou', 'mse-dt'])

    parser.add_argument("--model",
                        help="model codename",
                        type=str,
                        default='u-net-32')

    args = parser.parse_args()
    assert len(args.images) == len(args.labels) > 0
    assert len(args.images_validation) == len(args.labels_validation) >= 0
    return args
