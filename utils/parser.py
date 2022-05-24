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

    parser.add_argument("--save",
                        help="save location",
                        type=str,
                        default=None,
                        required=True)

    parser.add_argument("--patch-size",
                        help="patch size",
                        type=int,
                        nargs='+',
                        default=[],
                        choices=range(1, 4096))

    parser.add_argument("--batch-size",
                        help="batch size",
                        type=int,
                        default=1,
                        choices=range(1, 4096))

    parser.add_argument("--loss",
                        help="loss",
                        type=str,
                        default='crossentropy')

    parser.add_argument("--model",
                        help="model codename",
                        type=str,
                        default='unet')

    args = parser.parse_args()
    assert len(args.images) == len(args.labels) > 0
    return args


