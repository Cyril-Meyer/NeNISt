import os
import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from lii.lii import LargeImageInference as lii

import utils.parser
import datasets.reader
import loss.segmentation
import metrics.segmentation
import medpy.metric.binary

t0 = time.time()
args = utils.parser.args_eval()


def log(*strings):
    print(str(int(time.time() - t0)).rjust(10), *strings)


OUTPUT_FOLDER = args.save_folder
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
log("OUTPUT FOLDER :", OUTPUT_FOLDER)


# load data
log("DATA : Loading")
images, labels = datasets.reader.get_data(args.images, args.labels, create_background=args.create_background)
if args.create_background:
    assert np.all(np.sum(labels[0], axis=-1) == 1)
output_classes = labels[0].shape[-1]


log("LOAD MODEL :", args.load_model)
custom_objects = {"dice_coef_tf_meyer": loss.segmentation.dice_coef_tf_meyer}
with tf.keras.utils.custom_object_scope(custom_objects):
    model = tf.keras.models.load_model(args.load_model)

if args.verbose:
    log('input shape   :', model.input_shape)
    log('output shape  :', model.output_shape)
    log('parameters    :', model.count_params())
    log('model name    :', model.name)

assert len(model.input_shape) == len(model.output_shape)

predictions = []
# with tf.device("cpu:0"):
# with tf.device("gpu:0"):
for i in range(len(images)):
    image = images[i]
    pred = model
    # Y / X padding
    f = args.patch_size[-2]
    yp = 0 if image.shape[1] % f == 0 else f * (image.shape[1] // f + 1) - image.shape[1]
    f = args.patch_size[-1]
    xp = 0 if image.shape[2] % f == 0 else f * (image.shape[2] // f + 1) - image.shape[2]
    image_p = np.pad(image, pad_width=[(0, 0), (0, yp), (0, xp), (0, 0)], mode='reflect')

    if len(model.input_shape) == 4:
        overlap = (1, 2, 2)
        def pred_2d(img): return np.expand_dims(model(img[:, 0, :, :, :]), axis=1)
        pred = pred_2d
        predictions.append(
            lii.infer(image_p, (1, ) + tuple(args.patch_size), pred, overlap, verbose=int(args.verbose), dtype=np.float32)[:, 0:image.shape[1], 0:image.shape[2], :])
    elif len(model.input_shape) == 5:
        overlap = (1, 2, 2) if image.shape[0] == args.patch_size[0] else (2, 2, 2)
        predictions.append(
            lii.infer(image_p, args.patch_size, pred, overlap, verbose=int(args.verbose), dtype=np.float32)[:, 0:image.shape[1], 0:image.shape[2], :])
    else:
        raise NotImplementedError


for i in range(len(images)):
    if args.name_prefix is None:
        name = os.path.basename(args.images[i]).split('.')[0] + '_' + \
               os.path.basename(args.load_model).split('.')[0]
    else:
        name = args.name_prefix + '_' + \
               os.path.basename(args.images[i]).split('.')[0] + '_' + \
               os.path.basename(args.load_model).split('.')[0]
    log('name', name)

    file = open(f"{OUTPUT_FOLDER}/{name}.csv", "w")
    file.write("name,model,data,")
    for c in range(predictions[i].shape[-1]):
        file.write(f'precision_{c+1},recall_{c+1},f1_{c+1},iou_{c+1},assd_{c+1}\n')
    data_name = os.path.basename(args.images[i]).split('.')[0]
    file.write(f'{name},{model.name},{data_name},')

    prediction = predictions[i] > args.binary_threshold
    label = labels[i] > 0.5

    for c in range(prediction.shape[-1]):
        precision = metrics.segmentation.precision(label.flatten(), prediction.flatten())
        recall    = metrics.segmentation.recall(label.flatten(), prediction.flatten())
        f1        = metrics.segmentation.F1(label.flatten(), prediction.flatten())
        iou       = metrics.segmentation.IoU(label.flatten(), prediction.flatten())
        assd = np.inf
        try:
            assd  = medpy.metric.binary.assd(label, prediction)
        except:
            assd = np.inf
        
        # precision = sklearn.metrics.precision_score(label.flatten(), prediction.flatten())
        # recall    = sklearn.metrics.recall_score(label.flatten(), prediction.flatten())
        # f1        = sklearn.metrics.f1_score(label.flatten(), prediction.flatten())
        # iou       = sklearn.metrics.jaccard_score(label.flatten(), prediction.flatten())
        file.write(f'{precision},{recall},{f1},{iou},{assd}\n')

        if args.verbose:
            log('precision :', precision)
            log('recall    :', recall)
            log('f1        :', f1)
            log('iou       :', iou)
            log('assd      :', assd)
            '''
            plt.imshow(label[0, :, :, c].astype(np.float32))
            plt.show()
            plt.imshow(prediction[0, :, :, c].astype(np.float32))
            plt.show()
            '''
    file.close()
