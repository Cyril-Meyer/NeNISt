import os
import time

import numpy as np
import tensorflow as tf

import utils.parser
import datasets.reader
import data.patch
import data.distance_transform
import models.unet

t0 = time.time()

args = utils.parser.args()

OUTPUT_FOLDER = args.save
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
print(str(int(time.time() - t0)).rjust(10), "OUTPUT FOLDER :", OUTPUT_FOLDER)

# loss
loss, activation, multiple_outputs, threshold = utils.parser.get_loss(args.loss)

# load data
print(str(int(time.time() - t0)).rjust(10), "LOAD DATA")
images, labels = datasets.reader.get_data(args.images, args.labels)
images_validation, labels_validation = datasets.reader.get_data(args.images_validation, args.labels_validation)
output_classes = labels[0].shape[-1]

# label distance transform
'''
if '_dt' in args.loss:
    print(str(int(time.time() - t0)).rjust(10), "EDT DATA")
    edt = data.distance_transform.label_edt
    normalize = data.distance_transform.normalize_tanh
    labels_dt = []
    for label in labels:
        label_dt = []
        for cla in label:
            label_dt.append(normalize(edt((cla * 1.0).astype(np.float32)), 20))
        labels_dt.append(label_dt)

    labels = labels_dt
'''

# patch generator
if multiple_outputs:
    train = data.patch.gen_to_multiple_outputs(
        data.patch.gen_patch_batch(args.patch_size, images, labels, args.batch_size, augmentation=True))
else:
    train = data.patch.gen_patch_batch(args.patch_size, images, labels, args.batch_size, augmentation=True)

# create model
# 3D models
if args.model == 'unet-4-32' and len(args.patch_size) == 3:
    model = models.unet.get(input_shape=(None, None, None, 1), output_classes=output_classes, output_activation=activation, filters=32, depth=4, multiple_outputs=multiple_outputs)
elif args.model == 'unet-5-16' and len(args.patch_size) == 3:
    model = models.unet.get(input_shape=(None, None, None, 1), output_classes=output_classes, output_activation=activation, filters=16, depth=5, multiple_outputs=multiple_outputs)
# 2D models
elif args.model == 'unet-5-26' and len(args.patch_size) == 2:
    model = models.unet.get(input_shape=(      None, None, 1), output_classes=output_classes, output_activation=activation, filters=26, depth=5, multiple_outputs=multiple_outputs)
else:
    raise NotImplementedError
# model.summary(line_length=300)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
model.compile(optimizer=optimizer, loss=loss, loss_weights=[1/output_classes for i in range(output_classes)])

fit_history = model.fit(train, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs, verbose=args.verbose)

model.save(OUTPUT_FOLDER + "/model_last", save_format='h5')

'''
'''