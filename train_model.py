import os
import time

import numpy as np
import tensorflow as tf

import utils.parser
import datasets.reader
import data.patch
import data.distance_transform
import models.unet
import loss.segmentation

t0 = time.time()
args = utils.parser.args_train()


def log(*strings):
    print(str(int(time.time() - t0)).rjust(10), *strings)


OUTPUT_FOLDER = args.save_folder
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
log("OUTPUT FOLDER :", OUTPUT_FOLDER)

# activation
activation = args.activation
# loss
loss = utils.parser.get_loss(args.loss, activation)

# load data
log("DATA : Loading")
images, labels = datasets.reader.get_data(args.images, args.labels, create_background=args.create_background)
images_validation, labels_validation = datasets.reader.get_data(args.images_validation, args.labels_validation, create_background=args.create_background)
if args.create_background:
    assert np.all(np.sum(labels[0], axis=-1) == 1)
output_classes = labels[0].shape[-1]

log("DATA : Labels id")
labels_id = datasets.reader.get_labels_id(labels)
labels_validation_id = datasets.reader.get_labels_id(labels_validation)

# label distance transform
if 'dt' in args.loss or 'BoundaryDistance' in args.loss:
    edt = data.distance_transform.label_edt
    normalize = data.distance_transform.normalize_tanh

    log("DATA : Distance transform")
    labels_dt = []
    for label in labels:
        label_dt = []
        for cla in range(label.shape[-1]):
            label_dt.append(normalize(edt((label[..., cla] * 1.0).astype(np.float32)), 20))
        labels_dt.append(np.moveaxis(np.array(label_dt), 0, -1))
    labels = labels_dt

    log("DATA : Distance transform (valid)")
    labels_dt = []
    for label in labels_validation:
        label_dt = []
        for cla in range(label.shape[-1]):
            label_dt.append(normalize(edt((label[..., cla] * 1.0).astype(np.float32)), 20))
        labels_dt.append(np.moveaxis(np.array(label_dt), 0, -1))
    labels_validation = labels_dt

log("DATA : Ready")

model = utils.parser.get_model(args.model, output_classes, activation, name=args.model)
log("MODEL :", model.input_shape, "->", model.output_shape)
log("MODEL :", model.name, f"{model.count_params():,}")
log("MODEL :", activation, loss)

# patch generator
train = data.patch.gen_patch_batch(args.patch_size, images, labels, args.batch_size, augmentation=True,
                                   label_indexes=labels_id, label_indexes_prop=0.8)
valid = data.patch.gen_patch_batch(args.patch_size, images_validation, labels_validation, args.batch_size, augmentation=True,
                                   label_indexes=labels_validation_id, label_indexes_prop=0.8)
if args.multiple_outputs:
    train_ = train
    valid_ = valid
    train = data.patch.gen_to_multiple_outputs(train_)
    valid = data.patch.gen_to_multiple_outputs(valid_)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
if args.multiple_outputs:
    model.compile(optimizer=optimizer, loss=loss, loss_weights=[1/output_classes for i in range(output_classes)])
else:
    model.compile(optimizer=optimizer, loss=loss)

log("MODEL : Ready")

savebestmodel = tf.keras.callbacks.ModelCheckpoint(filepath=OUTPUT_FOLDER + "/model_best.h5",
                                                   save_weights_only=False, monitor='val_loss', mode='min',
                                                   save_best_only=True, verbose=args.verbose)

fit_history = model.fit(train, steps_per_epoch=args.steps_per_epoch, epochs=args.epochs, verbose=args.verbose,
                        validation_data=valid, validation_steps=args.steps_per_epoch_validation,
                        callbacks=[savebestmodel])

model.save(OUTPUT_FOLDER + "/model_last.h5", save_format='h5')
