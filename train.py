# ------------------------------------------------------------ #
#
# File : train.py
# Authors : CM
# Train, evaluate and, save model and prediction on test
#
# ------------------------------------------------------------ #

import sys
import os
import datetime
import time

import numpy as np
import tensorflow as tf

import segmentation_models as sm
import skimage as sk
import sklearn.metrics

import data.patch

import models.UNet
import models.EfficientUNet

if(os.uname()[1] == 'lythandas'):
    OUTPUT_FOLDER = ""
else:
    OUTPUT_FOLDER = "/b/home/miv/cmeyer/I_2021/"

dt = datetime.datetime.today().strftime("%H%M%f")

NAME = sys.argv[1] + "_" + dt
PATCH_SIZE = int(sys.argv[2])
DATASET = int(sys.argv[3])
MODEL = int(sys.argv[4])
EXP = int(sys.argv[5])

if EXP == 1:
    LAMBDAS = list(map(int, sys.argv[6:]))
    LAMBDAS_STR = str(LAMBDAS).replace(" ", "").replace("[", "").replace("]", "").replace(",", "_")

BATCH_SIZE = 4
EPOCHS = 100
TRAIN_PER_EPOCHS = 256
VALID_PER_EPOCHS = 64

print("NAME", NAME)
print("PATCH_SIZE", PATCH_SIZE)
print("DATASET", DATASET)
print("MODEL", MODEL)
print("EXP", EXP)

if EXP == 1:
    print("LAMBDAS", LAMBDAS)

# Dataset

if DATASET == 1:
    import data.datasets.I3 as D
    
    train_image = D.train_i3_image_normalized_f32
    train_labels = np.sum([D.train_i3_label_1, D.train_i3_label_2*2, D.train_i3_label_3*3], axis=0).astype(np.uint8)
    train_background = np.where(train_labels > 0, 0, 1)
    train_labels_indexes = [D.train_i3_label_1_indexes, D.train_i3_label_2_indexes, D.train_i3_label_3_indexes]
    train_labels_one_hot = np.stack([train_background, D.train_i3_label_1, D.train_i3_label_2, D.train_i3_label_3], axis=-1)

    valid_image = D.valid_i3_image_normalized_f32
    valid_labels = np.sum([D.valid_i3_label_1, D.valid_i3_label_2*2, D.valid_i3_label_3*3], axis=0).astype(np.uint8)
    valid_background = np.where(valid_labels > 0, 0, 1)
    valid_labels_indexes = [D.valid_i3_label_1_indexes, D.valid_i3_label_2_indexes, D.valid_i3_label_3_indexes]
    valid_labels_one_hot = np.stack([valid_background, D.valid_i3_label_1, D.valid_i3_label_2, D.valid_i3_label_3], axis=-1)

    test_image = D.test_i3_image_normalized_f32
    test_labels = np.sum([D.test_i3_label_1, D.test_i3_label_2*2, D.test_i3_label_3*3], axis=0).astype(np.uint8)
    test_background = np.where(test_labels > 0, 0, 1)
    test_labels_indexes = [D.test_i3_label_1_indexes, D.test_i3_label_2_indexes, D.test_i3_label_3_indexes]
    test_labels_one_hot = np.stack([test_background, D.test_i3_label_1, D.test_i3_label_2, D.test_i3_label_3], axis=-1)
    
elif DATASET == 2:
    import data.datasets.LW4 as D

    train_image = D.train_lw4_image_normalized_f32
    train_labels = np.sum([D.train_lw4_label_1, D.train_lw4_label_2*2, D.train_lw4_label_3*3], axis=0).astype(np.uint8)
    train_background = np.where(train_labels > 0, 0, 1)
    train_labels_indexes = [D.train_lw4_label_1_indexes, D.train_lw4_label_2_indexes, D.train_lw4_label_3_indexes]
    train_labels_one_hot = np.stack([train_background, D.train_lw4_label_1, D.train_lw4_label_2, D.train_lw4_label_3], axis=-1)

    valid_image = D.valid_lw4_image_normalized_f32
    valid_labels = np.sum([D.valid_lw4_label_1, D.valid_lw4_label_2*2, D.valid_lw4_label_3*3], axis=0).astype(np.uint8)
    valid_background = np.where(valid_labels > 0, 0, 1)
    valid_labels_indexes = [D.valid_lw4_label_1_indexes, D.valid_lw4_label_2_indexes, D.valid_lw4_label_3_indexes]
    valid_labels_one_hot = np.stack([valid_background, D.valid_lw4_label_1, D.valid_lw4_label_2, D.valid_lw4_label_3], axis=-1)
    
    test_image = D.test_lw4_image_normalized_f32
    test_labels = np.sum([D.test_lw4_label_1, D.test_lw4_label_2*2, D.test_lw4_label_3*3], axis=0).astype(np.uint8)
    test_background = np.where(test_labels > 0, 0, 1)
    test_labels_indexes = [D.test_lw4_label_1_indexes, D.test_lw4_label_2_indexes, D.test_lw4_label_3_indexes]
    test_labels_one_hot = np.stack([test_background, D.test_lw4_label_1, D.test_lw4_label_2, D.test_lw4_label_3], axis=-1)
    
else:
    print("ERROR: NO VALID DATASET SELECTED")
    sys.exit(os.EX_DATAERR)

if EXP == 1:
    train = data.patch.gen_patches_batch_augmented_tos_area_label_indexes_one_hot_p(PATCH_SIZE, train_image, train_labels_one_hot, train_labels_indexes, batch_size=BATCH_SIZE, lambdas=LAMBDAS)
    valid = data.patch.gen_patches_batch_augmented_tos_area_label_indexes_one_hot_p(PATCH_SIZE, valid_image, valid_labels_one_hot, valid_labels_indexes, batch_size=BATCH_SIZE, lambdas=LAMBDAS)
else:
    train = data.patch.gen_patches_batch_augmented_label_indexes_one_hot(PATCH_SIZE, train_image, train_labels_one_hot, train_labels_indexes, batch_size=BATCH_SIZE)
    valid = data.patch.gen_patches_batch_augmented_label_indexes_one_hot(PATCH_SIZE, valid_image, valid_labels_one_hot, valid_labels_indexes, batch_size=BATCH_SIZE)

# Model
# loss_cce = tf.keras.losses.CategoricalCrossentropy()
# loss_dice = sm.losses.DiceLoss()
loss_dice = sm.losses.DiceLoss(class_weights=np.array([0.10, 0.70, 1.0, 0.90]))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

if EXP == 1:
    if MODEL == 1:
        model = models.UNet.UNet(input_shape=(PATCH_SIZE, PATCH_SIZE, len(LAMBDAS)+1), output_classes=4, filters=64, depth=5, conv_per_block=2, padding='same', dropouts=0.50, batch_normalization=True)
    elif MODEL == 2:
        model = models.EfficientUNet.EfficientUNetB4(input_shape=(PATCH_SIZE, PATCH_SIZE, len(LAMBDAS)+1), classes=4, weight_name=None)
    elif MODEL == 3:
        base_model = sm.Unet('efficientnetb4', input_shape=(256, 256, 3), classes=4, activation='softmax', encoder_weights='imagenet')
        model_in = tf.keras.Input(shape=(PATCH_SIZE, PATCH_SIZE, len(LAMBDAS)+1))
        model_c1 = tf.keras.layers.Conv2D(3, (1, 1))(model_in)
        model_out = base_model(model_c1)
        model = tf.keras.Model(model_in, model_out, name=base_model.name)
    else:
        print("ERROR: NO VALID MODEL SELECTED")
        sys.exit(os.EX_DATAERR)
else:
    if MODEL == 1:
        model = models.UNet.UNet(input_shape=(PATCH_SIZE, PATCH_SIZE, 1), output_classes=4, filters=64, depth=5, conv_per_block=2, padding='same', dropouts=0.50, batch_normalization=True)
    elif MODEL == 2:
        model = models.EfficientUNet.EfficientUNetB4(input_shape=(PATCH_SIZE, PATCH_SIZE, 1), classes=4, weight_name=None)
    elif MODEL == 3:
        base_model = sm.Unet('efficientnetb4', input_shape=(256, 256, 3), classes=4, activation='softmax', encoder_weights='imagenet')
        model_in = tf.keras.Input(shape=(PATCH_SIZE, PATCH_SIZE, 1))
        model_c1 = tf.keras.layers.Conv2D(3, (1, 1))(model_in)
        model_out = base_model(model_c1)
        model = tf.keras.Model(model_in, model_out, name=base_model.name)
    else:
        print("ERROR: NO VALID MODEL SELECTED")
        sys.exit(os.EX_DATAERR)

model.compile(optimizer=optimizer, loss=loss_dice)

cb_best_model_val = tf.keras.callbacks.ModelCheckpoint(
    filepath= OUTPUT_FOLDER + NAME + "/",
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

model.fit(train, steps_per_epoch=TRAIN_PER_EPOCHS, epochs=EPOCHS, validation_data=valid, validation_steps=VALID_PER_EPOCHS, callbacks=[cb_best_model_val], verbose=0)

# Eval
model.load_weights(OUTPUT_FOLDER + NAME + "/")
f_results_name = OUTPUT_FOLDER + NAME + ".csv"

f_results = open(f_results_name, "w")
f_results.write("modelname")
for z in range(valid_image.shape[0]):
    f_results.write(",F1_C1_V" + str(z))
    f_results.write(",F1_C2_V" + str(z))
    f_results.write(",F1_C3_V" + str(z))
for z in range(test_image.shape[0]):
    f_results.write(",F1_C1_T" + str(z))
    f_results.write(",F1_C2_T" + str(z))
    f_results.write(",F1_C3_T" + str(z))
f_results.write("\n")
f_results.close()

f_results = open(f_results_name, "a")
f_results.write(NAME)

all_predicted_slice = np.zeros((valid_image.shape[0]+test_image.shape[0], valid_image.shape[1], valid_image.shape[2]), dtype=np.uint8)

for z in range(valid_image.shape[0]):
    if EXP == 1:
        predicted_slice = data.patch.predict_slice_one_hot_tos_area(256, valid_image[z], model, lambdas=LAMBDAS)
    else:
        predicted_slice = data.patch.predict_slice_one_hot(256, valid_image[z], model)
    
    all_predicted_slice[z] = predicted_slice.astype("uint8")
    
    for cl in range(1, valid_labels_one_hot[z].shape[-1]):
        x = ((predicted_slice == cl)*1).flatten().astype("uint8")
        y = valid_labels_one_hot[z,:,:,cl].flatten().astype("uint8")
        
        if np.sum(y) == 0:
            f1 = 0.0
        else:
            f1  = sklearn.metrics.f1_score(x, y)
        
        f_results.write("," + str(f1))

f_results.close()
f_results = open(f_results_name, "a")

for z in range(test_image.shape[0]):
    if EXP == 1:
        predicted_slice = data.patch.predict_slice_one_hot_tos_area(256, test_image[z], model, lambdas=LAMBDAS)
    else:
        predicted_slice = data.patch.predict_slice_one_hot(256, test_image[z], model)
    
    all_predicted_slice[valid_image.shape[0]+z] = predicted_slice.astype("uint8")
    
    for cl in range(1, test_labels_one_hot[z].shape[-1]):
        x = ((predicted_slice == cl)*1).flatten().astype("uint8")
        y = test_labels_one_hot[z,:,:,cl].flatten().astype("uint8")
        
        if np.sum(y) == 0:
            f1 = 0.0
        else:
            f1  = sklearn.metrics.f1_score(x, y)
        
        f_results.write("," + str(f1))

f_results.write("\n")
f_results.close()

np.save(OUTPUT_FOLDER + NAME + ".npy", all_predicted_slice)
