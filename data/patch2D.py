# ------------------------------------------------------------ #
#
# File : data/patch2D.py
# Authors : CM
# Extract and process batches of patches
# Predict slice using batch of patches
#
# ------------------------------------------------------------ #

from random import randint
import math
import itertools

import numpy as np
import skimage as sk


def gen_patches_batch_augmented_2d_bin(patch_size_y, patch_size_x, image, label, batch_size=32):
    batch_image = np.zeros((batch_size, patch_size_y, patch_size_x, 1), dtype=image.dtype)
    batch_label = np.zeros((batch_size, patch_size_y, patch_size_x, 1), dtype=np.float32)
    
    while True:
        for i in range(batch_size):
            x = randint(0, image.shape[2] - patch_size_x)
            y = randint(0, image.shape[1] - patch_size_y)
        
            batch_image[i, :, :, 0] = image[z:z + patch_size_z, y:y + patch_size_y, x:x + patch_size_x]
            batch_label[i, :, :, 0] = label[z:z + patch_size_z, y:y + patch_size_y, x:x + patch_size_x]

            rot = randint(0, 3)
            batch_image[i, :, :] = np.rot90(batch_image[i, :, :], rot)
            batch_label[i, :, :] = np.rot90(batch_label[i, :, :], rot)

            if randint(0, 1) == 1:
                batch_image[i, :, :] = np.fliplr(batch_image[i, :, :])
                batch_label[i, :, :] = np.fliplr(batch_label[i, :, :])

            if randint(0, 1) == 1:
                batch_image[i, :, :] = np.flipud(batch_image[i, :, :])
                batch_label[i, :, :] = np.flipud(batch_label[i, :, :])
        yield batch_image, batch_label

                
def gen_patches_batch_augmented_label_indexes_one_hot(patch_size, image, label, label_indexes, batch_size=32):
    n_label = label[0].shape[-1]
    label_indexes_iter = []
    for cla in range(len(label_indexes)):
        np.random.shuffle(label_indexes[cla])
        label_indexes_iter.append(itertools.cycle(label_indexes[cla]))
    batch_image = np.zeros((batch_size, patch_size, patch_size, 1))
    batch_label = np.zeros((batch_size, patch_size, patch_size, n_label))
    while True:
        for i in range(batch_size):
            # 20% of random patches
            if randint(0,9) < 2:
                x = randint(0, image.shape[2] - patch_size - 1)
                y = randint(0, image.shape[1] - patch_size - 1)
                z = randint(0, image.shape[0] - 1)
            else:
                cla = randint(0, len(label_indexes)-1)

                z, y, x = next(label_indexes_iter[cla])
                y = max(0, y - 1 - (patch_size // 2))
                x = max(0, x - 1 - (patch_size // 2))

                y = min(max(0, y), image.shape[1]-patch_size)
                x = min(max(0, x), image.shape[2]-patch_size)
            
            batch_image[i, :, :, 0] = image[z, y:y + patch_size, x:x + patch_size]
            batch_label[i, :, :, :] = label[z, y:y + patch_size, x:x + patch_size]
            
            # Augmentations
            # random 90 degree rotation
            # random flip
            rot = randint(0, 3)
            batch_image[i, :, :] = np.rot90(batch_image[i, :, :], rot)
            batch_label[i, :, :] = np.rot90(batch_label[i, :, :], rot)
            
            if randint(0, 1) == 1:
                batch_image[i, :, :] = np.fliplr(batch_image[i, :, :])
                batch_label[i, :, :] = np.fliplr(batch_label[i, :, :])
                
            if randint(0, 1) == 1:
                batch_image[i, :, :] = np.flipud(batch_image[i, :, :])
                batch_label[i, :, :] = np.flipud(batch_label[i, :, :])
        
        yield batch_image, batch_label

        
def predict_slice_one_hot(patch_size, image, model):
    current_image = np.expand_dims(image, -1)
    prediction_stride = patch_size // 3
    
    pad_y0 = patch_size
    pad_y1 = patch_size
    pad_x0 = patch_size
    pad_x1 = patch_size
    
    pad_y0 += 0 if (current_image.shape[0] % prediction_stride == 0) else math.floor((prediction_stride - current_image.shape[0] % prediction_stride)/2)
    pad_y1 += 0 if (current_image.shape[0] % prediction_stride == 0) else math.ceil((prediction_stride - current_image.shape[0] % prediction_stride)/2)
    pad_x0 += 0 if (current_image.shape[1] % prediction_stride == 0) else math.floor((prediction_stride - current_image.shape[1] % prediction_stride)/2)
    pad_x1 += 0 if (current_image.shape[1] % prediction_stride == 0) else math.ceil((prediction_stride - current_image.shape[1] % prediction_stride)/2)

    current_image_padded = sk.util.pad(current_image, pad_width=((pad_y0,pad_y1),(pad_x0, pad_x1),(0,0)), mode='symmetric')
    
    batch_size = (current_image_padded.shape[0] // prediction_stride) * (current_image_padded.shape[1] // prediction_stride)
    patches_batch = np.zeros((batch_size,patch_size,patch_size,1))
    prediction_image = np.zeros((current_image_padded.shape[0], current_image_padded.shape[1]), dtype=image.dtype)
        
    # deconstruct
    p = 0
    for y in range(0, current_image_padded.shape[0]-patch_size-1, prediction_stride):
        for x in range(0, current_image_padded.shape[1]-patch_size-1, prediction_stride):
            patches_batch[p, :, :, 0] = current_image_padded[y:y+patch_size,x:x+patch_size,0]
            p = p + 1

    # predict
    prediction_patches_batch = model.predict(patches_batch)
    n_label = prediction_patches_batch.shape[-1]
    patches_batch = np.argmax(prediction_patches_batch, axis=-1)
    
    # reconstruct
    ps = prediction_stride
    p = 0
    for y in range(0, current_image_padded.shape[0]-patch_size-1, prediction_stride):
        for x in range(0, current_image_padded.shape[1]-patch_size-1, prediction_stride):
            prediction_patch = patches_batch[p]
            p = p + 1
            prediction_image[y+ps:y+2*ps,x+ps:x+2*ps] = prediction_patch[ps:2*ps,ps:2*ps]
    
    prediction = prediction_image[pad_y0:pad_y0+current_image.shape[0],pad_x0:pad_x0+current_image.shape[1]]
    
    return prediction

