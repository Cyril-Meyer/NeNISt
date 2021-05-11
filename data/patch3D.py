# ------------------------------------------------------------ #
#
# File : data/patch3D.py
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


def gen_patches_batch_augmented_3d_bin(patch_size_z, patch_size_y, patch_size_x, image, label, batch_size=32, weights=None):
    batch_image = np.zeros((batch_size, patch_size_z, patch_size_y, patch_size_x, 1), dtype=image.dtype)
    batch_label = np.zeros((batch_size, patch_size_z, patch_size_y, patch_size_x, 1), dtype=np.float32)
    
    while True:
        for i in range(batch_size):
            x = randint(0, image.shape[2] - patch_size_x)
            y = randint(0, image.shape[1] - patch_size_y)
            z = randint(0, image.shape[0] - patch_size_z)
            
            batch_image[i, :, :, :, 0] = image[z:z + patch_size_z, y:y + patch_size_y, x:x + patch_size_x]
            batch_label[i, :, :, :, 0] = label[z:z + patch_size_z, y:y + patch_size_y, x:x + patch_size_x]

            rot = randint(0, 3)
            batch_image[i, :, :, :] = np.rot90(batch_image[i, :, :, :], rot, axes=(1, 2))
            batch_label[i, :, :, :] = np.rot90(batch_label[i, :, :, :], rot, axes=(1, 2))

            if randint(0, 1) == 1:
                batch_image[i, :, :, :] = np.flip(batch_image[i, :, :, :], 0)
                batch_label[i, :, :, :] = np.flip(batch_label[i, :, :, :], 0)

            if randint(0, 1) == 1:
                batch_image[i, :, :, :] = np.flip(batch_image[i, :, :, :], 1)
                batch_label[i, :, :, :] = np.flip(batch_label[i, :, :, :], 1)

            if randint(0, 1) == 1:
                batch_image[i, :, :, :] = np.flip(batch_image[i, :, :, :], 2)
                batch_label[i, :, :, :] = np.flip(batch_label[i, :, :, :], 2)
        
        if weights is None:
            yield batch_image, batch_label
        else:
            batch_weights = (batch_label*(weights[1]-weights[0])) + weights[0]
            yield batch_image, batch_label, batch_weights


def gen_patches_batch_augmented_3d_bin_nochan(patch_size_x, patch_size_y, patch_size_z, image, label, batch_size=32, weights=None):
    batch_image = np.zeros((batch_size, patch_size_x, patch_size_y, patch_size_z), dtype=image.dtype)
    batch_label = np.zeros((batch_size, patch_size_x, patch_size_y, patch_size_z), dtype=np.float32)
    
    while True:
        for i in range(batch_size):
            x = randint(0, image.shape[0] - patch_size_x)
            y = randint(0, image.shape[1] - patch_size_y)
            z = randint(0, image.shape[2] - patch_size_z)
            
            batch_image[i, :, :, :] = image[x:x + patch_size_x, y:y + patch_size_y, z:z + patch_size_z]
            batch_label[i, :, :, :] = label[x:x + patch_size_x, y:y + patch_size_y, z:z + patch_size_z]

            rot = randint(0, 3)
            batch_image[i, :, :, :] = np.rot90(batch_image[i, :, :, :], rot, axes=(1, 2))
            batch_label[i, :, :, :] = np.rot90(batch_label[i, :, :, :], rot, axes=(1, 2))

            if randint(0, 1) == 1:
                batch_image[i, :, :, :] = np.flip(batch_image[i, :, :, :], 0)
                batch_label[i, :, :, :] = np.flip(batch_label[i, :, :, :], 0)

            if randint(0, 1) == 1:
                batch_image[i, :, :, :] = np.flip(batch_image[i, :, :, :], 1)
                batch_label[i, :, :, :] = np.flip(batch_label[i, :, :, :], 1)

            if randint(0, 1) == 1:
                batch_image[i, :, :, :] = np.flip(batch_image[i, :, :, :], 2)
                batch_label[i, :, :, :] = np.flip(batch_label[i, :, :, :], 2)

        if weights is None:
            yield batch_image, batch_label
        else:
            batch_weights = (batch_label*(weights[1]-weights[0])) + weights[0]
            yield batch_image, batch_label, batch_weights


def gen_patches_batch_augmented_3d_label_indexes_one_hot(patch_size_z, patch_size_y, patch_size_x, image, label, label_indexes, batch_size=32):
    n_label = label[0].shape[-1]
    label_indexes_iter = []
    for cla in range(len(label_indexes)):
        label_indexes_iter.append(itertools.cycle(label_indexes[cla]))
    batch_image = np.zeros((batch_size, patch_size_z, patch_size_y, patch_size_x, 1))
    batch_label = np.zeros((batch_size, patch_size_z, patch_size_y, patch_size_x, n_label))
             
    while True:
        for i in range(batch_size):
            # 20% of random patches
            if randint(0,9) < 2:
                x = randint(0, image.shape[2] - patch_size_x - 1)
                y = randint(0, image.shape[1] - patch_size_y - 1)
                z = randint(0, image.shape[0] - patch_size_z)
            else:
                cla = randint(0, len(label_indexes)-1)
                
                z, y, x = next(label_indexes_iter[cla])
                
                # re center the patch on the interesting data
                x = max(0, x - 1 - (patch_size_x // 2))
                y = max(0, y - 1 - (patch_size_y // 2))
                z = max(0, z - 1 - (patch_size_z // 2))

                x = min(max(0, x), image.shape[2]-patch_size_x)
                y = min(max(0, y), image.shape[1]-patch_size_y)
                z = min(max(0, z), image.shape[0]-patch_size_z)

            batch_image[i, :, :, :, 0] = image[z:z + patch_size_z, y:y + patch_size_y, x:x + patch_size_x]
            batch_label[i, :, :, :, :] = label[z:z + patch_size_z, y:y + patch_size_y, x:x + patch_size_x]
            
            rot = randint(0, 3)
            batch_image[i, :, :, :] = np.rot90(batch_image[i, :, :, :], rot, axes=(1, 2))
            batch_label[i, :, :, :] = np.rot90(batch_label[i, :, :, :], rot, axes=(1, 2))
            
            if randint(0, 1) == 1:
                batch_image[i, :, :, :] = np.flip(batch_image[i, :, :, :], 0)
                batch_label[i, :, :, :] = np.flip(batch_label[i, :, :, :], 0)
                
            if randint(0, 1) == 1:
                batch_image[i, :, :, :] = np.flip(batch_image[i, :, :, :], 1)
                batch_label[i, :, :, :] = np.flip(batch_label[i, :, :, :], 1)
                
            if randint(0, 1) == 1:
                batch_image[i, :, :, :] = np.flip(batch_image[i, :, :, :], 2)
                batch_label[i, :, :, :] = np.flip(batch_label[i, :, :, :], 2)
            
        yield batch_image, batch_label


def gen_patches_batch_augmented_3d(patch_size_z, patch_size_y, patch_size_x, n_classes, image, label, batch_size=32, weights=None):
    batch_image = np.zeros((batch_size, patch_size_z, patch_size_y, patch_size_x, 1), dtype=image.dtype)
    batch_label = np.zeros((batch_size, patch_size_z, patch_size_y, patch_size_x, n_classes), dtype=np.float32)
    
    while True:
        for i in range(batch_size):
            x = randint(0, image.shape[2] - patch_size_x)
            y = randint(0, image.shape[1] - patch_size_y)
            z = randint(0, image.shape[0] - patch_size_z)
            
            batch_image[i, :, :, :, 0] = image[z:z + patch_size_z, y:y + patch_size_y, x:x + patch_size_x]
            batch_label[i, :, :, :, :] = label[z:z + patch_size_z, y:y + patch_size_y, x:x + patch_size_x]

            rot = randint(0, 3)
            batch_image[i, :, :, :] = np.rot90(batch_image[i, :, :, :], rot, axes=(1, 2))
            batch_label[i, :, :, :] = np.rot90(batch_label[i, :, :, :], rot, axes=(1, 2))

            if randint(0, 1) == 1:
                batch_image[i, :, :, :] = np.flip(batch_image[i, :, :, :], 0)
                batch_label[i, :, :, :] = np.flip(batch_label[i, :, :, :], 0)

            if randint(0, 1) == 1:
                batch_image[i, :, :, :] = np.flip(batch_image[i, :, :, :], 1)
                batch_label[i, :, :, :] = np.flip(batch_label[i, :, :, :], 1)

            if randint(0, 1) == 1:
                batch_image[i, :, :, :] = np.flip(batch_image[i, :, :, :], 2)
                batch_label[i, :, :, :] = np.flip(batch_label[i, :, :, :], 2)
        
        if weights is None:
            yield batch_image, batch_label
        else:
            batch_weights = (batch_label*(weights[1]-weights[0])) + weights[0]
            yield batch_image, batch_label, batch_weights
