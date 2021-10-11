# ------------------------------------------------------------ #
#
# File : data/patchMultipleOutputs.py
# Authors : CM
# Extract and process batches of patches
#
# ------------------------------------------------------------ #

from random import randint
import numpy as np

def gen_2d(patch_size, images, labels, batch_size=32):
    if not len(images) == len(labels):
        return
    patch_size_y, patch_size_x = patch_size
    n_images = len(images)
    n_classes = labels[0].shape[-1]
    
    batch_image = np.zeros((batch_size, patch_size_y, patch_size_x, 1), dtype=images[0].dtype)
    batch_label = np.zeros((batch_size, patch_size_y, patch_size_x, n_classes), dtype=labels[0].dtype)
    
    while True:
        for i in range(batch_size):
            imgid = randint(0, n_images-1)
            image = images[imgid]
            label = labels[imgid]
            
            x = randint(0, image.shape[2] - patch_size_x)
            y = randint(0, image.shape[1] - patch_size_y)
            z = randint(0, image.shape[0] - 1)
            
            batch_image[i, :, :, 0] = image[z, y:y + patch_size_y, x:x + patch_size_x]
            batch_label[i, :, :, :] = label[z, y:y + patch_size_y, x:x + patch_size_x]

            rot = randint(0, 3)
            batch_image[i, :, :] = np.rot90(batch_image[i, :, :], rot, axes=(0, 1))
            batch_label[i, :, :] = np.rot90(batch_label[i, :, :], rot, axes=(0, 1))

            if randint(0, 1) == 1:
                batch_image[i, :, :] = np.flip(batch_image[i, :, :], 0)
                batch_label[i, :, :] = np.flip(batch_label[i, :, :], 0)

            if randint(0, 1) == 1:
                batch_image[i, :, :] = np.flip(batch_image[i, :, :], 1)
                batch_label[i, :, :] = np.flip(batch_label[i, :, :], 1)
            
            batch_multi_label = []
            for c in range(n_classes):
                batch_multi_label.append(batch_label[:, :, :, c:c+1])
        
        yield batch_image, batch_multi_label


def gen_3d(patch_size, images, labels, batch_size=32):
    if not len(images) == len(labels):
        return
    patch_size_z, patch_size_y, patch_size_x = patch_size
    n_images = len(images)
    n_classes = labels[0].shape[-1]
    
    batch_image = np.zeros((batch_size, patch_size_z, patch_size_y, patch_size_x, 1), dtype=images[0].dtype)
    batch_label = np.zeros((batch_size, patch_size_z, patch_size_y, patch_size_x, labels[0].shape[-1]), dtype=labels[0].dtype)
    
    while True:
        for i in range(batch_size):
            imgid = randint(0, n_images-1)
            image = images[imgid]
            label = labels[imgid]
            
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
        
            batch_multi_label = []
            for c in range(n_classes):
                batch_multi_label.append(batch_label[:, :, :, :, c:c+1])
        
        yield batch_image, batch_multi_label
