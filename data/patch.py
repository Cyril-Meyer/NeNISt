# ------------------------------------------------------------ #
#
# File : data/patch.py
# Authors : CM
# Extract and process batches of patches
# Predict slice using batch of patches
#
# ------------------------------------------------------------ #

from random import randint
import math
import itertools

import higra as hg
import multiprocessing as mp
import numpy as np
import skimage as sk


# Patches

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


# TOS Patches (_p suffix = parallel calculations of filtered patches)

def patches_batch_tos_area(batch_image, batch_size, lambdas):
    for i in range(batch_size):
        tree, altitudes = hg.component_tree_tree_of_shapes_image2d(batch_image[i, :, :, 0])
        area = hg.attribute_area(tree)
        
        for j in range(len(lambdas)):
            batch_image[i, :, :, j+1] = hg.reconstruct_leaf_data(tree, altitudes, area < lambdas[j])
            
    return batch_image


def patches_batch_tos_area_p(batch_image, i, j, l):
    tree, altitudes = hg.component_tree_tree_of_shapes_image2d(batch_image[i, :, :, 0])
    area = hg.attribute_area(tree)
    batch_image[i, :, :, j+1] = hg.reconstruct_leaf_data(tree, altitudes, area < l)
    return batch_image[i, :, :, j+1]


def gen_patches_batch_augmented_tos_area_label_indexes_one_hot_p(patch_size, image, label, label_indexes, batch_size=32, lambdas=[]):
    n_label = label[0].shape[-1]
    label_indexes_iter = []
    for cla in range(len(label_indexes)):
        np.random.shuffle(label_indexes[cla])
        label_indexes_iter.append(itertools.cycle(label_indexes[cla]))
    batch_image = np.zeros((batch_size, patch_size, patch_size, len(lambdas)+1))
    batch_label = np.zeros((batch_size, patch_size, patch_size, n_label))
    pool = mp.Pool(min(mp.cpu_count(), 16))
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
        
        batch_image_p = (pool.starmap(patches_batch_tos_area_p, [(batch_image, ij//len(lambdas), ij%len(lambdas), lambdas[ij%len(lambdas)]) for ij in range(batch_size * len(lambdas))]))
        
        for i in range(batch_size):
            for j in range(len(lambdas)):
                x = j + i*len(lambdas)
                batch_image[i, :, :, j+1] = batch_image_p[x]
        
        del batch_image_p
        
        yield batch_image, batch_label
    pool.close()


def predict_slice_one_hot_tos_area(patch_size, image, model, lambdas=[]):
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
    patches_batch = np.zeros((batch_size,patch_size,patch_size,len(lambdas)+1))
    prediction_image = np.zeros((current_image_padded.shape[0], current_image_padded.shape[1]), dtype=image.dtype)
        
    # deconstruct
    p = 0
    for y in range(0, current_image_padded.shape[0]-patch_size-1, prediction_stride):
        for x in range(0, current_image_padded.shape[1]-patch_size-1, prediction_stride):
            patches_batch[p, :, :, 0] = current_image_padded[y:y+patch_size,x:x+patch_size,0]
            p = p + 1
    
    # tos aera convert
    patches_batch = patches_batch_tos_area(patches_batch, batch_size, lambdas)
    
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


def predict_slice_one_hot_tos_area_p(patch_size, image, model, lambdas=[]):
    pool = mp.Pool(min(mp.cpu_count(), 16))
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
    patches_batch = np.zeros((batch_size,patch_size,patch_size,len(lambdas)+1))
    prediction_image = np.zeros((current_image_padded.shape[0], current_image_padded.shape[1]), dtype=image.dtype)
        
    # deconstruct
    p = 0
    for y in range(0, current_image_padded.shape[0]-patch_size-1, prediction_stride):
        for x in range(0, current_image_padded.shape[1]-patch_size-1, prediction_stride):
            patches_batch[p, :, :, 0] = current_image_padded[y:y+patch_size,x:x+patch_size,0]
            p = p + 1
    
    # tos aera convert
    patches_batch_p = (pool.starmap(patches_batch_tos_area_p, [(patches_batch, ij//len(lambdas), ij%len(lambdas), lambdas[ij%len(lambdas)]) for ij in range(batch_size * len(lambdas))]))
        
    for i in range(batch_size):
        for j in range(len(lambdas)):
            x = j + i*len(lambdas)
            patches_batch[i, :, :, j+1] = patches_batch_p[x]
    
    del patches_batch_p
    
    
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
    
    pool.close()
    
    return prediction


# 2.5D Patches
def gen_patches_batch_augmented_25d_label_indexes_one_hot(patch_size, image, label, label_indexes, batch_size=32, z_slices=3):
    # valid z_slices is odd
    z_d2 = z_slices//2
    n_label = label[0].shape[-1]
    label_indexes_iter = []
    for cla in range(len(label_indexes)):
        label_indexes_iter.append(itertools.cycle(label_indexes[cla]))
    batch_image = np.zeros((batch_size, patch_size, patch_size, z_d2*2+1))
    batch_label = np.zeros((batch_size, patch_size, patch_size, n_label))
             
    while True:
        for i in range(batch_size):
            # 20% of random patches
            if randint(0,9) < 2:
                x = randint(0, image.shape[2] - patch_size - 1)
                y = randint(0, image.shape[1] - patch_size - 1)
                z = randint(z_d2, image.shape[0] - 1 - z_d2)
            else:
                cla = randint(0, len(label_indexes)-1)
                
                z, y, x = next(label_indexes_iter[cla])
                y = max(0, y - 1 - (patch_size // 2))
                x = max(0, x - 1 - (patch_size // 2))

                y = min(max(0, y), image.shape[1]-patch_size)
                x = min(max(0, x), image.shape[2]-patch_size)
                
                z = max(z_d2, z)
                z = min(z, image.shape[0] - 1 - z_d2)
            
            batch_image[i, :, :, z_d2] = image[z, y:y + patch_size, x:x + patch_size]
            for j in range(1, z_d2+1):
                batch_image[i, :, :, z_d2-j] = image[z-j, y:y + patch_size, x:x + patch_size]
                batch_image[i, :, :, z_d2+j] = image[z+j, y:y + patch_size, x:x + patch_size]
                
            batch_label[i, :, :, :] = label[z, y:y + patch_size, x:x + patch_size]
            
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


def predict_slice_one_hot_25d(patch_size, image, model, z_slices):
    # valid z_slices is odd
    z_d2 = z_slices//2
    current_image = np.expand_dims(image, -1)
    prediction_stride = patch_size // 3
    
    pad_y0 = patch_size
    pad_y1 = patch_size
    pad_x0 = patch_size
    pad_x1 = patch_size
    
    pad_y0 += 0 if (current_image[0].shape[0] % prediction_stride == 0) else math.floor((prediction_stride - current_image[0].shape[0] % prediction_stride)/2)
    pad_y1 += 0 if (current_image[0].shape[0] % prediction_stride == 0) else math.ceil((prediction_stride - current_image[0].shape[0] % prediction_stride)/2)
    pad_x0 += 0 if (current_image[0].shape[1] % prediction_stride == 0) else math.floor((prediction_stride - current_image[0].shape[1] % prediction_stride)/2)
    pad_x1 += 0 if (current_image[0].shape[1] % prediction_stride == 0) else math.ceil((prediction_stride - current_image[0].shape[1] % prediction_stride)/2)
    
    current_image_padded = np.zeros((z_slices, current_image[0].shape[0]+pad_y0+pad_y1, current_image[0].shape[1]+pad_x0+pad_x1,1))
    
    for z in range(z_slices):
        current_image_padded[z] = skimage.util.pad(current_image[z], pad_width=((pad_y0,pad_y1),(pad_x0, pad_x1),(0,0)), mode='symmetric')
    
    batch_size = (current_image_padded.shape[1] // prediction_stride) * (current_image_padded.shape[2] // prediction_stride)
    patches_batch = np.zeros((batch_size,patch_size,patch_size,z_d2*2+1))
    prediction_image = np.zeros((current_image_padded.shape[1], current_image_padded.shape[2]), dtype=image.dtype)
        
    # deconstruct
    p = 0
    for y in range(0, current_image_padded.shape[1]-patch_size-1, prediction_stride):
        for x in range(0, current_image_padded.shape[2]-patch_size-1, prediction_stride):
            patches_batch[p, :, :, z_d2] = current_image_padded[z_d2, y:y + patch_size, x:x + patch_size, 0]
            for j in range(1, z_d2+1):
                patches_batch[p, :, :, z_d2-j] = current_image_padded[z_d2-j, y:y + patch_size, x:x + patch_size, 0]
                patches_batch[p, :, :, z_d2+j] = current_image_padded[z_d2+j, y:y + patch_size, x:x + patch_size, 0]
            p = p + 1
    
    # predict
    prediction_patches_batch = model.predict(patches_batch)
    n_label = prediction_patches_batch.shape[-1]
    patches_batch = np.argmax(prediction_patches_batch, axis=-1)
    
    # reconstruct
    ps = prediction_stride
    p = 0
    for y in range(0, current_image_padded.shape[1]-patch_size-1, prediction_stride):
        for x in range(0, current_image_padded.shape[2]-patch_size-1, prediction_stride):
            prediction_patch = patches_batch[p]
            p = p + 1
            prediction_image[y+ps:y+2*ps,x+ps:x+2*ps] = prediction_patch[ps:2*ps,ps:2*ps]
    
    prediction = prediction_image[pad_y0:pad_y0+current_image.shape[1],pad_x0:pad_x0+current_image.shape[2]]
    
    return prediction
