from random import randint
import numpy as np


def check_valid(image, label):
    if type(image) is list and type(label) is list:
        if not len(image) == len(label):
            return False
        for i in range(len(image)):
            if not image[i].shape[:-1] == label[i].shape[:-1] and len(image[i].shape) == 4:
                return False
    elif type(image) is np.ndarray and type(label) is np.ndarray:
        return image.shape[:-1] == label.shape[:-1] and len(image.shape) == 4
    else:
        return False
    return True


def gen_patch_2d_batch(patch_size, image, label, batch_size, augmentation):
    n_channel = image[0].shape[-1]
    n_label = label[0].shape[-1]
    image_dtype = image[0].dtype
    label_dtype = label[0].dtype
    patch_size_y, patch_size_x = patch_size
    
    batch_image = np.zeros((batch_size,) + patch_size + (n_channel,), dtype=image_dtype)
    batch_label = np.zeros((batch_size,) + patch_size + (n_label,), dtype=label_dtype)
    img = image
    lbl = label
    while True:
        for i in range(batch_size):
            if type(image) is list:
                n = randint(0, len(image)-1)
                img = image[n]
                lbl = label[n]
        
            x = randint(0, img.shape[2] - patch_size_x)
            y = randint(0, img.shape[1] - patch_size_y)
            z = randint(0, img.shape[0] - 1)

            batch_image[i, :, :, :] = img[z, y:y + patch_size_y, x:x + patch_size_x, :]
            batch_label[i, :, :, :] = lbl[z, y:y + patch_size_y, x:x + patch_size_x, :]
            
            if augmentation:
                if patch_size_y == patch_size_x:
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


def gen_patch_3d_batch(patch_size, image, label, batch_size, augmentation):
    n_channel = image[0].shape[-1]
    n_label = label[0].shape[-1]
    image_dtype = image[0].dtype
    label_dtype = label[0].dtype
    patch_size_z, patch_size_y, patch_size_x = patch_size
    
    batch_image = np.zeros((batch_size,) + patch_size + (n_channel,), dtype=image_dtype)
    batch_label = np.zeros((batch_size,) + patch_size + (n_label,), dtype=label_dtype)
    img = image
    lbl = label
    while True:
        for i in range(batch_size):
            if type(image) is list:
                n = randint(0, len(image)-1)
                img = image[n]
                lbl = label[n]
        
            x = randint(0, img.shape[2] - patch_size_x)
            y = randint(0, img.shape[1] - patch_size_y)
            z = randint(0, img.shape[0] - patch_size_z)

            batch_image[i, :, :, :, :] = img[z:z + patch_size_z, y:y + patch_size_y, x:x + patch_size_x, :]
            batch_label[i, :, :, :, :] = lbl[z:z + patch_size_z, y:y + patch_size_y, x:x + patch_size_x, :]
            
            if augmentation:
                if patch_size_z == patch_size_x:
                    rot = randint(0, 3)
                    batch_image[i, :, :] = np.rot90(batch_image[i, :, :, :], rot, axes=(0, 2))
                    batch_label[i, :, :] = np.rot90(batch_label[i, :, :, :], rot, axes=(0, 2))
                if patch_size_z == patch_size_y:
                    rot = randint(0, 3)
                    batch_image[i, :, :] = np.rot90(batch_image[i, :, :, :], rot, axes=(0, 1))
                    batch_label[i, :, :] = np.rot90(batch_label[i, :, :, :], rot, axes=(0, 1))
                if patch_size_y == patch_size_x:
                    rot = randint(0, 3)
                    batch_image[i, :, :] = np.rot90(batch_image[i, :, :, :], rot, axes=(1, 2))
                    batch_label[i, :, :] = np.rot90(batch_label[i, :, :, :], rot, axes=(1, 2))

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


def gen_patch_batch(patch_size, image, label, batch_size=32, augmentation=True, label_indexes=None):
    gen = None
    if not (len(patch_size) == 2 or len(patch_size) == 3):
        raise ValueError
    if not check_valid(image, label):
        raise ValueError
    if len(patch_size) == 2:
        if label_indexes is None:
            gen = gen_patch_2d_batch(patch_size, image, label, batch_size, augmentation)
        else:
            raise NotImplementedError
    elif len(patch_size) == 3:
        if label_indexes is None:
            gen = gen_patch_3d_batch(patch_size, image, label, batch_size, augmentation)
        else:
            raise NotImplementedError
    else:
        raise ValueError
    return gen
