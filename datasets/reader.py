import numpy as np
import tifffile


def get_data(images_paths, labels_paths, normalize=255.0):
    images = []
    labels = []
    assert len(images_paths) == len(labels_paths)

    for i in range(len(images_paths)):
        images.append(np.array(tifffile.imread(images_paths[i]), dtype=np.float32) / normalize)
        labels.append(np.array(tifffile.imread(labels_paths[i]), dtype=np.uint8))

    return images, labels
