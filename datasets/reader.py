import numpy as np
import tifffile


def get_data(images_paths, labels_paths, normalize=255.0, create_background=False):
    images = []
    labels = []
    assert len(images_paths) == len(labels_paths)

    for i in range(len(images_paths)):
        images.append(np.expand_dims(np.array(tifffile.imread(images_paths[i]), dtype=np.float32) / normalize, axis=-1))
        if create_background:
            assert len(labels_paths[i]) > 1
            tmp = np.moveaxis(np.array(tifffile.imread(labels_paths[i]), dtype=np.float16), 0, -1)
            labels.append(np.concatenate([np.expand_dims(1 - np.max(tmp, axis=-1), -1), tmp], axis=-1))
        else:
            if len(labels_paths[i]) > 1:
                labels.append(np.moveaxis(np.array(tifffile.imread(labels_paths[i]), dtype=np.float16), 0, -1))
            else:
                labels.append(np.expand_dims(np.array(tifffile.imread(labels_paths[i]), dtype=np.float16), -1))

    return images, labels


def get_labels_id(labels):
    labels_id = []
    for i in range(len(labels)):
        tmp = []
        for cla in range(labels[i].shape[-1]):
            tmp.append(np.argwhere(labels[i][..., cla] == 1))
        labels_id.append(tmp)
    return labels_id
