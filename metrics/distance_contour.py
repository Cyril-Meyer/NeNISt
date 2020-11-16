import numpy as np
import scipy
import skimage as sk
import skimage.morphology


def internal_morphological_gradient(seg):
    return (seg*1.0 - skimage.morphology.erosion(seg*1.0, skimage.morphology.disk(1)))


to_contour = internal_morphological_gradient
to_distancemap = scipy.ndimage.distance_transform_edt


def distances_contour_distancemap(contour, distancemap):
    c = np.argwhere(contour==1)
    distances = []
    for x, y in c:
        distances.append(distancemap[x, y])
    return distances


def stats_distances_contour_distancemap(contour, distancemap):
    c = np.argwhere(contour==1)
    distances = []
    for x, y in c:
        distances.append(distancemap[x, y])
    distances = np.array(distances)
    return distances.mean(), distances.std(), distances.max()


def distance_contour_segmentation_2D(seg1, seg2):
    contour_1 = internal_morphological_gradient(seg1*1.0)
    contour_2 = internal_morphological_gradient(seg2*1.0)
    distance_map_1 = scipy.ndimage.distance_transform_edt(1-contour_1)
    distance_map_2 = scipy.ndimage.distance_transform_edt(1-contour_2)
    
    distances_1 = np.array(distances_contour_distancemap(contour_1, distance_map_2))
    distances_2 = np.array(distances_contour_distancemap(contour_2, distance_map_1))
    
    return (distances_1.mean(), distances_1.std(), distances_1.max()), (distances_2.mean(), distances_2.std(), distances_2.max())


def distance_contour_segmentation_slices(seg1, seg2):
    distances_1 = []
    distances_2 = []
    for z in range(seg1.shape[0]):
        contour_1 = to_contour(seg1[z]*1.0)
        contour_2 = to_contour(seg2[z]*1.0)
        distance_map_1 = scipy.ndimage.distance_transform_edt(1-contour_1)
        distance_map_2 = scipy.ndimage.distance_transform_edt(1-contour_2)

        distances_1 += distances_contour_distancemap(contour_1, distance_map_2)
        distances_2 += distances_contour_distancemap(contour_2, distance_map_1)
    distances_1 = np.array(distances_1)
    distances_2 = np.array(distances_2)
    return (distances_1.mean(), distances_1.std(), distances_1.max()), (distances_2.mean(), distances_2.std(), distances_2.max())


def print_md_distance_contour_segmentation(res):
    (seg1_mean, seg1_std, seg1_max), (seg2_mean, seg2_std, seg2_max) = res
    print("| mean     |", str(round(seg1_mean, 4)).ljust(16), "|", str(round(seg2_mean, 4)).ljust(16), "|")
    print("| std      |", str(round(seg1_std, 4)).ljust(16), "|", str(round(seg2_std, 4)).ljust(16), "|")
    print("| max      |", str(round(seg1_max, 4)).ljust(16), "|", str(round(seg2_max, 4)).ljust(16), "|")
