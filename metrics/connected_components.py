import numpy as np
import skimage as sk
import skimage.measure

def connected_components_detection(seg1, seg2, minimum_recall_tp=0.80):
    # results
    tp = 0
    fn = 0
    under_detected = 0

    if not (seg1.shape == seg2.shape):
        print("ERROR: segmentation shape not equal")
        return 1

    # The argument 'neighbors' is deprecated, use 'connectivity' instead. For neighbors=8, use connectivity=2.
    seg1_labels = skimage.measure.label(seg1, connectivity=2)
    seg1_n_cc = seg1_labels.max()

    for cc_label in range(1, seg1_n_cc + 1):
        # the current connected component
        current_cc = ((seg1_labels == cc_label) * 1).astype(np.uint8)
        # the intersection
        intersected_cc = current_cc * seg2
        
        recall = np.sum(current_cc * intersected_cc) / np.sum(current_cc)
        
        if np.sum(intersected_cc) == 0 or recall < 0.10:
            fn += 1
        elif recall > minimum_recall_tp:
            tp += 1
        else:
            under_detected += 1
    
    return (seg1_n_cc, tp, fn, under_detected)


def print_connected_components_detection(res):
    cc, tp, fn, ud = res
    print("(/"+str(cc)+") & "+str(round((tp/cc)*100, 1))+"\\% ("+str(tp)+") & "+str(round((fn/cc)*100, 1))+"\\% ("+str(fn)+") & "+str(round((ud/cc)*100, 1))+"\\% ("+str(ud)+") \\\\")
