import numpy as np

def complile_results(gt, bboxes, dist_threshold):

    l = len(gt)
    new_distances = np.zeros(l)
    new_ious = np.zeros(l)
    n_threshold = 50
    precisions_ths = np.zeros(n_threshold)
    robustness = 0

    for i in range(l):
        new_distances[i] = compute_distance(gt[i], bboxes[i])
        new_ious[i] = compute_iou(bboxes[i], gt[i])

    percision = sum(new_distances < dist_threshold) / l
    
    thresholds = np.linspace(0, 25, n_threshold + 1)
    thresholds = thresholds[-n_threshold:]
    thresholds = thresholds[::-1]

    for i in range(n_threshold):
        precisions_ths[i] = sum(new_distances < thresholds[i]) / l

    percision_auc = np.trapz(precisions_ths)
    iou = np.mean(new_ious) * 100

    return l, percision, percision_auc, iou
    

def compute_distance(boxA, boxB):
    a = np.array([boxA[0] + boxA[2]/2, boxA[1] + boxA[3]/2])
    b = np.array([boxB[0] + boxB[2]/2, boxB[1] + boxB[3]/2])
    
    dist = np.linalg.norm((a - b))
    return dist


def compute_iou(boxA, boxB):
    left = max(boxA[0], boxB[0])
    right = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    top = max(boxA[1], boxB[1])
    bound = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    if left < right and top < bound:
        interArea = (right - left) * (bound - top)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]

        iou = interArea / float(boxAArea + boxBArea - interArea)
    else:
        iou  = 0

    return iou
