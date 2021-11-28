import numpy as np
import json
from typing import Tuple, Union, List

from .mapping_utils import *


def get_category_weights_from_csb(classes, scores, bboxes, raw):
    """
    Count the number of pixels of each class in classes weighted by their scores/probabilities.
    Args:
        classes: list of categories
        scores: list of probabilities [0,1] for each category
        bboxes: list of 4 tuples for each category

    Returns:
        list of [class0: bbox0 area * prob0, class1: bbox1 area * prob1, ...]
    """
    ret = np.array([0 for _ in range(len(shortened_id_to_str))])
    if raw:
        for c, s, b in zip(classes, scores, bboxes):
            ret[raw_id_to_shortened_id[c]] = get_weight_from_bbox_score(b, s)
    else:
        for c, s, b in zip(classes, scores, bboxes):
            ret[c] = get_weight_from_bbox_score(b, s)
    return ret


# def get_category_points_from_csb(classes, scores, bboxes):
#     """
#
#     Args:
#         classes: category-only detectron output (0-35)
#         scores: scores for each classes[i] [0,1]
#         bboxes: 4 tuple (x1,y1,x2,y2) for each classes[i]
#
#     Returns:
#         num_categories(=36) tuple with normalized score for each class
#     """
#     ret = np.array([0 for _ in range(len(shortened_id_to_str))])
#     for c, s, b in zip(classes, scores, bboxes):
#         ret[raw_id_to_shortened_id[c]] = get_weight_from_bbox_score(b, s)
#     if len(classes) > 0:
#         ret = ret / sum(ret)
#     return ret


def get_weight_from_bbox_score(bbox: Union[Tuple, np.ndarray], score: float):
    """
    Area of bbox * probability
    Args:
        bbox: 4tuple bbox
        score: probability from instance detection

    Returns:
        Area of bbox * probability
    """
    return (bbox[2] - bbox[0])*(bbox[3] - bbox[1])*score


def gpu_to_numpy(item):
    return item.cpu().numpy()


if __name__ == "__main__":
    with open("config/id_to_OWN.json", "r") as f:
        id_dictionary = json.load(f)
        vals = id_dictionary.values()


