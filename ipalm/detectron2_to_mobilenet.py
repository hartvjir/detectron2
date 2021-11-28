import argparse
import os

from PIL import Image
import numpy as np
import torch
from typing import Tuple, List

from .dataset_loader import MincDataset, MincLoader, get_free_gpu
# from .material_utils import material_str2raw_id, material_raw_id2str, material_ipalm_ignore, material_all_str
from .mapping_utils import *
from .net import MobileNetV3Large

from .test import *


__all__ = ["get_materials_from_patches", "get_materials_from_patch"]


def get_materials_from_patches(patches_list, model=None) -> Tuple[Tuple[Tuple[Tuple[int, float]]], ...]:
    # list[image: list[patch:list[tuple[int,float]]], image: list[patch:list[tuple[int,float]]], ...]
    """

    Args:
        patches_list: list of N images, [N, H, W, C]
        model: mobilenet model.

    Returns:
        (
        i:0  ((material_id, probability), (material_id, probability), ...),
        i:1  ((material_id, probability), (material_id, probability), ...),
        ...
        )
    """
    # ignore non-ipalm materials:
    ipalm_ids = [i for i in range(len(material_all_str)) if material_all_str[i] not in material_ipalm_ignore]
    model_path = "/home/robot3/ipalm-vision-github/detectron2/ipalm/models/saved_model.pth"
    if model is None:
        model = MobileNetV3Large(n_classes=len(material_all_str))
        model.load_state_dict(torch.load(model_path))
    if torch.cuda.is_available():
        model = model.cuda()
    else:
        print("using cpu!")
    model.eval()     # Optional when not using Model Specific layer
    # print(ipalm_ids)
    per_image_materials = list()
    for k, patche_batch in enumerate(patches_list):
        # print(type(image))
        # N, H, W, C
        materials = list()
        for patch in patche_batch:
            material = get_materials_from_patch(model, patch, ipalm_ids)
            materials.append(material)
        per_image_materials.append(tuple(materials))
    return tuple(per_image_materials)


def get_materials_from_patch(model, image, selected_material_ids) -> Tuple[Tuple[int, float]]:
    """

    Args:
        model: classifier network w/ 23 inputs
        image: 362x362 image
        selected_material_ids: list of ids in [0,22] from which to calculate probabilities

    Returns:
        ((material_id, probability), (material_id, probability), ...)
    """
    # print(image.shape)
    data = np.transpose(np.array(image).astype('f4'), (2, 0, 1)) / 255.0
    data = torch.from_numpy(data)
    data.unsqueeze_(0)
    if torch.cuda.is_available():
        data = data.cuda()
    target = model(data)
    probs = get_probabilities_from_selection(target, selected_material_ids)
    class_probs = tuple(i for i in zip(map_from_selection(selected_material_ids, material_raw_id2str), to_numpy_cpu(probs)))
    return tuple(get_classes_above_threshold(class_probs))

