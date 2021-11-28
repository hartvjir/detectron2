import numpy as np
from typing import Tuple, Union, List

from .intermediate_data import IntermediateOutputs, SimpleInstance
# from .utils import shortened_id_to_str
# from .material_utils import *
from .mapping_utils import *


class BboxResult:
    def __init__(self):
        self.im_name = None
        self.initial_category = None
        self.initial_score = None
        self.initial_bbox = None
        self.category_list = None
        self.material_list = None

    def __str__(self):
        ret = ""
        ret += "    initial category: " + str(shortened_id_to_str[self.initial_category])
        ret += "\n    initial score: " + str(round(self.initial_score, 4))
        ret += "\n    initial bbox: " + str([int(i) for i in self.initial_bbox])
        return ret


class ImageInfo:
    def __init__(self):
        self.name: str = ""
        # shortened parameters: len
        self.box_results: List[BboxResult] = list()

    def __repr__(self):
        # image level
        ret = "\nname: "+self.name+"\n"
        for i in range(len(self.box_results)):
            # instance level
            temp = ""
            temp += "\n  instance:"
            temp += "\n" + str(self.box_results[i])
            temp += "\n    categories:"
            # categories in given box
            for j in range(len(self.box_results[i].category_list)):
                tmp = ""
                if self.box_results[i].category_list[j] > 0.0:
                    tmp += "\n      " + str(shortened_id_to_str[j]) + ": " + str(round(self.box_results[i].category_list[j], 4))
                temp += tmp
            temp += "\n    materials: "
            # materials in given box
            for j in range(len(self.box_results[i].material_list)):
                tmp = ""
                if self.box_results[i].material_list[j] > 0.0:
                    tmp += "\n      " + str(material_ipalm_id2str[j]) + ": " + str(round(self.box_results[i].material_list[j], 4))
                temp += tmp
            ret += temp + "\n  "
        return ret


class ImageInfos:
    def __init__(self, num):
        self.infos: List[ImageInfo] = list()
        for i in range(num):
            self.infos.append(ImageInfo())

    def update_with_im_names(self, im_names):
        for k, im_name in enumerate(im_names):
            self.infos[k].name = im_name

    def update_with_detectron_outputs(self, simple_outputs: IntermediateOutputs):
        for info, output in zip(self.infos, simple_outputs):
            for k, per_box_output in enumerate(output):
                # per_box_output: SimpleInstance: bbox, score, category
                if len(info.box_results) == k:
                    info.box_results.append(BboxResult())
                tmp_box_result = info.box_results[k]
                tmp_box_result.im_name = info.name
                tmp_box_result.initial_bbox = per_box_output.bbox  # iterable
                tmp_box_result.initial_score = per_box_output.score
                tmp_box_result.initial_category = per_box_output.category
                # REMAINING:
                # tmp_box_result.material_list -> update_with_mobile_outputs
                # tmp_box_result.category_list -> update_with_detectron_categories

    def update_with_mobile_outputs(self, material_outputs):
        # material_outputs: (number of images, number of boxes in image, number of materials > 0.1 in box)
        for info, material_output in zip(self.infos, material_outputs):
            # material_output : (instances, materials in bbox, 2tuple)
            # material_output : image
            for i in range(len(material_output)):
                # i-th instance in image
                # (str, float)
                tmp_box_result = info.box_results[i]
                instance_materials = [0 for _ in range(len(material_ipalm2raw))]
                for material_name, probability in material_output[i]:
                    # raw id => ipalm id
                    instance_materials[material_str2ipalm_id[material_name]] = probability
                instance_materials = instance_materials / np.sum(instance_materials)
                tmp_box_result.material_list = instance_materials
                # REMAINING:
                # tmp_box_result.category_list -> update_with_detectron_categories

    def update_with_detectron_categories(self, detectron_classes):
        """

        Args:
            detectron_classes: 3D array. Shape: (num_input_ims, instances_per_image, num of categories (36))
        """
        for info, categories in zip(self.infos, detectron_classes):
            for i in range(len(categories)):
                tmp_box_result = info.box_results[i]
                tmp_box_result.category_list = categories[i]

    def __getitem__(self, item):
        return self.infos[item]

    def __iter__(self):
        yield from self.infos

    def __str__(self):
        num_of_categories = len(shortened_id_to_str)
        num_of_materials = len(material_ipalm_id2str)
        ret = ""
        str_cats = str([shortened_id_to_str[i] for i in range(num_of_categories)])
        ret += "number of categories: " + str(num_of_categories)
        ret += "\n  List: " + str_cats
        str_mats = str([material_ipalm_id2str[i] for i in range(num_of_materials)])
        ret += "\nnumber of materials: " + str(num_of_materials)
        ret += "\n  List: " + str_mats + "\n"
        for info in self:
            ret += str(info)
        return ret




