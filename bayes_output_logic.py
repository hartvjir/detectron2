import torch

from detectron2.engine.defaults import DefaultPredictor

from ipalm.image_utils import ImageInfos

from ipalm.detectron2_to_mobilenet import get_materials_from_patches
from ipalm.utils import *
from ipalm.mapping_utils import material_all_str
from ipalm import mapping_utils
from ipalm.image_to_outputs import image_files2intermediate_data, get_detectron_categories, image2intermediate_data
from ipalm.net import MobileNetV3Large
from ipalm.setup_utils import setup

from typing import List, Tuple, Dict, Union, Iterable
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import matplotlib.patheffects as PathEffects


def get_confidence(probabilities):
    """

    Args:
        probabilities: list of probabilities

    Returns:
        highest probability - second highest probability
    """
    sorted_arr = np.sort(probabilities)[::-1]
    return sorted_arr[0] - sorted_arr[1]


def create_precision_list(confusion_matrix):
    """

    Args:
        confusion_matrix: (N+1) x (N+1) confusion matrix, used classification submatrix: M[1:][1:]

    Returns:
        precision for each class (N-1) precisions
    """
    ret = [0.0 for _ in range(len(confusion_matrix)-1)]
    for i in range(len(confusion_matrix)-1):
        col_sum = sum(confusion_matrix[1:][1+i])
        if col_sum == 0:
            col_sum = 1
        ret[i] = confusion_matrix[i+1][i+1] / col_sum
    return ret


def quick_plot_bboxes(instance_predictions, inp_img_path):
    """Emergency instant plotting. Not quite easy on the eye, but quite easy on the time! ;)
    For now, plotting only the material prediction.

    Args:
        instance_predictions: `dict` of instance predictions in format as in `bayes_output_format.txt`
        inp_img_path: `str` local path to the image

    Returns:
        plt, `matplotlib.pyplot` plot object containing the picture and labeled bounding boxes
    """
    img = mpimg.imread(inp_img_path)
    fig, ax = plt.subplots()
    ax.imshow(img)
    for pred in instance_predictions:
        y1, x1, y2, x2 = pred['bbox']
        w = abs(x1 - x2)
        h = abs(y1 - y2)
        max_prob = 0
        max_name = None
        for matname in pred['material']['names']:
            if pred['material']['prediction'][matname] >= max_prob:
                max_prob = pred['material']['prediction'][matname]
                max_name = matname
        pred_text = "{}, p = {:.2f}".format(max_name, max_prob)
        txt = ax.text(x1, y1, pred_text, color="black", fontsize=10, fontweight='bold')
        txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])

        rec = plt.Rectangle((x1, y1), w, h, facecolor='none', linewidth=1, edgecolor='r')
        ax.add_patch(rec)
    plt.show()
    return plt


class CatmatPredictor:
    """
    Persistent class containing Detectron2 for object detection and classification
        and a MobileNetV3 for material classification so they don't have to be reinitialized
    """
    def __init__(self, threshold, model_path="ipalm/models/model_final.pth", category_sensitivity=0.1,
                       material_model_path="ipalm/models/saved_model.pth",
                       confusion_matrices="ipalm/config/confusion_matrices.json"):
        self.threshold = threshold
        self.model_path = model_path
        cfg = setup(model_path=model_path)
        cfg.MODEL.WEIGHTS = model_path

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.threshold
        self.main_predictor = DefaultPredictor(cfg)
        # detectron2 repurposed for category classification
        self.category_sensitivity = category_sensitivity
        temp_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = category_sensitivity
        self.cat_predictor = DefaultPredictor(cfg)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = temp_thresh

        # mobile net for material classification
        self.material_model_path = material_model_path
        self.material_model = MobileNetV3Large(n_classes=len(material_all_str))
        self.material_model.load_state_dict(torch.load(material_model_path))

        # confusion matrices n shit
        cm_dict = dict()
        with open(confusion_matrices, "r") as f:
            cm_dict = json.load(f)
        self.category_cm = cm_dict["category_confusion_matrix"]
        self.category_precisions = create_precision_list(self.category_cm)
        self.category_precision = sum(self.category_precisions) / len(self.category_precisions)
            
        self.material_cm = cm_dict["material_confusion_matrix"]
        self.material_precisions = create_precision_list(self.material_cm)
        self.material_precision = sum(self.material_precisions) / len(self.material_precisions)


        # bayes format
        self.category_names = mapping_utils.category_ipalm_list
        self.material_names = mapping_utils.material_ipalm_list

    def get_image_boxes(self, src: Union[str, Iterable]) -> List[Dict]:
        """
            List[Dict["initial_bbox"/"category_list"/"material_list"]
        Args:
            src: path or the raw image data to be processed

        Returns:
            list of dictionaries with categories/materials for every bbox from detectron2
        """
        if type(src) == str:
            # from path
            intermediate_data = image_files2intermediate_data(self.main_predictor, [src, ])
        elif type(src) == np.ndarray or type(src) == List or type(src) == Tuple:
            # from array
            intermediate_data = image2intermediate_data(self.main_predictor, [src, ])
        else:
            raise NotImplementedError("other iterables not supported yet")
        im_names = intermediate_data.im_names
        detectron_instances = intermediate_data.outputs  # (number_of_images, number of instances per image)
        mobile_inputs = intermediate_data.inputs.mobile_inputs
        detectron_inputs = intermediate_data.inputs.detectron_inputs
        # ImageInfos are the data storage
        infos = ImageInfos(len(im_names))
        infos.update_with_im_names(im_names)
        infos.update_with_detectron_outputs(detectron_instances)

        # MOBILE NET MATERIAL SECTION
        # shape of `material_outputs`: (number of images, bboxes per image, materials per bbox)
        material_outputs = get_materials_from_patches(mobile_inputs, model=self.material_model)
        infos.update_with_mobile_outputs(material_outputs)

        sensitive_threshold = 0.10
        detectron_categories = get_detectron_categories(self.main_predictor, detectron_inputs, detectron_instances)
        infos.update_with_detectron_categories(detectron_categories)
        # it is a single image:
        info = infos[0]
        image_boxes = list()
        for box in info.box_results:
            tmp = dict()
            tmp["initial_bbox"] = np.array(box.initial_bbox).tolist()
            tmp["category_list"] = np.array(box.category_list).tolist()
            tmp["material_list"] = np.array(box.material_list).tolist()
            image_boxes.append(tmp)
        return image_boxes

    def get_bayes_output(self, src, output_target=None) -> Union[None, List[Dict]]:
        """
        Get dictionary in bayes_output_format.txt format
        Args:
            src: path or the raw image data to be processed
            output_target: None->return bayes dict, your_dict_name.json->writes to the file

        Returns:

        """
        if type(src) == str:
            boxresults = self.get_image_boxes(src)
            bayes_dicts = list()
            for boxresult in boxresults:
                bayes_dict = dict()
                category_dict = dict()
                cat_metrics_dict = dict()
                category_list: List = boxresult["category_list"]
                # Additional: bbox. Same level: "bbox", "category", "material"
                bayes_dict["bbox"] = boxresult["initial_bbox"]
                # "metrics"
                cat_metrics_dict["confidence"] = get_confidence(category_list)
                # precision of the whole classifier
                cat_metrics_dict["precision"] = self.category_precision
                category_dict["metrics"] = cat_metrics_dict
                # "names"
                category_dict["names"] = self.category_names
                # "prediction"
                prediction_dict = dict()
                for i in range(len(self.category_names)):
                    prediction_dict[self.category_names[i]] = category_list[i]
                category_dict["prediction"] = prediction_dict
                # finalize "category"
                bayes_dict["category"] = category_dict

                # material
                material_dict = dict()
                mat_metrics_dict = dict()
                material_list: List = boxresult["material_list"]
                mat_metrics_dict["confidence"] = get_confidence(material_list)
                # precision of the whole classifier
                mat_metrics_dict["precision"] = self.material_precision
                material_dict["metrics"] = mat_metrics_dict
                # "names"
                material_dict["names"] = self.material_names
                # "prediction"
                prediction_dict = dict()
                for i in range(len(self.material_names)):
                    prediction_dict[self.material_names[i]] = material_list[i]
                material_dict["prediction"] = prediction_dict
                # finalize "material"
                bayes_dict["material"] = material_dict

                # append dict to list of dicts for one image
                bayes_dicts.append(bayes_dict)
            if type(output_target) == str:
                with open(output_target, "w") as f:
                    json.dump(bayes_dicts, f)
                return None
            elif output_target is None:
                return bayes_dicts


if __name__ == "__main__":
    megapredictor = CatmatPredictor(0.6, model_path="ipalm/models/model_final.pth")
    input_imgs = ["images_input/" + f for f in listdir("images_input") if isfile(join("images_input", f))]
    for inp_img in input_imgs:
        predictions = megapredictor.get_bayes_output(inp_img)
        quick_plot_bboxes(predictions, inp_img)


