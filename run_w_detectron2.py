import os
from detectron2.engine.defaults import DefaultPredictor

from ipalm.image_utils import ImageInfos
from ipalm.intermediate_data import IntermediateOutputs
from train import setup

from ipalm.detectron2_to_mobilenet import get_materials_from_patches
from ipalm.utils import *
from ipalm.image_to_outputs import image_files2intermediate_data


def main():
    pred_dir = "images_input"
    image_names = []
    for (dirpath, dirnames, filenames) in os.walk(pred_dir):
        image_names.extend(filenames)
        break
    # choose the certainity threshold
    THRESHOLD = 0.6
    cfg = setup()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = THRESHOLD
    cfg.DATASETS.TEST = (
        "/local/temporary/DATASET/TRAIN",)  # must be a tuple, for the inner workings of detectron2 (stupid, we know :/)
    predictor = DefaultPredictor(cfg)

    # DETECTRON INSTANCE SECTION
    real_image_names = [pred_dir+"/"+im for im in image_names]
    intermediate_data = image_files2intermediate_data(predictor, real_image_names)
    # `im_files`: list of names of input images. Shape: (len(im_files), )
    # `mobile_inputs`: imgs to be fed into mobilenet. Shape: (imlen, number of predicted bboxes, *img_dims)
    # `detectron_outputs`: list of standard detectron outputs. Shape: (len(im_files), )
    im_names = intermediate_data.im_names
    detectron_instances = intermediate_data.outputs  # (number_of_images, number of instances per image)
    mobile_inputs = intermediate_data.inputs.mobile_inputs
    detectron_inputs = intermediate_data.inputs.detectron_inputs
    infos = ImageInfos(len(im_names))
    infos.update_with_im_names(im_names)
    infos.update_with_detectron_outputs(detectron_instances)
    print("Done: detectron instance detection")

    # MOBILE NET MATERIAL SECTION
    # shape of `material_outputs`: (number of images, bboxes per image, materials per bbox)
    material_outputs = get_materials_from_patches(mobile_inputs)
    infos.update_with_mobile_outputs(material_outputs)
    print("Done: material classification")

    sensitive_threshold = 0.10
    detectron_categories = get_detectron_categories(cfg, sensitive_threshold, detectron_inputs, detectron_instances)
    infos.update_with_detectron_categories(detectron_categories)
    print("Done: category classification")
    # print(infos)  # huge print
    dict_to_save = dict()
    for info in infos:
        image_boxes = list()
        for box in info.box_results:
            tmp = dict()
            tmp["initial_bbox"] = np.array(box.initial_bbox).tolist()
            tmp["category_list"] = np.array(box.category_list).tolist()
            tmp["material_list"] = np.array(box.material_list).tolist()
            image_boxes.append(tmp)
        dict_to_save[info.name] = image_boxes

    with open("deez_outputs.json", "w") as f:
        json.dump(dict_to_save, f)


def get_detectron_categories(cfg, sensitivity, intermediate_outputs, detectron_instances: IntermediateOutputs) -> np.ndarray:
    temp_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = sensitivity
    predictor = DefaultPredictor(cfg)
    detectron_categories = list()
    for k, (ims, detectron_output) in enumerate(zip(intermediate_outputs, detectron_instances)):
        per_im_classes = list()
        for im, instance in zip(ims, detectron_output):
            # detectron used as classifier
            outputs = predictor(im)
            instances = outputs["instances"]
            classes = gpu_to_numpy(instances.pred_classes)
            scores = gpu_to_numpy(instances.scores)
            bboxes = gpu_to_numpy(instances.pred_boxes.tensor)
            class_weights = get_category_weights_from_csb(classes, scores, bboxes, raw=True)
            # originally predicted by instance detection
            shortened_csb = [[instance.category, ], [instance.score, ], [instance.bbox, ]]
            original_instance_weights = get_category_weights_from_csb(*shortened_csb, raw=False)
            class_points = class_weights + original_instance_weights
            class_points = class_points / sum(class_points)
            per_im_classes.append(np.array(class_points))

        detectron_categories.append(np.array(per_im_classes))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = temp_thresh
    return np.array(detectron_categories)


if __name__ == "__main__":
    main()




