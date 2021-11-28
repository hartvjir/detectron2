import os
import cv2
import pickle

from ipalm.intermediate_data import *
from ipalm.utils import gpu_to_numpy, get_category_weights_from_csb
from train import setup
from detectron2.engine.defaults import DefaultPredictor
from detectron2.data import MetadataCatalog

import numpy as np
from PIL import Image
from typing import List, Tuple
import json


"""outputs["instances"]
   members: _image_size 2tuple, pred_boxes 4tuple,
   scores n-tuple, pred_classes n-tuple, pred_masks n-list of images"""


def get_detectron_categories(predictor, intermediate_outputs, detectron_instances: IntermediateOutputs) -> np.ndarray:
    """

    Args:
        predictor: detectron predictor(cfg)
        intermediate_outputs: list of images to plug into detectron2 again
        detectron_instances: list of detectron output instances corresponding to 'intermediate_outputs'

    Returns:
        List[List[probabilities]]
    """
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
    return np.array(detectron_categories)


def image2intermediate_data(predictor, image_arr) -> IntermediateData:
    """

    Args:
        predictor: detectron predictor(cfg)
        image_arr: raw image to convert to IntermediateData

    Returns:
        IntermediateData for one image
    """
    im_names: List[str] = list()
    inter_outputs: List[IntermediateOutput] = list()
    mobile_inputs: List[IntermediateInput] = list()
    detectron_inputs: List[IntermediateInput] = list()

    output = predictor(image_arr)
    output = output["instances"].to("cpu")
    inter_output, mobile_input, detectron_input = detectron2_output_to_mobile_input(image_arr, output)  # this converts BGR to RGB
    # save data
    im_names.append("image_from_array")
    mobile_inputs.append(mobile_input)
    detectron_inputs.append(detectron_input)
    inter_outputs.append(inter_output)

    # format data
    outputs = IntermediateOutputs(inter_outputs)
    inputs = IntermediateInputs(mobile_inputs, detectron_inputs)
    intermediate_data = IntermediateData(im_names, outputs, inputs)
    return intermediate_data


def image_files2intermediate_data(predictor, image_names) -> IntermediateData:
    """

    Args:
        predictor: detectron predictor(cfg)
        image_names: list of image file paths

    Returns:
        IntermediateData for one image
    """
    im_names: List[str] = list()
    inter_outputs: List[IntermediateOutput] = list()
    mobile_inputs: List[IntermediateInput] = list()
    detectron_inputs: List[IntermediateInput] = list()

    for cnt, im_name in enumerate(image_names):
        # predict
        im = cv2.imread(im_name)
        output = predictor(im)
        output = output["instances"].to("cpu")
        inter_output, mobile_input, detectron_input = detectron2_output_to_mobile_input(im, output)  # this converts BGR to RGB
        # save data
        im_names.append(im_name)
        mobile_inputs.append(mobile_input)
        detectron_inputs.append(detectron_input)
        inter_outputs.append(inter_output)

    # format data
    outputs = IntermediateOutputs(inter_outputs)
    inputs = IntermediateInputs(mobile_inputs, detectron_inputs)
    intermediate_data = IntermediateData(im_names, outputs, inputs)
    return intermediate_data


def detectron2_output_to_mobile_input(im, output, padding=50, outsize=362, max_deformation=3.5):
    # Boxes.tensor: (x1, y1, x2, y2)
    # outputs = outputs["instances"].to("cpu")
    # outputs should be in this format /\
    assert im.shape[2] == 3, "Image colour channels must be last"
    material_input = list()
    category_input = list()
    selected_detectron_outputs = list()
    height, width = im.shape[0:2]
    for i in range(len(output.pred_boxes)):
        # ORIGINAL detectron2 instance bboxes
        x1o, y1o, x2o, y2o = pred_box_to_bounding_box(output.pred_boxes[i])
        # category classification detectron input
        x1p, y1p, x2p, y2p = get_padded_bbox(padding, width, height, x1o, y1o, x2o, y2o)
        # square image for mobile net:
        x1s, y1s, x2s, y2s = extend_to_square(width, height, x1p, y1p, x2p, y2p)
        dx = x2p - x1p
        dy = y2p - y1p
        deformation = calculate_deformation(dx, dy, outsize)
        if deformation < max_deformation:
            # MOBILE NET input:
            square_image = im[y1s:y2s, x1s:x2s, ::-1]  # cut out instance bounding box, bgr => rgb
            resized_image = cv2.resize(square_image, dsize=(outsize, outsize), interpolation=cv2.INTER_CUBIC)
            # save next stage inputs
            category_input.append(im[y1p:y2p, x1p:x2p, ::-1])  # padded bbox
            material_input.append(resized_image)               # square cutout
            # save selected detectron output
            # print(output)
            simple_instance = SimpleInstance(int(output.pred_classes[i]), float(output.scores[i]), output.pred_boxes[i].tensor.numpy()[0])
            # print(simple_instance)
            selected_detectron_outputs.append(simple_instance)

    intermediate_output = IntermediateOutput(selected_detectron_outputs)
    intermediate_input = IntermediateInput(material_input)
    intermediate_input2 = IntermediateInput(category_input)
    return intermediate_output, intermediate_input, intermediate_input2


def calculate_deformation(dx, dy, new_side):
    return round(float(deformation_score(dx, dy, new_side*new_side)), 2)


def get_mobilenet_input(im, output, padding=30, outsize=362, max_deformation=3):
    """outputs["instances"]
       members: _image_size 2tuple, pred_boxes.tensor 4tuple,
       scores n-tuple, pred_classes n-tuple, pred_masks n-list of images"""
    # Boxes.tensor: (x1, y1, x2, y2)
    # outputs = outputs["instances"].to("cpu")
    # outputs should be in this format /\
    assert im.shape[2] == 3, "Image colour channels must be last"
    mobile_net_images = list()
    height, width = im.shape[0:2]
    square_ims = list()
    deformations = list()
    scores = list()
    """
    box1 = (xmin1, xmax1)
    box2 = (xmin2, xmax2)
    isOverlapping1D(box1, box2) = xmax1 >= xmin2 and xmax2 >= xmin1

    box1 = (x:(xmin1, xmax1), y:(ymin1, ymax1))
    box2 = (x:(xmin2, xmax2), y:(ymin2, ymax2))
    isOverlapping2D(box1, box2) = isOverlapping1D(box1.x, box2.x) and
    isOverlapping1D(box1.y, box2.y)
    """
    for i in range(len(output.pred_masks)):
        x1, y1, x2, y2 = pred_box_to_bounding_box(output.pred_boxes[i])
        x1, y1, x2, y2 = get_padded_extended_bbox(padding, width, height, x1, y1, x2, y2)
        dx = x2-x1
        dy = y2-y1
        deformation = round(float(deformation_score(dx, dy, outsize*outsize)), 2)
        print(output.pred_classes[i], "gonna get saved:", deformation < max_deformation)
        if deformation < max_deformation:
            sub_image = im[y1:y2, x1:x2, ::-1]  # cut out instance bounding box, bgr => rgb
            resized_image = cv2.resize(sub_image, dsize=(outsize, outsize), interpolation=cv2.INTER_CUBIC)
            square_ims.append(resized_image)
            deformations.append(deformation)
    return square_ims, deformations


def detectron2_output_2_mask(output):
    """outputs["instances"]
       members: _image_size 2tuple, pred_boxes 4tuple,
       scores n-tuple, pred_classes n-tuple, pred_masks n-list of images"""
    (pred_num, height, width, channels) = (len(output.pred_masks), *output._image_size, 3)
    out_shape = (pred_num, height, width, channels)
    out_ims = np.zeros(out_shape, dtype=np.uint8)
    for i in range(len(output.pred_masks)):
        out_im = out_ims[i]
        for j in range(len(out_im)):
            for k in range(len(out_im[0])):
                out_im[j, k] = np.array(list(255 if output.pred_masks[i, j, k] else 0 for _ in range(3)))
    return out_ims


def save_interstage_io(predictor, image_names):
    for cnt, im_name in enumerate(image_names):
        # path = pred_dir + '/' + im_name
        im = cv2.imread(im_name)
        output = predictor(im)
        output = output["instances"].to("cpu")
        # """outputs["instances"]
        #    members: _image_size 2tuple, pred_boxes 4tuple,
        #    scores n-tuple, pred_classes n-tuple, pred_masks n-list of images"""
        square_outputs, deformations = get_mobilenet_input(im, output)
        for i, sq in enumerate(square_outputs):
            out_im = Image.fromarray(sq)
            out_im.save("im-{}-bbox_vis-{}-deform-{}.png".format(cnt, i, deformations[i]))
            print("saved im-{}-bbox_vis-{}-deform-{}.png".format(cnt, i, deformations[i]))
        masks = detectron2_output_2_mask(output)
        for i, mask in enumerate(masks):
            im = Image.fromarray(mask)
            im.save("im-{}-mask-{}.png".format(cnt, i))
            print("saved im-{}-mask-{}.png".format(cnt, i))

        print("visualized interstage for image",str(cnt)+"/"+str(len(image_names)))


def deformation_score(w1, h1, w2h2):
    # w1=100, h1=400, w2=362, h2=362 =>
    # w1=362, h1=1448, w2=362, h2=362 =>
    # (w2h2 / (w1 * h1)) * max(w1 / h1, h1 / w1) = 0.25*4 = 1
    return (w2h2 / (w1 * h1)) * max(w1 / h1, h1 / w1)


def get_padded_extended_bbox(padding, w, h, x1, y1, x2, y2):
    return extend_to_square(w, h, *get_padded_bbox(padding, w, h, x1, y1, x2, y2))


def get_padded_bbox(padding, w, h, x1, y1, x2, y2):
    return max(x1-padding, 0), max(y1-padding, 0), min(x2+padding, w), min(y2+padding, h)


def extend_to_square(w, h, x1, y1, x2, y2):
    dx = x2-x1
    dy = y2-y1
    if dy > dx:
        x1 = max(x1 - (dy-dx)//2, 0)
        x2 = min(x2 + (dy-dx)//2, w)
    else:
        y1 = max(y1 - (dx-dy)//2, 0)
        y2 = min(y2 + (dx-dy)//2, h)
    return x1, y1, x2, y2


def visualize_bounding_boxes(im, outputs):
    assert im.shape[2] == 3, "Image colour channels must be last"
    for i in range(len(outputs.pred_masks)):
        x1, y1, x2, y2 = pred_box_to_bounding_box(outputs.pred_boxes[i])
        sub_image = im[y1:y2, x1:x2, ::-1]  # cut out instance bounding box, bgr => rgb
        out_im = Image.fromarray(sub_image)
        out_im.save("bbox_vis-{}-{}_{}_{}_{}.png".format(i, x1, y1, x2, y2))


def get_sub_images_detectron2(im, outputs):
    assert im.shape[2] == 3, "Image colour channels must be last"
    ret = list()
    for i in range(len(outputs.pred_masks)):
        x1, y1, x2, y2 = outputs.pred_boxes[i].tensor.int()[0]
        sub_image = im[y1:y2, x1:x2, ::-1]  # cut out instance bounding box, bgr => rgb
        ret.append(sub_image)
    return ret


def pred_box_to_bounding_box(pred_box):
    x1, y1, x2, y2 = pred_box.tensor.int()[0]
    return x1, y1, x2, y2


def main():
    with open("/local/temporary/DATASET/info/id_to_OWN.json") as json_labels:
        new_dict = json.load(json_labels)

    # PATHS SETUP
    trn_data = '/local/temporary/DATASET/TRAIN'
    pred_dir = "../images_input"
    img_output = "images_output"

    # video is having problems :/
    vid_dir = "videos"
    output = "videos_output"

    # lists all images to image_names list from the dictionary using os.walk
    image_names = []
    for (dirpath, dirnames, filenames) in os.walk(pred_dir):
        image_names.extend(filenames)
        break

    video_names = []
    for (dirpath, dirnames, filenames) in os.walk(vid_dir):
        video_names.extend(filenames)
        break

    # orders annotations for the METADATA
    ordered_list_of_names = []
    for i in range(len(new_dict)):
        ordered_list_of_names.append(new_dict[str(i)])

    ycb_metadata = MetadataCatalog.get(trn_data)

    # load annotations from any of the datasets (also train.data or val.data should work)
    def get_test_dict():
        with open("/local/temporary/DATASET/test.data", 'rb') as data:
            data = pickle.load(data)
        return data

    # choose the certainity threshold
    THRESHOLD = 0.6

    # translate threshold into a text form
    buff = ""
    for c in str(THRESHOLD):
        if c == '.':
            c = '_'
        buff += c
    THRESHOLD_TXT = buff

    cfg = setup()
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    # print(cfg.MODEL.WEIGHTS)
    # exit()
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = THRESHOLD
    cfg.DATASETS.TEST = (
    "/local/temporary/DATASET/TRAIN",)  # must be a tuple, for the inner workings of detectron2 (stupid, we know :/)
    predictor = DefaultPredictor(cfg)

    # testing image cutouts
    # im = cv2.imread(pred_dir + '/' + image_names[0])  # is numpy.ndarray
    # outputs = predictor(im)
    # detectron2_output_to_mobile_input(im, outputs)
    real_image_names = [pred_dir+"/"+im for im in image_names]
    # save_separate_masks(predictor, real_image_names)
    save_interstage_io(predictor, real_image_names)

    # # # ------------IMAGE PREDICTION-------------------
    # for cnt, im_name in enumerate(image_names):
    #     path = pred_dir + '/' + im_name
    #     im = cv2.imread(path)
    #     output = predictor(im)
    #     output = outputs["instances"].to("cpu")
    #     detectron2_output_to_mobile_input(im, output)


if __name__ == "__main__":
    main()
