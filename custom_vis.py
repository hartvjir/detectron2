import os
import cv2
import tqdm
import json
import random
import pickle
import logging
import detectron2
import pycocotools
import torch, torchvision
from collections import OrderedDict
from train import setup
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.utils.logger import setup_logger
from detectron2.model_zoo import model_zoo
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from predictor import VisualizationDemo
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)


with open("/local/temporary/DATASET/info/id_to_OWN.json") as json_labels:
    new_dict = json.load(json_labels)

# PATHS SETUP
trn_data = '/local/temporary/DATASET/TRAIN'
pred_dir = "images_input"
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
    buff+=c
THRESHOLD_TXT = buff

cfg = setup()
# DetectionCheckpointer(model).load(file_path_or_url) # you can also load your last checkopint with this line, if you havent yet finished the training
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
#cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = THRESHOLD
cfg.DATASETS.TEST = ("/local/temporary/DATASET/TRAIN", )  # must be a tuple, for the inner workings of detectron2 (stupid, we know :/)
predictor = DefaultPredictor(cfg)


# ------------IMAGE PREDICTION-------------------
cnt = 0
for im_name in image_names:
    path = pred_dir + '/' + im_name
    cnt += 1
    im = cv2.imread(path)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   scale=2,
                   metadata=ycb_metadata,
                   #instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
                   )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #print(outputs["instances"].__dict__)
    #out = v.draw_instance_predictions(outputs["instances"])
    #out = v.draw_instance_predictions(outputs["instances"])
    # cv2_imshow(out.get_image()[:, :, ::-1])
    img_name = "{0}-prediction_pred-{1}.png".format(THRESHOLD_TXT,cnt)
    appendix = "_pred-{}.png".format(cnt)
    img_name = THRESHOLD_TXT + im_name + appendix
    img_name = img_output + '/' + img_name
    out.save(img_name)
    if cnt%10 == 0:
        print("Infered Image num: ", cnt)


#------------VIDEO PREDICTION-------------------

"""
Not working yet... :/
uses visualiser.py
"""

# for vid_name in video_names:
#     path = vid_dir + '/' + vid_name
#     video = cv2.VideoCapture(path)
#     width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     frames_per_second = video.get(cv2.CAP_PROP_FPS)
#     num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#     basename = os.path.basename(path)

#     if os.path.isdir(output):
#         output_fname = os.path.join(output, basename)
#         output_fname = THRESHOLD_TXT + "-" + os.path.splitext(output_fname)[0] + ".mkv"
#     else:
#         output_fname = output
#     assert not os.path.isfile(output_fname), output_fname
#     output_file = cv2.VideoWriter(
#         filename=output_fname,
#         # some installation of opencv may not support x264 (due to its license),
#         # you can try other format (e.g. MPEG)
#         # fourcc=cv2.VideoWriter_fourcc(*"avc1"),
#         fourcc=cv2.VideoWriter_fourcc(*"x264"),
#         fps=float(frames_per_second),
#         frameSize=(width, height),
#         isColor=True,
#     )

#     demo = VisualizationDemo(cfg)
#     assert os.path.isfile(path)
#     for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
#         output_file.write(vis_frame)

#     video.release()
#     output_file.release()

# '''
# cd demo/
# python demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml \
#   --video-input version_3/videos/20210102_125741.mp4 \
#   --opts MODEL.WEIGHTS version_3/output/model_0004019.pth

# '''

print("Inference Done. Thanks for detecting with detectron2.")
print("Brought to you by FAIR, edited by Hartvich J, Kruzliak A, Pliska M 2021. <3")
