#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# customized by Kruzliak Andrej, Pliska Michal and Hartvich Jiri
import os
import json

from detectron2.model_zoo import model_zoo
from detectron2.config import get_cfg


# with open("ipalm/id_to_OWN.json") as json_labels:
#     new_dict = json.load(json_labels)


def setup(model_path, id_to_own_path="ipalm/config/id_to_OWN.json"):
    with open(id_to_own_path) as json_labels:
        new_dict = json.load(json_labels)
    """
    Create configs and perform basic setups.
    Use our weights 'model_final.pth' or start training
    from the baselines from the model_zoo below
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")) # not exaclty needed...

    # DATASET
    # cfg.DATASETS.TRAIN = (trn_data, )
    # cfg.DATASETS.TEST = (val_data, )
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.INPUT.MASK_FORMAT = "bitmask"

    # MODEL
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(new_dict)

    # SOLVER
    cfg.SOLVER.IMS_PER_BATCH = 12        # Batchsize basically - must be divisible by the n of GPUs used for training
    cfg.SOLVER.BASE_LR = 0.000025      # base learning rate default was 0.00000025
    cfg.SOLVER.WARMUP_ITERS = 500        # here the LR will start from 0 up to the set LR at the end of warmup
    cfg.SOLVER.MAX_ITER = 8001           # 30000 iterations - detectron2 is iteration based, not epochs- was 5000
    cfg.SOLVER.GAMMA = 0.2               # will produce: BASE_LR * GAMMA after SOLVER.STEPS - default 0.1
    cfg.SOLVER.STEPS = (1000,)           # denotes the interval of iterations, where the BASE_LR will be BASE_LR * GAMMA (lower_bound, upper_bound),-was 1000
    # if upper bound is left blank, this LR will stay till the end
    cfg.SOLVER.CHECKPOINT_PERIOD = 4000  # after 10005 the model will autosave into your output directory
    cfg.SOLVER.NESTEROV = True           # optional for SGD optimizer
    cfg.SOLVER.OPTIMIZER = "ADAM"        # "ADAM" is best choice, but "SGD" with momentum nad NESTEROV may also be a good choice
    cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True # enable rescaling gradient if too large

    cfg.TEST.EVAL_PERIOD = 1000          # mdoel evaluation after EVAL_PERIOD iterations (this practically divides the iterations into epochs)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)


    return cfg

