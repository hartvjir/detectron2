#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# customized by Kruzliak Andrej, Pliska Michal and Hartvich Jiri
import os
import json
import pickle
import logging
import pycocotools
import torch, torchvision
from collections import OrderedDict

import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.model_zoo import model_zoo
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.solver import get_default_optimizer_params, build_lr_scheduler
from detectron2.structures import BoxMode
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    verify_results,
)

#---------------------------------------------

# TRAINER SETUP
class MegatronTrainer(DefaultTrainer):
    """
    Custom, based on Panoptic-DeepLab
    """
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):

        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        # evaluator = COCOEvaluator("DATASET/TESTS", cfg, False, output_dir="./output/")

        return COCOEvaluator(val_data, cfg, False, output_dir="./output/")

    # @classmethod
    # def build_train_loader(cls, cfg):
    #     mapper = PanopticDeeplabDatasetMapper(cfg, augmentations=build_sem_seg_train_aug(cfg))
    #     return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Build an optimizer from config.
        """
        params = get_default_optimizer_params(
            model,
            base_lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
            bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
            weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
        )

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            return maybe_add_gradient_clipping(cfg, torch.optim.SGD)(
                params,
                cfg.SOLVER.BASE_LR,
                momentum=cfg.SOLVER.MOMENTUM,
                nesterov=cfg.SOLVER.NESTEROV,
            )
        elif optimizer_type == "ADAM":
            return maybe_add_gradient_clipping(cfg, torch.optim.Adam)(params, cfg.SOLVER.BASE_LR)
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")

# ---------------------------------------------


# PREPARE DICTIONARIES
with open("ipalm/config/id_to_OWN.json") as json_labels:
    new_dict = json.load(json_labels)

ordered_list_of_names = []
for i in range(len(new_dict)):
    ordered_list_of_names.append(new_dict[str(i)])


def get_train_dict():
    with open("/local/temporary/DATASET/train.data", 'rb') as data:
        data = pickle.load(data)
    return data


def get_val_dict():
    with open("/local/temporary/DATASET/val.data", 'rb') as data:
        data = pickle.load(data)
    return data

# ---------------------------------------------


# REGISTER DATASETS
trn_data = '/local/temporary/DATASET/TRAIN'
val_data = '/local/temporary/DATASET/VAL'

DatasetCatalog.register(trn_data, lambda: get_train_dict())
MetadataCatalog.get(trn_data).thing_classes = ordered_list_of_names

DatasetCatalog.register(val_data, lambda: get_val_dict())
MetadataCatalog.get(val_data).thing_classes = ordered_list_of_names
ycb_metadata = MetadataCatalog.get(trn_data)


#---------------------------------------------

def setup():
    """
    Create configs and perform basic setups.
    Use our weights 'model_final.pth' or start training
    from the baselines from the model_zoo below
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")) # not exaclty needed...

    # DATASET
    cfg.DATASETS.TRAIN = (trn_data, )
    cfg.DATASETS.TEST = (val_data, )
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.INPUT.MASK_FORMAT = "bitmask"

    # MODEL
    cfg.MODEL.WEIGHTS = "output/model_final.pth"
    #cfg.MODEL.WEIGHTS = "output/model_0004999.pth"
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
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

def main(args):
    """
    BEWARE:

    After changing the number of classes, certain layers in a pre-trained model
    will become incompatible and therefore cannot be loaded to the new model.
    This is EXPECTED, and loading such pre-trained models will produce warnings
    about such layers.

    """
    cfg = setup()
    Trainer = MegatronTrainer

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
