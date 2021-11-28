import numpy as np
import torch
import torch.nn as nn

import random

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision
from torchvision import transforms

from net.mobilenetv3 import MobileNetV3Large, MobileNetV3Small
# from trainer import Trainer
# import utils


# class MaterialPatchNet(MobileNetV3Large):
#     def __init__(self):
#         super().__init__()


if __name__ == "__main__":
    lrn = nn.LocalResponseNorm(2)
    signal_2d = torch.randn(32, 5, 24, 24)
    signal_4d = torch.randn(16, 5, 7, 7, 7, 7)
    output_2d = lrn(signal_2d)
    output_4d = lrn(signal_4d)